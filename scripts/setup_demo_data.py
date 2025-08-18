#!/usr/bin/env python3
"""
Setup Demo Data for DFS Prophet

This script automates preparing demo data:
- Load NFL player data using the collector
- Generate embeddings for all players
- Create Qdrant collections with proper configuration
- Load data into both regular and binary quantized collections
- Performance benchmarking and validation
- Progress reporting and error handling

Usage examples:
  uv run python scripts/setup_demo_data.py --seasons 2023 2024 --max-players 1200 --strategy hybrid
  uv run python scripts/setup_demo_data.py --max-players 500 --force-refresh --batch-size 64
"""

import argparse
import asyncio
import sys
import time
from typing import List, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dfs_prophet.utils import get_logger
from dfs_prophet.config import get_settings
from dfs_prophet.core import (
	get_embedding_generator,
	get_vector_engine,
	CollectionType,
)
from dfs_prophet.data.collectors import get_nfl_collector


logger = get_logger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="DFS Prophet Demo Data Setup")
	parser.add_argument(
		"--seasons",
		type=int,
		nargs="*",
		help="Season years to include (e.g., 2022 2023 2024)",
	)
	parser.add_argument(
		"--max-players",
		type=int,
		default=1000,
		help="Maximum number of players to process",
	)
	parser.add_argument(
		"--include-weeks",
		action="store_true",
		help="Collect weekly data instead of season totals",
	)
	parser.add_argument(
		"--force-refresh",
		action="store_true",
		help="Bypass cached NFL data and refetch",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=None,
		help="Batch size for embedding generation and upserts (defaults to config)",
	)
	parser.add_argument(
		"--strategy",
		choices=["statistical", "contextual", "hybrid", "text_only"],
		default="hybrid",
		help="Embedding strategy to use",
	)
	parser.add_argument(
		"--clear-collections",
		action="store_true",
		help="Clear existing Qdrant collections before loading",
	)
	parser.add_argument(
		"--compare-queries",
		type=int,
		default=10,
		help="Number of search queries to run during performance comparison",
	)
	return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
	settings = get_settings()
	logger.info("Starting demo data setup")
	logger.info(f"Environment: {settings.environment}")
	logger.info(f"Qdrant URL: {settings.get_qdrant_url()}")

	# Initialize engine and collections
	engine = get_vector_engine()
	logger.info("Initializing Qdrant collections ...")
	await engine.initialize_collections()

	if args.clear_collections:
		logger.info("Clearing existing collections as requested ...")
		try:
			await engine.clear_collection(CollectionType.REGULAR)
			await engine.clear_collection(CollectionType.BINARY_QUANTIZED)
		except Exception as e:
			logger.warning(f"Failed to clear collections: {e}")

	# Collect NFL player data
	collector = get_nfl_collector()
	logger.info("Collecting NFL player data ...")
	collect_start = time.time()
	players = await collector.collect_nfl_data(
		seasons=args.seasons or [2023, 2024],
		max_players=args.max_players,
		include_weeks=args.include_weeks or False,
	)
	collect_ms = (time.time() - collect_start) * 1000
	logger.info(f"Collected {len(players)} players in {collect_ms:.2f} ms")

	if not players:
		logger.error("No players collected; aborting.")
		return 2

	# Generate embeddings
	generator = get_embedding_generator()
	strategy_map = {
		"statistical": generator.EmbeddingStrategy.STATISTICAL if hasattr(generator, 'EmbeddingStrategy') else None,
		"contextual": generator.EmbeddingStrategy.CONTEXTUAL if hasattr(generator, 'EmbeddingStrategy') else None,
		"hybrid": generator.EmbeddingStrategy.HYBRID if hasattr(generator, 'EmbeddingStrategy') else None,
		"text_only": generator.EmbeddingStrategy.TEXT_ONLY if hasattr(generator, 'EmbeddingStrategy') else None,
	}
	# Fallback if accessing enum via instance fails: import from module
	try:
		from dfs_prophet.core.embedding_generator import EmbeddingStrategy as _ES
		selected_strategy = {
			"statistical": _ES.STATISTICAL,
			"contextual": _ES.CONTEXTUAL,
			"hybrid": _ES.HYBRID,
			"text_only": _ES.TEXT_ONLY,
		}[args.strategy]
	except Exception:
		selected_strategy = None

	logger.info(f"Generating embeddings with strategy: {args.strategy}")
	emb_start = time.time()
	embeddings = await generator.generate_batch_embeddings(
		players,
		strategy=selected_strategy,
		batch_size=args.batch_size,
	)
	emb_ms = (time.time() - emb_start) * 1000
	logger.info(f"Generated {len(embeddings)} embeddings in {emb_ms:.2f} ms")

	if len(embeddings) != len(players):
		logger.warning("Embeddings count does not match players; will align to shortest length.")
		min_len = min(len(embeddings), len(players))
		players = players[:min_len]
		embeddings = embeddings[:min_len]

	# Upsert into regular collection
	upsert_results = {"regular": 0, "quantized": 0}
	batch_size = args.batch_size or settings.vector_db.batch_size
	logger.info(f"Upserting into regular collection in batches of {batch_size} ...")
	try:
		upsert_regular = await engine.batch_upsert_vectors(
			players, embeddings, CollectionType.REGULAR, batch_size=batch_size
		)
		upsert_results["regular"] = upsert_regular
	except Exception as e:
		logger.warning(f"Batch upsert failed for regular collection: {e}; falling back to single upserts.")
		count = 0
		for p, v in zip(players, embeddings):
			ok = await engine.upsert_player_vector(p, v, CollectionType.REGULAR)
			if ok:
				count += 1
		upsert_results["regular"] = count

	# Upsert into quantized collection
	logger.info(f"Upserting into binary quantized collection in batches of {batch_size} ...")
	try:
		upsert_quant = await engine.batch_upsert_vectors(
			players, embeddings, CollectionType.BINARY_QUANTIZED, batch_size=batch_size
		)
		upsert_results["quantized"] = upsert_quant
	except Exception as e:
		logger.warning(f"Batch upsert failed for quantized collection: {e}; falling back to single upserts.")
		count = 0
		for p, v in zip(players, embeddings):
			ok = await engine.upsert_player_vector(p, v, CollectionType.BINARY_QUANTIZED)
			if ok:
				count += 1
		upsert_results["quantized"] = count

	logger.info(
		f"Upserted regular={upsert_results['regular']} | quantized={upsert_results['quantized']}"
	)

	# Validate collection stats
	reg_stats = await engine.get_collection_stats(CollectionType.REGULAR)
	bin_stats = await engine.get_collection_stats(CollectionType.BINARY_QUANTIZED)
	logger.info(
		f"Regular: points={reg_stats.get('points_count', 0)}, mem={reg_stats.get('memory_usage_mb', 0):.2f}MB; "
		f"Quantized: points={bin_stats.get('points_count', 0)}, mem={bin_stats.get('memory_usage_mb', 0):.2f}MB"
	)

	# Benchmark search performance
	logger.info("Running performance benchmark (search) ...")
	try:
		from dfs_prophet.core.embedding_generator import EmbeddingStrategy
		# Use the first player's hybrid embedding as a query vector proxy
		query_text = f"{players[0].name} {players[0].position.value} {players[0].team.value}"
		query_vector = await generator.generate_query_embedding(query_text)
		comparison = await engine.compare_collections_performance(query_vector, test_queries=args.compare_queries)
		logger.info(
			f"Speed improvement: {comparison.speed_improvement_percent:.1f}% | "
			f"Memory savings: {comparison.memory_savings_percent:.1f}% | "
			f"Compression ratio: {comparison.compression_ratio:.2f}"
		)
	except Exception as e:
		logger.warning(f"Benchmarking failed: {e}")

	logger.info("Demo data setup complete")
	return 0


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	return asyncio.run(main_async(args))


if __name__ == "__main__":
	sys.exit(main())



