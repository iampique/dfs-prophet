"""
DFS Prophet CLI

Common operations:
- setup-demo: Collect data, generate embeddings, and load into Qdrant
- health: Run a quick health check against Qdrant and the embedding pipeline
- clear-cache: Clear NFL collector and embedding caches
"""

import argparse
import asyncio
import sys
from typing import List, Optional

from .utils import get_logger
from .config import get_settings
from .core import get_vector_engine, get_embedding_generator
from .data.collectors import get_nfl_collector


logger = get_logger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dfs-prophet", description="DFS Prophet CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # setup-demo delegates to scripts/setup_demo_data.py recommended path
    demo = sub.add_parser("setup-demo", help="Prepare demo data and load into Qdrant")
    demo.add_argument("--max-players", type=int, default=500)
    demo.add_argument("--seasons", type=int, nargs="*")
    demo.add_argument("--include-weeks", action="store_true")
    demo.add_argument("--batch-size", type=int)
    demo.add_argument("--strategy", choices=["statistical", "contextual", "hybrid", "text_only"], default="hybrid")
    demo.add_argument("--clear-collections", action="store_true")

    # health
    sub.add_parser("health", help="Run a quick system health check")

    # clear-cache
    sub.add_parser("clear-cache", help="Clear NFL and embedding caches")

    return parser.parse_args(argv)


async def cmd_health() -> int:
    settings = get_settings()
    engine = get_vector_engine()
    logger.info("Running health check ...")
    result = await engine.health_check()
    logger.info(f"Health: {result}")
    return 0 if result.get("connection_healthy") else 1


async def cmd_clear_cache() -> int:
    collector = get_nfl_collector()
    removed = collector.clear_cache()
    logger.info(f"Cleared NFL cache files: {removed}")
    gen = get_embedding_generator()
    gen.clear_cache()
    logger.info("Cleared embedding cache")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "setup-demo":
        # Delegate to the existing script to avoid duplication
        import subprocess
        cmd = [sys.executable, "scripts/setup_demo_data.py"]
        if args.max_players:
            cmd += ["--max-players", str(args.max_players)]
        if args.seasons:
            cmd += ["--seasons", *map(str, args.seasons)]
        if args.include_weeks:
            cmd.append("--include-weeks")
        if args.batch_size:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.strategy:
            cmd += ["--strategy", args.strategy]
        if args.clear_collections:
            cmd.append("--clear-collections")
        return subprocess.call(cmd)
    elif args.command == "health":
        return asyncio.run(cmd_health())
    elif args.command == "clear-cache":
        return asyncio.run(cmd_clear_cache())
    else:
        logger.error(f"Unknown command: {args.command}")
        return 2


if __name__ == "__main__":
    sys.exit(main())


