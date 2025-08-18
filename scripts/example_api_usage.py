#!/usr/bin/env python3
"""
Example API Usage for DFS Prophet

Demonstrates calling the Player Search endpoints using httpx.
Run the API first:
  uvicorn src.dfs_prophet.main:app --reload

Then run:
  uv run python scripts/example_api_usage.py
"""

import asyncio
import httpx


BASE_URL = "http://localhost:8000/api/v1"


async def main() -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Basic health
        r = await client.get(f"{BASE_URL}/health")
        print("Health:", r.status_code, r.json().get("status"))

        # Binary search demo
        params = {
            "query": "Patrick Mahomes quarterback Kansas City Chiefs",
            "limit": 5,
        }
        r = await client.get(f"{BASE_URL}/players/search/binary", params=params)
        print("Binary search:", r.status_code)
        data = r.json()
        print("Results:", data.get("total_results"), "in", data.get("total_time_ms"), "ms")

        # Comparison
        r = await client.get(f"{BASE_URL}/players/compare", params={"query": params["query"], "limit": 5})
        print("Compare:", r.status_code)
        cmp = r.json()
        print("Speed improvement:", cmp.get("speed_improvement"), "%")


if __name__ == "__main__":
    asyncio.run(main())


