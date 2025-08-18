#!/bin/bash

echo "🧹 Cleaning up DFS Prophet project..."

# Remove Python cache files
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove old test files
rm -f test_*.sh 2>/dev/null || true
rm -f debug_*.py 2>/dev/null || true
rm -f regenerate_*.py 2>/dev/null || true

# Clear data cache (optional - uncomment if needed)
# rm -rf data/raw/nfl_cache/* 2>/dev/null || true
# rm -f data/processed/sample_* 2>/dev/null || true

echo "✅ Cleanup complete!"
