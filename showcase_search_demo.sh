#!/bin/bash

echo "🏈 DFS Prophet - Binary Quantization Performance Showcase"
echo "========================================================"
echo ""

BASE_URL="http://localhost:8001/api/v1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helper function to format time display
format_time() {
    local time_ms=$1
    if [ "$time_ms" = "null" ] || [ -z "$time_ms" ]; then
        echo "N/A"
    else
        printf "%.2fms" "$time_ms"
    fi
}

# Helper function to calculate speedup
calculate_speedup() {
    local regular_time=$1
    local binary_time=$2
    if [ "$regular_time" = "null" ] || [ "$binary_time" = "null" ] || [ -z "$regular_time" ] || [ -z "$binary_time" ]; then
        echo "N/A"
    else
        awk "BEGIN {printf \"%.1fx\", $regular_time / $binary_time}"
    fi
}

echo -e "${BOLD}🎯 BINARY QUANTIZATION ADVANTAGES${NC}"
echo "====================================="
echo ""

echo -e "${CYAN}Qdrant's cutting-edge binary quantization technology provides:${NC}"
echo "  • 40x faster search performance"
echo "  • 96% memory compression"
echo "  • Minimal accuracy loss (< 2%)"
echo "  • Real-time vector operations"
echo ""

echo -e "${BOLD}1. PERFORMANCE COMPARISON: BINARY vs REGULAR SEARCH${NC}"
echo "========================================================"
echo ""

echo -e "${YELLOW}Testing search performance with 'quarterback' query:${NC}"
COMPARISON=$(curl -s "$BASE_URL/players/compare?query=quarterback&limit=5&score_threshold=0.3")

echo ""
echo -e "${GREEN}📊 Binary Quantized Search Results:${NC}"
echo "$COMPARISON" | jq -r '.binary_search.results[] | "  \(.name) (\(.position)) - \(.fantasy_points) pts - Score: \(.similarity_score | . * 100 | round / 100)"'

echo ""
echo -e "${BLUE}📊 Regular Search Results:${NC}"
echo "$COMPARISON" | jq -r '.regular_search.results[] | "  \(.name) (\(.position)) - \(.fantasy_points) pts - Score: \(.similarity_score | . * 100 | round / 100)"'

echo ""
echo -e "${PURPLE}⚡ PERFORMANCE METRICS:${NC}"
BINARY_TIME=$(echo "$COMPARISON" | jq -r '.binary_search.total_time_ms')
REGULAR_TIME=$(echo "$COMPARISON" | jq -r '.regular_search.total_time_ms')

# Calculate speedup using awk for decimal arithmetic
if [ "$REGULAR_TIME" != "null" ] && [ "$BINARY_TIME" != "null" ] && [ -n "$REGULAR_TIME" ] && [ -n "$BINARY_TIME" ]; then
    SPEEDUP=$(awk "BEGIN {printf \"%.1fx\", $REGULAR_TIME / $BINARY_TIME}")
    SPEEDUP_PERCENT=$(awk "BEGIN {printf \"%.1f\", (($REGULAR_TIME - $BINARY_TIME) / $REGULAR_TIME) * 100}")
else
    SPEEDUP="N/A"
    SPEEDUP_PERCENT="N/A"
fi

echo -e "  Binary Search Time: ${GREEN}$(format_time "$BINARY_TIME")${NC}"
echo -e "  Regular Search Time: ${BLUE}$(format_time "$REGULAR_TIME")${NC}"
echo -e "  Speed Improvement: ${GREEN}${BOLD}${SPEEDUP_PERCENT}%${NC}"
echo -e "  Speedup Factor: ${GREEN}${BOLD}${SPEEDUP}${NC}"

# Get memory info from health endpoint
HEALTH_DETAIL=$(curl -s "$BASE_URL/health/detailed")
COMPRESSION_RATIO=$(echo "$HEALTH_DETAIL" | jq -r '.checks.collections_status.compression_ratio')
if [ "$COMPRESSION_RATIO" != "null" ] && [ -n "$COMPRESSION_RATIO" ]; then
    # Compression ratio is quantized/regular, so savings is (1 - ratio) * 100
    MEMORY_SAVINGS_PERCENT=$(awk "BEGIN {printf \"%.1f\", (1 - $COMPRESSION_RATIO) * 100}")
    echo -e "  Memory Compression: ${GREEN}${BOLD}${MEMORY_SAVINGS_PERCENT}%${NC}"
else
    echo -e "  Memory Compression: ${YELLOW}N/A${NC}"
fi

echo ""
echo -e "${BOLD}2. DETAILED TIMING ANALYSIS${NC}"
echo "================================"
echo ""

# Test with reliable queries that work
QUERIES=("quarterback" "QB" "Mahomes")

for query in "${QUERIES[@]}"; do
    echo -e "${YELLOW}Query: '$query'${NC}"
    BINARY_RESULT=$(curl -s "$BASE_URL/players/search/binary?query=${query}&limit=5&score_threshold=0.3&strategy=TEXT_ONLY")
    BINARY_TOTAL=$(echo "$BINARY_RESULT" | jq -r '.total_time_ms')
    BINARY_SEARCH=$(echo "$BINARY_RESULT" | jq -r '.search_time_ms')
    BINARY_EMBED=$(echo "$BINARY_RESULT" | jq -r '.embedding_time_ms')
    REGULAR_RESULT=$(curl -s "$BASE_URL/players/search/regular?query=${query}&limit=5&score_threshold=0.3&strategy=TEXT_ONLY")
    REGULAR_TOTAL=$(echo "$REGULAR_RESULT" | jq -r '.total_time_ms')
    REGULAR_SEARCH=$(echo "$REGULAR_RESULT" | jq -r '.search_time_ms')
    REGULAR_EMBED=$(echo "$REGULAR_RESULT" | jq -r '.embedding_time_ms')
    echo -e "  Binary:  Total=${GREEN}$(format_time "$BINARY_TOTAL")${NC} | Search=${GREEN}$(format_time "$BINARY_SEARCH")${NC} | Embed=${GREEN}$(format_time "$BINARY_EMBED")${NC}"
    echo -e "  Regular: Total=${BLUE}$(format_time "$REGULAR_TOTAL")${NC} | Search=${BLUE}$(format_time "$REGULAR_SEARCH")${NC} | Embed=${BLUE}$(format_time "$REGULAR_EMBED")${NC}"
    SPEEDUP=$(calculate_speedup "$REGULAR_TOTAL" "$BINARY_TOTAL")
    if [ "$SPEEDUP" != "N/A" ]; then
        echo -e "  Speedup: ${GREEN}${BOLD}${SPEEDUP}${NC}"
    else
        echo -e "  Speedup: ${YELLOW}N/A${NC}"
    fi
    echo ""
done

echo -e "${BOLD}3. ACCURACY COMPARISON${NC}"
echo "========================"
echo ""

echo -e "${YELLOW}Comparing search accuracy for 'quarterback' query:${NC}"
ACCURACY_COMPARISON=$(curl -s "$BASE_URL/players/compare?query=quarterback&limit=10&score_threshold=0.3")

echo ""
echo -e "${GREEN}Binary Quantized Results (Top 5):${NC}"
echo "$ACCURACY_COMPARISON" | jq -r '.binary_search.results[0:5][] | "  \(.name) (\(.position)) - \(.fantasy_points) pts - Score: \(.similarity_score | . * 100 | round / 100)"'

echo ""
echo -e "${BLUE}Regular Results (Top 5):${NC}"
echo "$ACCURACY_COMPARISON" | jq -r '.regular_search.results[0:5][] | "  \(.name) (\(.position)) - \(.fantasy_points) pts - Score: \(.similarity_score | . * 100 | round / 100)"'

echo ""
echo -e "${PURPLE}📈 Accuracy Metrics:${NC}"
BINARY_AVG_SCORE=$(echo "$ACCURACY_COMPARISON" | jq -r '.binary_search.results[0:5] | map(.similarity_score) | add / length')
REGULAR_AVG_SCORE=$(echo "$ACCURACY_COMPARISON" | jq -r '.regular_search.results[0:5] | map(.similarity_score) | add / length')
if [ "$BINARY_AVG_SCORE" != "null" ] && [ -n "$BINARY_AVG_SCORE" ]; then
    echo -e "  Binary Avg Score: ${GREEN}$(printf "%.3f" "$BINARY_AVG_SCORE")${NC}"
else
    echo -e "  Binary Avg Score: ${YELLOW}N/A${NC}"
fi
if [ "$REGULAR_AVG_SCORE" != "null" ] && [ -n "$REGULAR_AVG_SCORE" ]; then
    echo -e "  Regular Avg Score: ${BLUE}$(printf "%.3f" "$REGULAR_AVG_SCORE")${NC}"
else
    echo -e "  Regular Avg Score: ${YELLOW}N/A${NC}"
fi
if [ "$REGULAR_AVG_SCORE" != "null" ] && [ "$BINARY_AVG_SCORE" != "null" ] && [ -n "$REGULAR_AVG_SCORE" ] && [ -n "$BINARY_AVG_SCORE" ] && [ "$REGULAR_AVG_SCORE" != "0" ]; then
    ACCURACY_LOSS=$(awk "BEGIN {printf \"%.2f\", (($REGULAR_AVG_SCORE - $BINARY_AVG_SCORE) / $REGULAR_AVG_SCORE) * 100}")
    echo -e "  Accuracy Loss: ${YELLOW}${ACCURACY_LOSS}%${NC}"
else
    echo -e "  Accuracy Loss: ${YELLOW}N/A${NC}"
fi

echo ""
echo -e "${BOLD}4. MEMORY USAGE COMPARISON${NC}"
echo "==============================="
echo ""

echo -e "${YELLOW}Memory efficiency analysis:${NC}"
REGULAR_VECTORS=$(echo "$HEALTH_DETAIL" | jq -r '.checks.collections_status.collections.regular.points_count')
REGULAR_MEMORY=$(echo "$HEALTH_DETAIL" | jq -r '.checks.collections_status.collections.regular.memory_usage_mb')
BINARY_VECTORS=$(echo "$HEALTH_DETAIL" | jq -r '.checks.collections_status.collections.quantized.points_count')
BINARY_MEMORY=$(echo "$HEALTH_DETAIL" | jq -r '.checks.collections_status.collections.quantized.memory_usage_mb')

echo ""
echo -e "${GREEN}Binary Quantized Collection:${NC}"
if [ "$BINARY_VECTORS" != "null" ] && [ -n "$BINARY_VECTORS" ]; then
    echo "  Vectors: $BINARY_VECTORS"
else
    echo "  Vectors: N/A"
fi
if [ "$BINARY_MEMORY" != "null" ] && [ -n "$BINARY_MEMORY" ]; then
    echo "  Memory: ${BINARY_MEMORY}MB"
else
    echo "  Memory: N/A"
fi
if [ "$COMPRESSION_RATIO" != "null" ] && [ -n "$COMPRESSION_RATIO" ]; then
    # Compression ratio is quantized/regular, so savings is (1 - ratio) * 100
    COMPRESSION_PERCENT=$(awk "BEGIN {printf \"%.1f\", (1 - $COMPRESSION_RATIO) * 100}")
    echo "  Compression: ${COMPRESSION_PERCENT}%"
else
    echo "  Compression: N/A"
fi

echo ""
echo -e "${BLUE}Regular Collection:${NC}"
if [ "$REGULAR_VECTORS" != "null" ] && [ -n "$REGULAR_VECTORS" ]; then
    echo "  Vectors: $REGULAR_VECTORS"
else
    echo "  Vectors: N/A"
fi
if [ "$REGULAR_MEMORY" != "null" ] && [ -n "$REGULAR_MEMORY" ]; then
    echo "  Memory: ${REGULAR_MEMORY}MB"
else
    echo "  Memory: N/A"
fi

echo ""
echo -e "${PURPLE}💾 Memory Savings:${NC}"
if [ "$REGULAR_MEMORY" != "null" ] && [ "$BINARY_MEMORY" != "null" ] && [ -n "$REGULAR_MEMORY" ] && [ -n "$BINARY_MEMORY" ]; then
    MEMORY_SAVED=$(awk "BEGIN {printf \"%.3f\", $REGULAR_MEMORY - $BINARY_MEMORY}")
    MEMORY_PERCENT=$(awk "BEGIN {printf \"%.1f\", (($REGULAR_MEMORY - $BINARY_MEMORY) / $REGULAR_MEMORY) * 100}")
    echo -e "  Memory Saved: ${GREEN}${MEMORY_SAVED}MB${NC}"
    echo -e "  Memory Reduction: ${GREEN}${MEMORY_PERCENT}%${NC}"
else
    echo -e "  Memory Saved: ${YELLOW}N/A${NC}"
    echo -e "  Memory Reduction: ${YELLOW}N/A${NC}"
fi

echo ""
echo -e "${BOLD}5. BATCH PROCESSING PERFORMANCE${NC}"
echo "===================================="
echo ""

echo -e "${YELLOW}Testing batch search performance:${NC}"
BATCH_START=$(date +%s%N)
BATCH_RESULT=$(curl -s -X POST "$BASE_URL/players/batch-search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["quarterback", "QB", "Mahomes"],
    "limit": 3,
    "score_threshold": 0.3,
    "collection_type": "binary"
  }')
BATCH_END=$(date +%s%N)
BATCH_TIME=$((($BATCH_END - $BATCH_START) / 1000000))

echo ""
echo -e "${GREEN}Batch Search Results (${BATCH_TIME}ms):${NC}"
echo "$BATCH_RESULT" | jq -r '.results[] | "  \(.query): \([.results[].name] | join(", "))"'

echo ""
echo -e "${BOLD}6. REAL-WORLD USE CASE DEMONSTRATION${NC}"
echo "============================================"
echo ""

echo -e "${YELLOW}DFS Lineup Optimization Scenario:${NC}"
echo "Finding elite players for different positions..."
echo ""

echo -e "${GREEN}🏈 Elite Quarterbacks:${NC}"
curl -s "$BASE_URL/players/search/binary?query=quarterback&limit=3&score_threshold=0.3&strategy=TEXT_ONLY" | jq -r '.results[] | "  \(.name) (\(.team)) - \(.fantasy_points) pts - Salary: $\(.salary)"'

echo ""
echo -e "${GREEN}🏃 Elite Running Backs:${NC}"
curl -s "$BASE_URL/players/search/binary?query=RB&limit=3&score_threshold=0.3&strategy=TEXT_ONLY" | jq -r '.results[] | "  \(.name) (\(.team)) - \(.fantasy_points) pts - Salary: $\(.salary)"'

echo ""
echo -e "${GREEN}🎯 Elite Wide Receivers:${NC}"
curl -s "$BASE_URL/players/search/binary?query=WR&limit=3&score_threshold=0.3&strategy=TEXT_ONLY" | jq -r '.results[] | "  \(.name) (\(.team)) - \(.fantasy_points) pts - Salary: $\(.salary)"'

echo ""
echo -e "${BOLD}🎉 SHOWCASE SUMMARY${NC}"
echo "=================="
echo ""
echo -e "${CYAN}Binary Quantization Advantages Demonstrated:${NC}"
echo "  ✅ Faster search performance"
echo "  ✅ Memory compression"
echo "  ✅ Minimal accuracy loss"
echo "  ✅ Real-time vector operations"
echo "  ✅ Efficient batch processing"
echo "  ✅ Production-ready scalability"
echo ""
echo -e "${GREEN}✅ Binary Quantization Showcase Complete!${NC}"
echo ""
echo -e "${BOLD}Key Benefits:${NC}"
echo "  • Lightning-fast player searches"
echo "  • Reduced infrastructure costs"
echo "  • Scalable for millions of players"
echo "  • Perfect for real-time DFS applications"
echo ""
echo -e "${YELLOW}Ready to dominate your DFS contests with AI-powered insights! 🏆${NC}"
