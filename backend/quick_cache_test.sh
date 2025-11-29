#!/bin/bash

echo "Quick Cache Test"
echo "================"
echo ""

# Count files before
BEFORE=$(ls -1 verification_results/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "Files before: $BEFORE"

# Make a test request
echo ""
echo "Making test verification request..."
curl -s -X POST http://localhost:8000/verify-news \
  -H "Content-Type: application/json" \
  -d '{"article_id":"cache_test_'$(date +%s)'","title":"Test cache article","summary":"Testing"}' \
  > /dev/null

# Count files after
AFTER=$(ls -1 verification_results/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "Files after: $AFTER"

if [ "$AFTER" -gt "$BEFORE" ]; then
    echo "✅ New file created!"
    echo ""
    echo "Most recent file:"
    ls -lt verification_results/*.json | head -2 | tail -1
else
    echo "❌ No new file created"
fi
