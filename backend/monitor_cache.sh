#!/bin/bash

echo "Cache File Monitor"
echo "=================="
echo ""

FILE="verification_results/20251127_172541_One_genre_many_audiences_A_quarter_century_of_spor.json"

if [ ! -f "$FILE" ]; then
    echo "❌ File not found: $FILE"
    exit 1
fi

echo "📁 File: $(basename $FILE)"
echo "📏 Size: $(ls -lh $FILE | awk '{print $5}')"
echo "🕐 Modified: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" $FILE)"
echo ""

echo "Checking for article_id..."
if grep -q '"article_id"' "$FILE"; then
    ARTICLE_ID=$(grep -o '"article_id": "[^"]*"' "$FILE" | head -1 | cut -d'"' -f4)
    echo "✅ Has article_id: $ARTICLE_ID"
else
    echo "⏳ No article_id yet (verification in progress)"
fi

echo ""
echo "Checking for verification_result..."
if grep -q '"verification_result"' "$FILE"; then
    echo "✅ Has verification_result"
else
    echo "⏳ No verification_result yet"
fi
