import os
import json
import requests
import time
import asyncio
from datetime import datetime

# Import the REAL verification system
# We need to add the current directory to sys.path to import local modules
import sys
sys.path.append(os.path.dirname(__file__))
from search import SportsMisinformationDetector

# Load environment variables
# load_dotenv() - Removed, using shell exports instead

API_KEY = os.getenv("SCRAPINGDOG_API_KEY")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "articles.json")

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created directory: {DATA_DIR}")

def fetch_articles(count=30):
    print(f"📰 Fetching {count} articles from ScrapingDog...")
    
    if not API_KEY:
        print("❌ Error: SCRAPINGDOG_API_KEY not found in .env")
        return []

    url = "https://api.scrapingdog.com/google_news"
    all_articles = []
    
    # Fallback images
    fallbacks = [
        "https://images.unsplash.com/photo-1546519638-68e109498ffc?w=800", # Basketball
        "https://images.unsplash.com/photo-1574629810360-7efbbe195018?w=800", # Soccer
        "https://images.unsplash.com/photo-1530549387789-4c1017266635?w=800", # Swimming
        "https://images.unsplash.com/photo-1579952363873-27f3bade9f55?w=800", # Football
        "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=800", # Running
        "https://images.unsplash.com/photo-1531415074968-036ba1b575da?w=800", # Cricket
        "https://images.unsplash.com/photo-1517649763962-0c623066013b?w=800", # CrossFit/Gym
        "https://images.unsplash.com/photo-1599058945522-28d584b6f0ff?w=800", # American Football
        "https://images.unsplash.com/photo-1560272564-c83b66b1ad12?w=800", # Soccer Action
        "https://images.unsplash.com/photo-1519861531473-920026393112?w=800", # Basketball Action
        "https://images.unsplash.com/photo-1624526267942-ab4491853743?w=800", # Tennis
        "https://images.unsplash.com/photo-1552674605-db6ffd4facb5?w=800", # Running Track
    ]

    params = {
        "api_key": API_KEY,
        "query": "sports",
        "country": "us",
        "results": count
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return []

        data = response.json()
        raw_articles = data.get("news_results", [])
        print(f"✅ Received {len(raw_articles)} raw articles")

        for idx, item in enumerate(raw_articles, 1):
            # Handle source
            source_raw = item.get("source", "Unknown Source")
            source_name = source_raw.get("name", "Unknown Source") if isinstance(source_raw, dict) else str(source_raw)

            # Handle image
            image_url = item.get("thumbnail") or item.get("image") or item.get("img")
            if not image_url:
                image_url = fallbacks[idx % len(fallbacks)]

            article = {
                "id": f"art_{int(time.time())}_{idx}",
                "title": item.get("title", "No Title"),
                "description": item.get("description", "No description available."),
                "source": source_name,
                "publishedAt": item.get("date") or datetime.now().isoformat(),
                "url": item.get("link", "#"),
                "imageUrl": image_url,
                "verified": False
            }
            all_articles.append(article)

    except Exception as e:
        print(f"❌ Exception fetching articles: {e}")

    return all_articles

def verify_articles_real(articles):
    """Run REAL verification on articles"""
    print("🚀 Initializing AI Agents for verification...")
    detector = SportsMisinformationDetector()
    
    verified_articles = []
    total = len(articles)
    
    for idx, article in enumerate(articles, 1):
        print(f"\n[{idx}/{total}] Verifying: {article['title'][:50]}...")
        
        try:
            start_time = time.time()
            # Run the real verification pipeline (SYNC, not async)
            full_result = detector.verify_claim(article['title'])
            duration = time.time() - start_time
            
            # Transform complex result to frontend format
            evaluation = full_result.get("evaluation", {})
            sub_claims = full_result.get("sub_claim_results", [])
            
            # Calculate total sources
            total_sources = 0
            transformed_atomic_claims = []
            
            for sc in sub_claims:
                snippets = sc.get("snippets", [])
                total_sources += len(snippets)
                
                transformed_atomic_claims.append({
                    "claim": sc.get("statement", ""),
                    "verdict": sc.get("verdict", "UNKNOWN"), # Note: sub-claims might not have explicit verdict in all versions, fallback needed
                    "supporting_count": sc.get("support_count", 0),
                    "contradicting_count": sc.get("contradict_count", 0),
                    "sources": [
                        {
                            "url": s.get("url", ""),
                            "domain": s.get("domain", ""),
                            "title": s.get("title", ""),
                            "snippet": s.get("snippet", ""),
                            "classification": s.get("classification", "IRRELEVANT")
                        }
                        for s in snippets
                    ]
                })

            verification_result = {
                "final_verdict": evaluation.get("final_verdict", "UNVERIFIED"),
                "confidence_score": evaluation.get("confidence_score", 0),
                "explanation": evaluation.get("explanation", "No explanation provided."),
                "total_sources": total_sources,
                "verification_time": duration,
                "atomic_claims": transformed_atomic_claims,
                "original_claim": article['title']
            }
            
            # Add result to article
            article["verification_result"] = verification_result
            article["verified"] = verification_result["final_verdict"]
            
            print(f"   ✅ Verdict: {verification_result['final_verdict']} (Confidence: {verification_result['confidence_score']}%)")
            verified_articles.append(article)
            
            # Small delay to be nice to APIs
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ Verification failed: {e}")
            import traceback
            traceback.print_exc()
            # Keep the article but without verification result
            verified_articles.append(article)
            
    return verified_articles

def save_articles(articles):
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({"articles": articles, "last_updated": datetime.now().isoformat()}, f, indent=2)
    
    print(f"\n💾 Saved {len(articles)} articles to {OUTPUT_FILE}")

def main():
    ensure_data_dir()
    
    # 1. Fetch Articles
    articles = fetch_articles(30)  # Now use 30 as requested
    
    if articles:
        # 2. Verify them for real
        verified_articles = verify_articles_real(articles)
        
        # 3. Save to file
        save_articles(verified_articles)
        print("🎉 Seeding complete with REAL data!")
    else:
        print("⚠️ No articles fetched.")

if __name__ == "__main__":
    main()
