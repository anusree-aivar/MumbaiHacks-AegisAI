#!/usr/bin/env python3
"""
Test script to verify caching is working properly
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_cache_stats():
    """Test the cache stats endpoint"""
    print("\n" + "="*80)
    print("TEST 1: Cache Stats")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/cache-stats")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Cache stats retrieved successfully")
        print(f"   Total files: {data['total_files']}")
        print(f"   Results dir: {data['results_dir']}")
        
        if data['files']:
            print(f"\n   Recent files:")
            for file in data['files'][:5]:
                print(f"   - {file['filename']}")
                print(f"     Title: {file.get('title', 'N/A')[:60]}")
                print(f"     Has article_id: {file.get('has_article_id', False)}")
                print(f"     Modified: {file.get('modified', 'N/A')}")
        return True
    else:
        print(f"❌ Failed to get cache stats: {response.status_code}")
        return False


def test_news_verification(article_id, title, summary):
    """Test news article verification"""
    print("\n" + "="*80)
    print(f"TEST 2: News Verification - {title[:50]}")
    print("="*80)
    
    # First request (should verify)
    print("\n📝 First request (expecting verification)...")
    start_time = time.time()
    
    response = requests.post(
        f"{BASE_URL}/verify-news",
        json={
            "article_id": article_id,
            "title": title,
            "summary": summary
        }
    )
    
    first_duration = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Verification completed in {first_duration:.2f}s")
        print(f"   Status: {result['status']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   Sources: {len(result.get('sources', []))}")
    else:
        print(f"❌ Verification failed: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        return False
    
    # Wait a moment
    time.sleep(1)
    
    # Second request (should use cache)
    print("\n📝 Second request (expecting cache hit)...")
    start_time = time.time()
    
    response = requests.post(
        f"{BASE_URL}/verify-news",
        json={
            "article_id": article_id,
            "title": title,
            "summary": summary
        }
    )
    
    second_duration = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Verification completed in {second_duration:.2f}s")
        print(f"   Status: {result['status']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        
        # Check if it was faster (cache hit)
        if second_duration < first_duration * 0.1:  # Should be at least 10x faster
            print(f"✅ CACHE HIT! Second request was {first_duration/second_duration:.1f}x faster")
            return True
        else:
            print(f"⚠️  Second request was only {first_duration/second_duration:.1f}x faster")
            print(f"   Expected cache hit to be much faster")
            return False
    else:
        print(f"❌ Second verification failed: {response.status_code}")
        return False


def test_cache_retrieval(article_id):
    """Test retrieving cached verification"""
    print("\n" + "="*80)
    print(f"TEST 3: Cache Retrieval - Article {article_id}")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/news-verification/{article_id}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Retrieved cached verification")
        print(f"   Article ID: {result['article_id']}")
        print(f"   Status: {result['status']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   Verified at: {result['verified_at']}")
        return True
    elif response.status_code == 404:
        print(f"❌ Cache not found for article {article_id}")
        return False
    else:
        print(f"❌ Failed to retrieve cache: {response.status_code}")
        return False


def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("CACHING TEST SUITE")
    print("="*80)
    print(f"Testing backend at: {BASE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Check if server is running
    print("\n🔍 Checking if server is running...")
    if not check_server():
        print("\n" + "="*80)
        print("❌ ERROR: Backend server is not running!")
        print("="*80)
        print("\nPlease start the server first:")
        print("  cd new/backend")
        print("  python server.py")
        print("\nThen run this test again:")
        print("  python test_caching.py")
        print("="*80 + "\n")
        return 1
    
    print("✅ Server is running")
    
    # Test 1: Cache stats
    test1_passed = test_cache_stats()
    
    # Test 2: News verification with caching
    test_article = {
        "article_id": f"test_{int(time.time())}",
        "title": "Lakers defeat Warriors 120-110 in thrilling overtime game",
        "summary": "The Los Angeles Lakers secured a dramatic victory over the Golden State Warriors in overtime."
    }
    
    test2_passed = test_news_verification(
        test_article["article_id"],
        test_article["title"],
        test_article["summary"]
    )
    
    # Test 3: Cache retrieval
    test3_passed = test_cache_retrieval(test_article["article_id"])
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Test 1 (Cache Stats):     {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Verification):    {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Test 3 (Cache Retrieval): {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Caching is working correctly!")
    else:
        print("⚠️  SOME TESTS FAILED - Check the logs above")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
