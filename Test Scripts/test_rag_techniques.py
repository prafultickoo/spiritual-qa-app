#!/usr/bin/env python3
"""
Test script for RAG technique integration in Spiritual Q&A app.
Tests all available RAG techniques (Stuff, Refine, Map-Reduce, Map-Rerank, Selective) end-to-end.
"""

import requests
import json
import time
import sys

# Configuration
API_BASE = "http://localhost:60894"
TEST_QUESTION = "What is the meaning of Om in Hindu philosophy?"

# Available RAG techniques to test
RAG_TECHNIQUES = [
    ("stuff", "📦 Stuff (Fast & Cheap)"),
    ("refine", "🔄 Refine (High Quality)"),
    ("map_reduce", "🗺️ Map-Reduce (Parallel)"),
    ("map_rerank", "🏆 Map-Rerank (Best Answer)"),
    ("selective", "🎯 Selective (Smart)")
]

def test_api_health():
    """Test if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API health check passed")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check failed: {str(e)}")
        return False

def test_rag_technique(technique, description):
    """Test a specific RAG technique."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {description}")
    print(f"{'='*60}")
    
    payload = {
        "question": TEST_QUESTION,
        "model": "gpt-4.1",
        "use_mmr": True,
        "k": 5,  # Use fewer chunks for faster testing
        "diversity": 0.3,
        "rag_technique": technique
    }
    
    try:
        start_time = time.time()
        
        print(f"📤 Sending request with RAG technique: {technique}")
        response = requests.post(
            f"{API_BASE}/ask",
            json=payload,
            timeout=60
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            
            print(f"✅ SUCCESS ({duration:.1f}s)")
            print(f"📊 Status: {data.get('status')}")
            print(f"📚 Chunks used: {data.get('chunks_used', 'N/A')}")
            print(f"🤖 Model: {data.get('model', 'N/A')}")
            print(f"📝 Answer length: {len(answer)} chars")
            print(f"🔤 Answer preview: {answer[:200]}...")
            
            # Check if the answer contains spiritual content
            spiritual_keywords = ["Om", "Hindu", "philosophy", "spiritual", "sacred", "divine"]
            keyword_found = any(keyword.lower() in answer.lower() for keyword in spiritual_keywords)
            
            if keyword_found:
                print("✅ Answer contains relevant spiritual content")
            else:
                print("⚠️  Answer may not contain expected spiritual content")
                
            return True, duration, len(answer)
            
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"📥 Response: {response.text}")
            return False, duration, 0
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False, 0, 0

def compare_techniques():
    """Compare all RAG techniques and show results summary."""
    print(f"\n{'='*80}")
    print("📊 RAG TECHNIQUE COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    results = []
    
    for technique, description in RAG_TECHNIQUES:
        success, duration, answer_length = test_rag_technique(technique, description)
        results.append({
            "technique": technique,
            "description": description,
            "success": success,
            "duration": duration,
            "answer_length": answer_length
        })
    
    # Print comparison table
    print(f"\n{'Technique':<12} | {'Status':<7} | {'Time':<8} | {'Length':<8} | Description")
    print("-" * 80)
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration_str = f"{result['duration']:.1f}s" if result["duration"] > 0 else "N/A"
        length_str = f"{result['answer_length']}" if result['answer_length'] > 0 else "N/A"
        
        print(f"{result['technique']:<12} | {status:<7} | {duration_str:<8} | {length_str:<8} | {result['description']}")
    
    # Summary statistics
    successful_tests = sum(1 for r in results if r["success"])
    total_tests = len(results)
    
    print(f"\n📈 SUMMARY:")
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"⚡ Fastest technique: {min([r for r in results if r['success']], key=lambda x: x['duration'], default={'technique': 'N/A'})['technique'] if successful_tests > 0 else 'N/A'}")
    print(f"📝 Most detailed answer: {max([r for r in results if r['success']], key=lambda x: x['answer_length'], default={'technique': 'N/A'})['technique'] if successful_tests > 0 else 'N/A'}")

def main():
    """Main test execution."""
    print("🧪 RAG TECHNIQUE INTEGRATION TEST")
    print(f"Question: {TEST_QUESTION}")
    print(f"API Base: {API_BASE}")
    
    # Health check first
    if not test_api_health():
        print("❌ Cannot proceed - API is not healthy")
        sys.exit(1)
    
    # Test all RAG techniques
    compare_techniques()
    
    print(f"\n{'='*80}")
    print("🎉 RAG TECHNIQUE TEST COMPLETED!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
