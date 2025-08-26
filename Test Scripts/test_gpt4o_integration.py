#!/usr/bin/env python3
"""
Test script for GPT-4o integration in Spiritual Q&A app.
Tests GPT-4o model with different RAG techniques and settings.
"""

import requests
import json
import time
import sys

# Configuration
API_BASE = "http://localhost:60894"
TEST_QUESTIONS = [
    "What is the meaning of Om in Hindu philosophy?",
    "What does the Bhagavad Gita say about karma?",
    "Explain the concept of dharma in spiritual texts."
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

def test_gpt4o_basic(question):
    """Test GPT-4o with basic settings."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing GPT-4o Basic: {question[:50]}...")
    print(f"{'='*60}")
    
    payload = {
        "question": question,
        "model": "gpt-4o",
        "use_mmr": True,
        "k": 5,
        "diversity": 0.3,
        "rag_technique": "stuff"
    }
    
    try:
        start_time = time.time()
        
        print(f"📤 Sending request to GPT-4o...")
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
            
            # Check for spiritual content
            spiritual_keywords = ["Om", "karma", "dharma", "spiritual", "sacred", "divine", "Bhagavad", "Gita"]
            keyword_found = any(keyword.lower() in answer.lower() for keyword in spiritual_keywords)
            
            if keyword_found:
                print("✅ Answer contains relevant spiritual content")
                return True, duration, len(answer)
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

def test_gpt4o_with_different_rag_techniques(question):
    """Test GPT-4o with different RAG techniques."""
    print(f"\n{'='*70}")
    print(f"🔬 Testing GPT-4o with Different RAG Techniques")
    print(f"Question: {question}")
    print(f"{'='*70}")
    
    rag_techniques = [
        ("stuff", "📦 Stuff"),
        ("refine", "🔄 Refine"),
        ("selective", "🎯 Selective")
    ]
    
    results = []
    
    for technique, description in rag_techniques:
        print(f"\n🧪 Testing {description} with GPT-4o...")
        
        payload = {
            "question": question,
            "model": "gpt-4o",
            "use_mmr": True,
            "k": 3,  # Use fewer chunks for faster testing
            "diversity": 0.3,
            "rag_technique": technique
        }
        
        try:
            start_time = time.time()
            
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
                
                print(f"  ✅ {description}: {duration:.1f}s, {len(answer)} chars")
                results.append({
                    "technique": technique,
                    "description": description,
                    "success": True,
                    "duration": duration,
                    "answer_length": len(answer)
                })
            else:
                print(f"  ❌ {description}: FAILED ({response.status_code})")
                results.append({
                    "technique": technique,
                    "description": description,
                    "success": False,
                    "duration": duration,
                    "answer_length": 0
                })
                
        except Exception as e:
            print(f"  ❌ {description}: ERROR - {str(e)}")
            results.append({
                "technique": technique,
                "description": description,
                "success": False,
                "duration": 0,
                "answer_length": 0
            })
    
    return results

def compare_gpt4o_vs_o3mini(question):
    """Compare GPT-4o vs o3-mini performance."""
    print(f"\n{'='*70}")
    print(f"⚡ GPT-4o vs o3-mini Comparison")
    print(f"Question: {question}")
    print(f"{'='*70}")
    
    models = [
        ("gpt-4o", "GPT-4o", {}),
        ("o3-mini", "o3 mini", {"reasoning_effort": "medium"})
    ]
    
    results = []
    
    for model_id, model_name, extra_params in models:
        print(f"\n🤖 Testing {model_name}...")
        
        payload = {
            "question": question,
            "model": model_id,
            "use_mmr": True,
            "k": 3,
            "diversity": 0.3,
            "rag_technique": "stuff"
        }
        
        # Add extra parameters for specific models
        payload.update(extra_params)
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{API_BASE}/ask",
                json=payload,
                timeout=90
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                
                print(f"  ✅ {model_name}: {duration:.1f}s, {len(answer)} chars")
                results.append({
                    "model": model_id,
                    "name": model_name,
                    "success": True,
                    "duration": duration,
                    "answer_length": len(answer)
                })
            else:
                print(f"  ❌ {model_name}: FAILED ({response.status_code})")
                results.append({
                    "model": model_id,
                    "name": model_name,
                    "success": False,
                    "duration": duration,
                    "answer_length": 0
                })
                
        except Exception as e:
            print(f"  ❌ {model_name}: ERROR - {str(e)}")
            results.append({
                "model": model_id,
                "name": model_name,
                "success": False,
                "duration": 0,
                "answer_length": 0
            })
    
    return results

def main():
    """Main test execution."""
    print("🧪 GPT-4o INTEGRATION TEST")
    print(f"API Base: {API_BASE}")
    
    # Health check first
    if not test_api_health():
        print("❌ Cannot proceed - API is not healthy")
        sys.exit(1)
    
    # Test 1: Basic GPT-4o functionality
    print(f"\n{'='*80}")
    print("📋 TEST 1: Basic GPT-4o Functionality")
    print(f"{'='*80}")
    
    basic_results = []
    for question in TEST_QUESTIONS:
        success, duration, answer_length = test_gpt4o_basic(question)
        basic_results.append({
            "question": question,
            "success": success,
            "duration": duration,
            "answer_length": answer_length
        })
    
    # Test 2: GPT-4o with different RAG techniques
    print(f"\n{'='*80}")
    print("📋 TEST 2: GPT-4o with RAG Techniques")
    print(f"{'='*80}")
    
    rag_results = test_gpt4o_with_different_rag_techniques(TEST_QUESTIONS[0])
    
    # Test 3: GPT-4o vs o3-mini comparison
    print(f"\n{'='*80}")
    print("📋 TEST 3: GPT-4o vs o3-mini Comparison")
    print(f"{'='*80}")
    
    comparison_results = compare_gpt4o_vs_o3mini(TEST_QUESTIONS[1])
    
    # Final Summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}")
    
    # Basic functionality summary
    basic_success = sum(1 for r in basic_results if r["success"])
    print(f"✅ Basic GPT-4o tests: {basic_success}/{len(basic_results)} passed")
    
    # RAG techniques summary
    rag_success = sum(1 for r in rag_results if r["success"])
    print(f"✅ GPT-4o RAG techniques: {rag_success}/{len(rag_results)} passed")
    
    # Model comparison summary
    model_success = sum(1 for r in comparison_results if r["success"])
    print(f"✅ Model comparisons: {model_success}/{len(comparison_results)} passed")
    
    # Performance insights
    if basic_success > 0:
        avg_duration = sum(r["duration"] for r in basic_results if r["success"]) / basic_success
        avg_length = sum(r["answer_length"] for r in basic_results if r["success"]) / basic_success
        print(f"⚡ GPT-4o average response time: {avg_duration:.1f}s")
        print(f"📝 GPT-4o average answer length: {avg_length:.0f} chars")
    
    total_tests = len(basic_results) + len(rag_results) + len(comparison_results)
    total_success = basic_success + rag_success + model_success
    
    print(f"\n🎉 OVERALL RESULT: {total_success}/{total_tests} tests passed")
    
    if total_success == total_tests:
        print("✅ GPT-4o integration is FULLY FUNCTIONAL!")
    elif total_success >= total_tests * 0.8:
        print("✅ GPT-4o integration is MOSTLY FUNCTIONAL with minor issues")
    else:
        print("❌ GPT-4o integration has SIGNIFICANT ISSUES that need attention")

if __name__ == "__main__":
    main()
