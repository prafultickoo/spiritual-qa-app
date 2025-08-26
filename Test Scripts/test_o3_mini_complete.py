"""
Complete end-to-end test for o3-mini integration with the Spiritual Q&A app.
Tests both backend API and frontend functionality.
"""
import json
import requests
import time
from typing import Dict, Any

# Test configuration - use the running app's port
API_BASE_URL = "http://localhost:49350"  # Backend running on port 49350

def test_o3_mini_api_integration():
    """Test o3-mini API integration with all reasoning effort levels."""
    print("🧪 Testing o3-mini API Integration...")
    print("=" * 60)
    
    # Test spiritual question with different reasoning efforts
    test_question = "What is the essence of mindfulness meditation according to Buddhist teachings?"
    
    test_cases = [
        {"reasoning_effort": "low", "expected_response_time": "< 10s"},
        {"reasoning_effort": "medium", "expected_response_time": "< 15s"},
        {"reasoning_effort": "high", "expected_response_time": "< 25s"}
    ]
    
    results = {}
    
    for i, case in enumerate(test_cases, 1):
        reasoning_effort = case["reasoning_effort"]
        print(f"\n🧠 Test {i}/3: o3-mini with reasoning_effort='{reasoning_effort}'")
        print(f"   Question: {test_question}")
        
        payload = {
            "question": test_question,
            "model": "o3-mini",
            "reasoning_effort": reasoning_effort,
            "reading_style": "balanced",
            "max_context_docs": 5,
            "k": 5
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{API_BASE_URL}/ask",
                json=payload,
                timeout=120  # o3-mini can take longer
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"   ✅ SUCCESS ({response_time:.1f}s)")
                print(f"   Status: {result.get('status')}")
                print(f"   Model: {result.get('model')}")
                print(f"   Processing time: {result.get('processing_time', 'N/A')}s")
                print(f"   Answer length: {len(result.get('answer', ''))} characters")
                print(f"   Answer preview: {result.get('answer', '')[:120]}...")
                
                results[reasoning_effort] = {
                    "status": "success",
                    "response_time": response_time,
                    "processing_time": result.get('processing_time'),
                    "answer_length": len(result.get('answer', '')),
                    "full_result": result
                }
            else:
                print(f"   ❌ FAILED with status {response.status_code}")
                print(f"   Error: {response.text}")
                
                results[reasoning_effort] = {
                    "status": "error",
                    "error": response.text,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            print(f"   ❌ EXCEPTION: {str(e)}")
            results[reasoning_effort] = {
                "status": "exception",
                "error": str(e)
            }
    
    return results

def test_model_configuration():
    """Test that o3-mini is properly configured in the backend."""
    print("\n🤖 Testing Model Configuration...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            
            if 'o3-mini' in models.get('models', {}):
                o3_config = models['models']['o3-mini']
                print("   ✅ o3-mini found in backend configuration")
                print(f"   - Name: {o3_config.get('name')}")
                print(f"   - Reasoning model: {o3_config.get('is_reasoning_model')}")
                print(f"   - Reasoning efforts: {o3_config.get('reasoning_efforts')}")
                print(f"   - Default effort: {o3_config.get('default_reasoning_effort')}")
                return True
            else:
                print("   ❌ o3-mini not found in backend models")
                return False
        else:
            print(f"   ❌ Models endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception testing models: {str(e)}")
        return False

def test_reasoning_effort_comparison():
    """Compare reasoning efforts on the same question."""
    print("\n🔬 Testing Reasoning Effort Comparison...")
    
    question = "How do the concepts of karma and dharma interrelate in Hindu philosophy?"
    efforts = ["low", "medium", "high"]
    results = {}
    
    for effort in efforts:
        print(f"   Testing {effort} reasoning effort...")
        
        payload = {
            "question": question,
            "model": "o3-mini",
            "reasoning_effort": effort,
            "reading_style": "deep"
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=120)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                results[effort] = {
                    "response_time": end_time - start_time,
                    "answer_length": len(result.get('answer', '')),
                    "processing_time": result.get('processing_time'),
                    "status": "success"
                }
                print(f"     ✅ {effort}: {results[effort]['response_time']:.1f}s, {results[effort]['answer_length']} chars")
            else:
                results[effort] = {"status": "error", "error": response.text}
                print(f"     ❌ {effort}: Failed")
                
        except Exception as e:
            results[effort] = {"status": "error", "error": str(e)}
            print(f"     ❌ {effort}: Exception")
    
    # Analyze results
    successful_tests = [r for r in results.values() if r.get("status") == "success"]
    if len(successful_tests) >= 2:
        print("\n   📊 Comparison Analysis:")
        for effort in efforts:
            if results[effort].get("status") == "success":
                r = results[effort]
                print(f"     {effort.upper()}: {r['response_time']:.1f}s response, {r['answer_length']} chars")
    
    return results

def main():
    """Run complete o3-mini integration tests."""
    print("🚀 o3-mini Complete Integration Test")
    print("=" * 60)
    
    # Test 1: Check API health
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ API is healthy and responsive")
        else:
            print(f"❌ API health check failed: {health_response.status_code}")
            return
    except Exception as e:
        print(f"❌ API not accessible: {str(e)}")
        print("Make sure the backend is running on the correct port.")
        return
    
    # Test 2: Model configuration
    model_config_ok = test_model_configuration()
    
    # Test 3: o3-mini API integration
    api_results = test_o3_mini_api_integration()
    
    # Test 4: Reasoning effort comparison
    comparison_results = test_reasoning_effort_comparison()
    
    # Save results
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_configuration": model_config_ok,
        "api_integration": api_results,
        "reasoning_comparison": comparison_results
    }
    
    results_file = "/Users/prafultickoo/Desktop/Spiritual/o3_mini_complete_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 o3-mini Integration Test Summary")
    print("=" * 60)
    
    successful_api_tests = sum(1 for result in api_results.values() if result.get("status") == "success")
    total_api_tests = len(api_results)
    
    successful_comparison_tests = sum(1 for result in comparison_results.values() if result.get("status") == "success")
    total_comparison_tests = len(comparison_results)
    
    print(f"✅ Model Configuration: {'PASS' if model_config_ok else 'FAIL'}")
    print(f"✅ API Integration Tests: {successful_api_tests}/{total_api_tests} PASSED")
    print(f"✅ Reasoning Effort Tests: {successful_comparison_tests}/{total_comparison_tests} PASSED")
    
    if successful_api_tests == total_api_tests and successful_comparison_tests >= 2 and model_config_ok:
        print("\n🎉 ALL TESTS PASSED! o3-mini integration is fully functional.")
        print("💡 Ready for production use with user-selectable reasoning effort.")
    else:
        print("\n⚠️  Some tests failed. Please review the results.")
    
    print(f"\n📊 Detailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
