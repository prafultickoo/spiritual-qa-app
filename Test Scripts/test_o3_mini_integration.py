"""
Test script to validate o3-mini integration with the Spiritual Q&A backend.
This script tests the full integration flow from API request to answer generation.
"""
import json
import requests
import time
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8001"  # Adjust if your API runs on different port
TEST_QUESTIONS = [
    {
        "question": "What is the meaning of karma according to Hindu philosophy?",
        "reasoning_effort": "low",
        "expected_response_time": "< 10s"
    },
    {
        "question": "How can I apply the teachings of the Bhagavad Gita to overcome anxiety and fear in daily life?",
        "reasoning_effort": "medium", 
        "expected_response_time": "< 15s"
    },
    {
        "question": "Compare the concept of enlightenment in Buddhism, Hinduism, and Advaita Vedanta. What are the key similarities and differences in their approaches to spiritual awakening?",
        "reasoning_effort": "high",
        "expected_response_time": "< 20s"
    }
]

def test_api_health():
    """Test if the API is running and responsive."""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ API is healthy and responsive")
            return True
        else:
            print(f"❌ API health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check failed: {str(e)}")
        return False

def test_models_endpoint():
    """Test the /models endpoint to see if o3-mini is available."""
    print("\n🤖 Testing models endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Models endpoint successful")
            print(f"   Available models: {list(models.get('models', {}).keys())}")
            
            # Check if o3-mini is configured
            if 'o3-mini' in models.get('models', {}):
                o3_config = models['models']['o3-mini']
                print(f"✅ o3-mini is available with config:")
                print(f"   - Name: {o3_config.get('name')}")
                print(f"   - Reasoning model: {o3_config.get('is_reasoning_model')}")
                print(f"   - Available reasoning efforts: {o3_config.get('reasoning_efforts')}")
                return True
            else:
                print("❌ o3-mini not found in available models")
                return False
        else:
            print(f"❌ Models endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint failed: {str(e)}")
        return False

def test_o3_mini_question(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single question with o3-mini."""
    question = question_data["question"]
    reasoning_effort = question_data["reasoning_effort"]
    
    print(f"\n🧠 Testing o3-mini with reasoning_effort='{reasoning_effort}'...")
    print(f"   Question: {question[:60]}...")
    
    payload = {
        "question": question,
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
            timeout=60  # o3-mini might take longer
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ o3-mini response successful ({response_time:.1f}s)")
            print(f"   Status: {result.get('status')}")
            print(f"   Model used: {result.get('model')}")
            print(f"   Processing time: {result.get('processing_time', 'N/A')}s")
            print(f"   Answer preview: {result.get('answer', '')[:150]}...")
            
            return {
                "status": "success",
                "response_time": response_time,
                "reasoning_effort": reasoning_effort,
                "result": result
            }
        else:
            error_msg = response.text
            print(f"❌ o3-mini request failed with status {response.status_code}")
            print(f"   Error: {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "status_code": response.status_code,
                "reasoning_effort": reasoning_effort
            }
            
    except Exception as e:
        print(f"❌ o3-mini request failed: {str(e)}")
        return {
            "status": "error", 
            "error": str(e),
            "reasoning_effort": reasoning_effort
        }

def test_reasoning_effort_comparison():
    """Test all reasoning effort levels with the same question to compare."""
    print(f"\n🔬 Testing reasoning effort comparison...")
    
    test_question = "What is the essence of meditation according to spiritual traditions?"
    results = {}
    
    for effort in ["low", "medium", "high"]:
        print(f"\n   Testing reasoning_effort='{effort}'...")
        
        payload = {
            "question": test_question,
            "model": "o3-mini", 
            "reasoning_effort": effort,
            "reading_style": "balanced"
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=60)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                results[effort] = {
                    "status": "success",
                    "response_time": end_time - start_time,
                    "answer_length": len(result.get('answer', '')),
                    "processing_time": result.get('processing_time')
                }
                print(f"     ✅ {effort}: {results[effort]['response_time']:.1f}s, {results[effort]['answer_length']} chars")
            else:
                results[effort] = {"status": "error", "error": response.text}
                print(f"     ❌ {effort}: Failed - {response.status_code}")
                
        except Exception as e:
            results[effort] = {"status": "error", "error": str(e)}
            print(f"     ❌ {effort}: Exception - {str(e)}")
    
    return results

def main():
    """Run all o3-mini integration tests."""
    print("🚀 Starting o3-mini Backend Integration Tests")
    print("=" * 60)
    
    test_results = {
        "api_health": False,
        "models_available": False,
        "question_tests": [],
        "reasoning_comparison": {}
    }
    
    # Test 1: API Health
    test_results["api_health"] = test_api_health()
    if not test_results["api_health"]:
        print("\n❌ API not available. Please start the backend first.")
        print("   Run: cd /Users/prafultickoo/Desktop/Spiritual && python launch_app.py")
        return
    
    # Test 2: Models endpoint
    test_results["models_available"] = test_models_endpoint()
    if not test_results["models_available"]:
        print("\n❌ o3-mini not properly configured in backend")
        return
    
    # Test 3: Individual question tests
    print(f"\n🧪 Testing individual questions with different reasoning efforts...")
    for question_data in TEST_QUESTIONS:
        result = test_o3_mini_question(question_data)
        test_results["question_tests"].append(result)
    
    # Test 4: Reasoning effort comparison
    test_results["reasoning_comparison"] = test_reasoning_effort_comparison()
    
    # Save results
    results_file = "/Users/prafultickoo/Desktop/Spiritual/o3_mini_integration_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📊 Test results saved to: {results_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 o3-mini Backend Integration Test Complete")
    print("\n📋 SUMMARY:")
    
    successful_tests = sum(1 for test in test_results["question_tests"] if test.get("status") == "success")
    total_tests = len(test_results["question_tests"])
    
    print(f"  ✅ API Health: {'PASS' if test_results['api_health'] else 'FAIL'}")
    print(f"  ✅ Models Available: {'PASS' if test_results['models_available'] else 'FAIL'}")
    print(f"  ✅ Question Tests: {successful_tests}/{total_tests} PASSED")
    
    reasoning_success = sum(1 for effort, result in test_results["reasoning_comparison"].items() 
                           if result.get("status") == "success")
    print(f"  ✅ Reasoning Efforts: {reasoning_success}/3 PASSED")
    
    if successful_tests == total_tests and reasoning_success == 3:
        print(f"\n🎉 All tests PASSED! o3-mini integration is working correctly.")
        print(f"💡 Ready to proceed with frontend wiring.")
    else:
        print(f"\n⚠️  Some tests FAILED. Please review the results and fix issues before proceeding.")

if __name__ == "__main__":
    main()
