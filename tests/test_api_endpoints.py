"""
Test scripts for Spiritual Q&A API endpoints.
Comprehensive testing of all API functionality.
"""

import pytest
import requests
import json
import time
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

class SpiritualAPITester:
    """
    Comprehensive test suite for Spiritual Q&A API.
    """
    
    def __init__(self, base_url: str = BASE_URL):
        """Initialize the API tester."""
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_test_result(self, test_name: str, success: bool, response_time: float, details: str = ""):
        """Log test results for reporting."""
        result = {
            "test_name": test_name,
            "success": success,
            "response_time_ms": round(response_time * 1000, 2),
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({result['response_time_ms']}ms)")
        if details:
            print(f"    {details}")
    
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint."""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/health", timeout=TEST_TIMEOUT)
            response_time = time.time() - start_time
            
            success = response.status_code == 200
            data = response.json() if success else {}
            
            details = f"Status: {response.status_code}, Vector Store: {data.get('vector_store_status', 'unknown')}"
            self.log_test_result("Health Check", success, response_time, details)
            
            return success
            
        except Exception as e:
            self.log_test_result("Health Check", False, 0, f"Error: {str(e)}")
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test the root endpoint."""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/", timeout=TEST_TIMEOUT)
            response_time = time.time() - start_time
            
            success = response.status_code == 200
            data = response.json() if success else {}
            
            details = f"Status: {response.status_code}, Message: {data.get('message', 'N/A')[:50]}..."
            self.log_test_result("Root Endpoint", success, response_time, details)
            
            return success
            
        except Exception as e:
            self.log_test_result("Root Endpoint", False, 0, f"Error: {str(e)}")
            return False
    
    def test_models_endpoint(self) -> bool:
        """Test the models listing endpoint."""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/models", timeout=TEST_TIMEOUT)
            response_time = time.time() - start_time
            
            success = response.status_code == 200
            data = response.json() if success else {}
            
            models_count = len(data.get('available_models', []))
            details = f"Status: {response.status_code}, Models: {models_count}, Default: {data.get('default_model', 'N/A')}"
            self.log_test_result("Models Endpoint", success, response_time, details)
            
            return success
            
        except Exception as e:
            self.log_test_result("Models Endpoint", False, 0, f"Error: {str(e)}")
            return False
    
    def test_spiritual_question(self, question: str, model: str = "gpt-4o") -> bool:
        """Test asking a spiritual question."""
        try:
            start_time = time.time()
            
            payload = {
                "question": question,
                "model": model,
                "max_context_docs": 5
            }
            
            response = self.session.post(
                f"{self.base_url}/ask",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            response_time = time.time() - start_time
            
            success = response.status_code == 200
            data = response.json() if success else {}
            
            response_length = len(data.get('response', ''))
            sources_count = len(data.get('sources', []))
            
            details = f"Status: {response.status_code}, Model: {model}, Response: {response_length} chars, Sources: {sources_count}"
            self.log_test_result(f"Spiritual Question ({model})", success, response_time, details)
            
            return success
            
        except Exception as e:
            self.log_test_result(f"Spiritual Question ({model})", False, 0, f"Error: {str(e)}")
            return False
    
    def test_document_search(self, query: str) -> bool:
        """Test document search functionality."""
        try:
            start_time = time.time()
            
            payload = {
                "query": query,
                "num_docs": 5
            }
            
            response = self.session.post(
                f"{self.base_url}/search",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            response_time = time.time() - start_time
            
            success = response.status_code == 200
            data = response.json() if success else {}
            
            docs_found = data.get('documents_found', 0)
            details = f"Status: {response.status_code}, Documents Found: {docs_found}"
            self.log_test_result("Document Search", success, response_time, details)
            
            return success
            
        except Exception as e:
            self.log_test_result("Document Search", False, 0, f"Error: {str(e)}")
            return False
    
    def test_random_wisdom(self) -> bool:
        """Test random wisdom endpoint."""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/random-wisdom", timeout=TEST_TIMEOUT)
            response_time = time.time() - start_time
            
            success = response.status_code == 200
            data = response.json() if success else {}
            
            wisdom_length = len(str(data.get('wisdom', {}).get('quote', '')))
            details = f"Status: {response.status_code}, Wisdom Length: {wisdom_length} chars"
            self.log_test_result("Random Wisdom", success, response_time, details)
            
            return success
            
        except Exception as e:
            self.log_test_result("Random Wisdom", False, 0, f"Error: {str(e)}")
            return False
    
    def test_stats_endpoint(self) -> bool:
        """Test stats endpoint."""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/stats", timeout=TEST_TIMEOUT)
            response_time = time.time() - start_time
            
            success = response.status_code == 200
            data = response.json() if success else {}
            
            api_status = data.get('system_info', {}).get('status', 'unknown')
            details = f"Status: {response.status_code}, API Status: {api_status}"
            self.log_test_result("Stats Endpoint", success, response_time, details)
            
            return success
            
        except Exception as e:
            self.log_test_result("Stats Endpoint", False, 0, f"Error: {str(e)}")
            return False
    
    def test_quick_ask_endpoint(self, question: str, model: str = "gpt-4o") -> bool:
        """Test the quick ask endpoint with model in URL."""
        try:
            start_time = time.time()
            
            params = {"question": question}
            response = self.session.get(
                f"{self.base_url}/ask/{model}",
                params=params,
                timeout=TEST_TIMEOUT
            )
            response_time = time.time() - start_time
            
            success = response.status_code == 200
            data = response.json() if success else {}
            
            response_length = len(data.get('response', ''))
            details = f"Status: {response.status_code}, Model: {model}, Response: {response_length} chars"
            self.log_test_result(f"Quick Ask ({model})", success, response_time, details)
            
            return success
            
        except Exception as e:
            self.log_test_result(f"Quick Ask ({model})", False, 0, f"Error: {str(e)}")
            return False
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all API tests and return results."""
        print("🕉️ Starting Comprehensive API Tests for Spiritual Q&A System")
        print("=" * 60)
        
        # Test basic endpoints
        print("\n📍 Testing Basic Endpoints...")
        self.test_root_endpoint()
        self.test_health_endpoint()
        self.test_models_endpoint()
        self.test_stats_endpoint()
        self.test_random_wisdom()
        
        # Test document search
        print("\n🔍 Testing Document Search...")
        test_queries = [
            "meditation and mindfulness",
            "dharma and righteous living",
            "karma and its effects"
        ]
        
        for query in test_queries:
            self.test_document_search(query)
        
        # Test spiritual questions with different models
        print("\n🧘 Testing Spiritual Q&A...")
        spiritual_questions = [
            "What is the meaning of life?",
            "How can I find inner peace?",
            "What is the purpose of suffering?"
        ]
        
        # Test with different models
        test_models = ["gpt-4o", "gpt-4.1"]
        
        for question in spiritual_questions:
            for model in test_models:
                self.test_spiritual_question(question, model)
        
        # Test quick ask endpoint
        print("\n⚡ Testing Quick Ask Endpoint...")
        self.test_quick_ask_endpoint("How should I live righteously?", "gpt-4o")
        
        # Generate summary
        return self.generate_test_summary()
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        avg_response_time = sum(result['response_time_ms'] for result in self.test_results) / total_tests if total_tests > 0 else 0
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "average_response_time_ms": round(avg_response_time, 2),
            "detailed_results": self.test_results
        }
        
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Average Response Time: {summary['average_response_time_ms']:.2f}ms")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! API is functioning perfectly!")
        else:
            print(f"\n⚠️ {failed_tests} test(s) failed. Review the detailed results above.")
        
        print(f"{'='*60}")
        
        return summary

def run_api_tests():
    """Main function to run API tests."""
    try:
        tester = SpiritualAPITester()
        results = tester.run_comprehensive_tests()
        
        # Save results to file
        with open("api_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Detailed test results saved to: api_test_results.json")
        return results
        
    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Run tests if executed directly
    run_api_tests()
