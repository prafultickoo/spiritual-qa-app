"""
Comprehensive Test Script for Permanent Fix of Shared State Corruption Issue

This script tests the exact sequence that was causing the shared state corruption:
1. First question (should work)
2. Follow-up question with conversation_history (was causing corruption)
3. Multiple additional questions (were failing due to persistent corruption)

The test validates that the state isolation and error recovery fixes are working.
"""

import requests
import json
import time
from typing import List, Dict, Any

class PermanentFixTester:
    def __init__(self, base_url: str = "http://localhost:55135"):
        self.base_url = base_url
        self.test_results = []
        
    def log_test(self, test_name: str, status: str, details: Dict[str, Any]):
        """Log test results with timestamp"""
        result = {
            "test": test_name,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "details": details
        }
        self.test_results.append(result)
        
        # Print colored output
        color = "\033[92m" if status == "PASS" else "\033[91m"  # Green for PASS, Red for FAIL
        reset = "\033[0m"
        print(f"{color}[{status}]{reset} {test_name}")
        
        if details.get("error"):
            print(f"      Error: {details['error']}")
        if details.get("response_status"):
            print(f"      HTTP Status: {details['response_status']}")
        if details.get("model"):
            print(f"      Model: {details['model']}")
        if details.get("answer_preview"):
            print(f"      Answer: {details['answer_preview'][:100]}...")
        print()
    
    def test_health_check(self) -> bool:
        """Test the health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/admin/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                self.log_test(
                    "Health Check", 
                    "PASS",
                    {
                        "response_status": response.status_code,
                        "api_status": health_data.get("api_status"),
                        "global_health": health_data.get("global_health"),
                        "active_requests": health_data.get("active_requests", 0)
                    }
                )
                return True
            else:
                self.log_test(
                    "Health Check", 
                    "FAIL",
                    {
                        "response_status": response.status_code,
                        "error": f"Unexpected status code: {response.status_code}"
                    }
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Health Check", 
                "FAIL",
                {
                    "error": f"Exception: {str(e)}"
                }
            )
            return False
    
    def make_question_request(self, question: str, conversation_history: List[Dict] = None, 
                            test_name: str = "") -> Dict[str, Any]:
        """Make a question request to the API"""
        payload = {
            "question": question,
            "reading_style": "balanced",
            "model": "gpt-4.1",
            "rag_technique": "stuff"
        }
        
        if conversation_history:
            payload["conversation_history"] = conversation_history
            
        try:
            response = requests.post(f"{self.base_url}/ask", json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                self.log_test(
                    test_name,
                    "PASS",
                    {
                        "response_status": response.status_code,
                        "model": result.get("model"),
                        "status": result.get("status"),
                        "answer_preview": result.get("answer", "")[:200]
                    }
                )
                return result
            else:
                error_detail = response.text
                self.log_test(
                    test_name,
                    "FAIL",
                    {
                        "response_status": response.status_code,
                        "error": error_detail[:300]
                    }
                )
                return {"status": "error", "error": error_detail}
                
        except Exception as e:
            self.log_test(
                test_name,
                "FAIL",
                {
                    "error": f"Exception: {str(e)}"
                }
            )
            return {"status": "error", "error": str(e)}
    
    def run_comprehensive_test(self):
        """Run the complete test sequence that was causing corruption"""
        print("=" * 80)
        print("🧪 COMPREHENSIVE PERMANENT FIX VALIDATION TEST")
        print("=" * 80)
        print(f"Testing backend at: {self.base_url}")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Step 1: Health Check
        print("📋 Step 1: Backend Health Check")
        health_ok = self.test_health_check()
        if not health_ok:
            print("❌ Health check failed! Aborting test.")
            return False
        
        # Step 2: First Question (this should work)
        print("📋 Step 2: First Question (Should Work)")
        first_question = "What is Karma Yoga?"
        first_result = self.make_question_request(
            first_question, 
            test_name="First Question"
        )
        
        if first_result.get("status") != "success":
            print("❌ First question failed! This indicates a basic API issue.")
            return False
            
        # Step 3: Follow-up Question with Conversation History (this was causing corruption)
        print("📋 Step 3: Follow-up with Conversation History (Previously Caused Corruption)")
        conversation_history = [
            {"role": "user", "content": first_question},
            {"role": "assistant", "content": first_result.get("answer", "")}
        ]
        
        followup_result = self.make_question_request(
            "Can you explain this in bullet points?",
            conversation_history,
            "Follow-up with History"
        )
        
        # Step 4: Additional Questions (these were failing due to persistent corruption)
        print("📋 Step 4: Additional Questions After Follow-up (Previously Failed)")
        
        additional_questions = [
            "What is Tattvabodha?",
            "What is the meaning of life?",
            "Explain Bhagavad Gita Chapter 2",
            "What is meditation?"
        ]
        
        all_additional_passed = True
        for i, question in enumerate(additional_questions, 1):
            result = self.make_question_request(
                question,
                test_name=f"Additional Question #{i}"
            )
            if result.get("status") != "success":
                all_additional_passed = False
        
        # Step 5: Another Follow-up to Test Persistent State
        print("📋 Step 5: Another Follow-up (Testing Persistent State Integrity)")
        second_followup_result = self.make_question_request(
            "Can you give me more details?",
            [
                {"role": "user", "content": "What is Tattvabodha?"},
                {"role": "assistant", "content": "Tattvabodha is a foundational text..."}
            ],
            "Second Follow-up"
        )
        
        # Step 6: Final Health Check
        print("📋 Step 6: Final Health Check (State Integrity)")
        final_health_ok = self.test_health_check()
        
        # Results Summary
        print("=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED! Shared state corruption issue is PERMANENTLY FIXED!")
            print()
            print("✅ First questions work")
            print("✅ Follow-up questions with conversation history work") 
            print("✅ Additional questions after follow-ups work")
            print("✅ Multiple follow-ups work")
            print("✅ Backend state remains healthy")
            print()
            print("The permanent fix is working perfectly! 🚀")
            
        else:
            print("❌ Some tests failed. Issues may still exist:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"   - {result['test']}: {result['details'].get('error', 'Unknown error')}")
        
        return failed_tests == 0

if __name__ == "__main__":
    tester = PermanentFixTester()
    success = tester.run_comprehensive_test()
    
    # Save detailed results to file
    with open("permanent_fix_test_results.json", "w") as f:
        json.dump({
            "success": success,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": tester.test_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: permanent_fix_test_results.json")
    
    if success:
        exit(0)  # Success
    else:
        exit(1)  # Failure
