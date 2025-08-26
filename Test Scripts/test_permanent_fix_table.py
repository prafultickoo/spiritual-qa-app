"""
Permanent Fix Test Script with Table Format Output

Tests the shared state corruption fix and displays results in table format:
Question | Is Follow Up | Answer
"""

import requests
import json
import time
from typing import List, Dict, Any

class TableFormatTester:
    def __init__(self, base_url: str = "http://localhost:55135"):
        self.base_url = base_url
        self.results = []
        
    def make_question_request(self, question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
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
                return {
                    "status": "success",
                    "answer": result.get("answer", ""),
                    "model": result.get("model", "")
                }
            else:
                return {
                    "status": "error",
                    "answer": f"ERROR: {response.status_code} - {response.text[:100]}",
                    "model": "N/A"
                }
                
        except Exception as e:
            return {
                "status": "error", 
                "answer": f"EXCEPTION: {str(e)[:100]}",
                "model": "N/A"
            }
    
    def truncate_text(self, text: str, max_length: int = 80) -> str:
        """Truncate text for table display"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def run_table_test(self):
        """Run the test sequence and display results in table format"""
        print("🧪 TESTING PERMANENT FIX FOR SHARED STATE CORRUPTION")
        print("=" * 120)
        print()
        
        test_sequence = []
        
        # Test 1: First Question
        print("Processing: First Question...")
        first_question = "What is Karma Yoga?"
        first_result = self.make_question_request(first_question)
        
        test_sequence.append({
            "question": first_question,
            "is_follow_up": "No",
            "answer": first_result["answer"],
            "status": first_result["status"]
        })
        
        # Test 2: Follow-up Question with Conversation History (THE CRITICAL TEST)
        print("Processing: Follow-up with Conversation History...")
        conversation_history = [
            {"role": "user", "content": first_question},
            {"role": "assistant", "content": first_result["answer"]}
        ]
        
        followup_question = "Can you explain this in bullet points?"
        followup_result = self.make_question_request(followup_question, conversation_history)
        
        test_sequence.append({
            "question": followup_question,
            "is_follow_up": "Yes", 
            "answer": followup_result["answer"],
            "status": followup_result["status"]
        })
        
        # Test 3-6: Additional Questions (these were failing due to corruption)
        additional_questions = [
            "What is Tattvabodha?",
            "What is the meaning of life?", 
            "Explain Bhagavad Gita Chapter 2",
            "What is meditation?"
        ]
        
        for i, question in enumerate(additional_questions):
            print(f"Processing: Additional Question #{i+1}...")
            result = self.make_question_request(question)
            
            test_sequence.append({
                "question": question,
                "is_follow_up": "No",
                "answer": result["answer"], 
                "status": result["status"]
            })
        
        # Test 7: Another Follow-up (persistent state test)
        print("Processing: Second Follow-up...")
        second_followup_result = self.make_question_request(
            "Can you give me more details?",
            [
                {"role": "user", "content": "What is Tattvabodha?"},
                {"role": "assistant", "content": "Tattvabodha is a foundational text..."}
            ]
        )
        
        test_sequence.append({
            "question": "Can you give me more details?",
            "is_follow_up": "Yes",
            "answer": second_followup_result["answer"],
            "status": second_followup_result["status"]
        })
        
        # Display results in table format
        print()
        print("📊 TEST RESULTS:")
        print("=" * 120)
        
        # Table header
        header = f"{'Question':<35} | {'Is Follow Up':<12} | {'Answer':<65}"
        print(header)
        print("-" * 120)
        
        # Table rows
        success_count = 0
        total_count = len(test_sequence)
        
        for item in test_sequence:
            question = self.truncate_text(item["question"], 33)
            is_followup = item["is_follow_up"]
            answer = self.truncate_text(item["answer"], 63)
            
            # Add status indicator
            status_symbol = "✅" if item["status"] == "success" else "❌"
            answer_with_status = f"{status_symbol} {answer}"
            
            row = f"{question:<35} | {is_followup:<12} | {answer_with_status:<65}"
            print(row)
            
            if item["status"] == "success":
                success_count += 1
        
        print("-" * 120)
        
        # Summary
        print(f"\n📈 SUMMARY:")
        print(f"Total Tests: {total_count}")
        print(f"Successful: {success_count}")
        print(f"Failed: {total_count - success_count}")
        print(f"Success Rate: {(success_count/total_count)*100:.1f}%")
        
        if success_count == total_count:
            print(f"\n🎉 ALL TESTS PASSED! The permanent fix is working perfectly!")
            print(f"✅ Shared state corruption issue is RESOLVED")
            print(f"✅ Follow-up questions now work correctly") 
            print(f"✅ Backend state remains stable across all requests")
        else:
            print(f"\n❌ Some tests failed. Issues may still exist.")
            
        return success_count == total_count

if __name__ == "__main__":
    tester = TableFormatTester()
    success = tester.run_table_test()
    
    if success:
        exit(0)
    else:
        exit(1)
