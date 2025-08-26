"""
Debug follow-up question issue with detailed logging
"""
import requests
import json
import logging

# Set up logging to see detailed output
logging.basicConfig(level=logging.DEBUG)

def test_followup_with_logging():
    """Test follow-up question functionality with detailed logging"""
    print("=" * 70)
    print("DETAILED DEBUGGING OF FOLLOW-UP QUESTION ISSUE")
    print("=" * 70)
    
    base_url = "http://localhost:62566"
    
    # First, ask a primary question
    print("\n1. Testing Primary Question...")
    primary_payload = {
        "question": "What is apara vidya?",
        "reading_style": "balanced",
        "model": "gpt-4.1",
        "rag_technique": "stuff"
    }
    
    try:
        # Send primary question
        response = requests.post(f"{base_url}/ask", json=primary_payload, timeout=60)
        print(f"Primary Question Status: {response.status_code}")
        
        if response.status_code == 200:
            primary_result = response.json()
            print(f"\nPrimary Response Keys: {list(primary_result.keys())}")
            print(f"Primary Model Field: '{primary_result.get('model', 'MISSING')}'")
            print(f"Primary Answer (first 100 chars): {primary_result.get('answer', '')[:100]}...")
            
            # Now test follow-up question - try different approaches
            print("\n2. Testing Follow-up Questions with Different Approaches...")
            
            # Approach 1: Using conversation_history (as frontend does)
            print("\n2a. Testing with conversation_history...")
            followup_payload_1 = {
                "question": "Can you explain in bullets?",
                "conversation_history": [
                    {"role": "user", "content": primary_payload['question']},
                    {"role": "assistant", "content": primary_result.get('answer', '')}
                ],
                "reading_style": "balanced", 
                "model": "gpt-4.1",
                "rag_technique": "stuff"
            }
            
            print(f"Payload keys: {list(followup_payload_1.keys())}")
            print(f"Conversation history items: {len(followup_payload_1['conversation_history'])}")
            
            followup_response_1 = requests.post(f"{base_url}/ask", json=followup_payload_1, timeout=60)
            print(f"Follow-up 1 Status: {followup_response_1.status_code}")
            
            if followup_response_1.status_code == 200:
                followup_result_1 = followup_response_1.json()
                print(f"Follow-up 1 Response Keys: {list(followup_result_1.keys())}")
                print(f"Follow-up 1 Model Field: '{followup_result_1.get('model', 'MISSING')}'")
                print(f"Follow-up 1 Success: {followup_result_1.get('success', 'N/A')}")
            else:
                print(f"Follow-up 1 Failed: {followup_response_1.text[:500]}")
                
            # Approach 2: Simple follow-up without much context
            print("\n2b. Testing simple follow-up...")
            followup_payload_2 = {
                "question": "What about para vidya?",
                "reading_style": "balanced",
                "model": "gpt-4.1",
                "rag_technique": "stuff"
            }
            
            followup_response_2 = requests.post(f"{base_url}/ask", json=followup_payload_2, timeout=60)
            print(f"Follow-up 2 Status: {followup_response_2.status_code}")
            
            if followup_response_2.status_code == 200:
                followup_result_2 = followup_response_2.json()
                print(f"Follow-up 2 Model Field: '{followup_result_2.get('model', 'MISSING')}'")
                
        else:
            print(f"Primary Question Failed: {response.text}")
            
    except Exception as e:
        print(f"Exception during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_followup_with_logging()
