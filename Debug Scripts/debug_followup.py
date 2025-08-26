"""
Debug follow-up question issue
"""
import requests
import json

def test_followup_questions():
    """Test follow-up question functionality"""
    print("=" * 70)
    print("DEBUGGING FOLLOW-UP QUESTION ISSUE")
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
        response = requests.post(f"{base_url}/ask", json=primary_payload, timeout=60)
        print(f"Primary Question Status: {response.status_code}")
        
        if response.status_code == 200:
            primary_result = response.json()
            print(f"Primary Answer (first 200 chars): {primary_result.get('answer', '')[:200]}...")
            
            # Now test follow-up question with context
            print("\n2. Testing Follow-up Question...")
            
            # Build conversation context
            conversation_context = f"User: {primary_payload['question']}\nAssistant: {primary_result.get('answer', '')}"
            
            followup_payload = {
                "question": "Can you explain in bullets with examples?",
                "conversation_history": [
                    {"role": "user", "content": primary_payload['question']},
                    {"role": "assistant", "content": primary_result.get('answer', '')}
                ],
                "reading_style": "balanced", 
                "model": "gpt-4.1",
                "rag_technique": "stuff"
            }
            
            print(f"Conversation Context Length: {len(conversation_context)} characters")
            print(f"Follow-up Query: {followup_payload['question']}")
            
            followup_response = requests.post(f"{base_url}/ask", json=followup_payload, timeout=60)
            print(f"Follow-up Status: {followup_response.status_code}")
            
            if followup_response.status_code == 200:
                followup_result = followup_response.json()
                print(f"Follow-up Answer: {followup_result.get('answer', '')[:300]}...")
                print(f"Success: {followup_result.get('success', 'N/A')}")
                print(f"Error: {followup_result.get('error', 'None')}")
            else:
                print(f"Follow-up Failed: {followup_response.text}")
                
        else:
            print(f"Primary Question Failed: {response.text}")
            
    except Exception as e:
        print(f"Exception during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_followup_questions()
