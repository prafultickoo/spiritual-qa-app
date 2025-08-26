#!/usr/bin/env python3
"""
Quick test to verify if answers start with "Om!" as per updated prompt template.
"""

import requests
import json

# Configuration
API_BASE = "http://localhost:56680"
TEST_QUESTION = "What is the meaning of life?"

def test_om_greeting():
    """Test if the response starts with Om! greeting."""
    print("🧪 Testing Om! greeting in responses...")
    
    payload = {
        "question": TEST_QUESTION,
        "model": "gpt-4.1",
        "use_mmr": True,
        "k": 3,
        "diversity": 0.3,
        "rag_technique": "stuff"
    }
    
    try:
        print(f"📤 Asking: {TEST_QUESTION}")
        
        response = requests.post(
            f"{API_BASE}/ask",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            
            print(f"📥 Response received ({len(answer)} chars)")
            print(f"🔍 First 100 chars: {answer[:100]}...")
            
            # Check if starts with Om!
            if answer.strip().startswith("Om!"):
                print("✅ SUCCESS: Response starts with 'Om!'")
                return True
            else:
                print("❌ ISSUE: Response does NOT start with 'Om!'")
                print(f"🔤 Actual start: '{answer[:20]}...'")
                return False
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"📥 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def main():
    """Main test execution."""
    print("🔍 DEBUGGING: Om! GREETING ISSUE")
    print(f"API Base: {API_BASE}")
    
    # Test Om! greeting
    success = test_om_greeting()
    
    if success:
        print("\n✅ Om! greeting is working correctly!")
    else:
        print("\n❌ Om! greeting is NOT working - prompt template issue detected!")
        print("\n🔧 POSSIBLE SOLUTIONS:")
        print("1. Restart the application to reload prompt templates")
        print("2. Clear any caching mechanisms")
        print("3. Check if prompt templates are being loaded correctly")

if __name__ == "__main__":
    main()
