"""
Simple test to verify o3-mini integration works.
"""
import requests
import time

def test_simple_o3_mini():
    """Simple test with longer timeout."""
    print("🧪 Simple o3-mini Test")
    print("=" * 40)
    
    # Simple health check first
    print("1. Testing API health...")
    try:
        response = requests.get("http://localhost:49350/health", timeout=30)
        if response.status_code == 200:
            print("   ✅ API is healthy")
        else:
            print(f"   ❌ API health failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ API not accessible: {str(e)}")
        return
    
    # Test o3-mini with simple question
    print("\n2. Testing o3-mini with simple question...")
    
    payload = {
        "question": "What is meditation?",
        "model": "o3-mini",
        "reasoning_effort": "low"
    }
    
    try:
        print("   Sending request...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:49350/ask",
            json=payload,
            timeout=60  # Longer timeout for o3-mini
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ SUCCESS! ({end_time - start_time:.1f}s)")
            print(f"   Model used: {result.get('model')}")
            print(f"   Answer: {result.get('answer', '')[:100]}...")
            return True
        else:
            print(f"   ❌ Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_simple_o3_mini()
    if success:
        print("\n🎉 o3-mini integration is working!")
    else:
        print("\n⚠️ o3-mini test failed")
