"""
Test script to validate OpenAI o3-mini API parameters and integration.
This script will test o3-mini specific parameters before integrating into the main system.
"""
import os
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test_o3_mini_basic():
    """Test basic o3-mini API call with minimal parameters."""
    print("🧪 Testing o3-mini basic API call...")
    
    try:
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2? Explain briefly."}
            ],
            max_completion_tokens=200
        )
        
        end_time = time.time()
        
        result = {
            "status": "success",
            "model": response.model,
            "content": response.choices[0].message.content,
            "usage": response.usage.dict() if response.usage else None,
            "response_time": round(end_time - start_time, 2)
        }
        
        print("✅ Basic o3-mini test PASSED")
        print(f"   Model: {result['model']}")
        print(f"   Response time: {result['response_time']}s")
        print(f"   Tokens: {result['usage']}")
        print(f"   Answer: {result['content'][:100]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Basic o3-mini test FAILED: {str(e)}")
        return {"status": "error", "error": str(e)}

def test_o3_mini_reasoning_effort():
    """Test o3-mini with different reasoning_effort levels."""
    print("\n🧠 Testing o3-mini reasoning_effort parameter...")
    
    reasoning_levels = ["low", "medium", "high"]
    results = {}
    
    for effort in reasoning_levels:
        print(f"  Testing reasoning_effort={effort}...")
        
        try:
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are a logical reasoning expert."},
                    {"role": "user", "content": "If all roses are flowers, and some flowers are red, can we conclude that some roses are red? Explain your reasoning step by step."}
                ],
                max_completion_tokens=500,
                reasoning_effort=effort
            )
            
            end_time = time.time()
            
            results[effort] = {
                "status": "success",
                "content": response.choices[0].message.content,
                "usage": response.usage.dict() if response.usage else None,
                "response_time": round(end_time - start_time, 2),
                "reasoning_tokens": getattr(response.usage, 'reasoning_tokens', None) if response.usage else None
            }
            
            print(f"    ✅ reasoning_effort={effort} PASSED ({results[effort]['response_time']}s)")
            
        except Exception as e:
            print(f"    ❌ reasoning_effort={effort} FAILED: {str(e)}")
            results[effort] = {"status": "error", "error": str(e)}
    
    return results

def test_o3_mini_unsupported_params():
    """Test parameters that might not be supported by o3-mini."""
    print("\n⚠️  Testing potentially unsupported parameters...")
    
    unsupported_tests = []
    
    # Test temperature
    print("  Testing temperature parameter...")
    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ],
            max_completion_tokens=100,
            temperature=0.7
        )
        print("    ✅ temperature=0.7 SUPPORTED")
        unsupported_tests.append({"parameter": "temperature", "supported": True})
    except Exception as e:
        print(f"    ❌ temperature UNSUPPORTED: {str(e)}")
        unsupported_tests.append({"parameter": "temperature", "supported": False, "error": str(e)})
    
    # Test top_p
    print("  Testing top_p parameter...")
    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ],
            max_completion_tokens=100,
            top_p=0.9
        )
        print("    ✅ top_p=0.9 SUPPORTED")
        unsupported_tests.append({"parameter": "top_p", "supported": True})
    except Exception as e:
        print(f"    ❌ top_p UNSUPPORTED: {str(e)}")
        unsupported_tests.append({"parameter": "top_p", "supported": False, "error": str(e)})
    
    # Test frequency_penalty
    print("  Testing frequency_penalty parameter...")
    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ],
            max_completion_tokens=100,
            frequency_penalty=0.2
        )
        print("    ✅ frequency_penalty=0.2 SUPPORTED")
        unsupported_tests.append({"parameter": "frequency_penalty", "supported": True})
    except Exception as e:
        print(f"    ❌ frequency_penalty UNSUPPORTED: {str(e)}")
        unsupported_tests.append({"parameter": "frequency_penalty", "supported": False, "error": str(e)})
    
    return unsupported_tests

def test_o3_mini_spiritual_query():
    """Test o3-mini with a spiritual reasoning query similar to our use case."""
    print("\n🕉️  Testing o3-mini with spiritual reasoning query...")
    
    spiritual_prompt = """
Based on the following spiritual context, provide a thoughtful answer to the question:

Context:
"The Bhagavad Gita teaches that one should perform their duty (dharma) without attachment to results. Krishna advises Arjuna to act according to his nature as a warrior, but to surrender the fruits of action to the divine."

Question: How can someone apply the principle of detached action in modern work life?
"""
    
    try:
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a wise spiritual teacher who explains ancient wisdom in practical, modern terms. Provide thoughtful, compassionate guidance."},
                {"role": "user", "content": spiritual_prompt}
            ],
            max_completion_tokens=800,
            reasoning_effort="medium"
        )
        
        end_time = time.time()
        
        result = {
            "status": "success",
            "content": response.choices[0].message.content,
            "usage": response.usage.dict() if response.usage else None,
            "response_time": round(end_time - start_time, 2),
            "reasoning_tokens": getattr(response.usage, 'reasoning_tokens', None) if response.usage else None
        }
        
        print("✅ Spiritual reasoning test PASSED")
        print(f"   Response time: {result['response_time']}s")
        print(f"   Total tokens: {result['usage']['total_tokens'] if result['usage'] else 'N/A'}")
        print(f"   Reasoning tokens: {result['reasoning_tokens'] or 'N/A'}")
        print(f"   Answer preview: {result['content'][:200]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Spiritual reasoning test FAILED: {str(e)}")
        return {"status": "error", "error": str(e)}

def test_o3_mini_streaming():
    """Test o3-mini streaming capability."""
    print("\n🌊 Testing o3-mini streaming...")
    
    try:
        start_time = time.time()
        
        stream = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "user", "content": "Explain meditation in 3 steps."}
            ],
            max_completion_tokens=300,
            reasoning_effort="low",
            stream=True
        )
        
        print("  Streaming response:")
        content_chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                content_chunks.append(content)
        
        end_time = time.time()
        
        print(f"\n✅ Streaming test PASSED ({round(end_time - start_time, 2)}s)")
        return {
            "status": "success", 
            "streaming_supported": True,
            "content": "".join(content_chunks),
            "response_time": round(end_time - start_time, 2)
        }
        
    except Exception as e:
        print(f"❌ Streaming test FAILED: {str(e)}")
        return {"status": "error", "error": str(e), "streaming_supported": False}

def main():
    """Run all o3-mini validation tests."""
    print("🚀 Starting o3-mini API Validation Tests")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    test_results = {}
    
    # Run all tests
    test_results["basic"] = test_o3_mini_basic()
    test_results["reasoning_effort"] = test_o3_mini_reasoning_effort()
    test_results["unsupported_params"] = test_o3_mini_unsupported_params()
    test_results["spiritual_query"] = test_o3_mini_spiritual_query()
    test_results["streaming"] = test_o3_mini_streaming()
    
    # Save results to file
    results_file = "/Users/prafultickoo/Desktop/Spiritual/o3_mini_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📊 Test results saved to: {results_file}")
    print("\n" + "=" * 60)
    print("🏁 o3-mini API Validation Complete")
    
    # Summary
    print("\n📋 SUMMARY:")
    for test_name, result in test_results.items():
        if isinstance(result, dict) and result.get("status") == "success":
            print(f"  ✅ {test_name}: PASSED")
        elif isinstance(result, dict) and result.get("status") == "error":
            print(f"  ❌ {test_name}: FAILED")
        elif isinstance(result, list):  # unsupported_params returns list
            supported_count = sum(1 for item in result if item.get("supported", False))
            print(f"  ⚠️  {test_name}: {supported_count}/{len(result)} parameters supported")
        else:
            print(f"  ❓ {test_name}: UNKNOWN")

if __name__ == "__main__":
    main()
