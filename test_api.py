import requests
import json
import time

# API endpoint
API_URL = "https://junaid121dark--llama-3-1-legal-inference-inference-api.modal.run"

def test_api(user_prompt, system_prompt=None, chat_history=None, rag_context="", max_tokens=2048):
    """
    Test the Llama 3.1 Legal API
    
    Args:
        user_prompt: Your question/prompt
        system_prompt: System instructions (optional)
        chat_history: Previous conversation (optional)
        rag_context: Additional context (optional)
        max_tokens: Max response length
    """
    
    # Default system prompt if not provided
    if system_prompt is None:
        system_prompt = "You are a helpful legal assistant specializing in Bangladeshi law. Provide accurate, detailed responses in user query language (bengali or english) when appropriate."
    
    # Default empty chat history if not provided
    if chat_history is None:
        chat_history = []
    
    # Prepare request payload
    payload = {
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "chat_history": chat_history,
        "rag_context": rag_context,
        "max_new_tokens": max_tokens,
        "temperature": 0.8,
        "top_p": 0.9,
        "do_sample": True
    }
    
    print(f"🔍 Testing API with prompt: {user_prompt[:50]}...")
    print(f"📤 Sending request to: {API_URL}")
    
    start_time = time.time()
    
    try:
        # Make POST request
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2 minute timeout
        )
        
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n{'='*60}")
            print(f"✅ SUCCESS - Request completed in {request_time:.2f}s")
            print(f"{'='*60}")
            print(f"👤 User Prompt: {result['user_prompt']}")
            if rag_context:
                print(f"📄 RAG Context: {rag_context[:100]}...")
            if chat_history:
                print(f"💬 Chat History: {len(chat_history)} messages")
            print(f"\n🤖 Model Response:")
            print(f"{result['response']}")
            print(f"\n📊 Stats:")
            print(f"   - Model: {result['model_name']}")
            print(f"   - Inference Time: {result['inference_time']:.2f}s")
            print(f"   - Total Request Time: {request_time:.2f}s")
            print(f"{'='*60}")
            
            return result
            
        else:
            print(f"❌ ERROR {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (>2 minutes)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"🔌 Network error: {e}")
        return None
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return None

def test_with_context():
    """Test with RAG context"""
    user_prompt = "বাংলাদেশে সম্পত্তির উত্তরাধিকার আইন কী?"
    rag_context = """
    বাংলাদেশ সংবিধানের ২৭ নম্বর অনুচ্ছেদ: আইনের দৃষ্টিতে সকল নাগরিক সমান।
    সম্পত্তি আইন: মুসলিম ব্যক্তিগত আইন অনুযায়ী উত্তরাধিকার নির্ধারিত হয়।
    """
    
    return test_api(
        user_prompt=user_prompt,
        rag_context=rag_context,
        max_tokens=2048
    )

def test_with_chat_history():
    """Test with previous conversation"""
    chat_history = [
        {"role": "user", "content": "আমার একটি আইনি সমস্যা আছে।"},
        {"role": "assistant", "content": "আমি আপনাকে বাংলাদেশি আইন বিষয়ে সহায়তা করতে পারি। আপনার সমস্যাটি কী?"}
    ]
    
    user_prompt = "আমার পিতার সম্পত্তি নিয়ে বিরোধ হচ্ছে। কী করব?"
    
    return test_api(
        user_prompt=user_prompt,
        chat_history=chat_history,
        max_tokens=2048
    )

def test_simple():
    """Simple test without context"""
    user_prompt = "find me case references for criminal code"
    
    return test_api(
        user_prompt=user_prompt,
        max_tokens=2048
    )

if __name__ == "__main__":
    print("🧪 Testing Llama 3.1 Legal API")
    print(f"📡 API URL: {API_URL}")
    print("\n" + "="*60)
    
    # Test 1: Simple question
    print("TEST 1: Simple Question")
    test_simple()
    
    # time.sleep(2)  # Brief pause between tests
    
    # Test 2: With RAG context
    # print("\nTEST 2: With RAG Context")
    # # test_with_context()
    
    # # time.sleep(2)
    
    # # Test 3: With chat history
    # print("\nTEST 3: With Chat History")
    # # test_with_chat_history()
    
    # print("\n🎉 All tests completed!")