"""
Simulate exact API flow for follow-up questions
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from answer_generator import AnswerGenerator
import json
import time
from api.spiritual_api import AskRequest, ConversationHistoryItem

def simulate_api_flow():
    """Simulate the exact API flow for follow-up questions"""
    print("=" * 70)
    print("SIMULATING EXACT API FLOW")
    print("=" * 70)
    
    # Simulate API request with conversation history
    request = AskRequest(
        question="Can you explain in bullets?",
        conversation_history=[
            ConversationHistoryItem(role="user", content="What is apara vidya?"),
            ConversationHistoryItem(role="assistant", content="Apara vidya is worldly knowledge...")
        ],
        reading_style="balanced",
        model="gpt-4.1",
        rag_technique="stuff",
        max_context_docs=5,
        use_mmr=True,
        diversity=0.3,
        reasoning_effort=None
    )
    
    print(f"\nRequest model: {request.model}")
    print(f"Has conversation history: {len(request.conversation_history)} items")
    
    # Simulate API initialization
    vector_store_dir = "vector_store"
    
    # Initialize generator (simulating API behavior)
    generator = AnswerGenerator(
        vector_store_dir=vector_store_dir,
        llm_model=request.model,  # This is how API does it
        enable_dual_source=True
    )
    
    print(f"\nGenerator initialized with llm_model: {generator.llm_model}")
    
    # Build conversation context (exactly as API does)
    conversation_context = ""
    if request.conversation_history and len(request.conversation_history) > 0:
        history_text = []
        for item in request.conversation_history[-10:]:
            role_prefix = "User" if item.role == "user" else "Assistant"
            history_text.append(f"{role_prefix}: {item.content}")
        
        conversation_context = "\n\n".join(history_text)
        print(f"\nConversation context built: {len(conversation_context)} chars")
    
    # Call generate_answer_with_context (as API does)
    print("\nCalling generate_answer_with_context...")
    try:
        result = generator.generate_answer_with_context(
            query=request.question,
            conversation_context=conversation_context,
            max_docs=request.max_context_docs,
            k=10,
            use_mmr=request.use_mmr,
            diversity=request.diversity,
            reasoning_effort=request.reasoning_effort
        )
        
        print(f"\nResult received:")
        print(f"- Status: {result.get('status')}")
        print(f"- Result keys: {list(result.keys())}")
        print(f"- Model in result: '{result.get('model')}'")
        print(f"- Model type: {type(result.get('model'))}")
        
        # Try to create QuestionResponse as API would
        if result.get("status") == "success":
            print("\nCreating QuestionResponse...")
            from api.spiritual_api import QuestionResponse
            
            response = QuestionResponse(
                status="success",
                answer=result.get("answer"),
                sources=result.get("sources"),
                verses=result.get("verses"),
                model=result.get("model"),
                processing_time=1.0,
                conversation_id="test_conv_123"
            )
            
            print("QuestionResponse created successfully!")
            print(f"Response model: {response.model}")
            
    except Exception as e:
        print(f"\nException occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_api_flow()
