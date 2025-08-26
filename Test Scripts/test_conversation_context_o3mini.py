#!/usr/bin/env python3
"""
Extended comprehensive test script with o3-mini (medium reasoning) to test 
conversation context functionality across different spiritual topics and follow-up types.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from answer_generator import AnswerGenerator
import logging

# Set up logging to reduce noise
logging.basicConfig(level=logging.WARNING)

def test_conversation_context_o3mini():
    """Test conversation context with o3-mini medium reasoning - 10 diverse question scenarios"""
    
    print("=" * 100)
    print("🧪 CONVERSATION CONTEXT TEST - O3-MINI (MEDIUM REASONING) - 10 DIVERSE SCENARIOS")
    print("=" * 100)
    
    # Initialize answer generator with o3-mini
    generator = AnswerGenerator(
        vector_store_dir="vector_store",
        llm_model="o3-mini"  # Using o3-mini instead of gpt-4.1
    )
    
    # Test cases: (question, is_follow_up, conversation_context, description)
    test_cases = [
        # 1. Basic spiritual concept
        ("What is moksha?", False, "", "Basic spiritual concept question"),
        
        # 2. Follow-up: Convert to list format
        ("Convert this to a numbered list", True, 
         "User: What is moksha?\nAssistant: Moksha is the ultimate spiritual liberation in Hindu philosophy. It represents the soul's release from the cycle of birth, death, and rebirth (samsara). When you achieve moksha, you realize your true divine nature and unite with the infinite consciousness (Brahman). It's the end of all suffering and the beginning of eternal bliss and knowledge.",
         "Follow-up: Convert to numbered list"),
        
        # 3. Different spiritual topic
        ("Explain the concept of Maya", False, "", "Different spiritual concept"),
        
        # 4. Follow-up: Ask for practical examples
        ("Give me practical examples of this in daily life", True,
         "User: Explain the concept of Maya\nAssistant: Maya is the cosmic illusion that veils our true spiritual nature. It's like a dream or mirage that makes us believe the temporary world is permanent and real. Maya causes us to identify with our body and mind instead of our eternal soul. Through Maya, we experience separation from the Divine when in reality we are always one with it.",
         "Follow-up: Ask for practical examples"),
        
        # 5. Verse-specific query  
        ("What does Bhagavad Gita say about detachment?", False, "", "Scripture-specific question"),
        
        # 6. Follow-up: Explain in simple terms
        ("Explain this using simple everyday language", True,
         "User: What does Bhagavad Gita say about detachment?\nAssistant: The Bhagavad Gita teaches that detachment (vairagya) is essential for spiritual growth. Krishna advises Arjuna to perform his duties without attachment to results. True detachment means acting with full commitment while remaining unaffected by success or failure. It's not indifference but rather freedom from the anxiety of outcomes.",
         "Follow-up: Simplify language"),
        
        # 7. Meditation technique question
        ("How do I practice mindfulness meditation?", False, "", "Practical meditation question"),
        
        # 8. Follow-up: Address challenges/difficulties
        ("What should I do when my mind keeps wandering?", True,
         "User: How do I practice mindfulness meditation?\nAssistant: Mindfulness meditation involves sitting quietly and observing your thoughts, feelings, and sensations without judgment. Start by focusing on your breath - feel the air entering and leaving your nostrils. When thoughts arise, simply notice them and gently return attention to your breath. Practice regularly, starting with just 10-15 minutes daily.",
         "Follow-up: Address meditation challenges"),
        
        # 9. Philosophical question
        ("What is the relationship between karma and free will?", False, "", "Philosophical question"),
        
        # 10. Follow-up: Create a comparison
        ("Create a comparison table between these two concepts", True,
         "User: What is the relationship between karma and free will?\nAssistant: Karma and free will work together in spiritual philosophy. While karma represents the law of cause and effect from past actions, free will is your power to make conscious choices in the present moment. Your past karma influences your current circumstances, but your free will determines how you respond to those circumstances, thereby creating new karma.",
         "Follow-up: Create comparison table"),
    ]
    
    # Print header
    print(f"{'Question':<50} | {'Is Follow-up':<12} | {'Answer Generated':<40}")
    print("-" * 110)
    
    # Test each case
    for i, (question, is_follow_up, conversation_context, description) in enumerate(test_cases, 1):
        if is_follow_up and conversation_context:
            # Use conversation context for follow-up questions with o3-mini and medium reasoning
            result = generator.generate_answer_with_context(
                query=question,
                conversation_context=conversation_context,
                k=5,
                use_mmr=True,
                reasoning_effort="medium"  # Medium reasoning effort for o3-mini
            )
        else:
            # Normal question without context, but with o3-mini and medium reasoning
            result = generator.generate_answer(
                query=question,
                k=5,
                reasoning_effort="medium"  # Medium reasoning effort for o3-mini
            )
        
        # Extract answer and truncate for display
        answer = result.get("answer", "ERROR: No answer generated")
        answer_preview = (answer[:37] + "...") if len(answer) > 40 else answer
        answer_preview = answer_preview.replace('\n', ' ')
        
        # Format and print result
        is_follow_up_str = "Yes" if is_follow_up else "No"
        print(f"{question:<50} | {is_follow_up_str:<12} | {answer_preview:<40}")
    
    print("\n" + "=" * 110)
    print("✅ O3-Mini (Medium Reasoning) test completed! All 10 diverse scenarios tested.")
    print("=" * 110)

if __name__ == "__main__":
    test_conversation_context_o3mini()
