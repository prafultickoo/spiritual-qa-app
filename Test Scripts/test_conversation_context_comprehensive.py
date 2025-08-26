#!/usr/bin/env python3
"""
Comprehensive test script to verify conversation context functionality.

Tests both normal questions and follow-up questions to ensure:
1. Normal questions work as expected
2. Follow-up questions properly use conversation history
3. Context enhancement actions work (reformat_previous, clarify_previous)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from answer_generator import AnswerGenerator
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.WARNING)  # Reduce noise, only show warnings/errors

def test_conversation_context():
    """Test conversation context with various question types"""
    
    print("=" * 100)
    print("🧪 COMPREHENSIVE CONVERSATION CONTEXT TEST")
    print("=" * 100)
    
    # Initialize answer generator
    generator = AnswerGenerator(
        vector_store_dir="vector_store",
        llm_model="gpt-4.1"
    )
    
    # Test cases: (question, is_follow_up, conversation_context)
    test_cases = [
        # Normal questions (not follow-ups)
        ("What is Karma Yoga?", False, ""),
        ("What is meditation?", False, ""),
        ("Explain the concept of dharma", False, ""),
        
        # Follow-up questions that should use conversation context
        ("Can you explain this in bullet points?", True, "User: What is Karma Yoga?\nAssistant: Karma Yoga is the path of doing your daily actions with a spirit of devotion, without being attached to the results. It means you do your work sincerely, but you let go of worrying about what you will get in return. You offer your actions to something higher than yourself, like the Divine or the greater good. This practice helps purify your mind and eventually leads to spiritual liberation. The key is to act with love and dedication while remaining unattached to success or failure."),
        
        ("Give me more details about this", True, "User: What is meditation?\nAssistant: Meditation is a practice of training your mind to focus and achieve a state of peaceful awareness. It involves sitting quietly, usually with closed eyes, and directing your attention inward. Through meditation, you learn to observe your thoughts without getting caught up in them, cultivating inner stillness and clarity. Regular practice helps reduce stress, increase self-awareness, and connect you with your true nature."),
        
        ("Summarize the main points", True, "User: Explain the concept of dharma\nAssistant: Dharma is your life's purpose and righteous duty according to your nature, circumstances, and stage of life. It represents the moral and ethical principles that guide right living. Dharma is not the same for everyone - a teacher's dharma differs from a soldier's or a parent's. Following your dharma means acting in harmony with your true nature while contributing to the greater good of society and the universe."),
        
        ("Can you make this simpler to understand?", True, "User: What is Karma Yoga?\nAssistant: Karma Yoga is the path of doing your daily actions with a spirit of devotion, without being attached to the results. It means you do your work sincerely, but you let go of worrying about what you will get in return."),
    ]
    
    # Print header
    print(f"{'Question':<40} | {'Is Follow-up':<15} | {'Answer Generated':<50}")
    print("-" * 110)
    
    # Test each case
    for question, is_follow_up, conversation_context in test_cases:
        if is_follow_up and conversation_context:
            # Use conversation context for follow-up questions
            result = generator.generate_answer_with_context(
                query=question,
                conversation_context=conversation_context,
                k=5,
                use_mmr=True
            )
        else:
            # Normal question without context
            result = generator.generate_answer(
                query=question,
                k=5
            )
        
        # Extract answer and truncate for display
        answer = result.get("answer", "ERROR: No answer generated")
        answer_preview = (answer[:47] + "...") if len(answer) > 50 else answer
        answer_preview = answer_preview.replace('\n', ' ')
        
        # Format and print result
        is_follow_up_str = "Yes" if is_follow_up else "No"
        print(f"{question:<40} | {is_follow_up_str:<15} | {answer_preview:<50}")
    
    print("\n" + "=" * 100)
    print("✅ Test completed! Check if follow-up questions properly use conversation context.")
    print("=" * 100)

if __name__ == "__main__":
    test_conversation_context()
