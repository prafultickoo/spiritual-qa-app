"""
Conversation Context Processor - Stage 1: Fast Heuristic Filter
Detects follow-up queries without any LLM calls
"""
import logging
from typing import List, Dict, Tuple, Optional
import re

# Configure logging
logger = logging.getLogger(__name__)


class ConversationContextProcessor:
    """
    Stage 1: Fast heuristic-based follow-up detection.
    No LLM calls, just pattern matching and simple rules.
    """
    
    def __init__(self):
        # Pre-compiled patterns for speed
        self.spiritual_topics = {
            'dharma', 'karma', 'yoga', 'meditation', 'turiya', 'atman', 'brahman',
            'vedanta', 'upanishad', 'gita', 'krishna', 'buddha', 'enlightenment',
            'moksha', 'nirvana', 'samsara', 'bhakti', 'jnana', 'raja', 'mantra',
            'chakra', 'prana', 'kundalini', 'samadhi', 'consciousness', 'awareness',
            'mindfulness', 'devotion', 'liberation', 'self-realization', 'guru',
            'vedas', 'sutras', 'mantras', 'asana', 'pranayama', 'dhyana'
        }
        
        self.follow_up_indicators = {
            # Direct follow-up language
            'explain', 'more detail', 'elaborate', 'expand on', 'clarify',
            'what do you mean', 'can you', 'tell me more', 'go deeper',
            'further', 'additional', 'continue', 'more about',
            # Referential phrases
            'explain this', 'explain that', 'what about this', 'regarding this',
            'about that', 'in this context', 'you mentioned', 'you said',
            # Question continuations
            'how does this', 'why is this', 'what makes this', 'when does this',
            # Simple references
            'this', 'that', 'it', 'these', 'those'
        }
        
        self.new_topic_indicators = {
            'what is', 'who is', 'define', 'meaning of', 'concept of',
            'tell me about', 'explain the concept', 'new question',
            'different topic', 'another question', 'switching to',
            'i want to know about', 'can you tell me about'
        }

    def is_follow_up_query(self, query: str, conversation_history: List[Dict]) -> Dict[str, any]:
        """
        Stage 1: Fast heuristic check if query needs context enhancement.
        
        Args:
            query: Current user query
            conversation_history: List of previous conversation turns
            
        Returns:
            Dict with 'is_follow_up', 'confidence', 'reasoning'
        """
        
        # No history = definitely new topic
        if not conversation_history or len(conversation_history) < 2:
            return {
                'is_follow_up': False,
                'confidence': 1.0,
                'reasoning': 'No conversation history available',
                'stage': 1
            }
        
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        # Rule 1: Check for explicit new topic indicators
        for pattern in self.new_topic_indicators:
            if query_lower.startswith(pattern):
                # But check if it might still reference previous context
                if any(ref in query_lower for ref in ['this', 'that', 'it']):
                    # e.g., "What is this concept called?" - might be follow-up
                    return {
                        'is_follow_up': True,
                        'confidence': 0.6,
                        'reasoning': f'New topic pattern "{pattern}" but contains reference words',
                        'stage': 1
                    }
                return {
                    'is_follow_up': False,
                    'confidence': 0.9,
                    'reasoning': f'Starts with new topic pattern: "{pattern}"',
                    'stage': 1
                }
        
        # Rule 2: Contains specific spiritual topic = likely new topic
        spiritual_topics_found = self.spiritual_topics.intersection(query_words)
        if spiritual_topics_found:
            # Check if it's asking for more detail about the topic
            if any(indicator in query_lower for indicator in ['more about', 'explain more', 'tell me more']):
                return {
                    'is_follow_up': True,
                    'confidence': 0.7,
                    'reasoning': f'Contains spiritual topic {spiritual_topics_found} but asks for more detail',
                    'stage': 1
                }
            return {
                'is_follow_up': False,
                'confidence': 0.8,
                'reasoning': f'Contains specific spiritual topic: {spiritual_topics_found}',
                'stage': 1
            }
        
        # Rule 3: Check for follow-up indicators
        follow_up_found = []
        for indicator in self.follow_up_indicators:
            if indicator in query_lower:
                follow_up_found.append(indicator)
        
        if follow_up_found:
            # Strong follow-up indicators
            strong_indicators = ['explain this', 'explain that', 'tell me more', 'more detail', 'elaborate']
            if any(ind in query_lower for ind in strong_indicators):
                return {
                    'is_follow_up': True,
                    'confidence': 0.9,
                    'reasoning': f'Contains strong follow-up indicators: {follow_up_found}',
                    'stage': 1
                }
            return {
                'is_follow_up': True,
                'confidence': 0.7,
                'reasoning': f'Contains follow-up indicators: {follow_up_found}',
                'stage': 1
            }
        
        # Rule 4: Very short queries often reference context
        word_count = len(query.split())
        if word_count <= 5:
            # Check if it's a complete question despite being short
            if query_lower.startswith(('what', 'who', 'when', 'where', 'why', 'how')):
                return {
                    'is_follow_up': False,
                    'confidence': 0.6,
                    'reasoning': 'Short but complete question',
                    'stage': 1
                }
            return {
                'is_follow_up': True,
                'confidence': 0.8,
                'reasoning': f'Very short query ({word_count} words) likely references previous context',
                'stage': 1
            }
        
        # Rule 5: Questions that lack subject/object
        has_question_word = query_lower.startswith(('what', 'who', 'when', 'where', 'why', 'how', 'can', 'could', 'would'))
        if has_question_word and word_count < 8 and not spiritual_topics_found:
            return {
                'is_follow_up': True,
                'confidence': 0.6,
                'reasoning': 'Question lacks specific subject/topic',
                'stage': 1
            }
        
        # Default: Not a follow-up
        return {
            'is_follow_up': False,
            'confidence': 0.5,
            'reasoning': 'No clear follow-up indicators found',
            'stage': 1
        }
    
    def extract_last_topic(self, conversation_history: List[Dict]) -> Optional[str]:
        """
        Extract the main topic from the last assistant response.
        This is a simple extraction for Stage 1.
        
        Args:
            conversation_history: Conversation history
            
        Returns:
            Main topic string or None
        """
        if not conversation_history or len(conversation_history) < 2:
            return None
        
        # Find last assistant response
        last_assistant_response = None
        for item in reversed(conversation_history):
            if item.get('role') == 'assistant':
                last_assistant_response = item.get('content', '')
                break
        
        if not last_assistant_response:
            return None
        
        # Simple extraction: Look for spiritual topics mentioned
        response_lower = last_assistant_response.lower()
        topics_mentioned = []
        
        for topic in self.spiritual_topics:
            if topic in response_lower:
                # Count occurrences to find most discussed topic
                count = response_lower.count(topic)
                topics_mentioned.append((topic, count))
        
        if topics_mentioned:
            # Return the most frequently mentioned topic
            topics_mentioned.sort(key=lambda x: x[1], reverse=True)
            return topics_mentioned[0][0]
        
        # Fallback: Try to extract from user's last question
        for item in reversed(conversation_history):
            if item.get('role') == 'user':
                user_query = item.get('content', '').lower()
                for topic in self.spiritual_topics:
                    if topic in user_query:
                        return topic
                break
        
        return None
    
    def process_query(self, query: str, conversation_history: List[Dict]) -> Dict[str, any]:
        """
        Main entry point for Stage 1 processing.
        
        Args:
            query: Current user query
            conversation_history: Conversation history
            
        Returns:
            Dict with processing result including whether to proceed to Stage 2
        """
        # Check if it's a follow-up
        follow_up_result = self.is_follow_up_query(query, conversation_history)
        
        # Log the decision
        logger.info(f"Stage 1 Heuristic Result: is_follow_up={follow_up_result['is_follow_up']}, "
                   f"confidence={follow_up_result['confidence']}, "
                   f"reasoning={follow_up_result['reasoning']}")
        
        result = {
            'original_query': query,
            'processed_query': query,  # Stage 1 doesn't modify queries
            'is_follow_up': follow_up_result['is_follow_up'],
            'confidence': follow_up_result['confidence'],
            'reasoning': follow_up_result['reasoning'],
            'stage_completed': 1,
            'proceed_to_stage_2': False,
            'proceed_to_stage_3': False
        }
        
        # Decide if we need Stage 2 (semantic analysis)
        # Only if we're uncertain (confidence between 0.5 and 0.8)
        if follow_up_result['is_follow_up'] and 0.5 <= follow_up_result['confidence'] <= 0.8:
            result['proceed_to_stage_2'] = True
            logger.info("Stage 1: Uncertainty detected, recommending Stage 2 semantic analysis")
        
        # Decide if we need Stage 3 (LLM enhancement)
        # Only if we're confident it's a follow-up
        elif follow_up_result['is_follow_up'] and follow_up_result['confidence'] > 0.8:
            result['proceed_to_stage_3'] = True
            # Try to extract topic for context
            topic = self.extract_last_topic(conversation_history)
            if topic:
                result['detected_topic'] = topic
            logger.info(f"Stage 1: High confidence follow-up detected, recommending Stage 3 LLM enhancement. Topic: {topic}")
        
        return result


# Test function for verification
def test_stage1_heuristics():
    """Test Stage 1 heuristic detection with various scenarios."""
    
    processor = ConversationContextProcessor()
    
    # Test scenarios
    test_cases = [
        # Scenario 1: Your exact Turiya case
        {
            'query': 'Can you explain this in much more detail?',
            'history': [
                {'role': 'user', 'content': 'What is the meaning of Turiya?'},
                {'role': 'assistant', 'content': 'Turiya is the fourth state of consciousness beyond waking, dreaming, and deep sleep...'}
            ],
            'expected': True,
            'description': 'Turiya follow-up case'
        },
        # Scenario 2: New topic
        {
            'query': 'What is karma yoga?',
            'history': [
                {'role': 'user', 'content': 'Tell me about meditation'},
                {'role': 'assistant', 'content': 'Meditation is...'}
            ],
            'expected': False,
            'description': 'New spiritual topic'
        },
        # Scenario 3: Short follow-up
        {
            'query': 'More details please',
            'history': [
                {'role': 'user', 'content': 'What is dharma?'},
                {'role': 'assistant', 'content': 'Dharma refers to...'}
            ],
            'expected': True,
            'description': 'Short follow-up request'
        },
        # Scenario 4: No history
        {
            'query': 'Tell me about enlightenment',
            'history': [],
            'expected': False,
            'description': 'No conversation history'
        }
    ]
    
    print("="*60)
    print("STAGE 1 HEURISTIC FILTER TEST RESULTS")
    print("="*60)
    
    for i, test in enumerate(test_cases, 1):
        result = processor.process_query(test['query'], test['history'])
        
        print(f"\nTest {i}: {test['description']}")
        print(f"Query: '{test['query']}'")
        print(f"Expected follow-up: {test['expected']}")
        print(f"Detected follow-up: {result['is_follow_up']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Proceed to Stage 2: {result['proceed_to_stage_2']}")
        print(f"Proceed to Stage 3: {result['proceed_to_stage_3']}")
        if 'detected_topic' in result:
            print(f"Detected topic: {result['detected_topic']}")
        print(f"✅ PASS" if result['is_follow_up'] == test['expected'] else "❌ FAIL")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run tests when executed directly
    test_stage1_heuristics()
