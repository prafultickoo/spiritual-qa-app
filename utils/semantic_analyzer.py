"""
Stage 2: Semantic Analysis for Ambiguous Queries
Deep linguistic analysis without any LLM calls
"""
import logging
from typing import List, Dict, Tuple, Optional, Set
import re
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """
    Stage 2: Semantic analysis for ambiguous follow-up detection.
    Uses linguistic analysis, NO external API calls.
    """
    
    def __init__(self):
        # Pronouns that often indicate references
        self.pronouns = {
            'subjective': {'it', 'this', 'that', 'these', 'those', 'they', 'them'},
            'possessive': {'its', 'their'},
            'interrogative': {'what', 'which', 'who', 'whom', 'whose'},
            'demonstrative': {'this', 'that', 'these', 'those'}
        }
        
        # Question patterns that need objects
        self.incomplete_patterns = [
            r'^(what|which|who) (is|are|was|were)$',  # "What is?" 
            r'^(how|why|when|where) (does|do|did|is|are|was|were)$',  # "How does?"
            r'^can you (explain|describe|tell)$',  # "Can you explain?"
            r'^(explain|describe|tell me about)$',  # "Explain"
            r'^(more about|details about|information about)$',  # "More about"
        ]
        
        # Completeness indicators
        self.completeness_markers = {
            'specific_objects': ['dharma', 'karma', 'yoga', 'meditation', 'consciousness', 
                               'enlightenment', 'moksha', 'brahman', 'atman', 'vedanta'],
            'complete_phrases': ['the concept of', 'the practice of', 'the meaning of',
                               'the difference between', 'the relationship between']
        }
        
        # Common ambiguous references
        self.ambiguous_refs = {
            'concept', 'idea', 'practice', 'technique', 'method', 'approach',
            'philosophy', 'teaching', 'principle', 'aspect', 'element', 'part'
        }

    def analyze_linguistic_completeness(self, query: str) -> Dict[str, any]:
        """
        Analyze if a query is linguistically complete.
        
        Args:
            query: The query to analyze
            
        Returns:
            Dict with completeness analysis
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        result = {
            'is_complete': True,
            'missing_elements': [],
            'confidence': 0.0,
            'analysis': {}
        }
        
        # Check 1: Very short queries
        if len(words) <= 3:
            result['is_complete'] = False
            result['missing_elements'].append('too_short')
            result['confidence'] = 0.8
            result['analysis']['reason'] = f'Very short query ({len(words)} words)'
        
        # Check 2: Incomplete patterns
        for pattern in self.incomplete_patterns:
            if re.match(pattern, query_lower):
                result['is_complete'] = False
                result['missing_elements'].append('incomplete_pattern')
                result['confidence'] = 0.9
                result['analysis']['pattern'] = pattern
                break
        
        # Check 3: Dangling pronouns without antecedent
        pronouns_found = []
        for word in words:
            if word in self.pronouns['subjective'] or word in self.pronouns['demonstrative']:
                pronouns_found.append(word)
        
        if pronouns_found and not any(obj in query_lower for obj in self.completeness_markers['specific_objects']):
            result['is_complete'] = False
            result['missing_elements'].append('dangling_pronouns')
            result['confidence'] = 0.7
            result['analysis']['pronouns'] = pronouns_found
        
        # Check 4: Questions without objects
        if query_lower.startswith(('what', 'which', 'how', 'why')):
            # Check if there's a meaningful object after the question word
            if len(words) <= 4 and not any(obj in query_lower for obj in self.completeness_markers['specific_objects']):
                result['is_complete'] = False
                result['missing_elements'].append('missing_object')
                result['confidence'] = 0.8
                result['analysis']['question_type'] = words[0]
        
        # Check 5: References to ambiguous concepts
        ambiguous_found = []
        for word in words:
            if word in self.ambiguous_refs:
                ambiguous_found.append(word)
        
        if ambiguous_found and len(words) < 6:
            result['is_complete'] = False
            result['missing_elements'].append('ambiguous_reference')
            result['confidence'] = 0.6
            result['analysis']['ambiguous_terms'] = ambiguous_found
        
        # If query has complete phrases, boost completeness
        if any(phrase in query_lower for phrase in self.completeness_markers['complete_phrases']):
            result['is_complete'] = True
            result['confidence'] = 0.9
            result['analysis']['has_complete_phrase'] = True
        
        return result

    def calculate_semantic_overlap(self, query: str, previous_content: str) -> float:
        """
        Calculate semantic overlap between query and previous content.
        Simple word-based overlap without embeddings.
        
        Args:
            query: Current query
            previous_content: Previous conversation content
            
        Returns:
            Overlap score (0-1)
        """
        # Simple tokenization and normalization
        query_words = set(query.lower().split())
        content_words = set(previous_content.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but'}
        
        query_words = query_words - stop_words
        content_words = content_words - stop_words
        
        # Calculate Jaccard similarity
        if not query_words or not content_words:
            return 0.0
        
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)
        
        return intersection / union if union > 0 else 0.0

    def extract_key_entities(self, text: str) -> Set[str]:
        """
        Extract key entities/concepts from text.
        Simple rule-based extraction.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Set of key entities
        """
        text_lower = text.lower()
        entities = set()
        
        # Extract spiritual concepts
        spiritual_terms = [
            'dharma', 'karma', 'yoga', 'meditation', 'consciousness', 'awareness',
            'enlightenment', 'moksha', 'nirvana', 'brahman', 'atman', 'vedanta',
            'bhagavad gita', 'upanishad', 'buddha', 'krishna', 'shiva', 'vedas'
        ]
        
        for term in spiritual_terms:
            if term in text_lower:
                entities.add(term)
        
        # Extract capitalized words (likely proper nouns)
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.add(word.lower())
        
        return entities

    def analyze_query_ambiguity(self, query: str, conversation_history: List[Dict]) -> Dict[str, any]:
        """
        Main Stage 2 analysis: Determine if query is ambiguous and needs context.
        
        Args:
            query: Current query
            conversation_history: Previous conversation
            
        Returns:
            Analysis results
        """
        result = {
            'stage': 2,
            'is_ambiguous': False,
            'confidence': 0.0,
            'needs_context': False,
            'analysis_details': {}
        }
        
        # Step 1: Linguistic completeness check
        completeness = self.analyze_linguistic_completeness(query)
        result['analysis_details']['completeness'] = completeness
        
        # Step 2: Extract previous context if available
        if conversation_history and len(conversation_history) >= 2:
            # Get last exchange
            last_user_query = None
            last_assistant_response = None
            
            for i in range(len(conversation_history) - 1, -1, -1):
                item = conversation_history[i]
                if item.get('role') == 'assistant' and not last_assistant_response:
                    last_assistant_response = item.get('content', '')
                elif item.get('role') == 'user' and not last_user_query:
                    last_user_query = item.get('content', '')
                
                if last_user_query and last_assistant_response:
                    break
            
            # Step 3: Calculate semantic overlap
            if last_assistant_response:
                overlap_score = self.calculate_semantic_overlap(query, last_assistant_response)
                result['analysis_details']['semantic_overlap'] = overlap_score
                
                # Extract entities from previous response
                prev_entities = self.extract_key_entities(last_assistant_response)
                curr_entities = self.extract_key_entities(query)
                
                result['analysis_details']['previous_entities'] = list(prev_entities)
                result['analysis_details']['current_entities'] = list(curr_entities)
                
                # High overlap but incomplete query = likely follow-up
                if overlap_score > 0.3 and not completeness['is_complete']:
                    result['is_ambiguous'] = True
                    result['needs_context'] = True
                    result['confidence'] = 0.8
                    result['analysis_details']['reason'] = 'High semantic overlap with incomplete query'
        
        # Step 4: Pronoun resolution check
        query_lower = query.lower()
        pronouns_needing_resolution = []
        
        for pronoun in ['it', 'this', 'that', 'they', 'them']:
            if f' {pronoun} ' in f' {query_lower} ' or query_lower.startswith(f'{pronoun} '):
                pronouns_needing_resolution.append(pronoun)
        
        if pronouns_needing_resolution and not completeness['is_complete']:
            result['is_ambiguous'] = True
            result['needs_context'] = True
            result['confidence'] = max(result['confidence'], 0.7)
            result['analysis_details']['unresolved_pronouns'] = pronouns_needing_resolution
        
        # Step 5: Final decision based on completeness
        if not completeness['is_complete']:
            # Query is incomplete
            if 'missing_object' in completeness['missing_elements'] or \
               'dangling_pronouns' in completeness['missing_elements']:
                result['is_ambiguous'] = True
                result['needs_context'] = True
                result['confidence'] = max(result['confidence'], completeness['confidence'])
        
        return result

    def process_stage2(self, query: str, conversation_history: List[Dict], 
                      stage1_result: Dict) -> Dict[str, any]:
        """
        Main entry point for Stage 2 processing.
        Only called when Stage 1 is uncertain.
        
        Args:
            query: Current query
            conversation_history: Conversation history
            stage1_result: Results from Stage 1
            
        Returns:
            Enhanced analysis results
        """
        # Only process if Stage 1 recommended it
        if not stage1_result.get('proceed_to_stage_2', False):
            logger.info("Stage 2: Not needed, Stage 1 was conclusive")
            return stage1_result
        
        logger.info(f"Stage 2: Analyzing ambiguity for query: '{query}'")
        
        # Perform semantic analysis
        ambiguity_analysis = self.analyze_query_ambiguity(query, conversation_history)
        
        # Merge with Stage 1 results
        result = {
            'original_query': query,
            'processed_query': query,  # Stage 2 doesn't modify queries
            'stage_completed': 2,
            'stage1_result': stage1_result,
            'stage2_analysis': ambiguity_analysis,
            'proceed_to_stage_3': False
        }
        
        # Decision logic
        if ambiguity_analysis['is_ambiguous'] and ambiguity_analysis['confidence'] >= 0.7:
            result['is_follow_up'] = True
            result['confidence'] = ambiguity_analysis['confidence']
            result['reasoning'] = f"Stage 2: {ambiguity_analysis['analysis_details'].get('reason', 'Query is ambiguous and needs context')}"
            result['proceed_to_stage_3'] = True
            logger.info(f"Stage 2: Confirmed follow-up (confidence: {ambiguity_analysis['confidence']})")
        else:
            result['is_follow_up'] = False
            result['confidence'] = 0.9
            result['reasoning'] = "Stage 2: Query is complete and self-contained"
            logger.info("Stage 2: Query is self-contained, no context needed")
        
        return result


# Test function for Stage 2
def test_stage2_semantic_analysis():
    """Test Stage 2 semantic analysis with various scenarios."""
    
    analyzer = SemanticAnalyzer()
    
    # Test scenarios
    test_cases = [
        # Scenario 1: Ambiguous pronoun reference
        {
            'query': 'How does it work?',
            'history': [
                {'role': 'user', 'content': 'What is karma yoga?'},
                {'role': 'assistant', 'content': 'Karma yoga is the path of selfless action...'}
            ],
            'stage1_result': {'proceed_to_stage_2': True, 'confidence': 0.6},
            'expected_ambiguous': True,
            'description': 'Ambiguous pronoun "it"'
        },
        # Scenario 2: Incomplete question
        {
            'query': 'What are the benefits?',
            'history': [
                {'role': 'user', 'content': 'Tell me about meditation'},
                {'role': 'assistant', 'content': 'Meditation is a practice of focused awareness...'}
            ],
            'stage1_result': {'proceed_to_stage_2': True, 'confidence': 0.7},
            'expected_ambiguous': True,
            'description': 'Incomplete - benefits of what?'
        },
        # Scenario 3: Complete question despite pronouns
        {
            'query': 'What is the meaning of dharma in this context?',
            'history': [
                {'role': 'user', 'content': 'Explain the Bhagavad Gita'},
                {'role': 'assistant', 'content': 'The Bhagavad Gita discusses dharma...'}
            ],
            'stage1_result': {'proceed_to_stage_2': True, 'confidence': 0.5},
            'expected_ambiguous': False,
            'description': 'Complete despite "this context"'
        },
        # Scenario 4: Very short ambiguous query
        {
            'query': 'Why?',
            'history': [
                {'role': 'user', 'content': 'Should I practice yoga daily?'},
                {'role': 'assistant', 'content': 'Yes, daily yoga practice is beneficial...'}
            ],
            'stage1_result': {'proceed_to_stage_2': True, 'confidence': 0.6},
            'expected_ambiguous': True,
            'description': 'Ultra-short query needing context'
        }
    ]
    
    print("="*60)
    print("STAGE 2 SEMANTIC ANALYSIS TEST RESULTS")
    print("="*60)
    
    for i, test in enumerate(test_cases, 1):
        result = analyzer.process_stage2(
            test['query'], 
            test['history'], 
            test['stage1_result']
        )
        
        is_ambiguous = result.get('stage2_analysis', {}).get('is_ambiguous', False)
        
        print(f"\nTest {i}: {test['description']}")
        print(f"Query: '{test['query']}'")
        print(f"Expected ambiguous: {test['expected_ambiguous']}")
        print(f"Detected ambiguous: {is_ambiguous}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print(f"Reasoning: {result.get('reasoning', 'N/A')}")
        print(f"Proceed to Stage 3: {result.get('proceed_to_stage_3', False)}")
        
        # Show detailed analysis
        if 'stage2_analysis' in result:
            analysis = result['stage2_analysis']['analysis_details']
            if 'completeness' in analysis:
                print(f"Completeness: {analysis['completeness']['is_complete']}")
                if not analysis['completeness']['is_complete']:
                    print(f"  Missing: {analysis['completeness']['missing_elements']}")
            if 'unresolved_pronouns' in analysis:
                print(f"Unresolved pronouns: {analysis['unresolved_pronouns']}")
            if 'semantic_overlap' in analysis:
                print(f"Semantic overlap: {analysis['semantic_overlap']:.2f}")
        
        print(f"✅ PASS" if is_ambiguous == test['expected_ambiguous'] else "❌ FAIL")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run tests when executed directly
    test_stage2_semantic_analysis()
