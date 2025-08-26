"""
Stage 3: LLM-Powered Query Intent Classification
Uses multi-shot learning to classify follow-up queries
"""
import json
import logging
from typing import List, Dict, Optional, Any
from utils.llm_timeout import (
    call_chat_completion_with_timeout,
    get_llm_timeout_default,
)
import uuid

# Configure logging
logger = logging.getLogger(__name__)


class LLMQueryClassifier:
    """
    Stage 3: LLM-based intent classification for follow-up queries.
    Uses multi-shot examples to classify intent and determine action.
    """
    
    def __init__(self):
        # Multi-shot examples for training the LLM
        self.classification_examples = """
CLASSIFICATION EXAMPLES:

Example 1:
History: User asked "What is karma yoga?". Assistant explained karma yoga as the path of selfless action.
Query: "Summarize this in 3 bullet points"
Output: {
  "intent": "reformatting",
  "action": "reformat_previous",
  "needs_rag": false,
  "enhancement_needed": false,
  "explanation": "User wants the previous answer reformatted as bullet points"
}

Example 2:
History: User asked about meditation techniques. Assistant described various meditation practices.
Query: "Give examples from the corporate world"
Output: {
  "intent": "perspective_application",
  "action": "apply_perspective",
  "needs_rag": false,
  "enhancement_needed": false,
  "explanation": "User wants corporate/practical examples of the spiritual concept discussed"
}

Example 3:
History: User asked about dharma. Assistant provided detailed explanation with Sanskrit verses.
Query: "Can you explain this without Sanskrit terms?"
Output: {
  "intent": "content_modification",
  "action": "modify_content",
  "needs_rag": false,
  "enhancement_needed": false,
  "explanation": "User wants same content but filtered to remove Sanskrit terminology"
}

Example 4:
History: User asked "What is Turiya?". Assistant explained it as the fourth state of consciousness.
Query: "Tell me more about this"
Output: {
  "intent": "information_expansion",
  "action": "enhance_and_retrieve",
  "needs_rag": true,
  "enhancement_needed": true,
  "explanation": "Ambiguous 'this' refers to Turiya - need to enhance query before RAG search"
}

Example 5:
History: User asked about enlightenment. Assistant gave comprehensive answer.
Query: "How does it work?"
Output: {
  "intent": "information_expansion",
  "action": "enhance_and_retrieve",
  "needs_rag": true,
  "enhancement_needed": true,
  "explanation": "Pronoun 'it' needs to be resolved to 'enlightenment' before searching"
}

Example 6:
History: User asked about different yoga types. Assistant listed and explained them.
Query: "Which one is best for beginners?"
Output: {
  "intent": "specific_question",
  "action": "enhance_and_retrieve",
  "needs_rag": true,
  "enhancement_needed": true,
  "explanation": "Need to specify 'which yoga type' for proper context in search"
}

Example 7:
History: User asked about consciousness. Assistant explained various states.
Query: "Compare this with modern neuroscience"
Output: {
  "intent": "comparison_request",
  "action": "hybrid_approach",
  "needs_rag": true,
  "enhancement_needed": true,
  "explanation": "Need both document search for consciousness info and LLM knowledge of neuroscience"
}

Example 8:
History: User learned about meditation benefits. Assistant listed physical and mental benefits.
Query: "How can I start practicing?"
Output: {
  "intent": "application_guidance",
  "action": "enhance_and_retrieve",
  "needs_rag": true,
  "enhancement_needed": true,
  "explanation": "User wants practical guidance on starting meditation practice"
}

Example 9:
History: User asked about moksha. Assistant explained liberation concept.
Query: "Is this similar to nirvana?"
Output: {
  "intent": "concept_comparison",
  "action": "enhance_and_retrieve",
  "needs_rag": true,
  "enhancement_needed": true,
  "explanation": "Need to search for both moksha and nirvana to provide comparison"
}

Example 10:
History: User asked about Bhagavad Gita. Assistant provided overview.
Query: "What are the main teachings?"
Output: {
  "intent": "information_expansion",
  "action": "enhance_and_retrieve",
  "needs_rag": true,
  "enhancement_needed": false,
  "explanation": "Clear question about Bhagavad Gita teachings - no ambiguity to resolve"
}"""

        # Classification prompt template
        self.classification_prompt_template = """You are a query intent classifier for a spiritual Q&A system. 
Given a conversation history and a follow-up query, classify the user's intent and determine the appropriate action.

{examples}

CLASSIFICATION TASK:
History: {conversation_summary}
Query: "{user_query}"

Analyze the query and return a JSON object with these fields:
- intent: The type of follow-up (reformatting, perspective_application, content_modification, information_expansion, specific_question, comparison_request, application_guidance, concept_comparison, etc.)
- action: What the system should do (reformat_previous, apply_perspective, modify_content, enhance_and_retrieve, hybrid_approach)
- needs_rag: Whether document retrieval is needed (true/false)
- enhancement_needed: Whether the query needs context enhancement before processing (true/false)
- explanation: Brief explanation of your classification decision

Output only the JSON object, no additional text:"""

    def summarize_conversation(self, conversation_history: List[Dict]) -> str:
        """
        Create a concise summary of the conversation for the classifier.
        
        Args:
            conversation_history: List of conversation turns
            
        Returns:
            Summary string
        """
        if not conversation_history:
            return "No previous conversation"
        
        # Get last 2 exchanges (4 messages max)
        recent_history = conversation_history[-4:]
        summary_parts = []
        
        for item in recent_history:
            role = item.get('role', 'unknown')
            content = item.get('content', '')
            # Truncate long content
            if len(content) > 200:
                content = content[:197] + "..."
            
            if role == 'user':
                summary_parts.append(f"User asked: {content}")
            elif role == 'assistant':
                summary_parts.append(f"Assistant explained: {content}")
        
        return " ".join(summary_parts)

    def classify_query_intent(self, query: str, conversation_history: List[Dict], 
                            llm_client: Any, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Classify the intent of a follow-up query using LLM.
        
        Args:
            query: The follow-up query
            conversation_history: Previous conversation
            llm_client: LLM client for classification
            model: LLM model to use for classification
            
        Returns:
            Classification result dict
        """
        try:
            # Build the classification prompt
            prompt = self.classification_prompt_template.format(
                examples=self.classification_examples,
                conversation_summary=self.summarize_conversation(conversation_history),
                user_query=query
            )
            
            # Debug: print first part of prompt to see query
            # print(f"\nDEBUG: Prompt preview:\n{prompt[-200:]}\n")
            
            # Call LLM for classification using centralized timeout wrapper
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a query classification assistant. Analyze queries and return JSON classification."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,  # Deterministic classification
                "max_tokens": 200,  # Classification output is small
                "response_format": {"type": "json_object"},  # Ensure JSON response
            }
            timeout_seconds = get_llm_timeout_default()
            request_id = f"intent-classify-{uuid.uuid4().hex[:8]}"
            response = call_chat_completion_with_timeout(
                client=llm_client,
                params=params,
                timeout_seconds=timeout_seconds,
                request_id=request_id,
                logger=logger,
            )
            
            # Parse JSON response from chat completion
            classification = json.loads(response.choices[0].message.content.strip())
            
            # Validate required fields
            required_fields = ['intent', 'action', 'needs_rag', 'enhancement_needed', 'explanation']
            for field in required_fields:
                if field not in classification:
                    raise ValueError(f"Missing required field: {field}")
            
            logger.info(f"Query classification: intent={classification['intent']}, "
                       f"action={classification['action']}, needs_rag={classification['needs_rag']}")
            
            return classification
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM classification response: {e}")
            # Fallback classification
            return {
                "intent": "information_expansion",
                "action": "enhance_and_retrieve",
                "needs_rag": True,
                "enhancement_needed": True,
                "explanation": "Failed to classify - defaulting to safe information expansion"
            }
        except TimeoutError as e:
            rid = locals().get("request_id", "n/a")
            to = locals().get("timeout_seconds", "n/a")
            logger.error(f"LLM classification timed out after {to}s (request_id={rid}): {e}")
            # Safe fallback on timeout
            return {
                "intent": "unknown",
                "action": "enhance_and_retrieve",
                "needs_rag": True,
                "enhancement_needed": False,
                "explanation": "Classification timed out; using safe default action"
            }
        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}")
            # Safe fallback
            return {
                "intent": "unknown",
                "action": "enhance_and_retrieve",
                "needs_rag": True,
                "enhancement_needed": False,
                "explanation": f"Classification error: {str(e)}"
            }

    def process_stage3(self, query: str, conversation_history: List[Dict], 
                       stage2_result: Dict, llm_client: Any, 
                       model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Main entry point for Stage 3 processing.
        
        Args:
            query: Current query
            conversation_history: Conversation history
            stage2_result: Results from Stage 2
            llm_client: LLM client for classification
            model: LLM model to use for classification
            
        Returns:
            Enhanced result with classification
        """
        # Only process if previous stages recommended it
        if not stage2_result.get('proceed_to_stage_3', False):
            logger.info("Stage 3: Not needed based on Stage 2 results")
            return stage2_result
        
        logger.info(f"Stage 3: Classifying query intent for: '{query}'")
        
        # Classify the query
        classification = self.classify_query_intent(query, conversation_history, llm_client, model)
        
        # Merge results
        result = {
            'original_query': query,
            'processed_query': query,  # Will be enhanced later if needed
            'stage_completed': 3,
            'stage1_result': stage2_result.get('stage1_result'),
            'stage2_result': stage2_result.get('stage2_analysis'),
            'stage3_classification': classification,
            'final_action': classification['action'],
            'needs_rag': classification['needs_rag'],
            'needs_enhancement': classification['enhancement_needed']
        }
        
        return result


# Mock LLM client for testing
class MockLLMClient:
    """Mock LLM client for testing without actual API calls."""
    
    def complete(self, prompt: str, temperature: float = 0, max_tokens: int = 200) -> str:
        """Mock LLM completion based on query content."""
        prompt_lower = prompt.lower()
        
        # Find the LAST occurrence of 'query: "' (the actual query, not from examples)
        query_marker = 'query: "'
        last_query_pos = prompt_lower.rfind(query_marker)  # rfind gets last occurrence
        
        if last_query_pos != -1:
            query_start = last_query_pos + len(query_marker)
            query_end = prompt_lower.find('"', query_start)
            if query_end > query_start:
                query = prompt_lower[query_start:query_end]
            else:
                query = prompt_lower
        else:
            query = prompt_lower
        
        # Debug output
        # print(f"DEBUG: Extracted query: '{query}'")
        
        # Pattern matching on the extracted query
        if "summarize" in query and ("bullet" in query or "points" in query):
            return json.dumps({
                "intent": "reformatting",
                "action": "reformat_previous",
                "needs_rag": False,
                "enhancement_needed": False,
                "explanation": "User wants bullet point summary of previous answer"
            })
        elif "tell me more" in query:
            return json.dumps({
                "intent": "information_expansion",
                "action": "enhance_and_retrieve",
                "needs_rag": True,
                "enhancement_needed": True,
                "explanation": "Ambiguous request needs context enhancement"
            })
        elif "corporate" in query and "example" in query:
            return json.dumps({
                "intent": "perspective_application",
                "action": "apply_perspective",
                "needs_rag": False,
                "enhancement_needed": False,
                "explanation": "Apply corporate context to spiritual concept"
            })
        elif "without sanskrit" in query:
            return json.dumps({
                "intent": "content_modification",
                "action": "modify_content",
                "needs_rag": False,
                "enhancement_needed": False,
                "explanation": "Filter Sanskrit from previous response"
            })
        
        # Default response
        return json.dumps({
            "intent": "information_expansion",
            "action": "enhance_and_retrieve",
            "needs_rag": True,
            "enhancement_needed": True,
            "explanation": "Default classification for unknown query"
        })


# Test function
def test_stage3_classification():
    """Test Stage 3 LLM classification with various scenarios."""
    
    classifier = LLMQueryClassifier()
    mock_llm = MockLLMClient()
    
    # Test scenarios
    test_cases = [
        # Scenario 1: Reformatting request
        {
            'query': 'Summarize this in bullet points',
            'history': [
                {'role': 'user', 'content': 'What is karma yoga?'},
                {'role': 'assistant', 'content': 'Karma yoga is the path of selfless action...'}
            ],
            'stage2_result': {'proceed_to_stage_3': True},
            'expected_action': 'reformat_previous',
            'expected_rag': False,
            'description': 'Reformatting request'
        },
        # Scenario 2: Information expansion
        {
            'query': 'Tell me more about this',
            'history': [
                {'role': 'user', 'content': 'What is Turiya?'},
                {'role': 'assistant', 'content': 'Turiya is the fourth state of consciousness...'}
            ],
            'stage2_result': {'proceed_to_stage_3': True},
            'expected_action': 'enhance_and_retrieve',
            'expected_rag': True,
            'description': 'Information expansion with ambiguity'
        },
        # Scenario 3: Perspective shift
        {
            'query': 'Give corporate examples',
            'history': [
                {'role': 'user', 'content': 'Explain dharma'},
                {'role': 'assistant', 'content': 'Dharma refers to righteous living...'}
            ],
            'stage2_result': {'proceed_to_stage_3': True},
            'expected_action': 'apply_perspective',
            'expected_rag': False,
            'description': 'Perspective application'
        },
        # Scenario 4: Content modification
        {
            'query': 'Explain without Sanskrit terms',
            'history': [
                {'role': 'user', 'content': 'What is moksha?'},
                {'role': 'assistant', 'content': 'Moksha (मोक्ष) is liberation...'}
            ],
            'stage2_result': {'proceed_to_stage_3': True},
            'expected_action': 'modify_content',
            'expected_rag': False,
            'description': 'Content filtering request'
        }
    ]
    
    print("="*60)
    print("STAGE 3 LLM CLASSIFICATION TEST RESULTS")
    print("="*60)
    
    for i, test in enumerate(test_cases, 1):
        result = classifier.process_stage3(
            test['query'],
            test['history'],
            test['stage2_result'],
            mock_llm
        )
        
        classification = result.get('stage3_classification', {})
        
        print(f"\nTest {i}: {test['description']}")
        print(f"Query: '{test['query']}'")
        print(f"Expected action: {test['expected_action']}")
        print(f"Detected action: {classification.get('action', 'N/A')}")
        print(f"Expected RAG: {test['expected_rag']}")
        print(f"Detected RAG: {classification.get('needs_rag', 'N/A')}")
        print(f"Intent: {classification.get('intent', 'N/A')}")
        print(f"Enhancement needed: {classification.get('enhancement_needed', 'N/A')}")
        print(f"Explanation: {classification.get('explanation', 'N/A')}")
        
        # Check if it matches expected
        action_match = classification.get('action') == test['expected_action']
        rag_match = classification.get('needs_rag') == test['expected_rag']
        
        print(f"✅ PASS" if action_match and rag_match else "❌ FAIL")
    
    print("\n" + "="*60)


def test_full_pipeline_integration():
    """
    Integration test demonstrating the full 3-stage pipeline.
    """
    from conversation_context import ConversationContextProcessor
    from semantic_analyzer import SemanticAnalyzer
    
    print("\n" + "="*60)
    print("FULL PIPELINE INTEGRATION TEST")
    print("="*60)
    
    # Initialize all components
    context_processor = ConversationContextProcessor()
    semantic_analyzer = SemanticAnalyzer()
    classifier = LLMQueryClassifier()
    mock_llm = MockLLMClient()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Clear standalone query',
            'query': 'What is the meaning of karma in Hinduism?',
            'history': [],
            'expected_flow': 'Direct to RAG (no enhancement needed)'
        },
        {
            'name': 'Simple follow-up with pronoun',
            'query': 'Tell me more about this',
            'history': [
                {'role': 'user', 'content': 'What is karma?'},
                {'role': 'assistant', 'content': 'Karma is the law of cause and effect...'}
            ],
            'expected_flow': 'Stage 1 → Stage 2 → Stage 3 → Enhancement → RAG'
        },
        {
            'name': 'Reformatting request',
            'query': 'Can you summarize that in bullet points?',
            'history': [
                {'role': 'user', 'content': 'Explain dharma'},
                {'role': 'assistant', 'content': 'Dharma is a complex concept...'}
            ],
            'expected_flow': 'Stage 1 → Stage 2 → Stage 3 → Reformat (no RAG)'
        },
        {
            'name': 'Perspective application',
            'query': 'How can I apply these principles in my workplace?',
            'history': [
                {'role': 'user', 'content': 'What are the Yamas and Niyamas?'},
                {'role': 'assistant', 'content': 'The Yamas and Niyamas are ethical guidelines...'}
            ],
            'expected_flow': 'Stage 1 → Stage 2 → Stage 3 → Apply perspective (no RAG)'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n\nSCENARIO: {scenario['name']}")
        print("-" * 40)
        print(f"Query: '{scenario['query']}'")
        print(f"Expected flow: {scenario['expected_flow']}")
        print("\nPipeline execution:")
        
        # Stage 1: Heuristic detection
        stage1_result = context_processor.is_follow_up_query(scenario['query'], scenario['history'])
        is_followup = stage1_result['is_follow_up']
        print(f"  Stage 1 (Heuristic): Follow-up detected = {is_followup}")
        print(f"                      Confidence = {stage1_result['confidence']}")
        
        if is_followup:
            # Stage 2: Semantic analysis
            stage2_result = semantic_analyzer.analyze_query_ambiguity(
                scenario['query'], 
                scenario['history']
            )
            print(f"  Stage 2 (Semantic): Is ambiguous = {stage2_result.get('is_ambiguous', False)}")
            print(f"                     Needs context = {stage2_result.get('needs_context', False)}")
            print(f"                     Confidence = {stage2_result.get('confidence', 0)}")
            
            if stage2_result.get('needs_context', False):
                # Add flag for Stage 3 processing
                stage2_result['proceed_to_stage_3'] = True
                
                # Stage 3: LLM classification
                stage3_result = classifier.process_stage3(
                    scenario['query'],
                    scenario['history'],
                    stage2_result,
                    mock_llm
                )
                classification = stage3_result.get('stage3_classification', {})
                
                print(f"  Stage 3 (LLM): Intent = {classification['intent']}")
                print(f"                 Action = {classification['action']}")
                print(f"                 Needs RAG = {classification['needs_rag']}")
                print(f"                 Enhancement needed = {classification['enhancement_needed']}")
                print(f"                 Explanation: {classification['explanation']}")
                
                # Final decision
                if classification['action'] == 'enhance_and_retrieve':
                    print("\n  → DECISION: Enhance query with context, then retrieve from RAG")
                elif classification['action'] == 'reformat_previous':
                    print("\n  → DECISION: Reformat previous answer (no RAG needed)")
                elif classification['action'] == 'apply_perspective':
                    print("\n  → DECISION: Apply new perspective to previous answer")
                else:
                    print(f"\n  → DECISION: {classification['action']}")
            else:
                print("\n  → DECISION: Query is clear enough, proceed to RAG directly")
        else:
            print("\n  → DECISION: Not a follow-up, proceed to RAG directly")
    
    print("\n" + "="*60)
    print("Pipeline integration test complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run tests when executed directly
    print("\nRunning Stage 3 classification tests...")
    test_stage3_classification()
    
    print("\n\nRunning full pipeline integration test...")
    test_full_pipeline_integration()
