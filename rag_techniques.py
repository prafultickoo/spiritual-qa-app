"""
RAG (Retrieval-Augmented Generation) Techniques Implementation
Provides multiple strategies for processing retrieved documents and generating answers.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import os
from dotenv import load_dotenv

# Import prompt templates
from prompts.answer_prompts import (
    select_prompt_template,
    format_context_from_chunks,
    ANSWER_BASE_PROMPT
)

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RAGTechniqueHandler:
    """Handler for different RAG techniques."""
    
    def __init__(self, llm_completion_func):
        """
        Initialize RAG technique handler.
        
        Args:
            llm_completion_func: Function to call LLM (should be the _create_llm_completion method)
        """
        self.llm_completion_func = llm_completion_func
        
    def generate_answer(self, 
                       technique: str,
                       question: str,
                       chunks: List[Dict[str, Any]],
                       **kwargs) -> Dict[str, Any]:
        """
        Generate answer using specified RAG technique.
        
        Args:
            technique: RAG technique to use
            question: User question
            chunks: Retrieved document chunks
            **kwargs: Additional parameters (reasoning_effort, etc.)
            
        Returns:
            Dict with generated answer and metadata
        """
        technique = technique.lower()
        
        if technique == 'stuff':
            return self._stuff_technique(question, chunks, **kwargs)
        elif technique == 'refine':
            return self._refine_technique(question, chunks, **kwargs)
        elif technique == 'map_reduce':
            return self._map_reduce_technique(question, chunks, **kwargs)
        elif technique == 'map_rerank':
            return self._map_rerank_technique(question, chunks, **kwargs)
        elif technique == 'selective':
            return self._selective_technique(question, chunks, **kwargs)
        else:
            logger.warning(f"Unknown RAG technique: {technique}, falling back to 'stuff'")
            return self._stuff_technique(question, chunks, **kwargs)
    
    def _stuff_technique(self, question: str, chunks: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Stuff technique: Combine all chunks into single context.
        ✅ Fast, cheap, simple
        ❌ Token limit constraints
        """
        if not chunks:
            return {
                "status": "success",
                "answer": "I couldn't find any relevant information to answer your question.",
                "technique_used": "stuff",
                "chunks_processed": 0,
                "api_calls": 0
            }
        
        # Use proper context formatting from prompt templates
        context = format_context_from_chunks(chunks)
        
        # Use proper prompt template that includes "Om!" greeting
        prompt_template = select_prompt_template(question)
        formatted_prompt = prompt_template.format(
            context=context,
            question=question
        )
        
        try:
            response = self.llm_completion_func(
                messages=[
                    {"role": "system", "content": "You are a spiritual expert assistant specializing in understanding and explaining complex spiritual concepts from various traditions."},
                    {"role": "user", "content": formatted_prompt}
                ],
                **kwargs
            )
            
            return {
                "status": "success",
                "answer": response.choices[0].message.content,
                "technique_used": "stuff",
                "chunks_processed": len(chunks),
                "api_calls": 1,
                "context_length": len(context)
            }
            
        except Exception as e:
            logger.error(f"Error in stuff technique: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "technique_used": "stuff",
                "chunks_processed": len(chunks),
                "api_calls": 0
            }
    
    def _refine_technique(self, question: str, chunks: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Refine technique: Iteratively refine answer with each chunk.
        ✅ High quality, handles many documents
        ❌ Expensive, slow
        """
        if not chunks:
            return {
                "status": "success",
                "answer": "I couldn't find any relevant information to answer your question.",
                "technique_used": "refine",
                "chunks_processed": 0,
                "api_calls": 0
            }
        
        try:
            api_calls = 0
            
            # Start with first chunk
            first_chunk = chunks[0]
            first_context = f"SOURCE: {first_chunk.get('metadata', {}).get('source', 'Unknown')}\n{first_chunk.get('content', '')}"
            
            initial_prompt = f"""Based on the following spiritual text, provide an initial answer to the question.

CONTEXT:
{first_context}

QUESTION: {question}

Provide a thoughtful initial answer based on this context."""
            
            response = self.llm_completion_func(
                messages=[
                    {"role": "system", "content": "You are a wise spiritual teacher providing guidance based on ancient wisdom."},
                    {"role": "user", "content": initial_prompt}
                ],
                **kwargs
            )
            api_calls += 1
            
            current_answer = response.choices[0].message.content
            
            # Refine with remaining chunks
            for i, chunk in enumerate(chunks[1:], 2):
                chunk_context = f"SOURCE: {chunk.get('metadata', {}).get('source', 'Unknown')}\n{chunk.get('content', '')}"
                
                refine_prompt = f"""You previously provided this answer to the question "{question}":

PREVIOUS ANSWER:
{current_answer}

Now, please refine and improve your answer using this additional spiritual text:

ADDITIONAL CONTEXT:
{chunk_context}

Provide an enhanced, more comprehensive answer that incorporates insights from both your previous response and this new context."""
                
                response = self.llm_completion_func(
                    messages=[
                        {"role": "system", "content": "You are refining a spiritual answer with additional wisdom texts."},
                        {"role": "user", "content": refine_prompt}
                    ],
                    **kwargs
                )
                api_calls += 1
                
                current_answer = response.choices[0].message.content
            
            return {
                "status": "success",
                "answer": current_answer,
                "technique_used": "refine",
                "chunks_processed": len(chunks),
                "api_calls": api_calls,
                "refinement_steps": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in refine technique: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "technique_used": "refine",
                "chunks_processed": len(chunks),
                "api_calls": api_calls if 'api_calls' in locals() else 0
            }
    
    def _map_reduce_technique(self, question: str, chunks: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Map-Reduce technique: Generate partial answers, then combine.
        ✅ Parallelizable, handles many documents
        ❌ Expensive, may lose nuance
        """
        if not chunks:
            return {
                "status": "success",
                "answer": "I couldn't find any relevant information to answer your question.",
                "technique_used": "map_reduce",
                "chunks_processed": 0,
                "api_calls": 0
            }
        
        try:
            # Phase 1: Map - Generate partial answers for each chunk
            partial_answers = []
            
            for chunk in chunks:
                chunk_context = f"SOURCE: {chunk.get('metadata', {}).get('source', 'Unknown')}\n{chunk.get('content', '')}"
                
                map_prompt = f"""Based solely on the following spiritual text, provide a partial answer to the question.

CONTEXT:
{chunk_context}

QUESTION: {question}

Provide a concise answer based only on this specific text. Focus on the key insights this source offers."""
                
                response = self.llm_completion_func(
                    messages=[
                        {"role": "system", "content": "You are analyzing a single spiritual text to provide focused insights."},
                        {"role": "user", "content": map_prompt}
                    ],
                    **kwargs
                )
                
                partial_answers.append(response.choices[0].message.content)
            
            # Phase 2: Reduce - Combine all partial answers
            combined_answers = "\n\n".join([f"INSIGHT {i+1}: {answer}" for i, answer in enumerate(partial_answers)])
            
            reduce_prompt = f"""You have received multiple partial answers to the question "{question}" from different spiritual sources. Please synthesize these insights into a comprehensive, coherent answer.

PARTIAL INSIGHTS:
{combined_answers}

Provide a unified, thoughtful answer that weaves together the key insights from all sources while maintaining spiritual wisdom and coherence."""
            
            final_response = self.llm_completion_func(
                messages=[
                    {"role": "system", "content": "You are synthesizing multiple spiritual insights into a unified wisdom teaching."},
                    {"role": "user", "content": reduce_prompt}
                ],
                **kwargs
            )
            
            return {
                "status": "success",
                "answer": final_response.choices[0].message.content,
                "technique_used": "map_reduce",
                "chunks_processed": len(chunks),
                "api_calls": len(chunks) + 1,  # Map calls + Reduce call
                "partial_answers_generated": len(partial_answers)
            }
            
        except Exception as e:
            logger.error(f"Error in map_reduce technique: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "technique_used": "map_reduce",
                "chunks_processed": len(chunks),
                "api_calls": len(chunks) if chunks else 0
            }
    
    def _map_rerank_technique(self, question: str, chunks: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Map-Rerank technique: Generate multiple answers, rank and select best.
        ✅ High quality (picks best answer)
        ❌ Expensive, may discard good insights
        """
        if not chunks:
            return {
                "status": "success",
                "answer": "I couldn't find any relevant information to answer your question.",
                "technique_used": "map_rerank",
                "chunks_processed": 0,
                "api_calls": 0
            }
        
        try:
            # Phase 1: Generate candidate answers from different chunks
            candidate_answers = []
            
            # Use up to 5 chunks to avoid too many candidates
            top_chunks = chunks[:5]
            
            for i, chunk in enumerate(top_chunks):
                chunk_context = f"SOURCE: {chunk.get('metadata', {}).get('source', 'Unknown')}\n{chunk.get('content', '')}"
                
                candidate_prompt = f"""Based on the following spiritual text, provide a complete answer to the question.

CONTEXT:
{chunk_context}

QUESTION: {question}

Provide a thoughtful, complete answer based on this spiritual source."""
                
                response = self.llm_completion_func(
                    messages=[
                        {"role": "system", "content": "You are providing spiritual guidance based on a specific wisdom text."},
                        {"role": "user", "content": candidate_prompt}
                    ],
                    **kwargs
                )
                
                candidate_answers.append({
                    "answer": response.choices[0].message.content,
                    "source": chunk.get('metadata', {}).get('source', f'Source {i+1}')
                })
            
            # Phase 2: Rank candidates and select/combine the best
            candidates_text = "\n\n".join([
                f"CANDIDATE {i+1} (from {candidate['source']}):\n{candidate['answer']}" 
                for i, candidate in enumerate(candidate_answers)
            ])
            
            rerank_prompt = f"""You have {len(candidate_answers)} candidate answers to the question "{question}" from different spiritual sources. 

CANDIDATE ANSWERS:
{candidates_text}

Please analyze these candidates and either:
1. Select the single best answer and explain why, OR
2. Synthesize the best elements from multiple candidates into a superior combined answer

Provide the final answer along with a brief explanation of your selection/synthesis reasoning."""
            
            final_response = self.llm_completion_func(
                messages=[
                    {"role": "system", "content": "You are selecting and refining the best spiritual wisdom from multiple candidate answers."},
                    {"role": "user", "content": rerank_prompt}
                ],
                **kwargs
            )
            
            return {
                "status": "success",
                "answer": final_response.choices[0].message.content,
                "technique_used": "map_rerank",
                "chunks_processed": len(chunks),
                "api_calls": len(top_chunks) + 1,  # Candidate calls + Rerank call
                "candidates_generated": len(candidate_answers)
            }
            
        except Exception as e:
            logger.error(f"Error in map_rerank technique: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "technique_used": "map_rerank",
                "chunks_processed": len(chunks),
                "api_calls": len(chunks[:5]) if chunks else 0
            }
    
    def _selective_technique(self, question: str, chunks: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Selective technique: Smart chunk selection based on relevance and token limits.
        ✅ Efficient token usage, good quality
        ❌ Requires relevance scoring logic
        """
        if not chunks:
            return {
                "status": "success",
                "answer": "I couldn't find any relevant information to answer your question.",
                "technique_used": "selective",
                "chunks_processed": 0,
                "api_calls": 0
            }
        
        try:
            # Estimate token usage (rough approximation: 4 chars = 1 token)
            max_tokens = kwargs.get('max_context_tokens', 6000)  # Conservative limit
            
            # Score chunks by relevance (simple scoring based on question keywords)
            question_words = set(question.lower().split())
            scored_chunks = []
            
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                metadata = chunk.get('metadata', {})
                
                # Simple relevance scoring
                relevance_score = sum(1 for word in question_words if word in content)
                
                # Boost score for chunks with verses (important for spiritual content)
                if metadata.get('has_verses', False):
                    relevance_score += 2
                
                # Estimate token count
                estimated_tokens = len(chunk.get('content', '')) // 4
                
                scored_chunks.append({
                    'chunk': chunk,
                    'relevance_score': relevance_score,
                    'estimated_tokens': estimated_tokens
                })
            
            # Sort by relevance score (descending)
            scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Select chunks within token limit
            selected_chunks = []
            total_tokens = 0
            
            for scored_chunk in scored_chunks:
                if total_tokens + scored_chunk['estimated_tokens'] <= max_tokens:
                    selected_chunks.append(scored_chunk['chunk'])
                    total_tokens += scored_chunk['estimated_tokens']
                else:
                    break
            
            # If no chunks selected, take the most relevant one
            if not selected_chunks and scored_chunks:
                selected_chunks = [scored_chunks[0]['chunk']]
            
            # Use stuff technique on selected chunks
            result = self._stuff_technique(question, selected_chunks, **kwargs)
            
            # Update technique metadata
            result.update({
                "technique_used": "selective",
                "chunks_available": len(chunks),
                "chunks_selected": len(selected_chunks),
                "token_budget": max_tokens,
                "estimated_tokens_used": total_tokens
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in selective technique: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "technique_used": "selective",
                "chunks_processed": len(chunks),
                "api_calls": 0
            }

# Technique descriptions for frontend
RAG_TECHNIQUES = {
    "stuff": {
        "name": "📦 Stuff (Fast)",
        "description": "Combine all context into one request. Fastest and cheapest option.",
        "pros": ["⚡ Fastest response", "💰 Lowest cost", "🔧 Simple and reliable"],
        "cons": ["📏 Token limit constraints"],
        "best_for": "Quick questions, limited context",
        "cost_multiplier": 1.0,
        "speed_rating": 5
    },
    "refine": {
        "name": "🔄 Refine (Quality)",
        "description": "Iteratively improve answer with each document. Highest quality.",
        "pros": ["⭐ Highest quality", "📚 Handles many documents", "🎯 Thorough analysis"],
        "cons": ["💸 Most expensive", "⏳ Slowest", "🧠 High reasoning token usage"],
        "best_for": "Complex questions requiring deep analysis",
        "cost_multiplier": 3.0,
        "speed_rating": 1
    },
    "map_reduce": {
        "name": "🗺️ Map-Reduce (Parallel)",
        "description": "Generate partial answers, then combine. Good for broad questions.",
        "pros": ["⚡ Parallelizable", "📚 Handles many documents", "🎯 Comprehensive coverage"],
        "cons": ["💸 Expensive", "🤔 May lose nuance"],
        "best_for": "Broad questions with multiple aspects",
        "cost_multiplier": 2.5,
        "speed_rating": 3
    },
    "map_rerank": {
        "name": "🏆 Map-Rerank (Best Answer)",
        "description": "Generate multiple answers, select the best. Excellent quality.",
        "pros": ["🎯 Picks best answer", "⭐ High quality", "🔍 Multiple perspectives"],
        "cons": ["💸 Expensive", "🗑️ May discard good insights"],
        "best_for": "Questions with multiple valid approaches",
        "cost_multiplier": 2.8,
        "speed_rating": 2
    },
    "selective": {
        "name": "🎯 Selective (Smart)",
        "description": "Intelligently select most relevant chunks. Balanced approach.",
        "pros": ["🧠 Smart selection", "💰 Efficient token usage", "⚖️ Good balance"],
        "cons": ["🤔 May miss some context"],
        "best_for": "Large document sets, balanced quality/cost",
        "cost_multiplier": 1.2,
        "speed_rating": 4
    }
}
