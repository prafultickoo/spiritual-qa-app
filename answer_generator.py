"""
Answer generator for creating detailed responses to spiritual questions.
"""
import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from rag_techniques import RAGTechniqueHandler
from enhanced_document_retriever import EnhancedDocumentRetriever
from document_retriever import DocumentRetriever, retrieve_for_query
import uuid

# Timeout helper for OpenAI calls
from utils.llm_timeout import (
    call_chat_completion_with_timeout,
    get_llm_timeout_default,
)

# Import prompt templates
from prompts.answer_prompts import (
    select_prompt_template,
    format_context_from_chunks,
    ANSWER_BASE_PROMPT
)

# Initialize OpenAI integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("answer_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AnswerGenerator:
    """Class for generating answers to spiritual questions using LLMs."""
    
    def __init__(self, 
                vector_store_dir: str,
                embedding_model: str = "openai",
                llm_model: str = "gpt-5",
                temperature: float = 0.0,
                max_tokens: int = 1024,
                reasoning_effort: str = "medium",
                enable_dual_source: bool = True):
        """
        Initialize the answer generator.
        
        Args:
            vector_store_dir: Directory containing the vector database
            embedding_model: Name of embedding model to use ('openai' or 'huggingface')
            llm_model: LLM model for generating answers
            temperature: Temperature parameter for the LLM (not used for o3-mini)
            max_tokens: Maximum tokens for LLM response
            reasoning_effort: For o3-mini reasoning models ("low", "medium", "high")
            enable_dual_source: Whether to enable dual-source retrieval
        """
        self.vector_store_dir = vector_store_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.enable_dual_source = enable_dual_source
        
        # Store OpenAI client for context enhancement
        self.llm_client = client
        # Default timeout for LLM requests (seconds)
        self.request_timeout_seconds = get_llm_timeout_default()
        
        # Initialize document retriever (Enhanced or Standard based on flag)
        if self.enable_dual_source:
            try:
                self.retriever = EnhancedDocumentRetriever(
                    vector_store_dir=vector_store_dir,
                    embedding_model=embedding_model,
                    enable_dual_source=True,
                    enable_context_enhancement=True  # Enable context-aware retrieval
                )
                logger.info("Initialized Enhanced DocumentRetriever with dual-source support")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced DocumentRetriever: {str(e)}")
                logger.warning("Falling back to standard DocumentRetriever")
                self.retriever = DocumentRetriever(
                    vector_store_dir=vector_store_dir,
                    embedding_model=embedding_model
                )
                self.enable_dual_source = False
        else:
            self.retriever = DocumentRetriever(
                vector_store_dir=vector_store_dir,
                embedding_model=embedding_model
            )
            logger.info("Initialized standard DocumentRetriever (dual-source disabled)")
        
        # Initialize RAG technique handler
        self.rag_handler = RAGTechniqueHandler(self._create_llm_completion)
        
        logger.info(f"Initialized answer generator with LLM: {llm_model}")
    
    def _create_llm_completion(self, messages: List[Dict[str, str]], **kwargs):
        """
        Create LLM completion with model-specific parameters.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            OpenAI completion response
        """
        # Per-call timeout override; falls back to instance default
        timeout_seconds = kwargs.pop("timeout", self.request_timeout_seconds)
        request_id = kwargs.pop("request_id", str(uuid.uuid4())[:8])
        # Check if this is o3-mini reasoning model
        if "o3-mini" in self.llm_model.lower():
            # o3-mini specific parameters
            params = {
                "model": self.llm_model,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                "reasoning_effort": kwargs.get("reasoning_effort", self.reasoning_effort)
            }
            
            # Add optional parameters if supported
            if "stream" in kwargs:
                params["stream"] = kwargs["stream"]
                
        elif "gpt-5" in self.llm_model.lower():
            # GPT-5 uses max_completion_tokens and needs higher limits for reasoning + content
            # GPT-5 uses ~300-500 tokens for reasoning, so we need buffer for actual content
            gpt5_tokens = kwargs.get("max_tokens", self.max_tokens)
            if gpt5_tokens < 1000:
                gpt5_tokens = 2000  # Ensure enough tokens for reasoning + content
            params = {
                "model": self.llm_model,
                "messages": messages,
                "max_completion_tokens": gpt5_tokens
            }
            # Remove temperature from kwargs to avoid passing it to GPT-5
            kwargs.pop("temperature", None)
            
            # Add optional standard parameters
            if "top_p" in kwargs:
                params["top_p"] = kwargs["top_p"]
            if "frequency_penalty" in kwargs:
                params["frequency_penalty"] = kwargs["frequency_penalty"]
            if "presence_penalty" in kwargs:
                params["presence_penalty"] = kwargs["presence_penalty"]
            if "stream" in kwargs:
                params["stream"] = kwargs["stream"]
                
        else:
            # Standard GPT models (gpt-4, gpt-4o, etc.)
            params = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens)
            }
            
            # Add optional standard parameters
            if "top_p" in kwargs:
                params["top_p"] = kwargs["top_p"]
            if "frequency_penalty" in kwargs:
                params["frequency_penalty"] = kwargs["frequency_penalty"]
            if "presence_penalty" in kwargs:
                params["presence_penalty"] = kwargs["presence_penalty"]
            if "stream" in kwargs:
                params["stream"] = kwargs["stream"]
                
        logger.info(
            f"Creating LLM completion for {self.llm_model} with params: {list(params.keys())} "
            f"timeout={timeout_seconds}s request_id={request_id}"
        )
        
        try:
            return call_chat_completion_with_timeout(
                client=self.llm_client,
                params=params,
                timeout_seconds=timeout_seconds,
                request_id=request_id,
                logger=logger,
            )
        except TimeoutError as te:
            logger.exception(f"LLM call timed out for model {self.llm_model} (request_id={request_id}): {te}")
            raise
        except Exception as e:
            logger.exception(f"LLM call failed for model {self.llm_model} (request_id={request_id}): {e}")
            raise
    
    def retrieve_relevant_chunks(self, 
                                query: str, 
                                k: int = 5,
                                use_mmr: bool = True,
                                conversation_history: Optional[List[Dict]] = None,
                                llm_client: Optional[Any] = None) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a query with optional context enhancement.
        
        Args:
            query: User query text
            k: Number of chunks to retrieve
            use_mmr: Whether to use MMR for diverse retrieval
            conversation_history: Optional conversation history for context enhancement
            llm_client: Optional LLM client for advanced context processing
            
        Returns:
            Dict containing retrieved chunks
        """
        # Check if retriever supports context-aware retrieval
        if hasattr(self.retriever, 'retrieve_chunks_with_context') and conversation_history:
            # Use context-aware retrieval
            return self.retriever.retrieve_chunks_with_context(
                query=query,
                conversation_history=conversation_history,
                k=k,
                use_mmr=use_mmr,
                llm_client=llm_client,
                model=self.llm_model  # Pass the user's selected model
            )
        else:
            # Fallback to standard retrieval
            return self.retriever.retrieve_chunks(
                query=query,
                k=k,
                use_mmr=use_mmr
            )
    
    def generate_answer(self, 
                       query: str, 
                       k: int = 5,
                       use_mmr: bool = True,
                       diversity: float = 0.3,
                       query_type: str = None,
                       conversation_history: List[Dict[str, str]] = None,
                       rag_technique: str = "stuff",
                       reasoning_effort: str = "medium",
                       reading_style: str = "balanced") -> Dict[str, Any]:
        """
        Generate an answer for a user query.
        
        Args:
            query: User query text
            k: Number of chunks to retrieve
            use_mmr: Whether to use MMR for diverse retrieval
            diversity: Diversity parameter for MMR (higher = more diverse)
            chunks: Optional pre-retrieved chunks (if None, retrieves chunks)
            k: Number of chunks to retrieve if needed
            query_type: Type of query for prompt selection
            custom_prompt: Optional custom prompt template
            
        Returns:
            Dict with generated answer and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Initialize chunks variable
            chunks = None
            
            # Retrieve chunks if not provided
            if not chunks:
                retrieval_start = time.time()
                retrieval_result = self.retrieve_relevant_chunks(query, k)
                retrieval_time = time.time() - retrieval_start
                print(f"🔍 Document retrieval took: {retrieval_time:.2f} seconds")
                
                if retrieval_result.get("status") != "success":
                    return {
                        "status": "error",
                        "error": retrieval_result.get("error", "Failed to retrieve chunks"),
                        "answer": "I couldn't find relevant information to answer your question."
                    }
                chunks = retrieval_result.get("chunks", [])
            
            # Use RAG technique handler to generate answer
            rag_start = time.time()
            result = self.rag_handler.generate_answer(
                technique=rag_technique,
                question=query,
                chunks=chunks,
                reasoning_effort=reasoning_effort
            )
            rag_time = time.time() - rag_start
            print(f"🤖 LLM generation took: {rag_time:.2f} seconds (Model: {self.llm_model})")
            
            # Add model information
            if result.get("status") == "success":
                result.update({
                    "model": self.llm_model,
                    "prompt_type": query_type or "automatic"
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "answer": "An error occurred while generating the answer."
            }
    
    def generate_answer_with_context(self,
                               query: str,
                               conversation_context: str,
                               max_docs: int = 5,
                               k: int = 10,
                               use_mmr: bool = True,
                               diversity: float = 0.3,
                               reasoning_effort: str = None,
                               rag_technique: str = 'stuff') -> Dict[str, Any]:
        """
        Generate an answer for a user query with conversation history context.
        
        Args:
            query: User query text
            conversation_context: Previous conversation history
            max_docs: Maximum number of documents to include in context
            k: Number of chunks to retrieve initially
            use_mmr: Whether to use MMR for diverse retrieval
            diversity: Diversity parameter for MMR (higher = more diverse)
            
        Returns:
            Dict with generated answer, sources, and metadata
        """
        try:
            # Parse conversation history from string format to list of dicts
            conversation_history = []
            if conversation_context:
                # Convert conversation context string to structured format
                lines = conversation_context.strip().split('\n')
                for line in lines:
                    if line.startswith('User:'):
                        conversation_history.append({
                            'role': 'user',
                            'content': line[5:].strip()
                        })
                    elif line.startswith('Assistant:'):
                        conversation_history.append({
                            'role': 'assistant',
                            'content': line[10:].strip()
                        })
            
            # Retrieve chunks for the query with conversation history
            retrieval_result = self.retrieve_relevant_chunks(
                query, 
                k, 
                use_mmr,
                conversation_history=conversation_history,
                llm_client=self.llm_client  # Pass OpenAI client for Stage 3 classification
            )
            
            if retrieval_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": retrieval_result.get("error", "Failed to retrieve chunks"),
                    "answer": "I couldn't find relevant information to answer your question.",
                    "model": self.llm_model
                }
            
            # CHECK FOR CONTEXT ENHANCEMENT ACTIONS
            context_action = retrieval_result.get("context_action")
            if context_action and context_action != "retrieve_and_answer":
                logger.info(f"Handling context enhancement action: {context_action}")
                
                if context_action == "reformat_previous":
                    # Extract previous answer from conversation history
                    previous_answer = ""
                    if conversation_history:
                        for item in reversed(conversation_history):
                            if item.get('role') == 'assistant':
                                previous_answer = item.get('content', '')
                                break
                    
                    if previous_answer:
                        # Use LLM to reformat the previous answer according to the user's request
                        return self._reformat_previous_answer(query, previous_answer)
                    else:
                        return {
                            "status": "error",
                            "error": "No previous answer found to reformat",
                            "answer": "I don't have a previous answer to reformat.",
                            "model": self.llm_model
                        }
                
                elif context_action == "clarify_previous":
                    # Extract previous answer from conversation history
                    previous_answer = ""
                    if conversation_history:
                        for item in reversed(conversation_history):
                            if item.get('role') == 'assistant':
                                previous_answer = item.get('content', '')
                                break
                    
                    if previous_answer:
                        # Use LLM to clarify or expand the previous answer
                        return self._clarify_previous_answer(query, previous_answer)
                    else:
                        return {
                            "status": "error",
                            "error": "No previous answer found to clarify",
                            "answer": "I don't have a previous answer to clarify.",
                            "model": self.llm_model
                        }
                
                elif context_action == "modify_content":
                    # Extract previous answer from conversation history
                    previous_answer = ""
                    if conversation_history:
                        for item in reversed(conversation_history):
                            if item.get('role') == 'assistant':
                                previous_answer = item.get('content', '')
                                break
                    
                    if previous_answer:
                        # Use LLM to modify the previous answer according to the user's request
                        return self._modify_previous_answer(query, previous_answer)
                    else:
                        return {
                            "status": "error",
                            "error": "No previous answer found to modify",
                            "answer": "I don't have a previous answer to modify.",
                            "model": self.llm_model
                        }
                
                # Add more context actions as needed
                
            chunks = retrieval_result.get("chunks", [])
            
            # If no chunks were retrieved, return early
            if not chunks:
                return {
                    "status": "success",
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "chunks_used": 0,
                    "model": self.llm_model,
                    "sources": [],
                    "verses": []
                }
            
            # Check relevance scores to determine if chunks are actually relevant
            # Set threshold based on retrieval method (MMR scores tend to be higher)
            relevance_threshold = 0.78 if use_mmr else 0.55
            
            # Calculate average relevance score
            relevance_scores = []
            for chunk in chunks:
                if isinstance(chunk, dict) and 'relevance_score' in chunk:
                    relevance_scores.append(chunk['relevance_score'])
            
            # If we have relevance scores, check if they meet threshold
            if relevance_scores:
                avg_relevance = sum(relevance_scores) / len(relevance_scores)
                max_relevance = max(relevance_scores)
                
                logger.info(f"Relevance scores - Avg: {avg_relevance:.3f}, Max: {max_relevance:.3f}, Threshold: {relevance_threshold}")
                
                # If all chunks are below threshold, return spiritual fallback
                if max_relevance < relevance_threshold:
                    logger.info(f"Low relevance detected. Triggering spiritual fallback response.")
                    return {
                        "status": "success",
                        "answer": "I don't find specific guidance on this in our spiritual texts. Perhaps this question invites us to reflect on what truly nourishes our soul.",
                        "chunks_used": 0,
                        "model": self.llm_model,
                        "sources": [],
                        "verses": [],
                        "fallback_triggered": True,
                        "relevance_info": {
                            "max_score": max_relevance,
                            "avg_score": avg_relevance,
                            "threshold": relevance_threshold
                        }
                    }
            
            # Limit chunks to max_docs to avoid context length issues
            chunks = chunks[:max_docs]
            
            # Format document context from chunks
            document_context = format_context_from_chunks(chunks)
            
            # Extract source information for the response
            sources = []
            verses = []
            for chunk in chunks:
                # Extract source information
                if "metadata" in chunk and "source" in chunk["metadata"]:
                    source_info = {
                        "title": chunk["metadata"].get("title", "Unknown"),
                        "source": chunk["metadata"].get("source", "Unknown"),
                        "page": chunk["metadata"].get("page", None)
                    }
                    
                    if source_info not in sources:
                        sources.append(source_info)
                
                # Extract verse information if available
                if "metadata" in chunk and "verses" in chunk["metadata"]:
                    verses_data = chunk["metadata"].get("verses", [])
                    # Ensure verses_data is a list, not a string
                    if isinstance(verses_data, list):
                        for verse in verses_data:
                            if isinstance(verse, dict) and verse not in verses:
                                verses.append(verse)
                    elif isinstance(verses_data, str):
                        # If verses is a string, wrap it in a dict
                        verse_dict = {"content": verses_data}
                        if verse_dict not in verses:
                            verses.append(verse_dict)
            
            # Create a special conversational prompt
            prompt = f"""You are a spiritual expert assistant specializing in understanding and explaining complex spiritual concepts from various traditions. You have access to a vast library of spiritual texts and teachings. Answer the user's question thoughtfully and accurately, drawing on the relevant information provided.

Previous conversation:
{conversation_context}

Current question: {query}

Relevant information from spiritual texts:
{document_context}

Using the information provided, answer the current question while maintaining context from the previous conversation. Explain concepts in a way that's accessible yet profound. Include specific references to texts or verses when appropriate. If you don't have enough information to answer fully, acknowledge this limitation."""
            
            # Generate answer using LLM with conversation context
            response = self._create_llm_completion(
                messages=[
                    {"role": "system", "content": "You are a spiritual expert assistant specializing in understanding and explaining complex spiritual concepts from various traditions."},
                    {"role": "user", "content": prompt}
                ],
                reasoning_effort=reasoning_effort  # Pass user-provided reasoning effort for o3-mini
            )
            
            # Extract answer from response
            answer = response.choices[0].message.content
            
            return {
                "status": "success",
                "answer": answer,
                "chunks_used": len(chunks),
                "model": self.llm_model,
                "sources": sources,
                "verses": verses
            }
            
        except Exception as e:
            logger.error(f"Error generating contextualized answer: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "answer": "An error occurred while generating the answer.",
                "model": self.llm_model
            }
    
    def save_answer(self, answer_result: Dict[str, Any], output_path: str) -> bool:
        """
        Save generated answer to a file.
        
        Args:
            answer_result: Dict containing the answer and metadata
            output_path: Path to save the answer
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(answer_result, f, indent=2)
            
            logger.info(f"Saved answer to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving answer: {str(e)}")
            return False
    
    def _reformat_previous_answer(self, query: str, previous_answer: str) -> Dict[str, Any]:
        """
        Reformat a previous answer according to user's request (e.g., bullet points, summary, etc.)
        
        Args:
            query: User's reformatting request (e.g., "explain in bullet points")
            previous_answer: The previous answer to reformat
            
        Returns:
            Dict with reformatted answer
        """
        try:
            logger.info(f"Reformatting previous answer for query: {query}")
            
            # Create prompt for reformatting
            reformat_prompt = f"""You are a spiritual guide. The user has asked you to reformat a previous answer.

Previous Answer:
{previous_answer}

User's Request: {query}

Please reformat the previous answer according to the user's request. Maintain the spiritual, compassionate tone and all the important information, but present it in the requested format.

Reformatted Answer:"""
            
            # Generate reformatted answer using LLM
            messages = [
                {"role": "system", "content": "You are a wise and compassionate spiritual guide who helps seekers understand spiritual teachings."},
                {"role": "user", "content": reformat_prompt}
            ]
            
            completion = self._create_llm_completion(
                messages=messages
            )
            
            reformatted_answer = completion.choices[0].message.content.strip()
            
            logger.info(f"Successfully reformatted previous answer")
            
            return {
                "status": "success",
                "answer": reformatted_answer,
                "chunks_used": 0,  # No chunks used for reformatting
                "model": self.llm_model,
                "sources": [],
                "verses": [],
                "context_action": "reformat_previous"
            }
            
        except Exception as e:
            logger.error(f"Error reformatting previous answer: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to reformat previous answer: {str(e)}",
                "answer": "I apologize, but I couldn't reformat the previous answer.",
                "model": self.llm_model
            }
    
    def _clarify_previous_answer(self, query: str, previous_answer: str) -> Dict[str, Any]:
        """
        Clarify or expand on a previous answer based on user's request
        
        Args:
            query: User's clarification request (e.g., "can you give me more details?")
            previous_answer: The previous answer to clarify
            
        Returns:
            Dict with clarified/expanded answer
        """
        try:
            logger.info(f"Clarifying previous answer for query: {query}")
            
            # Create prompt for clarification
            clarify_prompt = f"""You are a spiritual guide. The user has asked for clarification or more details about a previous answer.

Previous Answer:
{previous_answer}

User's Request: {query}

Please provide clarification, more details, or expand on the previous answer according to the user's request. Maintain the spiritual, compassionate tone and build upon the previous answer with additional insights or explanations.

Clarified Answer:"""
            
            # Generate clarified answer using LLM
            messages = [
                {"role": "system", "content": "You are a wise and compassionate spiritual guide who helps seekers understand spiritual teachings."},
                {"role": "user", "content": clarify_prompt}
            ]
            
            completion = self._create_llm_completion(
                messages=messages
            )
            
            clarified_answer = completion.choices[0].message.content.strip()
            
            logger.info(f"Successfully clarified previous answer")
            
            return {
                "status": "success",
                "answer": clarified_answer,
                "chunks_used": 0,  # No chunks used for clarification
                "model": self.llm_model,
                "sources": [],
                "verses": [],
                "context_action": "clarify_previous"
            }
            
        except Exception as e:
            logger.error(f"Error clarifying previous answer: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to clarify previous answer: {str(e)}",
                "answer": "I apologize, but I couldn't clarify the previous answer.",
                "model": self.llm_model
            }
    
    def _modify_previous_answer(self, query: str, previous_answer: str) -> Dict[str, Any]:
        """
        Modify or simplify a previous answer based on user's request
        
        Args:
            query: User's modification request (e.g., "make this simpler to understand")
            previous_answer: The previous answer to modify
            
        Returns:
            Dict with modified answer
        """
        try:
            logger.info(f"Modifying previous answer for query: {query}")
            
            # Create prompt for modification
            modify_prompt = f"""You are a spiritual guide. The user has asked you to modify a previous answer.

Previous Answer:
{previous_answer}

User's Request: {query}

Please modify the previous answer according to the user's request. This might involve:
- Making it simpler or easier to understand
- Changing the tone or style
- Restructuring the content
- Adding or removing details

Maintain the spiritual, compassionate tone and all the important information, but adapt it according to the user's specific request.

Modified Answer:"""
            
            # Generate modified answer using LLM
            messages = [
                {"role": "system", "content": "You are a wise and compassionate spiritual guide who helps seekers understand spiritual teachings."},
                {"role": "user", "content": modify_prompt}
            ]
            
            completion = self._create_llm_completion(
                messages=messages
            )
            
            modified_answer = completion.choices[0].message.content.strip()
            
            logger.info(f"Successfully modified previous answer")
            
            return {
                "status": "success",
                "answer": modified_answer,
                "chunks_used": 0,  # No chunks used for modification
                "model": self.llm_model,
                "sources": [],
                "verses": [],
                "context_action": "modify_content"
            }
            
        except Exception as e:
            logger.error(f"Error modifying previous answer: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to modify previous answer: {str(e)}",
                "answer": "I apologize, but I couldn't modify the previous answer.",
                "model": self.llm_model
            }


def generate_answer_for_query(query: str,
                             vector_store_dir: str,
                             output_path: Optional[str] = None,
                             k: int = 5,
                             llm_model: str = "gpt-5",
                             query_type: str = None) -> Dict[str, Any]:
    """
    Generate an answer for a user query.
    
    Args:
        query: User query text
        vector_store_dir: Directory containing the vector database
        output_path: Optional path to save the answer
        k: Number of chunks to retrieve
        llm_model: LLM model for generating answers
        query_type: Type of query for prompt selection
        
    Returns:
        Dict with generated answer
    """
    # Initialize answer generator
    try:
        generator = AnswerGenerator(
            vector_store_dir=vector_store_dir,
            llm_model=llm_model
        )
        
        # Generate answer
        result = generator.generate_answer(
            query=query,
            k=k,
            query_type=query_type
        )
        
        # Save answer if output path is provided
        if output_path and result.get("status") == "success":
            generator.save_answer(result, output_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in generate_answer_for_query: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "answer": "An error occurred while generating the answer."
        }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate answers to spiritual questions")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--vector-dir", required=True, help="Vector store directory")
    parser.add_argument("--output", help="Output file path for answer")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--model", default="gpt-5", help="LLM model to use")
    parser.add_argument("--query-type", choices=["general", "verse_focused", "comparative", "practical"],
                        help="Type of query for prompt selection")
    
    args = parser.parse_args()
    
    # Generate answer
    result = generate_answer_for_query(
        query=args.query,
        vector_store_dir=args.vector_dir,
        output_path=args.output,
        k=args.k,
        llm_model=args.model,
        query_type=args.query_type
    )
    
    # Print answer
    if result.get("status") == "success":
        print("\n" + "=" * 50)
        print("ANSWER:")
        print("=" * 50)
        print(result.get("answer", ""))
        print("\n" + "=" * 50)
        if args.output:
            print(f"Answer saved to {args.output}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
