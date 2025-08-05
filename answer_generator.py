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

# Import prompt templates
from prompts.answer_prompts import (
    select_prompt_template,
    format_context_from_chunks,
    ANSWER_BASE_PROMPT
)

# Initialize OpenAI integration
import openai
from openai import OpenAI

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
                llm_model: str = "gpt-4.1",
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
        
        # Initialize document retriever (Enhanced or Standard based on flag)
        if self.enable_dual_source:
            try:
                self.retriever = EnhancedDocumentRetriever(
                    vector_store_dir=vector_store_dir,
                    embedding_model=embedding_model,
                    enable_dual_source=True
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
                
        logger.info(f"Creating LLM completion for {self.llm_model} with params: {list(params.keys())}")
        
        return client.chat.completions.create(**params)
    
    def retrieve_relevant_chunks(self, 
                                query: str, 
                                k: int = 5,
                                use_mmr: bool = True) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query text
            k: Number of chunks to retrieve
            use_mmr: Whether to use MMR for diverse retrieval
            
        Returns:
            Dict containing retrieved chunks
        """
        return self.retriever.retrieve_chunks(
            query=query,
            k=k,
            use_mmr=use_mmr
        )
    
    def generate_answer(self, 
                       query: str, 
                       chunks: List[Dict[str, Any]] = None,
                       k: int = 5,
                       query_type: str = None,
                       custom_prompt: str = None,
                       reasoning_effort: str = None,
                       rag_technique: str = 'stuff') -> Dict[str, Any]:
        """
        Generate an answer for a user query.
        
        Args:
            query: User query text
            chunks: Optional pre-retrieved chunks (if None, retrieves chunks)
            k: Number of chunks to retrieve if needed
            query_type: Type of query for prompt selection
            custom_prompt: Optional custom prompt template
            
        Returns:
            Dict with generated answer and metadata
        """
        try:
            # Retrieve chunks if not provided
            if not chunks:
                retrieval_result = self.retrieve_relevant_chunks(query, k)
                if retrieval_result.get("status") != "success":
                    return {
                        "status": "error",
                        "error": retrieval_result.get("error", "Failed to retrieve chunks"),
                        "answer": "I couldn't find relevant information to answer your question."
                    }
                chunks = retrieval_result.get("chunks", [])
            
            # Use RAG technique handler to generate answer
            result = self.rag_handler.generate_answer(
                technique=rag_technique,
                question=query,
                chunks=chunks,
                reasoning_effort=reasoning_effort
            )
            
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
            # Retrieve chunks for the query
            retrieval_result = self.retrieve_relevant_chunks(query, k, use_mmr)
            
            if retrieval_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": retrieval_result.get("error", "Failed to retrieve chunks"),
                    "answer": "I couldn't find relevant information to answer your question."
                }
                
            chunks = retrieval_result.get("chunks", [])
            
            # If no chunks were retrieved, return early
            if not chunks:
                return {
                    "status": "success",
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "chunks_used": 0,
                    "sources": [],
                    "verses": []
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
                    for verse in chunk["metadata"].get("verses", []):
                        if verse not in verses:
                            verses.append(verse)
            
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
                "answer": "An error occurred while generating the answer."
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


def generate_answer_for_query(query: str,
                             vector_store_dir: str,
                             output_path: Optional[str] = None,
                             k: int = 5,
                             llm_model: str = "gpt-4.1",
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
    parser.add_argument("--model", default="gpt-4.1", help="LLM model to use")
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
