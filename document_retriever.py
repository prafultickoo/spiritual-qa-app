"""
Document retriever for retrieving relevant document chunks based on user queries.
"""
import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import our query processing utilities
from utils.query_utils import create_query_processor, QueryProcessor
from utils.langchain_utils import DocumentChunk
from utils.query_processor import enhance_spiritual_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_retriever.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class DocumentRetriever:
    """Document retrieval system for retrieving relevant chunks to answer user queries."""
    
    def __init__(self, 
                vector_store_dir: str,
                embedding_model: str = "openai",
                collection_name: str = "spiritual_texts"):
        """
        Initialize the document retriever.
        
        Args:
            vector_store_dir: Directory containing the vector database
            embedding_model: Name of embedding model to use ('openai' or 'huggingface')
            collection_name: Name of the ChromaDB collection
        """
        self.vector_store_dir = vector_store_dir
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Check if vector store exists
        if not os.path.exists(os.path.join(vector_store_dir, "chroma.sqlite3")):
            logger.error(f"Vector store not found at {vector_store_dir}")
            raise FileNotFoundError(f"Vector store not found at {vector_store_dir}")
        
        # Initialize query processor
        self.query_processor = create_query_processor(
            vector_store_dir=vector_store_dir,
            embedding_model=embedding_model,
            collection_name=collection_name
        )
        
        logger.info(f"Initialized document retriever using {embedding_model} embeddings")
    
    def retrieve_chunks(self, 
                       query: str, 
                       k: int = 5,
                       use_mmr: bool = True,
                       diversity: float = 0.7,
                       filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a user query with enhanced chapter/verse processing.
        
        Args:
            query: User query text
            k: Number of chunks to retrieve
            use_mmr: Whether to use Maximum Marginal Relevance for diverse retrieval
            diversity: MMR diversity parameter (0 = maximal diversity, 1 = maximal relevance)
            filter_metadata: Optional metadata filter for retrieval
            
        Returns:
            Dict with retrieved chunks and metadata
        """
        try:
            # Step 1: Enhanced query processing for chapter/verse queries
            enhanced_query_info = enhance_spiritual_query(query)
            
            logger.info(f"Processing query: {query}")
            logger.info(f"Enhanced queries: {enhanced_query_info.get('enhanced_queries', [])}")
            
            # Step 2: Try multiple query variations for better retrieval
            all_results = []
            unique_chunks = {}
            
            # Try each enhanced query
            for enhanced_query in enhanced_query_info.get('enhanced_queries', [query]):
                try:
                    # Process the enhanced query
                    query_result = self.query_processor.process_query(enhanced_query)
                    
                    if query_result.get("status") != "success":
                        logger.warning(f"Failed to process enhanced query '{enhanced_query}': {query_result.get('error', 'Unknown error')}")
                        continue
                    
                    # Retrieve chunks based on method
                    if use_mmr:
                        fetch_k = max(k * 2, 10)  # Fetch more candidates for MMR
                        result = self.query_processor.mmr_retrieval(
                            query_text=enhanced_query,
                            k=k,
                            fetch_k=fetch_k,
                            diversity=diversity,
                            filter_metadata=filter_metadata
                        )
                    else:
                        result = self.query_processor.retrieve_relevant_chunks(
                            query_text=enhanced_query,
                            k=k,
                            filter_metadata=filter_metadata
                        )
                    
                    # Collect unique chunks
                    if result.get("status") == "success":
                        for chunk in result.get("chunks", []):
                            chunk_id = chunk.get("chunk_id") or chunk.get("content", "")[:100]
                            if chunk_id not in unique_chunks:
                                unique_chunks[chunk_id] = chunk
                                all_results.append(chunk)
                    
                    # If we have enough relevant results, break early
                    if len(all_results) >= k * 2:
                        break
                        
                except Exception as query_error:
                    logger.warning(f"Error processing enhanced query '{enhanced_query}': {str(query_error)}")
                    continue
            
            # Step 3: If chapter/verse query, prioritize results with verse content
            if enhanced_query_info.get('is_chapter_verse_query'):
                # Sort results to prioritize those with verse content
                verse_results = []
                other_results = []
                
                for chunk in all_results:
                    content = chunk.get("content", "").lower()
                    metadata = chunk.get("metadata", {})
                    
                    # Check if this chunk contains verse-like content
                    has_verse_indicators = any([
                        "verse" in content,
                        "chapter" in content,
                        metadata.get("verses"),
                        any(pattern in content for pattern in ["karm", "yoga", "dharma", "moksh"])
                    ])
                    
                    if has_verse_indicators:
                        verse_results.append(chunk)
                    else:
                        other_results.append(chunk)
                
                # Combine with verse results first
                final_results = (verse_results + other_results)[:k]
            else:
                final_results = all_results[:k]
            
            # Step 4: Return enhanced results
            return {
                "status": "success",
                "chunks": final_results,
                "query_info": enhanced_query_info,
                "enhanced_queries_used": enhanced_query_info.get('enhanced_queries', []),
                "total_unique_chunks": len(unique_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "chunks": []
            }
    
    def get_verse_context(self, chunk: Dict[str, Any], context_window: int = 1) -> List[Dict[str, Any]]:
        """
        Get surrounding verses for context if a chunk contains verses.
        
        Args:
            chunk: Document chunk containing verses
            context_window: Number of verses to include before and after each verse
            
        Returns:
            List of additional context chunks with surrounding verses
        """
        # Not implemented yet - this would require access to the original documents
        # or a special verses database to retrieve context around specific verses
        return []
    
    def save_retrieved_chunks(self, chunks: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Save retrieved chunks to a JSON file.
        
        Args:
            chunks: List of retrieved chunks
            output_path: Path to save the chunks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(chunks, f, indent=2)
            
            logger.info(f"Saved {len(chunks)} chunks to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving chunks: {str(e)}")
            return False


def retrieve_for_query(query: str, 
                      vector_store_dir: str,
                      output_path: Optional[str] = None,
                      k: int = 5,
                      use_mmr: bool = True,
                      embedding_model: str = "openai") -> Dict[str, Any]:
    """
    Retrieve relevant chunks for a user query.
    
    Args:
        query: User query text
        vector_store_dir: Directory containing the vector database
        output_path: Optional path to save retrieved chunks
        k: Number of chunks to retrieve
        use_mmr: Whether to use MMR for diverse retrieval
        embedding_model: Embedding model to use
        
    Returns:
        Dict with retrieved chunks
    """
    # Initialize retriever
    try:
        retriever = DocumentRetriever(
            vector_store_dir=vector_store_dir,
            embedding_model=embedding_model
        )
        
        # Retrieve chunks
        result = retriever.retrieve_chunks(
            query=query,
            k=k,
            use_mmr=use_mmr
        )
        
        # Save chunks if output path is provided
        if output_path and result.get("status") == "success" and result.get("chunks"):
            retriever.save_retrieved_chunks(result["chunks"], output_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in retrieve_for_query: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "chunks": []
        }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Retrieve relevant document chunks for a query")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--vector-dir", required=True, help="Vector store directory")
    parser.add_argument("--output", help="Output file path for retrieved chunks")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR retrieval")
    parser.add_argument("--model", default="openai", choices=["openai", "huggingface"], 
                        help="Embedding model to use")
    
    args = parser.parse_args()
    
    # Retrieve chunks
    result = retrieve_for_query(
        query=args.query,
        vector_store_dir=args.vector_dir,
        output_path=args.output,
        k=args.k,
        use_mmr=not args.no_mmr,
        embedding_model=args.model
    )
    
    # Print results
    if result.get("status") == "success":
        print(f"Retrieved {len(result.get('chunks', []))} chunks for query: '{args.query}'")
        if args.output:
            print(f"Results saved to {args.output}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
