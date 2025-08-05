"""
Direct document vectorization script - bypasses CrewAI for now to create vector DB.
"""
import os
import json
import argparse
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import our utilities directly
from utils.langchain_utils import load_documents, chunk_documents, DocumentChunk
from utils.vectorization_utils import create_vectorizer, DocumentVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vectorization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def vectorize_all_documents(input_dir: str, 
                           vector_store_dir: str,
                           output_dir: str = None,
                           chunk_size: int = 1500,
                           chunk_overlap: int = 200,
                           embedding_model: str = "openai") -> Dict[str, Any]:
    """
    Load, chunk, and vectorize all documents from input directory.
    
    Args:
        input_dir: Directory containing PDF documents
        vector_store_dir: Directory to store the vector database
        output_dir: Optional directory to save processed chunks
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        embedding_model: Embedding model to use ('openai' or 'huggingface')
        
    Returns:
        Dict with processing results
    """
    try:
        logger.info(f"Starting document vectorization process...")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Vector store directory: {vector_store_dir}")
        
        # Step 1: Load documents
        logger.info("Step 1: Loading documents...")
        documents = load_documents(input_dir)
        logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            return {
                "status": "error",
                "error": "No documents found in input directory",
                "documents_processed": 0
            }
        
        # Step 2: Chunk documents
        logger.info("Step 2: Chunking documents...")
        chunks = chunk_documents(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_verses=True
        )
        logger.info(f"Created {len(chunks)} chunks from documents")
        
        if not chunks:
            return {
                "status": "error",
                "error": "No chunks created from documents",
                "documents_processed": len(documents)
            }
        
        # Step 3: Save chunks if output directory specified
        if output_dir:
            logger.info("Step 3: Saving chunks...")
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert chunks to serializable format
            chunks_data = []
            for chunk in chunks:
                chunk_data = {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "verses": chunk.verses if hasattr(chunk, 'verses') else []
                }
                chunks_data.append(chunk_data)
            
            chunks_file = os.path.join(output_dir, "all_chunks.json")
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")
        
        # Step 4: Initialize vectorizer
        logger.info("Step 4: Initializing vectorizer...")
        vectorizer = create_vectorizer(
            persist_directory=vector_store_dir,
            model_name=embedding_model,
            collection_name="spiritual_texts"
        )
        
        # Step 5: Vectorize and store chunks
        logger.info("Step 5: Vectorizing and storing chunks...")
        vectorization_result = vectorizer.vectorize_chunks(chunks)
        
        if vectorization_result.get("status") == "success":
            logger.info("Successfully vectorized and stored all documents!")
            
            return {
                "status": "success",
                "documents_processed": len(documents),
                "chunks_created": len(chunks),
                "chunks_vectorized": vectorization_result.get("chunks_processed", 0),
                "vector_store_dir": vector_store_dir,
                "embedding_model": embedding_model
            }
        else:
            return {
                "status": "error",
                "error": vectorization_result.get("error", "Vectorization failed"),
                "documents_processed": len(documents),
                "chunks_created": len(chunks)
            }
            
    except Exception as e:
        logger.error(f"Error in vectorization process: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "documents_processed": 0
        }


def verify_vector_store(vector_store_dir: str) -> Dict[str, Any]:
    """
    Verify that the vector store was created successfully.
    
    Args:
        vector_store_dir: Directory containing the vector database
        
    Returns:
        Dict with verification results
    """
    try:
        # Check if vector store directory exists
        if not os.path.exists(vector_store_dir):
            return {
                "status": "error",
                "error": "Vector store directory does not exist"
            }
        
        # Check for ChromaDB files
        chroma_files = []
        for file in os.listdir(vector_store_dir):
            if file.endswith('.sqlite3') or file.startswith('chroma'):
                chroma_files.append(file)
        
        if chroma_files:
            logger.info(f"Vector store verified! Found files: {chroma_files}")
            
            # Try to load and check collection
            vectorizer = create_vectorizer(
                persist_directory=vector_store_dir,
                model_name="openai",
                collection_name="spiritual_texts"
            )
            
            # Test a simple search
            test_result = vectorizer.similarity_search("dharma", k=1)
            
            return {
                "status": "success",
                "vector_store_files": chroma_files,
                "test_search_results": len(test_result.get("documents", [])),
                "message": "Vector store created and accessible"
            }
        else:
            return {
                "status": "error",
                "error": "No ChromaDB files found in vector store directory"
            }
            
    except Exception as e:
        logger.error(f"Error verifying vector store: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vectorize spiritual documents")
    parser.add_argument("--input-dir", required=True, help="Directory containing PDF documents")
    parser.add_argument("--vector-dir", required=True, help="Directory to store vector database")
    parser.add_argument("--output-dir", help="Directory to save processed chunks")
    parser.add_argument("--chunk-size", type=int, default=1500, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--embedding-model", default="openai", choices=["openai", "huggingface"],
                        help="Embedding model to use")
    parser.add_argument("--verify-only", action="store_true", 
                        help="Only verify existing vector store")
    
    args = parser.parse_args()
    
    if args.verify_only:
        # Only verify existing vector store
        result = verify_vector_store(args.vector_dir)
        print(json.dumps(result, indent=2))
    else:
        # Run full vectorization process
        result = vectorize_all_documents(
            input_dir=args.input_dir,
            vector_store_dir=args.vector_dir,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model
        )
        
        print("VECTORIZATION RESULTS:")
        print("=" * 50)
        print(json.dumps(result, indent=2))
        
        # Verify the result
        if result.get("status") == "success":
            print("\nVERIFYING VECTOR STORE:")
            print("=" * 50)
            verification = verify_vector_store(args.vector_dir)
            print(json.dumps(verification, indent=2))
