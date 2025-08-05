"""
Vectorization utilities for converting document chunks to embeddings.
"""
import os
import json
import logging
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Import our document chunk models
from utils.langchain_utils import DocumentChunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Dictionary of available embedding models
EMBEDDING_MODELS = {
    "openai": OpenAIEmbeddings,
    "huggingface": HuggingFaceEmbeddings
}

class DocumentVectorizer:
    """Class for vectorizing document chunks and storing in a vector database."""
    
    def __init__(self, 
                 persist_directory: str, 
                 embedding_model: str = "openai",
                 embedding_model_kwargs: Optional[Dict[str, Any]] = None,
                 collection_name: str = "spiritual_texts"):
        """
        Initialize the document vectorizer.
        
        Args:
            persist_directory: Directory to persist vector database
            embedding_model: Name of embedding model to use ('openai' or 'huggingface')
            embedding_model_kwargs: Additional keyword arguments for embedding model
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding model
        if embedding_model not in EMBEDDING_MODELS:
            logger.warning(f"Unknown embedding model: {embedding_model}. Defaulting to OpenAI.")
            self.embedding_model_name = "openai"
            
        # Default kwargs for embedding models
        default_kwargs = {
            "openai": {"model": "text-embedding-ada-002"},
            "huggingface": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}
        }
        
        # Use provided kwargs or defaults
        model_kwargs = embedding_model_kwargs or default_kwargs.get(self.embedding_model_name, {})
        
        # Initialize the embedding model
        try:
            EmbeddingClass = EMBEDDING_MODELS[self.embedding_model_name]
            self.embedding_model = EmbeddingClass(**model_kwargs)
            logger.info(f"Initialized {self.embedding_model_name} embedding model")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
        
        # Initialize vector store
        self.vector_store = None
    
    def _convert_chunks_to_documents(self, chunks: List[DocumentChunk]) -> List[Document]:
        """
        Convert DocumentChunk objects to Langchain Document objects.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of Langchain Document objects
        """
        documents = []
        for chunk in chunks:
            # Get base metadata
            metadata = dict(chunk.metadata) if hasattr(chunk, 'metadata') else {}
            
            # Add chunk-specific metadata
            metadata['chunk_id'] = str(uuid.uuid4())
            metadata['has_verses'] = chunk.has_verses if hasattr(chunk, 'has_verses') else False
            
            # If chunk has verses, convert them to string format for ChromaDB compatibility
            if hasattr(chunk, 'verses') and chunk.verses:
                # Convert list of verses to a single string
                metadata['verses'] = ' | '.join(str(verse) for verse in chunk.verses)
                metadata['verse_count'] = len(chunk.verses)
            
            # Convert any remaining complex metadata to strings
            for key, value in metadata.items():
                if isinstance(value, (list, dict, tuple)):
                    metadata[key] = str(value)
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    metadata[key] = str(value)
            
            # Create Document object
            doc = Document(
                page_content=chunk.content if hasattr(chunk, 'content') else str(chunk),
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def vectorize_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100) -> Dict[str, Any]:
        """
        Vectorize document chunks and store them in the vector database.
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Batch size for vectorization (reduced to avoid token limits)
            
        Returns:
            Dict with vectorization results
        """
        logger.info(f"Vectorizing {len(chunks)} document chunks in batches of {batch_size}")
        
        # Convert chunks to Langchain Document objects
        documents = self._convert_chunks_to_documents(chunks)
        
        try:
            # Initialize or load existing Chroma DB
            if os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
                logger.info(f"Loading existing Chroma DB from {self.persist_directory}")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model,
                    collection_name=self.collection_name
                )
                # Process documents in batches to avoid token limits
                total_processed = 0
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                    self.vector_store.add_documents(batch)
                    total_processed += len(batch)
                    
                    # Persist after each batch to avoid data loss
                    self.vector_store.persist()
                    
            else:
                # Create new DB with first batch
                logger.info(f"Creating new Chroma DB at {self.persist_directory}")
                first_batch = documents[:batch_size]
                self.vector_store = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embedding_model,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
                self.vector_store.persist()
                total_processed = len(first_batch)
                logger.info(f"Created DB with first batch of {len(first_batch)} documents")
                
                # Process remaining documents in batches
                remaining_documents = documents[batch_size:]
                for i in range(0, len(remaining_documents), batch_size):
                    batch = remaining_documents[i:i + batch_size]
                    batch_num = (i // batch_size) + 2  # +2 because we already processed batch 1
                    total_batches = ((len(documents) + batch_size - 1) // batch_size)
                    logger.info(f"Processing batch {batch_num}/{total_batches}")
                    
                    self.vector_store.add_documents(batch)
                    total_processed += len(batch)
                    
                    # Persist after each batch
                    self.vector_store.persist()
            
            # Final persist
            self.vector_store.persist()
            logger.info(f"Successfully vectorized and stored {total_processed} documents")
            
            return {
                "status": "success",
                "chunks_processed": total_processed,
                "persist_directory": self.persist_directory,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error vectorizing chunks: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "chunks_processed": 0
            }
    
    def vectorize_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        Vectorize document chunks from a JSON file.
        
        Args:
            json_path: Path to JSON file containing document chunks
            
        Returns:
            Dict with vectorization results
        """
        logger.info(f"Vectorizing document chunks from {json_path}")
        
        try:
            # Load chunks from JSON
            with open(json_path, 'r') as f:
                chunk_dicts = json.load(f)
            
            # Convert dictionaries to DocumentChunk objects
            chunks = [DocumentChunk(**chunk_dict) for chunk_dict in chunk_dicts]
            
            # Vectorize chunks
            return self.vectorize_chunks(chunks)
            
        except Exception as e:
            logger.error(f"Error vectorizing from JSON: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "chunks_vectorized": 0
            }
    
    def load_vector_store(self) -> bool:
        """
        Load the vector store from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of Documents most similar to query
        """
        if not self.vector_store:
            if not self.load_vector_store():
                logger.error("Cannot perform search: Vector store not loaded")
                return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []


# Factory function to create vectorizer with appropriate model
def create_vectorizer(persist_directory: str, 
                      model_name: str = "openai",
                      collection_name: str = "spiritual_texts",
                      **kwargs) -> DocumentVectorizer:
    """
    Create a document vectorizer with the specified embedding model.
    
    Args:
        persist_directory: Directory to persist vector database
        model_name: Name of embedding model ('openai', 'huggingface')
        collection_name: Name of the ChromaDB collection
        **kwargs: Additional keyword arguments for the embedding model
        
    Returns:
        DocumentVectorizer instance
    """
    return DocumentVectorizer(
        persist_directory=persist_directory,
        embedding_model=model_name,
        embedding_model_kwargs=kwargs,
        collection_name=collection_name
    )
