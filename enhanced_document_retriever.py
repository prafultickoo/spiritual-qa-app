#!/usr/bin/env python3
"""
Enhanced Document Retriever with Dual-Source Logic
Extends DocumentRetriever to support both clean_verses and spiritual_texts collections
while maintaining exact interface compatibility.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Import existing DocumentRetriever as base
from document_retriever import DocumentRetriever

# Import our dual-source utilities
from utils.dual_source_retriever import DualSourceRetriever, RetrievalResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_document_retriever.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class EnhancedDocumentRetriever(DocumentRetriever):
    """
    Enhanced document retriever with dual-source logic.
    
    Maintains exact interface compatibility with DocumentRetriever while adding:
    - Dual-source retrieval (clean_verses + spiritual_texts)
    - Intelligent query routing
    - Enhanced chapter/verse detection
    - Graceful fallback to single-source if needed
    """
    
    def __init__(self, 
                 vector_store_dir: str,
                 embedding_model: str = "openai",
                 collection_name: str = "spiritual_texts",
                 enable_dual_source: bool = True):
        """
        Initialize enhanced document retriever.
        
        Args:
            vector_store_dir: Directory containing the vector database
            embedding_model: Name of embedding model to use ('openai' or 'huggingface')
            collection_name: Name of the primary ChromaDB collection (for backward compatibility)
            enable_dual_source: Whether to enable dual-source functionality
        """
        # Initialize parent class (existing functionality)
        super().__init__(vector_store_dir, embedding_model, collection_name)
        
        self.enable_dual_source = enable_dual_source
        self.dual_source_retriever = None
        
        # Initialize dual-source retriever if enabled
        if self.enable_dual_source:
            try:
                self.dual_source_retriever = DualSourceRetriever(
                    vector_store_dir=vector_store_dir,
                    verses_collection="clean_verses",
                    explanations_collection="spiritual_texts"
                )
                logger.info("Dual-source retriever initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize dual-source retriever: {str(e)}")
                logger.warning("Falling back to single-source mode")
                self.enable_dual_source = False
        
        logger.info(f"Enhanced document retriever initialized (dual-source: {self.enable_dual_source})")
    
    def _convert_documents_to_chunks(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert Langchain Document objects to chunk format expected by existing API.
        
        Args:
            documents: List of Langchain Document objects
            
        Returns:
            List of chunk dictionaries in expected format
        """
        chunks = []
        
        for i, doc in enumerate(documents):
            # Handle both Document objects and dict formats
            if hasattr(doc, 'page_content'):
                # Langchain Document object
                content = doc.page_content
                metadata = doc.metadata
            else:
                # Already in dict format (from parent class)
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
            
            chunk = {
                'content': content,
                'metadata': metadata
            }
            
            # Add chunk_id if not present
            if 'chunk_id' not in chunk['metadata']:
                source = metadata.get('source', 'unknown')
                page = metadata.get('page', 0)
                chunk['metadata']['chunk_id'] = f"enhanced_{i}_{source}_{page}"
            
            chunks.append(chunk)
        
        return chunks
    
    def _validate_and_sanitize_query(self, query: str) -> Dict[str, Any]:
        """
        Validate and sanitize user query before processing.
        
        Args:
            query: Raw user query
            
        Returns:
            Dict with validation results and sanitized query
        """
        validation = {
            'is_valid': True,
            'sanitized_query': query.strip(),
            'issues': []
        }
        
        # Check for empty or whitespace-only queries
        if not query or not query.strip():
            validation['is_valid'] = False
            validation['issues'].append('Empty query')
            validation['sanitized_query'] = ''
            return validation
        
        # Check for minimum length
        clean_query = query.strip()
        if len(clean_query) < 3:
            validation['is_valid'] = False
            validation['issues'].append('Query too short (minimum 3 characters)')
            return validation
        
        # Check for maximum length
        max_length = 500  # Reasonable limit
        if len(clean_query) > max_length:
            validation['sanitized_query'] = clean_query[:max_length].strip()
            validation['issues'].append(f'Query truncated to {max_length} characters')
        
        # Remove excessive repetition
        words = clean_query.split()
        if len(words) > 10:  # Only check for repetition in longer queries
            # Count word frequency
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Check if any word appears more than 5 times
            max_repetitions = max(word_counts.values()) if word_counts else 0
            if max_repetitions > 5:
                # Remove excessive repetition by keeping only first 5 occurrences
                deduped_words = []
                word_usage = {}
                for word in words:
                    word_usage[word] = word_usage.get(word, 0) + 1
                    if word_usage[word] <= 3:  # Keep max 3 occurrences
                        deduped_words.append(word)
                    elif len(deduped_words) < 20:  # Keep reasonable length
                        deduped_words.append(word)
                
                validation['sanitized_query'] = ' '.join(deduped_words)
                validation['issues'].append('Removed excessive word repetition')
        
        return validation
    
    def _analyze_query_for_dual_source(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine if dual-source retrieval would be beneficial.
        
        Args:
            query: User query (already validated and sanitized)
            
        Returns:
            Analysis results with recommendation
        """
        query_lower = query.lower()
        
        analysis = {
            'use_dual_source': False,
            'priority': 'balanced',
            'chapter_verse_detected': False,
            'verse_keywords': False,
            'explanation_keywords': False
        }
        
        # Chapter/verse detection
        chapter_verse_patterns = [
            'chapter', 'verse', 'श्लोक', 'adhyaya', 'bhagavad gita', 'गीता'
        ]
        
        verse_keywords = [
            'sanskrit', 'devanagari', 'original', 'verse', 'sloka', 'mantra'
        ]
        
        explanation_keywords = [
            'meaning', 'explanation', 'commentary', 'explain', 'what is', 'tell me about'
        ]
        
        # Check for chapter/verse indicators
        if any(pattern in query_lower for pattern in chapter_verse_patterns):
            analysis['chapter_verse_detected'] = True
            analysis['use_dual_source'] = True
            analysis['priority'] = 'verses_first'
        
        # Check for verse-specific keywords
        if any(keyword in query_lower for keyword in verse_keywords):
            analysis['verse_keywords'] = True
            analysis['use_dual_source'] = True
            analysis['priority'] = 'verses_first'
        
        # Check for explanation-specific keywords
        if any(keyword in query_lower for keyword in explanation_keywords):
            analysis['explanation_keywords'] = True
            if not analysis['use_dual_source']:
                analysis['use_dual_source'] = True
                analysis['priority'] = 'explanations_first'
        
        return analysis
    
    def retrieve_chunks(self, 
                       query: str, 
                       k: int = 5,
                       use_mmr: bool = True,
                       diversity: float = 0.7,
                       filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced retrieve_chunks method with dual-source logic.
        Maintains exact interface compatibility with parent class.
        
        Args:
            query: User query text
            k: Number of chunks to retrieve
            use_mmr: Whether to use Maximum Marginal Relevance for diverse retrieval
            diversity: MMR diversity parameter (0 = maximal diversity, 1 = maximal relevance)
            filter_metadata: Optional metadata filter for retrieval
            
        Returns:
            Dict with retrieved chunks in same format as parent class
        """
        logger.info(f"Enhanced retrieval for query: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        # Step 1: Validate and sanitize query
        validation = self._validate_and_sanitize_query(query)
        
        if not validation['is_valid']:
            logger.warning(f"Invalid query rejected: {validation['issues']}")
            return {
                'status': 'error',
                'chunks': [],
                'query_info': {
                    'original_query': query,
                    'validation_issues': validation['issues'],
                    'error': 'Invalid query - ' + ', '.join(validation['issues'])
                },
                'enhanced_queries_used': [],
                'total_unique_chunks': 0
            }
        
        # Use sanitized query for processing
        sanitized_query = validation['sanitized_query']
        if validation['issues']:
            logger.info(f"Query sanitized: {validation['issues']}")
        
        # Step 2: Analyze query for dual-source potential
        query_analysis = self._analyze_query_for_dual_source(sanitized_query)
        
        # Step 3: Decide whether to use dual-source or fallback to parent
        if (self.enable_dual_source and 
            self.dual_source_retriever and 
            query_analysis.get('use_dual_source', False)):
            
            try:
                result = self._dual_source_retrieve(sanitized_query, k, use_mmr, diversity, filter_metadata, query_analysis)
                # Add validation info to result
                if validation['issues']:
                    result['query_info']['query_sanitization'] = validation['issues']
                return result
            except Exception as e:
                logger.warning(f"Dual-source retrieval failed: {str(e)}")
                logger.warning("Falling back to single-source retrieval")
        
        # Fallback to parent class method (existing functionality)
        logger.info("Using single-source retrieval")
        result = super().retrieve_chunks(sanitized_query, k, use_mmr, diversity, filter_metadata)
        
        # Add validation info to result
        if validation['issues'] and isinstance(result, dict):
            if 'query_info' not in result:
                result['query_info'] = {}
            result['query_info']['query_sanitization'] = validation['issues']
        
        return result
    
    def _dual_source_retrieve(self, 
                             query: str, 
                             k: int,
                             use_mmr: bool,
                             diversity: float,
                             filter_metadata: Optional[Dict[str, Any]],
                             query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform dual-source retrieval and format results.
        
        Args:
            query: User query
            k: Number of results
            use_mmr: MMR flag
            diversity: Diversity parameter
            filter_metadata: Metadata filter
            query_analysis: Query analysis results
            
        Returns:
            Results in same format as parent class
        """
        logger.info("Performing dual-source retrieval")
        
        # Use dual-source retriever
        retrieval_result = self.dual_source_retriever.retrieve(
            query=query,
            k=k,
            reading_style="balanced"  # Map from MMR parameters if needed
        )
        
        # Convert merged context to expected chunk format
        enhanced_chunks = self._convert_documents_to_chunks(retrieval_result.merged_context)
        
        # Create result in same format as parent class
        result = {
            'status': 'success',
            'chunks': enhanced_chunks,
            'query_info': {
                'original_query': query,
                'dual_source_used': True,
                'query_type': retrieval_result.query_type,
                'chapter': retrieval_result.chapter,
                'verse': retrieval_result.verse,
                'verses_found': len(retrieval_result.clean_verses),
                'explanations_found': len(retrieval_result.explanations)
            },
            'enhanced_queries_used': [query],  # Maintain compatibility
            'total_unique_chunks': len(enhanced_chunks)
        }
        
        logger.info(f"Dual-source retrieval successful: {len(enhanced_chunks)} chunks returned")
        return result
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval system.
        
        Returns:
            Dict with system statistics
        """
        stats = {
            'dual_source_enabled': self.enable_dual_source,
            'collections': []
        }
        
        try:
            # Get collection info
            client = chromadb.PersistentClient(path=self.vector_store_dir)
            collections = client.list_collections()
            
            for collection in collections:
                stats['collections'].append({
                    'name': collection.name,
                    'count': collection.count()
                })
        except Exception as e:
            stats['error'] = str(e)
        
        return stats


def create_enhanced_retriever(vector_store_dir: str = "./vector_store",
                            embedding_model: str = "openai",
                            enable_dual_source: bool = True) -> EnhancedDocumentRetriever:
    """
    Factory function to create enhanced document retriever.
    
    Args:
        vector_store_dir: Vector store directory
        embedding_model: Embedding model to use
        enable_dual_source: Whether to enable dual-source functionality
        
    Returns:
        EnhancedDocumentRetriever instance
    """
    return EnhancedDocumentRetriever(
        vector_store_dir=vector_store_dir,
        embedding_model=embedding_model,
        enable_dual_source=enable_dual_source
    )


if __name__ == "__main__":
    # Test the enhanced retriever
    retriever = create_enhanced_retriever()
    
    # Test queries
    test_queries = [
        "Chapter 2 verse 47",
        "What is karma yoga?",
        "Explain meditation",
        "Bhagavad Gita 4.7"
    ]
    
    print("=== ENHANCED DOCUMENT RETRIEVER TEST ===")
    print()
    
    for query in test_queries:
        print(f"🔍 Testing: {query}")
        try:
            result = retriever.retrieve_chunks(query, k=3)
            print(f"   ✅ Status: {result.get('status', 'unknown')}")
            print(f"   📊 Chunks: {len(result.get('chunks', []))}")
            print(f"   🔧 Dual-source: {result.get('query_info', {}).get('dual_source_used', False)}")
            if result.get('query_info', {}).get('dual_source_used'):
                query_info = result['query_info']
                print(f"   📝 Type: {query_info.get('query_type', 'unknown')}")
                print(f"   📚 Verses: {query_info.get('verses_found', 0)}")
                print(f"   💡 Explanations: {query_info.get('explanations_found', 0)}")
            print()
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            print()
    
    # Show stats
    stats = retriever.get_retrieval_stats()
    print("📊 System Stats:")
    print(f"   Dual-source enabled: {stats.get('dual_source_enabled', False)}")
    for collection in stats.get('collections', []):
        print(f"   Collection {collection['name']}: {collection['count']} documents")
