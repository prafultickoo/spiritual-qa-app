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

# Import context enhancement pipeline
from utils.conversation_context import ConversationContextProcessor
from utils.semantic_analyzer import SemanticAnalyzer
from utils.query_classifier import LLMQueryClassifier

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
                 enable_dual_source: bool = True,
                 enable_context_enhancement: bool = False):
        """
        Initialize enhanced document retriever.
        
        Args:
            vector_store_dir: Directory containing the vector database
            embedding_model: Name of embedding model to use ('openai' or 'huggingface')
            collection_name: Name of the primary ChromaDB collection (for backward compatibility)
            enable_dual_source: Whether to enable dual-source functionality
            enable_context_enhancement: Whether to enable context enhancement pipeline
        """
        # Initialize parent class (existing functionality)
        super().__init__(vector_store_dir, embedding_model, collection_name)
        
        self.enable_dual_source = enable_dual_source
        self.dual_source_retriever = None
        
        # Initialize context enhancement pipeline components
        self.enable_context_enhancement = enable_context_enhancement
        self.context_processor = None
        self.semantic_analyzer = None
        self.query_classifier = None
        
        if self.enable_context_enhancement:
            try:
                self.context_processor = ConversationContextProcessor()
                self.semantic_analyzer = SemanticAnalyzer()
                self.query_classifier = LLMQueryClassifier()
                logger.info("Context enhancement pipeline initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize context enhancement: {str(e)}")
                logger.warning("Context enhancement disabled")
                self.enable_context_enhancement = False
        
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
            
            # Propagate relevance score if present in metadata
            if 'relevance_score' in metadata:
                chunk['relevance_score'] = metadata['relevance_score']
            
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
    
    def _process_context_enhancement(self, 
                                    query: str, 
                                    conversation_history: List[Dict],
                                    llm_client: Optional[Any] = None,
                                    model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Process query through the 3-stage context enhancement pipeline.
        
        Args:
            query: User query
            conversation_history: Previous conversation exchanges
            llm_client: LLM client for Stage 3 classification (optional)
            
        Returns:
            Dict with enhancement results and processed query
        """
        result = {
            'original_query': query,
            'processed_query': query,
            'enhancement_applied': False,
            'pipeline_stage': 0,
            'action': 'direct_retrieval',
            'needs_rag': True
        }
        
        if not self.enable_context_enhancement:
            logger.info("Context enhancement disabled, using original query")
            return result
        
        try:
            # Stage 1: Fast heuristic detection
            stage1_result = self.context_processor.is_follow_up_query(query, conversation_history)
            logger.info(f"Stage 1: Follow-up={stage1_result['is_follow_up']}, confidence={stage1_result['confidence']}")
            
            if not stage1_result['is_follow_up']:
                result['pipeline_stage'] = 1
                return result
            
            # Stage 2: Semantic analysis
            stage2_result = self.semantic_analyzer.analyze_query_ambiguity(query, conversation_history)
            logger.info(f"Stage 2: Ambiguous={stage2_result.get('is_ambiguous')}, needs_context={stage2_result.get('needs_context')}")
            
            if not stage2_result.get('needs_context', False):
                result['pipeline_stage'] = 2
                return result
            
            # Stage 3: LLM classification (only if we have an LLM client)
            if llm_client is None:
                logger.warning("No LLM client provided for Stage 3, using default enhancement")
                # Default to information expansion if no LLM available
                result['enhancement_applied'] = True
                result['pipeline_stage'] = 2
                result['action'] = 'enhance_and_retrieve'
                
                # Simple context enhancement: append last topic
                if conversation_history:
                    last_topic = self.context_processor.extract_last_topic(conversation_history)
                    if last_topic:
                        result['processed_query'] = f"{query} (in context of {last_topic})"
                return result
            
            # Run Stage 3 classification
            stage2_result['proceed_to_stage_3'] = True
            stage3_result = self.query_classifier.process_stage3(
                query, conversation_history, stage2_result, llm_client, model
            )
            
            classification = stage3_result.get('stage3_classification', {})
            logger.info(f"Stage 3: Intent={classification.get('intent')}, action={classification.get('action')}")
            
            result['pipeline_stage'] = 3
            result['action'] = classification.get('action', 'enhance_and_retrieve')
            result['needs_rag'] = classification.get('needs_rag', True)
            result['intent'] = classification.get('intent')
            result['explanation'] = classification.get('explanation')
            
            # Process based on action
            if classification.get('action') == 'enhance_and_retrieve':
                # Enhance query with context
                if conversation_history:
                    context_summary = self.query_classifier.summarize_conversation(conversation_history)
                    last_topic = self.context_processor.extract_last_topic(conversation_history)
                    
                    # Create enhanced query
                    enhanced_parts = [query]
                    if last_topic:
                        enhanced_parts.append(f"(regarding {last_topic})")
                    if context_summary:
                        enhanced_parts.append(f"[Context: {context_summary}]")
                    
                    result['processed_query'] = " ".join(enhanced_parts)
                    result['enhancement_applied'] = True
                    logger.info(f"Enhanced query: {result['processed_query']}")
            
            elif classification.get('action') == 'reformat_previous':
                # This will be handled by the answer generator
                result['needs_rag'] = False
                result['enhancement_applied'] = False
                result['context_action'] = 'reformat_previous'  # THE MISSING LINK!
                logger.info("Action: Reformat previous answer (no RAG needed)")
            
            elif classification.get('action') == 'apply_perspective':
                # Enhance query to apply new perspective
                if conversation_history:
                    last_topic = self.context_processor.extract_last_topic(conversation_history)
                    result['processed_query'] = f"{query} (apply to {last_topic})" if last_topic else query
                    result['enhancement_applied'] = True
                result['needs_rag'] = False  # Will use previous answer as base
            
            return result
            
        except Exception as e:
            logger.error(f"Error in context enhancement pipeline: {str(e)}")
            logger.exception("Full traceback:")
            # Fallback to original query
            return result
    
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
    
    def retrieve_chunks_with_context(self,
                                   query: str,
                                   conversation_history: Optional[List[Dict]] = None,
                                   k: int = 5,
                                   use_mmr: bool = True,
                                   diversity: float = 0.7,
                                   filter_metadata: Optional[Dict[str, Any]] = None,
                                   llm_client: Optional[Any] = None,
                                   model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Retrieve chunks with context enhancement capabilities.
        
        This method adds context-aware query processing while maintaining
        full compatibility with the base retrieve_chunks interface.
        
        Args:
            query: User query text
            conversation_history: Optional conversation history for context enhancement
            k: Number of chunks to retrieve
            use_mmr: Whether to use Maximum Marginal Relevance
            diversity: MMR diversity parameter
            filter_metadata: Optional metadata filter
            llm_client: Optional LLM client for Stage 3 classification
            model: LLM model to use for Stage 3 classification
            
        Returns:
            Dict with retrieval results and context enhancement info
        """
        # Process through context enhancement pipeline if enabled
        context_result = self._process_context_enhancement(
            query, 
            conversation_history or [],
            llm_client,
            model
        )
        
        # Use the processed query for retrieval if enhancement was applied
        retrieval_query = context_result['processed_query']
        
        # Only retrieve if the action requires RAG
        if not context_result['needs_rag']:
            logger.info(f"Action '{context_result['action']}' does not require RAG retrieval")
            return {
                'status': 'success',
                'chunks': [],
                'context_action': context_result['action'],  # TOP-LEVEL FIELD FOR ANSWER GENERATOR!
                'query_info': {
                    'original_query': query,
                    'processed_query': retrieval_query,
                    'context_enhanced': context_result['enhancement_applied'],
                    'pipeline_stage': context_result['pipeline_stage'],
                    'action': context_result['action'],
                    'intent': context_result.get('intent'),
                    'explanation': context_result.get('explanation')
                },
                'enhanced_queries_used': [retrieval_query],
                'total_unique_chunks': 0,
                'requires_special_handling': True
            }
        
        # Perform regular retrieval with the (possibly enhanced) query
        result = self.retrieve_chunks(
            query=retrieval_query,
            k=k,
            use_mmr=use_mmr,
            diversity=diversity,
            filter_metadata=filter_metadata
        )
        
        # Add context enhancement information to the result
        if 'query_info' not in result:
            result['query_info'] = {}
        
        result['query_info'].update({
            'original_query': query,
            'processed_query': retrieval_query,
            'context_enhanced': context_result['enhancement_applied'],
            'pipeline_stage': context_result['pipeline_stage'],
            'action': context_result['action'],
            'intent': context_result.get('intent'),
            'explanation': context_result.get('explanation')
        })
        
        # Add top-level context_action field for answer generator
        result['context_action'] = context_result['action']
        
        return result


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
