#!/usr/bin/env python3
"""
Dual-Source Retrieval Handler for Hybrid Knowledge Base
Combines clean verses from new collection with explanations from existing collection.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import re

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Container for dual-source retrieval results."""
    clean_verses: List[Document]
    explanations: List[Document]
    merged_context: List[Document]
    query_type: str
    chapter: Optional[int] = None
    verse: Optional[int] = None

class DualSourceRetriever:
    """
    Intelligent dual-source retrieval system.
    
    Combines:
    - Clean verses from 'clean_verses' collection (new Bhagavad Gita)
    - Rich explanations from 'spiritual_texts' collection (existing database)
    """
    
    def __init__(self, 
                 vector_store_dir: str = "./vector_store",
                 verses_collection: str = "clean_verses",
                 explanations_collection: str = "spiritual_texts"):
        """
        Initialize dual-source retriever.
        
        Args:
            vector_store_dir: Vector store directory
            verses_collection: Collection name for clean verses
            explanations_collection: Collection name for explanations
        """
        self.vector_store_dir = vector_store_dir
        self.verses_collection = verses_collection
        self.explanations_collection = explanations_collection
        
        # Initialize embeddings (must match existing database)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Initialize vector stores
        self._init_vector_stores()
        
        logger.info("DualSourceRetriever initialized successfully")
    
    def _init_vector_stores(self):
        """Initialize both vector store connections."""
        try:
            # Clean verses collection (new Bhagavad Gita)
            self.verses_store = Chroma(
                collection_name=self.verses_collection,
                embedding_function=self.embeddings,
                persist_directory=self.vector_store_dir
            )
            
            # Explanations collection (existing database)
            self.explanations_store = Chroma(
                collection_name=self.explanations_collection,
                embedding_function=self.embeddings,
                persist_directory=self.vector_store_dir
            )
            
            logger.info(f"Vector stores initialized: {self.verses_collection}, {self.explanations_collection}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector stores: {str(e)}")
            raise
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine retrieval strategy.
        
        Args:
            query: User query
            
        Returns:
            Query analysis results
        """
        query_lower = query.lower()
        
        analysis = {
            'query_type': 'general',
            'chapter': None,
            'verse': None,
            'is_verse_specific': False,
            'keywords': [],
            'priority_source': 'both'
        }
        
        # Chapter/verse detection patterns
        chapter_verse_patterns = [
            r'chapter\s+(\d+).*verse\s+(\d+)',
            r'chapter\s+(\d+).*श्लोक\s+(\d+)',
            r'adhyaya\s+(\d+).*verse\s+(\d+)',
            r'(?:bhagavad\s+gita|gita).*(\d+)\.(\d+)',
            r'बीजी\s+(\d+)\.(\d+)',
            r'(\d+)\s*[\.\-]\s*(\d+)'  # Simple 2.47 format
        ]
        
        for pattern in chapter_verse_patterns:
            match = re.search(pattern, query_lower)
            if match:
                analysis['chapter'] = int(match.group(1))
                analysis['verse'] = int(match.group(2))
                analysis['is_verse_specific'] = True
                analysis['query_type'] = 'chapter_verse'
                analysis['priority_source'] = 'verses'
                break
        
        # Chapter-only detection
        if not analysis['is_verse_specific']:
            chapter_patterns = [
                r'chapter\s+(\d+)',
                r'adhyaya\s+(\d+)',
                r'अध्याय\s+(\d+)'
            ]
            
            for pattern in chapter_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    analysis['chapter'] = int(match.group(1))
                    analysis['query_type'] = 'chapter'
                    analysis['priority_source'] = 'verses'
                    break
        
        # Sanskrit/verse keywords
        verse_keywords = [
            'verse', 'श्लोक', 'sloka', 'mantra', 'sanskrit', 'devanagari',
            'original', 'bhagavad', 'gita', 'krishna', 'arjuna'
        ]
        
        spiritual_keywords = [
            'meaning', 'explanation', 'commentary', 'teaching', 'wisdom',
            'understanding', 'interpretation', 'philosophy', 'spiritual'
        ]
        
        found_verse_keywords = [kw for kw in verse_keywords if kw in query_lower]
        found_spiritual_keywords = [kw for kw in spiritual_keywords if kw in query_lower]
        
        analysis['keywords'] = found_verse_keywords + found_spiritual_keywords
        
        # Adjust priority based on keywords
        if found_verse_keywords and not found_spiritual_keywords:
            analysis['priority_source'] = 'verses'
        elif found_spiritual_keywords and not found_verse_keywords:
            analysis['priority_source'] = 'explanations'
        
        logger.info(f"Query analysis: {analysis}")
        return analysis
    
    def retrieve_from_verses(self, 
                           query: str, 
                           k: int = 5,
                           chapter: Optional[int] = None,
                           verse: Optional[int] = None) -> List[Document]:
        """
        Retrieve from clean verses collection.
        
        Args:
            query: Search query
            k: Number of results
            chapter: Chapter filter
            verse: Verse filter
            
        Returns:
            List of verse documents
        """
        try:
            # Build search query
            search_query = query
            
            # Enhance query for chapter/verse searches
            if chapter and verse:
                search_query = f"chapter {chapter} verse {verse} {query}"
            elif chapter:
                search_query = f"chapter {chapter} {query}"
            
            # Search verses collection
            results = self.verses_store.similarity_search(
                search_query, 
                k=k
            )
            
            # Filter by metadata if specified
            if chapter or verse:
                filtered_results = []
                for doc in results:
                    doc_chapter = doc.metadata.get('chapter_number')
                    doc_verses = doc.metadata.get('verse_numbers', [])
                    
                    # Convert verse_numbers string back to list if needed
                    if isinstance(doc_verses, str):
                        doc_verses = eval(doc_verses) if doc_verses != "[]" else []
                    
                    match = True
                    
                    if chapter and doc_chapter != chapter:
                        match = False
                    
                    if verse and match:
                        # Check if verse is in the document
                        verse_found = False
                        for v in doc_verses:
                            if str(verse) in str(v) or f"{chapter}.{verse}" in str(v):
                                verse_found = True
                                break
                        if not verse_found:
                            match = False
                    
                    if match:
                        filtered_results.append(doc)
                
                if filtered_results:
                    results = filtered_results[:k]
            
            logger.info(f"Retrieved {len(results)} verses for: {search_query}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving from verses: {str(e)}")
            return []
    
    def retrieve_from_explanations(self, 
                                 query: str, 
                                 k: int = 5,
                                 chapter: Optional[int] = None) -> List[Document]:
        """
        Retrieve from explanations collection.
        
        Args:
            query: Search query
            k: Number of results
            chapter: Chapter filter
            
        Returns:
            List of explanation documents
        """
        try:
            # Build search query
            search_query = query
            
            # Enhance query for chapter searches
            if chapter:
                search_query = f"chapter {chapter} {query}"
            
            # Search explanations collection
            results = self.explanations_store.similarity_search(
                search_query,
                k=k
            )
            
            logger.info(f"Retrieved {len(results)} explanations for: {search_query}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving from explanations: {str(e)}")
            return []
    
    def merge_results(self, 
                     verses: List[Document], 
                     explanations: List[Document],
                     query_analysis: Dict[str, Any]) -> List[Document]:
        """
        Intelligently merge results from both sources.
        
        Args:
            verses: Documents from verses collection
            explanations: Documents from explanations collection
            query_analysis: Query analysis results
            
        Returns:
            Merged and prioritized document list
        """
        merged = []
        
        query_type = query_analysis.get('query_type', 'general')
        priority_source = query_analysis.get('priority_source', 'both')
        
        if query_type == 'chapter_verse':
            # For chapter/verse queries: verses first, then explanations
            merged.extend(verses[:3])  # Top 3 verses
            merged.extend(explanations[:2])  # Top 2 explanations
            
        elif query_type == 'chapter':
            # For chapter queries: balanced mix
            merged.extend(verses[:2])  # Top 2 verses
            merged.extend(explanations[:3])  # Top 3 explanations
            
        elif priority_source == 'verses':
            # Verse-focused queries
            merged.extend(verses[:4])  # Top 4 verses
            merged.extend(explanations[:1])  # Top 1 explanation
            
        elif priority_source == 'explanations':
            # Explanation-focused queries
            merged.extend(explanations[:4])  # Top 4 explanations
            merged.extend(verses[:1])  # Top 1 verse
            
        else:
            # General balanced approach
            # Interleave results for variety
            max_len = max(len(verses), len(explanations))
            for i in range(max_len):
                if i < len(verses):
                    merged.append(verses[i])
                if i < len(explanations):
                    merged.append(explanations[i])
                if len(merged) >= 5:  # Limit total results
                    break
        
        # Remove duplicates while preserving order
        seen_content = set()
        unique_merged = []
        
        for doc in merged:
            content_hash = hash(doc.page_content[:100])  # Hash first 100 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_merged.append(doc)
        
        logger.info(f"Merged results: {len(unique_merged)} total documents")
        return unique_merged[:5]  # Return top 5
    
    def retrieve(self, 
                query: str, 
                k: int = 5,
                reading_style: str = "balanced") -> RetrievalResult:
        """
        Main retrieval method with dual-source intelligence.
        
        Args:
            query: User query
            k: Number of results per source
            reading_style: Reading style preference
            
        Returns:
            RetrievalResult with all components
        """
        logger.info(f"Dual-source retrieval for: {query}")
        
        # Analyze query
        analysis = self.analyze_query(query)
        
        # Retrieve from both sources
        verses = self.retrieve_from_verses(
            query, 
            k=k,
            chapter=analysis.get('chapter'),
            verse=analysis.get('verse')
        )
        
        explanations = self.retrieve_from_explanations(
            query,
            k=k,
            chapter=analysis.get('chapter')
        )
        
        # Merge results intelligently
        merged_context = self.merge_results(verses, explanations, analysis)
        
        # Create result object
        result = RetrievalResult(
            clean_verses=verses,
            explanations=explanations,
            merged_context=merged_context,
            query_type=analysis['query_type'],
            chapter=analysis.get('chapter'),
            verse=analysis.get('verse')
        )
        
        logger.info(f"Retrieval complete: {len(verses)} verses, {len(explanations)} explanations, {len(merged_context)} merged")
        return result

# Convenience function for easy usage
def create_dual_source_retriever() -> DualSourceRetriever:
    """Create and return a DualSourceRetriever instance."""
    return DualSourceRetriever()

if __name__ == "__main__":
    # Test the dual-source retriever
    retriever = create_dual_source_retriever()
    
    # Test queries
    test_queries = [
        "Chapter 2 verse 47",
        "What is karma yoga?",
        "Chapter 13 verse 11",
        "Explain the nature of the soul",
        "Bhagavad Gita 4.7"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: {query}")
        result = retriever.retrieve(query)
        print(f"   Type: {result.query_type}")
        print(f"   Verses: {len(result.clean_verses)}")
        print(f"   Explanations: {len(result.explanations)}")
        print(f"   Merged: {len(result.merged_context)}")
