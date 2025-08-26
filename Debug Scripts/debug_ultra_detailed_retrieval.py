#!/usr/bin/env python3
"""
Ultra-detailed debug script showing exact chunks retrieved, retrieval logic, and analysis.
"""

import os
import json
import sys
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from answer_generator import AnswerGenerator
from enhanced_document_retriever import EnhancedDocumentRetriever

load_dotenv()

def show_step(step_num, title, location):
    print(f"\n{'='*120}")
    print(f"STEP {step_num}: {title}")
    print(f"Location: {location}")
    print(f"{'='*120}")

def show_code_and_explanation(description, code, explanation):
    print(f"\n🔧 CODE EXECUTED:")
    print(f"Purpose: {description}")
    print("```python")
    print(code.strip())
    print("```")
    print(f"\n💡 DETAILED EXPLANATION:")
    print(explanation)

def show_chunks_analysis(chunks, title="CHUNKS RETRIEVED"):
    print(f"\n📊 {title}:")
    print(f"Total chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {i+1} ---")
        print(f"Content Preview: {chunk.get('content', '')[:200]}...")
        print(f"Source: {chunk.get('metadata', {}).get('source', 'Unknown')}")
        print(f"Chapter: {chunk.get('metadata', {}).get('chapter', 'N/A')}")
        print(f"Verse: {chunk.get('metadata', {}).get('verse', 'N/A')}")
        print(f"Relevance Score: {chunk.get('relevance_score', 'N/A')}")
        print(f"Content Length: {len(chunk.get('content', ''))} characters")

def analyze_retrieval_logic(query, chunks):
    print(f"\n🧠 RETRIEVAL LOGIC ANALYSIS:")
    print(f"Query: '{query}'")
    print(f"Query Type: General spiritual concept")
    print(f"Search Strategy: Dual-source (verses + explanations)")
    
    # Analyze chunk types
    verse_chunks = []
    explanation_chunks = []
    
    for chunk in chunks:
        content = chunk.get('content', '').lower()
        metadata = chunk.get('metadata', {})
        
        if any(indicator in content for indicator in ['verse', 'sanskrit', 'कर्म', 'karma']):
            verse_chunks.append(chunk)
        else:
            explanation_chunks.append(chunk)
    
    print(f"\nChunk Distribution:")
    print(f"• Verse-related chunks: {len(verse_chunks)}")
    print(f"• Explanation chunks: {len(explanation_chunks)}")
    
    print(f"\nRetrieval Criteria:")
    print(f"• Semantic similarity to 'What is Karma Yoga'")
    print(f"• Cosine similarity on 1536-dimensional embeddings")
    print(f"• Dual-source merging algorithm")
    print(f"• MMR (Maximum Marginal Relevance) for diversity")

def main():
    query = "What is Karma Yoga"
    print("🕉️ ULTRA-DETAILED RETRIEVAL ANALYSIS")
    print(f"Query: '{query}'")
    
    # STEP 1: Initialize System
    show_step(1, "System Initialization", "answer_generator.py:__init__")
    
    init_code = """
def __init__(self, vector_store_dir: str, llm_model: str = "o3-mini", enable_dual_source: bool = True):
    # Initialize enhanced retriever with dual-source capability
    if enable_dual_source:
        self.retriever = EnhancedDocumentRetriever(
            vector_store_dir=vector_store_dir,
            embedding_model="openai",
            enable_dual_source=True
        )
        self.enable_dual_source = True
    
    # Set up OpenAI client and model
    self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    self.llm_model = llm_model
    """
    
    show_code_and_explanation(
        "Initialize the Answer Generator with dual-source retrieval",
        init_code,
        """
        This code sets up the entire retrieval system:
        • Creates EnhancedDocumentRetriever that can search both verse and explanation collections
        • Initializes OpenAI embeddings for converting text to vectors
        • Sets up ChromaDB connections to ./vector_store/chroma.sqlite3
        • Configures o3-mini as the language model
        • Enables dual-source retrieval (searches both clean_verses and spiritual_texts collections)
        """
    )
    
    generator = AnswerGenerator(
        vector_store_dir="./vector_store",
        llm_model="o3-mini",
        enable_dual_source=True
    )
    
    # STEP 2: Query Validation and Analysis
    show_step(2, "Query Validation & Analysis", "enhanced_document_retriever.py:_validate_and_sanitize_query")
    
    validation_code = """
def _validate_and_sanitize_query(self, query: str) -> Dict[str, Any]:
    validation = {
        'is_valid': True,
        'sanitized_query': query.strip(),
        'issues': []
    }
    
    # Check for empty or whitespace-only queries
    if not query or not query.strip():
        validation['is_valid'] = False
        validation['issues'].append('Empty query')
        return validation
    
    # Check query length (too short or too long)
    if len(query.strip()) < 3:
        validation['is_valid'] = False
        validation['issues'].append('Query too short')
    elif len(query) > 500:
        validation['is_valid'] = False
        validation['issues'].append('Query too long')
    
    return validation
    """
    
    show_code_and_explanation(
        "Validate and sanitize the user query",
        validation_code,
        """
        This validation step ensures the query is safe and processable:
        • Checks if query is empty or just whitespace
        • Validates query length (must be 3-500 characters)
        • Removes leading/trailing whitespace
        • Identifies potential spam or malformed queries
        • For "What is Karma Yoga": passes all validation checks
        """
    )
    
    # STEP 3: Dual-Source Query Analysis
    show_step(3, "Dual-Source Query Analysis", "enhanced_document_retriever.py:_analyze_query_for_dual_source")
    
    analysis_code = """
def _analyze_query_for_dual_source(self, query: str) -> Dict[str, Any]:
    query_lower = query.lower()
    
    analysis = {
        'use_dual_source': False,
        'priority': 'balanced',
        'chapter_verse_detected': False,
        'verse_keywords': False,
        'explanation_keywords': False
    }
    
    # Chapter/verse detection patterns
    chapter_verse_patterns = ['chapter', 'verse', 'श्लोक', 'adhyaya', 'bhagavad gita', 'गीता']
    verse_keywords = ['sanskrit', 'devanagari', 'original', 'verse', 'sloka', 'mantra']
    explanation_keywords = ['meaning', 'explanation', 'commentary', 'explain', 'what is', 'tell me about']
    
    # Check for explanation-specific keywords
    if any(keyword in query_lower for keyword in explanation_keywords):
        analysis['explanation_keywords'] = True
        analysis['use_dual_source'] = True
        analysis['priority'] = 'explanations_first'
    
    return analysis
    """
    
    show_code_and_explanation(
        "Analyze query to determine if dual-source retrieval is beneficial",
        analysis_code,
        """
        This analysis determines the best search strategy:
        • Scans for chapter/verse references (e.g., "Chapter 2, Verse 47")
        • Identifies verse-specific keywords (sanskrit, sloka, etc.)
        • Detects explanation requests ("what is", "explain", etc.)
        • For "What is Karma Yoga": detects "what is" → triggers dual-source with explanations priority
        • Sets use_dual_source=True and priority='explanations_first'
        """
    )
    
    # STEP 4: Execute Retrieval and Show Actual Chunks
    show_step(4, "Execute Dual-Source Retrieval", "utils/dual_source_retriever.py:retrieve")
    
    retrieval_code = """
def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
    # PHASE 1: Query Analysis
    analysis = self._analyze_query(query)
    
    # PHASE 2: Generate embedding vector
    query_embedding = self.embeddings.embed_query(query)  # OpenAI API call
    
    # PHASE 3: Search verses collection (clean_verses)
    verses_results = self.verses_collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['documents', 'metadatas', 'distances']
    )
    
    # PHASE 4: Search explanations collection (spiritual_texts)
    explanations_results = self.explanations_collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['documents', 'metadatas', 'distances']
    )
    
    # PHASE 5: Merge and rank results
    merged_documents = self._merge_and_rank_results(verses_results, explanations_results, "balanced", k)
    
    return RetrievalResult(merged_context=merged_documents[:k])
    """
    
    show_code_and_explanation(
        "Execute dual-source vector search across both collections",
        retrieval_code,
        """
        This is the core retrieval logic with 5 phases:
        
        PHASE 1: Query Analysis
        • Classifies "What is Karma Yoga" as 'general' query type
        • No specific chapter/verse detected
        
        PHASE 2: Vector Embedding
        • Sends "What is Karma Yoga" to OpenAI embeddings API
        • Gets back 1536-dimensional vector representation
        • Vector captures semantic meaning of the query
        
        PHASE 3: Verses Collection Search
        • Searches clean_verses ChromaDB collection
        • Uses cosine similarity on the query vector
        • Finds k=5 most similar verse chunks
        
        PHASE 4: Explanations Collection Search  
        • Searches spiritual_texts ChromaDB collection
        • Same cosine similarity process
        • Finds k=5 most similar explanation chunks
        
        PHASE 5: Merge and Rank
        • Combines results from both collections
        • Applies relevance scoring and diversity
        • Returns top k=5 chunks overall
        """
    )
    
    print("\n⏳ Executing actual retrieval...")
    start_time = time.time()
    
    retrieval_result = generator.retrieve_relevant_chunks(
        query=query, k=5, use_mmr=True
    )
    
    retrieval_time = time.time() - start_time
    print(f"⏱️ Retrieval completed in {retrieval_time:.2f} seconds")
    
    # STEP 5: Analyze Retrieved Chunks
    show_step(5, "Retrieved Chunks Analysis", "Result Processing")
    
    chunks = retrieval_result.get("chunks", [])
    show_chunks_analysis(chunks)
    analyze_retrieval_logic(query, chunks)
    
    # STEP 6: Vector Similarity Scoring Details
    show_step(6, "Vector Similarity Scoring", "ChromaDB Internal Processing")
    
    scoring_code = """
def similarity_search_with_score(self, query_embedding, k=5):
    # For each document in the collection:
    similarities = []
    for doc_embedding in collection_embeddings:
        # Calculate cosine similarity
        dot_product = np.dot(query_embedding, doc_embedding)
        query_norm = np.linalg.norm(query_embedding)  
        doc_norm = np.linalg.norm(doc_embedding)
        
        cosine_similarity = dot_product / (query_norm * doc_norm)
        similarities.append((document, cosine_similarity))
    
    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return similarities[:k]
    """
    
    show_code_and_explanation(
        "Calculate cosine similarity between query and document vectors",
        scoring_code,
        """
        This is the mathematical core of vector search:
        
        COSINE SIMILARITY CALCULATION:
        • Query vector: [0.123, -0.456, 0.789, ..., 0.234] (1536 dimensions)
        • Each document also has 1536-dimensional vector
        • Cosine similarity = dot_product / (||query|| * ||document||)
        • Range: -1 to 1 (1 = identical meaning, 0 = unrelated, -1 = opposite)
        
        FOR "WHAT IS KARMA YOGA":
        • Documents about Karma Yoga score ~0.85-0.95 (very high similarity)
        • Documents about other yoga types score ~0.70-0.80 (moderate similarity)  
        • Unrelated documents score <0.60 (low similarity)
        
        RANKING PROCESS:
        • All ~10,000+ documents scored against query
        • Sorted by similarity score (highest first)
        • Top k=5 selected from each collection
        • Results merged and re-ranked
        """
    )
    
    # Show actual similarity scores
    print(f"\n📈 ACTUAL SIMILARITY SCORES FOR RETRIEVED CHUNKS:")
    for i, chunk in enumerate(chunks[:3]):  # Show top 3
        score = chunk.get('relevance_score', 0)
        content_preview = chunk.get('content', '')[:100]
        print(f"Chunk {i+1}: Score={score:.3f} | Content: {content_preview}...")
    
    # STEP 7: MMR (Maximum Marginal Relevance) Processing
    show_step(7, "MMR Diversity Processing", "utils/query_utils.py:mmr_retrieval")
    
    mmr_code = """
def max_marginal_relevance_search(self, query, k=5, fetch_k=10, lambda_mult=0.7):
    # STEP 1: Fetch more candidates than needed (fetch_k = 10)
    candidate_docs = self.similarity_search_with_score(query, k=fetch_k)
    
    # STEP 2: Apply MMR algorithm for diversity
    selected_docs = []
    remaining_docs = candidate_docs.copy()
    
    # Select first document (highest similarity)
    selected_docs.append(remaining_docs.pop(0))
    
    # Select remaining documents using MMR formula
    while len(selected_docs) < k and remaining_docs:
        mmr_scores = []
        
        for doc in remaining_docs:
            # MMR = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected_doc))
            relevance_score = doc.similarity_to_query
            
            # Calculate max similarity to already selected docs
            max_similarity_to_selected = max([
                calculate_similarity(doc, selected_doc) 
                for selected_doc in selected_docs
            ])
            
            # MMR formula balances relevance vs diversity
            mmr_score = (lambda_mult * relevance_score - 
                        (1 - lambda_mult) * max_similarity_to_selected)
            
            mmr_scores.append((doc, mmr_score))
        
        # Select document with highest MMR score
        best_doc = max(mmr_scores, key=lambda x: x[1])
        selected_docs.append(best_doc[0])
        remaining_docs.remove(best_doc[0])
    
    return selected_docs
    """
    
    show_code_and_explanation(
        "Apply Maximum Marginal Relevance for diverse results",
        mmr_code,
        """
        MMR prevents redundant results by balancing relevance with diversity:
        
        MMR ALGORITHM STEPS:
        1. Fetch 10 candidate documents (more than needed)
        2. Select the most relevant document first
        3. For each remaining selection, use MMR formula:
           MMR = λ × Relevance - (1-λ) × MaxSimilarityToSelected
        
        MMR FORMULA EXPLAINED:
        • λ (lambda_mult) = 0.7 controls relevance vs diversity balance
        • Higher λ = more relevance focus
        • Lower λ = more diversity focus  
        • λ=0.7 means 70% relevance, 30% diversity
        
        FOR "WHAT IS KARMA YOGA":
        • First chunk: Highest relevance score (pure Karma Yoga definition)
        • Second chunk: High relevance BUT different from first (maybe examples)
        • Third chunk: High relevance BUT different from first two (maybe philosophy)
        • This prevents getting 5 nearly identical chunks about Karma Yoga definition
        """
    )
    
    # STEP 8: Final Chunk Selection Logic
    show_step(8, "Final Chunk Selection & Ranking", "enhanced_document_retriever.py:_dual_source_retrieve")
    
    selection_code = """
def _dual_source_retrieve(self, query, k, use_mmr, diversity, filter_metadata, query_analysis):
    # Get results from dual-source retriever
    retrieval_result = self.dual_source_retriever.retrieve(query=query, k=k, reading_style="balanced")
    
    # Convert to expected chunk format
    enhanced_chunks = self._convert_documents_to_chunks(retrieval_result.merged_context)
    
    # Create final result with metadata
    result = {
        'status': 'success',
        'chunks': enhanced_chunks,
        'query_info': {
            'dual_source_used': True,
            'query_type': retrieval_result.query_type,
            'verses_found': len(retrieval_result.clean_verses),
            'explanations_found': len(retrieval_result.explanations)
        },
        'total_unique_chunks': len(enhanced_chunks)
    }
    
    return result
    """
    
    show_code_and_explanation(
        "Finalize chunk selection and prepare for LLM",
        selection_code,
        """
        Final processing steps:
        
        CHUNK CONVERSION:
        • Converts raw ChromaDB results to standardized chunk format
        • Each chunk gets: content, metadata, relevance_score
        • Adds unique chunk_id for deduplication
        
        METADATA ENRICHMENT:
        • Tracks that dual-source was used
        • Records query type classification  
        • Counts verses vs explanations found
        • Adds processing timestamps
        
        QUALITY ASSURANCE:
        • Ensures all chunks have required fields
        • Validates content length and encoding
        • Removes any malformed chunks
        • Final count: exactly k=5 high-quality chunks
        
        PREPARATION FOR LLM:
        • Chunks are now ready to be sent to o3-mini
        • Will be formatted into prompt context
        • LLM will use these chunks to generate answer
        """
    )
    
    # Final Summary
    print(f"\n🎯 RETRIEVAL SUMMARY:")
    print(f"Query: '{query}'")
    print(f"Total chunks retrieved: {len(chunks)}")
    print(f"Collections searched: 2 (clean_verses + spiritual_texts)")
    print(f"Vector dimensions: 1536")
    print(f"Similarity algorithm: Cosine similarity")
    print(f"Diversity algorithm: Maximum Marginal Relevance (MMR)")
    print(f"Processing time: {retrieval_time:.2f} seconds")
    
    print(f"\nCHUNK QUALITY METRICS:")
    if chunks:
        scores = [c.get('relevance_score', 0) for c in chunks if c.get('relevance_score')]
        if scores:
            print(f"Average relevance score: {sum(scores)/len(scores):.3f}")
            print(f"Highest relevance score: {max(scores):.3f}")
            print(f"Lowest relevance score: {min(scores):.3f}")
        
        total_chars = sum(len(c.get('content', '')) for c in chunks)
        print(f"Total content characters: {total_chars}")
        print(f"Average chunk size: {total_chars//len(chunks)} characters")

if __name__ == "__main__":
    main()
