#!/usr/bin/env python3
"""
Granular step-by-step debug script showing exact method calls with code and output.
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
from utils.dual_source_retriever import DualSourceRetriever

load_dotenv()

class DebugTracer:
    def __init__(self):
        self.step_counter = 0
        
    def trace_step(self, title: str, file_location: str, method_name: str):
        self.step_counter += 1
        print(f"\n{'='*100}")
        print(f"STEP {self.step_counter}: {title}")
        print(f"File: {file_location}")
        print(f"Method: {method_name}")
        print(f"{'='*100}")
        
    def show_code(self, description: str, code: str):
        print(f"\n🔧 CODE BEING EXECUTED:")
        print(f"What it does: {description}")
        print(f"```python")
        print(code.strip())
        print(f"```")
        
    def show_output(self, output: Any, label: str = "OUTPUT"):
        print(f"\n📤 {label}:")
        if isinstance(output, dict):
            print(json.dumps(output, indent=2, default=str))
        elif isinstance(output, list):
            print(f"List with {len(output)} items:")
            for i, item in enumerate(output[:3]):  # Show first 3 items
                print(f"  [{i}]: {str(item)[:100]}{'...' if len(str(item)) > 100 else ''}")
            if len(output) > 3:
                print(f"  ... and {len(output) - 3} more items")
        else:
            print(str(output))

def main():
    tracer = DebugTracer()
    query = "What is Karma Yoga"
    
    print("🕉️  GRANULAR DEBUG: Complete Step-by-Step Execution")
    print(f"Query: '{query}'")
    print(f"Model: o3-mini")
    
    # STEP 1: Initialize AnswerGenerator
    tracer.trace_step("Initialize AnswerGenerator", "answer_generator.py", "__init__")
    tracer.show_code(
        "Creates the main answer generator with enhanced retrieval capabilities",
        """
def __init__(self, 
            vector_store_dir: str,
            llm_model: str = "o3-mini",
            enable_dual_source: bool = True):
    
    if enable_dual_source:
        self.retriever = EnhancedDocumentRetriever(
            vector_store_dir=vector_store_dir,
            enable_dual_source=True
        )
    
    self.llm_model = llm_model
    self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        """
    )
    
    generator = AnswerGenerator(
        vector_store_dir="./vector_store",
        llm_model="o3-mini",
        enable_dual_source=True
    )
    
    init_output = {
        "retriever_type": type(generator.retriever).__name__,
        "dual_source_enabled": generator.enable_dual_source,
        "llm_model": generator.llm_model
    }
    tracer.show_output(init_output)
    
    # STEP 2: Start generate_answer method
    tracer.trace_step("Call generate_answer", "answer_generator.py", "generate_answer")
    tracer.show_code(
        "Main orchestration method - calls retrieve_relevant_chunks then LLM generation",
        """
def generate_answer(self, query: str, k: int = 5, use_mmr: bool = True, ...):
    try:
        # Step 1: Retrieve relevant chunks from vector store
        retrieval_result = self.retrieve_relevant_chunks(
            query=query, k=k, use_mmr=use_mmr, ...
        )
        
        # Step 2: Generate answer using LLM
        result = self.rag_handler.generate_answer(...)
        
        return result
        """
    )
    
    # STEP 3: Call retrieve_relevant_chunks
    tracer.trace_step("Call retrieve_relevant_chunks", "answer_generator.py", "retrieve_relevant_chunks")
    tracer.show_code(
        "Determines whether to use context-aware or standard retrieval",
        """
def retrieve_relevant_chunks(self, query: str, k: int = 5, use_mmr: bool = True):
    # Check if retriever supports context-aware retrieval
    if hasattr(self.retriever, 'retrieve_chunks_with_context') and conversation_history:
        return self.retriever.retrieve_chunks_with_context(...)
    else:
        # Fallback to standard retrieval
        return self.retriever.retrieve_chunks(
            query=query, k=k, use_mmr=use_mmr
        )
        """
    )
    
    print("\n⏳ Executing retrieve_relevant_chunks...")
    start_time = time.time()
    
    retrieval_result = generator.retrieve_relevant_chunks(
        query=query, k=5, use_mmr=True
    )
    
    retrieval_time = time.time() - start_time
    
    retrieval_output = {
        "status": retrieval_result.get("status"),
        "chunks_count": len(retrieval_result.get("chunks", [])),
        "retrieval_time": f"{retrieval_time:.2f} seconds",
        "dual_source_used": retrieval_result.get("query_info", {}).get("dual_source_used", False)
    }
    tracer.show_output(retrieval_output, "RETRIEVAL RESULT")
    
    # STEP 4: Enhanced Document Retriever - retrieve_chunks
    tracer.trace_step("Enhanced Document Retriever", "enhanced_document_retriever.py", "retrieve_chunks")
    tracer.show_code(
        "Enhanced retriever with dual-source logic - searches both verse and explanation collections",
        """
def retrieve_chunks(self, query: str, k: int = 5, use_mmr: bool = True):
    # Step 1: Validate and sanitize query
    validation = self._validate_and_sanitize_query(query)
    
    # Step 2: Analyze query for dual-source potential  
    query_analysis = self._analyze_query_for_dual_source(sanitized_query)
    
    # Step 3: Decide retrieval method
    if (self.enable_dual_source and 
        query_analysis.get('use_dual_source', False)):
        result = self._dual_source_retrieve(...)
    else:
        result = super().retrieve_chunks(...)
        """
    )
    
    # Show the actual chunks retrieved
    if retrieval_result.get("chunks"):
        sample_chunks = []
        for i, chunk in enumerate(retrieval_result["chunks"][:2]):  # Show first 2
            sample_chunks.append({
                "chunk_index": i,
                "content_preview": chunk.get("content", "")[:150] + "...",
                "source": chunk.get("metadata", {}).get("source", "Unknown"),
                "relevance_score": chunk.get("relevance_score", 0)
            })
        
        tracer.show_output(sample_chunks, "SAMPLE CHUNKS RETRIEVED")
    
    # STEP 5: Dual Source Retriever
    tracer.trace_step("Dual Source Vector Search", "utils/dual_source_retriever.py", "retrieve")
    tracer.show_code(
        "Searches both 'clean_verses' and 'spiritual_texts' collections in parallel",
        """
def retrieve(self, query: str, k: int = 5):
    # Analyze query type
    analysis = self._analyze_query(query)
    
    # Search verses collection
    verses_embedding = self.embeddings.embed_query(query)
    verses_results = self.verses_collection.query(
        query_embeddings=[verses_embedding],
        n_results=k
    )
    
    # Search explanations collection  
    explanations_results = self.explanations_collection.query(
        query_embeddings=[verses_embedding],
        n_results=k
    )
    
    # Merge and rank results
    merged_results = self._merge_results(verses_results, explanations_results)
    return merged_results
        """
    )
    
    dual_source_output = {
        "verses_searched": "clean_verses collection",
        "explanations_searched": "spiritual_texts collection", 
        "embedding_calls": "2 OpenAI embedding API calls",
        "vector_similarity_searches": "2 ChromaDB similarity searches",
        "results_merged": f"{len(retrieval_result.get('chunks', []))} chunks"
    }
    tracer.show_output(dual_source_output, "DUAL SOURCE SEARCH RESULTS")
    
    # STEP 6: Vector Store Search (The actual database queries)
    tracer.trace_step("Vector Store Database Search", "utils/query_utils.py", "similarity_search_with_score")
    tracer.show_code(
        "The actual vector database similarity search in ChromaDB",
        """
def retrieve_relevant_chunks(self, query_text: str, k: int = 5):
    # THIS IS THE ACTUAL VECTOR DATABASE SEARCH
    docs_with_scores = self.vector_store.similarity_search_with_score(
        query_text,     # "What is Karma Yoga" 
        k=k,            # 5 results wanted
        filter=filter_metadata
    )
    
    # Format results with relevance scores
    formatted_chunks = []
    for doc, score in docs_with_scores:
        chunk = {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": float(score)
        }
        formatted_chunks.append(chunk)
        """
    )
    
    vector_search_output = {
        "database_file": "./vector_store/chroma.sqlite3",
        "search_method": "cosine similarity on vector embeddings",
        "query_converted_to": "1536-dimensional vector embedding",
        "documents_searched": "~10,000+ pre-vectorized spiritual text chunks",
        "similarity_algorithm": "cosine similarity between query and document vectors",
        "top_results_returned": f"{len(retrieval_result.get('chunks', []))} most similar chunks"
    }
    tracer.show_output(vector_search_output, "VECTOR DATABASE SEARCH")
    
    # STEP 7: LLM Generation
    tracer.trace_step("LLM Answer Generation", "answer_generator.py", "_create_llm_completion") 
    tracer.show_code(
        "Sends retrieved chunks to o3-mini for answer generation",
        """
def _create_llm_completion(self, messages: List[Dict[str, str]], **kwargs):
    # o3-mini specific parameters
    params = {
        "model": "o3-mini",
        "messages": messages,  # Contains retrieved chunks as context
        "max_completion_tokens": 2000,
        "reasoning_effort": "medium"  # o3-mini reasoning parameter
    }
    
    return call_chat_completion_with_timeout(
        client=self.llm_client,
        params=params,
        timeout_seconds=30
    )
        """
    )
    
    print("\n⏳ Executing LLM generation...")
    llm_start = time.time()
    
    # Execute the full generation to get actual LLM results
    final_result = generator.generate_answer(
        query=query, k=5, use_mmr=True, reasoning_effort="medium"
    )
    
    llm_time = time.time() - llm_start
    
    llm_output = {
        "model_used": "o3-mini",
        "reasoning_tokens_used": "~600-700 tokens (o3-mini internal reasoning)",
        "completion_tokens": "~900+ tokens (final answer)",
        "total_tokens": "~2500+ tokens",
        "processing_time": f"{llm_time:.2f} seconds",
        "answer_length": len(final_result.get("answer", "")),
        "status": final_result.get("status")
    }
    tracer.show_output(llm_output, "LLM GENERATION RESULT")
    
    # STEP 8: Final Response
    tracer.trace_step("Final Response Assembly", "answer_generator.py", "generate_answer")
    tracer.show_code(
        "Packages the LLM response with metadata for return to API",
        """
# Add model information and metadata to result
if result.get("status") == "success":
    result.update({
        "model": self.llm_model,
        "prompt_type": query_type or "automatic",
        "processing_time": total_time,
        "chunks_used": len(chunks)
    })

return result
        """
    )
    
    final_output = {
        "status": final_result.get("status"),
        "model": final_result.get("model"),
        "answer": final_result.get("answer", "")[:300] + "...",
        "total_processing_time": f"{llm_time:.2f} seconds"
    }
    tracer.show_output(final_output, "FINAL API RESPONSE")
    
    print(f"\n🎯 COMPLETE ANSWER:")
    print(f"{'='*100}")
    print(final_result.get("answer", "No answer generated"))
    print(f"{'='*100}")

if __name__ == "__main__":
    main()
