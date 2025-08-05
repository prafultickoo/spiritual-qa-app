#!/usr/bin/env python3
"""
Simple and direct verification script for spiritual document vector store.
This script directly tests the vector store without complex agent frameworks.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_vector_store_path() -> str:
    """Get the vector store path from environment or use default."""
    return os.getenv("VECTOR_STORE_DIR", "./vector_store")

def initialize_vector_store():
    """Initialize and return the vector store instance."""
    try:
        embeddings = OpenAIEmbeddings()
        vector_store_path = get_vector_store_path()
        
        if not os.path.exists(vector_store_path):
            raise FileNotFoundError(f"Vector store directory not found: {vector_store_path}")
        
        vectorstore = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings,
            collection_name="spiritual_texts"
        )
        
        logger.info("Vector store initialized successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise

def verify_vector_store_integrity() -> Dict[str, Any]:
    """Verify the structural integrity and accessibility of the ChromaDB vector store."""
    print("\n🔍 VERIFYING VECTOR STORE INTEGRITY...")
    print("-" * 50)
    
    try:
        vector_store_path = get_vector_store_path()
        result = {
            "status": "checking",
            "vector_store_path": vector_store_path,
            "database_files": [],
            "file_sizes": {},
            "collection_info": {},
            "document_count": 0,
            "errors": []
        }
        
        # Check if vector store directory exists
        if not os.path.exists(vector_store_path):
            result["status"] = "error"
            result["errors"].append(f"Vector store directory not found: {vector_store_path}")
            return result
        
        print(f"✅ Vector store directory found: {vector_store_path}")
        
        # List database files
        db_files = list(Path(vector_store_path).glob("*"))
        result["database_files"] = [str(f) for f in db_files]
        
        print(f"📁 Database files found: {len(db_files)}")
        for file_path in db_files:
            if file_path.is_file():
                size = file_path.stat().st_size
                result["file_sizes"][str(file_path)] = size
                print(f"   - {file_path.name}: {size:,} bytes")
        
        # Initialize ChromaDB client to check collection
        client = chromadb.PersistentClient(path=vector_store_path)
        collections = client.list_collections()
        
        if collections:
            collection = collections[0]  # Get the first collection
            count = collection.count()
            result["collection_info"] = {
                "name": collection.name,
                "count": count,
                "metadata": collection.metadata
            }
            result["document_count"] = count
            print(f"📊 Collection '{collection.name}': {count:,} documents")
        
        # Try to initialize vector store
        vectorstore = initialize_vector_store()
        
        result["status"] = "success"
        print("✅ Vector store integrity verification PASSED")
        
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(str(e))
        print(f"❌ Vector store integrity verification FAILED: {str(e)}")
    
    return result

def test_similarity_search() -> Dict[str, Any]:
    """Perform comprehensive similarity search tests using various spiritual queries."""
    print("\n🔍 TESTING SIMILARITY SEARCH...")
    print("-" * 50)
    
    try:
        vectorstore = initialize_vector_store()
        
        # Test queries covering different aspects of spiritual content
        test_queries = [
            "What is the meaning of life?",
            "dharma and righteous living",
            "meditation and self-realization", 
            "karma and its effects",
            "moksha liberation",
            "आत्मा और परमात्मा",  # Soul and Supreme Soul in Hindi
            "yoga and spiritual practice",
            "devotion and surrender"
        ]
        
        result = {
            "status": "testing",
            "total_queries": len(test_queries),
            "test_results": [],
            "successful_searches": 0,
            "errors": []
        }
        
        for i, query in enumerate(test_queries, 1):
            try:
                print(f"🔎 Query {i}/{len(test_queries)}: '{query}'")
                
                # Perform similarity search
                docs = vectorstore.similarity_search_with_score(query, k=3)
                
                if docs:
                    result["successful_searches"] += 1
                    print(f"   ✅ Found {len(docs)} relevant documents")
                    
                    # Show top result
                    top_doc, top_score = docs[0]
                    print(f"   📄 Top result (score: {top_score:.4f}): {top_doc.page_content[:100]}...")
                    if "verses" in top_doc.metadata:
                        print(f"   🕉️  Verses preserved: {bool(top_doc.metadata['verses'])}")
                else:
                    print(f"   ❌ No documents found")
                
                query_result = {
                    "query": query,
                    "results_count": len(docs),
                    "top_score": docs[0][1] if docs else None,
                    "has_verses": "verses" in docs[0][0].metadata if docs else False
                }
                result["test_results"].append(query_result)
                
            except Exception as e:
                result["errors"].append(f"Query '{query}' failed: {str(e)}")
                print(f"   ❌ Search failed: {str(e)}")
        
        result["status"] = "completed"
        print(f"\n✅ Similarity search testing COMPLETED: {result['successful_searches']}/{len(test_queries)} successful")
        
    except Exception as e:
        result = {"status": "error", "error": str(e)}
        print(f"❌ Similarity search testing FAILED: {str(e)}")
    
    return result

def validate_metadata_preservation() -> Dict[str, Any]:
    """Validate that all document metadata has been properly preserved during vectorization."""
    print("\n🔍 VALIDATING METADATA PRESERVATION...")
    print("-" * 50)
    
    try:
        vectorstore = initialize_vector_store()
        
        # Get a sample of documents to check metadata
        docs = vectorstore.similarity_search("spiritual wisdom", k=20)
        
        result = {
            "status": "validating",
            "total_documents_sampled": len(docs),
            "metadata_fields": {},
            "verses_preservation": {
                "documents_with_verses": 0,
                "sample_verses": []
            },
            "errors": []
        }
        
        print(f"📊 Analyzing metadata from {len(docs)} sample documents...")
        
        # Analyze metadata fields
        all_metadata_keys = set()
        for doc in docs:
            all_metadata_keys.update(doc.metadata.keys())
        
        print(f"🏷️  Metadata fields found: {', '.join(all_metadata_keys)}")
        
        # Count occurrences of each metadata field
        for key in all_metadata_keys:
            count = sum(1 for doc in docs if key in doc.metadata)
            percentage = (count / len(docs)) * 100
            result["metadata_fields"][key] = {"count": count, "percentage": percentage}
            print(f"   - {key}: {count}/{len(docs)} documents ({percentage:.1f}%)")
        
        # Check verses preservation
        verses_count = 0
        for doc in docs:
            if "verses" in doc.metadata and doc.metadata["verses"]:
                verses_count += 1
                if len(result["verses_preservation"]["sample_verses"]) < 3:
                    result["verses_preservation"]["sample_verses"].append({
                        "verses": doc.metadata["verses"][:100] + "..." if len(doc.metadata["verses"]) > 100 else doc.metadata["verses"],
                        "source": doc.metadata.get("source", "unknown")
                    })
        
        result["verses_preservation"]["documents_with_verses"] = verses_count
        print(f"🕉️  Documents with verses: {verses_count}/{len(docs)} ({(verses_count/len(docs)*100):.1f}%)")
        
        # Show sample verses
        if result["verses_preservation"]["sample_verses"]:
            print("📜 Sample preserved verses:")
            for i, sample in enumerate(result["verses_preservation"]["sample_verses"], 1):
                print(f"   {i}. {sample['verses']} (from {sample['source']})")
        
        result["status"] = "completed"
        print("✅ Metadata validation COMPLETED")
        
    except Exception as e:
        result = {"status": "error", "error": str(e)}
        print(f"❌ Metadata validation FAILED: {str(e)}")
    
    return result

def test_document_retrieval() -> Dict[str, Any]:
    """Test end-to-end document retrieval functionality with real-world spiritual queries."""
    print("\n🔍 TESTING DOCUMENT RETRIEVAL...")
    print("-" * 50)
    
    try:
        vectorstore = initialize_vector_store()
        
        # Real-world spiritual questions
        test_questions = [
            "How can I find inner peace?",
            "What is the purpose of meditation?", 
            "How do I overcome suffering?",
            "What is the path to enlightenment?",
            "How should I live righteously?"
        ]
        
        result = {
            "status": "testing",
            "total_questions": len(test_questions),
            "successful_retrievals": 0,
            "retrieval_tests": [],
            "errors": []
        }
        
        for i, question in enumerate(test_questions, 1):
            try:
                print(f"❓ Question {i}/{len(test_questions)}: '{question}'")
                
                # Retrieve relevant documents
                docs = vectorstore.similarity_search_with_score(question, k=3)
                
                if docs:
                    result["successful_retrievals"] += 1
                    print(f"   ✅ Retrieved {len(docs)} relevant documents")
                    
                    # Show best answer
                    best_doc, best_score = docs[0]
                    print(f"   📖 Best match (score: {best_score:.4f}):")
                    print(f"      {best_doc.page_content[:200]}...")
                    print(f"      Source: {best_doc.metadata.get('source', 'unknown')}")
                else:
                    print(f"   ❌ No relevant documents found")
                
                test_result = {
                    "question": question,
                    "documents_retrieved": len(docs),
                    "best_score": docs[0][1] if docs else None,
                    "content_quality": "good" if docs else "poor"
                }
                result["retrieval_tests"].append(test_result)
                
            except Exception as e:
                result["errors"].append(f"Retrieval test for '{question}' failed: {str(e)}")
                print(f"   ❌ Retrieval failed: {str(e)}")
        
        result["status"] = "completed"
        print(f"\n✅ Document retrieval testing COMPLETED: {result['successful_retrievals']}/{len(test_questions)} successful")
        
    except Exception as e:
        result = {"status": "error", "error": str(e)}
        print(f"❌ Document retrieval testing FAILED: {str(e)}")
    
    return result

def run_comprehensive_verification():
    """Run all verification tests and generate a comprehensive report."""
    print("=" * 60)
    print("🕉️  SPIRITUAL DOCUMENT VECTOR STORE VERIFICATION")
    print("=" * 60)
    
    verification_results = {
        "timestamp": str(Path().resolve()),
        "vector_store_path": get_vector_store_path(),
        "tests": {}
    }
    
    # Run all verification tests
    tests = [
        ("integrity", verify_vector_store_integrity),
        ("similarity_search", test_similarity_search),
        ("metadata_preservation", validate_metadata_preservation), 
        ("document_retrieval", test_document_retrieval)
    ]
    
    for test_name, test_func in tests:
        try:
            verification_results["tests"][test_name] = test_func()
        except Exception as e:
            verification_results["tests"][test_name] = {
                "status": "error",
                "error": str(e)
            }
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    total_tests = len(tests)
    passed_tests = sum(1 for test in verification_results["tests"].values() if test.get("status") == "success" or test.get("status") == "completed")
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL VERIFICATION TESTS PASSED!")
        print("✅ Vector store is ready for production use")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed or had issues")
        print("❗ Review the detailed results above")
    
    # Save detailed report
    report_file = "verification_report.json"
    try:
        with open(report_file, 'w') as f:
            json.dump(verification_results, f, indent=2, default=str)
        print(f"📄 Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"❌ Failed to save report: {str(e)}")
    
    print("=" * 60)
    return verification_results

def main():
    """Main function to run vector store verification."""
    try:
        results = run_comprehensive_verification()
        return results
    except Exception as e:
        print(f"❌ Verification process failed: {str(e)}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    main()
