#!/usr/bin/env python3
"""
Complete re-vectorization script with enhanced Unicode/Sanskrit support.
This script will replace your existing vector database with properly preserved Sanskrit text.
"""

import os
import shutil
import logging
import argparse
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Import enhanced utilities
from utils.langchain_utils import load_documents, chunk_documents
from utils.vectorization_utils import create_vectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"re_vectorization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def backup_existing_vector_store(vector_store_dir: str) -> str:
    """
    Create a backup of the existing vector store before re-vectorization.
    
    Args:
        vector_store_dir: Path to existing vector store
        
    Returns:
        Path to backup directory
    """
    if not os.path.exists(vector_store_dir):
        logger.info("No existing vector store found - creating fresh database")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f"{vector_store_dir}_backup_{timestamp}"
    
    try:
        logger.info(f"Creating backup of existing vector store...")
        shutil.copytree(vector_store_dir, backup_dir)
        logger.info(f"✅ Backup created at: {backup_dir}")
        return backup_dir
    except Exception as e:
        logger.error(f"❌ Failed to create backup: {str(e)}")
        return None

def remove_existing_vector_store(vector_store_dir: str):
    """Remove existing vector store directory."""
    if os.path.exists(vector_store_dir):
        try:
            shutil.rmtree(vector_store_dir)
            logger.info(f"✅ Removed existing vector store: {vector_store_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to remove existing vector store: {str(e)}")
            raise

def test_sanskrit_preservation(documents, sample_size: int = 5) -> Dict[str, Any]:
    """
    Test if Sanskrit text is properly preserved in loaded documents.
    
    Args:
        documents: List of loaded documents
        sample_size: Number of documents to sample for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("🔍 Testing Sanskrit preservation in loaded documents...")
    
    test_results = {
        'total_documents': len(documents),
        'documents_with_sanskrit': 0,
        'documents_with_transliteration': 0,
        'sample_sanskrit_texts': [],
        'unicode_validation': 'PASS'
    }
    
    # Sample documents for testing
    sample_docs = documents[:sample_size] if len(documents) > sample_size else documents
    
    for i, doc in enumerate(sample_docs):
        content = doc.page_content
        metadata = doc.metadata
        
        # Check for Devanagari characters
        devanagari_chars = [char for char in content if 0x0900 <= ord(char) <= 0x097F]
        if devanagari_chars:
            test_results['documents_with_sanskrit'] += 1
            # Store sample Sanskrit text
            sanskrit_sample = ''.join(devanagari_chars[:20])  # First 20 Sanskrit chars
            test_results['sample_sanskrit_texts'].append({
                'document_index': i,
                'sanskrit_sample': sanskrit_sample,
                'source': metadata.get('source', 'unknown')
            })
        
        # Check for transliteration patterns
        import re
        transliteration_patterns = [
            r'\b[a-zA-Z]*[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ][a-zA-Z]*\b',
            r'\b(om|aum|namah|namaha|svaha|svadha)\b'
        ]
        
        for pattern in transliteration_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                test_results['documents_with_transliteration'] += 1
                break
    
    # Validation
    if test_results['documents_with_sanskrit'] == 0:
        test_results['unicode_validation'] = 'FAIL - No Sanskrit found'
        logger.warning("⚠️ No Sanskrit characters found in sample documents")
    else:
        logger.info(f"✅ Found Sanskrit in {test_results['documents_with_sanskrit']}/{sample_size} sample documents")
    
    return test_results

def re_vectorize_with_unicode_support(
    input_dir: str,
    vector_store_dir: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    create_backup: bool = True,
    embedding_model: str = "huggingface"  # Default to free HuggingFace to save credits
) -> Dict[str, Any]:
    """
    Complete re-vectorization with enhanced Unicode/Sanskrit support.
    
    Args:
        input_dir: Directory containing PDF documents
        vector_store_dir: Directory for vector database
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        create_backup: Whether to backup existing vector store
        
    Returns:
        Dictionary with processing results
    """
    logger.info("🚀 Starting enhanced re-vectorization with Unicode support")
    logger.info("=" * 80)
    
    results = {
        'status': 'started',
        'backup_created': None,
        'documents_loaded': 0,
        'chunks_created': 0,
        'chunks_vectorized': 0,
        'sanskrit_test_results': {},
        'errors': []
    }
    
    try:
        # Step 1: Backup existing vector store
        if create_backup:
            backup_path = backup_existing_vector_store(vector_store_dir)
            results['backup_created'] = backup_path
        
        # Step 2: Remove existing vector store
        remove_existing_vector_store(vector_store_dir)
        
        # Step 3: Load documents with enhanced Unicode support
        logger.info("📚 Loading documents with enhanced Unicode support...")
        documents = load_documents(input_dir, use_enhanced_loader=True)
        
        if not documents:
            raise ValueError("No documents loaded - check input directory and PDF files")
        
        results['documents_loaded'] = len(documents)
        logger.info(f"✅ Loaded {len(documents)} documents")
        
        # Step 4: Test Sanskrit preservation
        sanskrit_test = test_sanskrit_preservation(documents)
        results['sanskrit_test_results'] = sanskrit_test
        
        if sanskrit_test['unicode_validation'] == 'FAIL - No Sanskrit found':
            logger.warning("⚠️ Sanskrit preservation test failed - proceeding anyway")
        
        # Step 5: Chunk documents with enhanced verse detection
        logger.info("✂️ Chunking documents with enhanced verse detection...")
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_verses=True
        )
        
        results['chunks_created'] = len(chunks)
        logger.info(f"✅ Created {len(chunks)} chunks")
        
        # Step 6: Vectorize and store with enhanced metadata
        logger.info(f"🔄 Vectorizing chunks with enhanced metadata using {embedding_model} embeddings...")
        if embedding_model == "openai":
            logger.warning("⚠️ Using OpenAI embeddings - this will consume API credits!")
        else:
            logger.info("✅ Using free HuggingFace embeddings - no API credits needed")
            
        vectorizer = create_vectorizer(
            persist_directory=vector_store_dir,
            model_name=embedding_model,
            collection_name="spiritual_texts"
        )
        
        vectorization_result = vectorizer.vectorize_chunks(chunks, batch_size=50)
        
        if vectorization_result['status'] == 'success':
            results['chunks_vectorized'] = vectorization_result['chunks_processed']
            results['status'] = 'success'
            logger.info(f"✅ Successfully vectorized {results['chunks_vectorized']} chunks")
        else:
            raise ValueError("Vectorization failed")
        
        # Step 7: Verify the new vector store
        logger.info("🔍 Verifying new vector store...")
        verification_result = verify_enhanced_vector_store(vector_store_dir)
        results['verification'] = verification_result
        
        logger.info("🎉 Re-vectorization completed successfully!")
        logger.info("=" * 80)
        
        # Print summary
        print_summary(results)
        
        return results
        
    except Exception as e:
        error_msg = f"Re-vectorization failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        results['status'] = 'failed'
        results['errors'].append(error_msg)
        return results

def verify_enhanced_vector_store(vector_store_dir: str) -> Dict[str, Any]:
    """
    Verify the enhanced vector store with Sanskrit text queries.
    
    Args:
        vector_store_dir: Path to vector store directory
        
    Returns:
        Dictionary with verification results
    """
    logger.info("Verifying enhanced vector store...")
    
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        
        # Initialize vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            collection_name="spiritual_texts",
            embedding_function=embeddings,
            persist_directory=vector_store_dir
        )
        
        # Test queries
        test_queries = [
            "Bhagavad Gita verse",
            "Sanskrit shloka",
            "Chapter 2 verse 48",
            "योग", # Sanskrit word for "yoga"
            "karma yoga"
        ]
        
        verification_results = {
            'vector_store_accessible': True,
            'query_results': {},
            'sanskrit_content_found': False,
            'total_documents': 0
        }
        
        # Get total document count
        try:
            collection = vectorstore._collection
            verification_results['total_documents'] = collection.count()
            logger.info(f"Vector store contains {verification_results['total_documents']} documents")
        except:
            logger.warning("Could not get document count")
        
        # Test each query
        for query in test_queries:
            try:
                results = vectorstore.similarity_search(query, k=3)
                verification_results['query_results'][query] = {
                    'results_found': len(results),
                    'sample_content': results[0].page_content[:200] if results else None
                }
                
                # Check for Sanskrit in results
                for result in results:
                    if any(0x0900 <= ord(char) <= 0x097F for char in result.page_content):
                        verification_results['sanskrit_content_found'] = True
                        break
                
                logger.info(f"Query '{query}': {len(results)} results")
                
            except Exception as e:
                logger.error(f"Query '{query}' failed: {str(e)}")
                verification_results['query_results'][query] = {'error': str(e)}
        
        if verification_results['sanskrit_content_found']:
            logger.info("✅ Sanskrit content verified in vector store")
        else:
            logger.warning("⚠️ No Sanskrit content found in verification queries")
        
        return verification_results
        
    except Exception as e:
        logger.error(f"Vector store verification failed: {str(e)}")
        return {'error': str(e), 'vector_store_accessible': False}

def print_summary(results: Dict[str, Any]):
    """Print a summary of the re-vectorization results."""
    print("\n" + "=" * 80)
    print("📋 RE-VECTORIZATION SUMMARY")
    print("=" * 80)
    
    print(f"Status: {'✅ SUCCESS' if results['status'] == 'success' else '❌ FAILED'}")
    print(f"Documents Loaded: {results['documents_loaded']}")
    print(f"Chunks Created: {results['chunks_created']}")
    print(f"Chunks Vectorized: {results['chunks_vectorized']}")
    
    if results.get('backup_created'):
        print(f"Backup Created: {results['backup_created']}")
    
    # Sanskrit test results
    if 'sanskrit_test_results' in results:
        sanskrit_results = results['sanskrit_test_results']
        print(f"\n📜 SANSKRIT PRESERVATION TEST:")
        print(f"Documents with Sanskrit: {sanskrit_results['documents_with_sanskrit']}")
        print(f"Documents with Transliteration: {sanskrit_results['documents_with_transliteration']}")
        print(f"Unicode Validation: {sanskrit_results['unicode_validation']}")
        
        if sanskrit_results['sample_sanskrit_texts']:
            print("Sample Sanskrit found:")
            for sample in sanskrit_results['sample_sanskrit_texts'][:2]:
                print(f"  - {sample['sanskrit_sample']} (from {sample['source']})")
    
    # Verification results
    if 'verification' in results:
        verification = results['verification']
        if verification.get('vector_store_accessible'):
            print(f"\n🔍 VERIFICATION RESULTS:")
            print(f"Total Documents in Vector Store: {verification.get('total_documents', 'Unknown')}")
            print(f"Sanskrit Content Found: {'✅ YES' if verification.get('sanskrit_content_found') else '❌ NO'}")
    
    if results.get('errors'):
        print(f"\n❌ ERRORS:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print("=" * 80)

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Re-vectorize documents with enhanced Unicode support")
    parser.add_argument("--input-dir", default="./documents", help="Input directory containing PDFs")
    parser.add_argument("--vector-store-dir", default="./vector_store", help="Vector store directory")
    parser.add_argument("--chunk-size", type=int, default=1500, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    parser.add_argument("--embedding-model", default="huggingface", choices=["openai", "huggingface"], 
                       help="Embedding model: 'openai' (uses credits) or 'huggingface' (free)")
    
    args = parser.parse_args()
    
    # Run re-vectorization
    results = re_vectorize_with_unicode_support(
        input_dir=args.input_dir,
        vector_store_dir=args.vector_store_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        create_backup=not args.no_backup,
        embedding_model=args.embedding_model
    )
    
    # Exit with appropriate code
    exit_code = 0 if results['status'] == 'success' else 1
    exit(exit_code)

if __name__ == "__main__":
    main()
