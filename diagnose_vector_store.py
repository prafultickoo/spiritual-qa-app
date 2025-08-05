#!/usr/bin/env python3
"""
Diagnostic script to identify and fix vector store retrieval issues.
"""

import os
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def diagnose_vector_store():
    """Diagnose vector store retrieval issues."""
    print("🔍 DIAGNOSING VECTOR STORE RETRIEVAL ISSUES...")
    print("=" * 60)
    
    vector_store_path = os.getenv("VECTOR_STORE_DIR", "./vector_store")
    
    # 1. Direct ChromaDB inspection
    print("\n1️⃣ DIRECT CHROMADB INSPECTION:")
    print("-" * 40)
    
    try:
        client = chromadb.PersistentClient(path=vector_store_path)
        collections = client.list_collections()
        
        print(f"📁 Collections found: {len(collections)}")
        for collection in collections:
            print(f"   - Name: {collection.name}")
            print(f"   - Count: {collection.count()}")
            print(f"   - Metadata: {collection.metadata}")
            
            # Get a few sample documents
            print("   - Sample documents:")
            try:
                # Get first 3 documents
                results = collection.get(limit=3)
                print(f"     Retrieved {len(results['ids'])} sample documents")
                for i, doc_id in enumerate(results['ids']):
                    content = results['documents'][i] if results['documents'] else "No content"
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    print(f"     Doc {i+1}: ID={doc_id[:20]}...")
                    print(f"              Content: {content[:100]}...")
                    print(f"              Metadata: {list(metadata.keys()) if metadata else 'None'}")
            except Exception as e:
                print(f"     ❌ Error getting sample docs: {e}")
    
    except Exception as e:
        print(f"❌ ChromaDB inspection failed: {e}")
        return
    
    # 2. Langchain Vector Store Inspection
    print("\n2️⃣ LANGCHAIN VECTOR STORE INSPECTION:")
    print("-" * 40)
    
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings
        )
        
        print("✅ Vector store initialized successfully")
        
        # Test basic query without embeddings
        print("🔎 Testing basic query...")
        try:
            # Try to get some documents directly
            results = vectorstore._collection.get(limit=5)
            print(f"   Direct collection query returned {len(results['ids'])} documents")
            
            if results['ids']:
                # Test similarity search with a simple query
                print("🔎 Testing similarity search...")
                docs = vectorstore.similarity_search("test", k=3)
                print(f"   Similarity search returned {len(docs)} documents")
                
                if docs:
                    print(f"   First result: {docs[0].page_content[:100]}...")
                else:
                    print("   ❌ No documents returned from similarity search")
                    
                    # Test with different embedding model
                    print("🔎 Testing with different approaches...")
                    
                    # Try without score
                    docs_no_score = vectorstore.similarity_search("spiritual", k=5)
                    print(f"   Search without score: {len(docs_no_score)} documents")
                    
                    # Try different embedding model
                    try:
                        embeddings_alt = OpenAIEmbeddings(model="text-embedding-3-small")
                        vectorstore_alt = Chroma(
                            persist_directory=vector_store_path,
                            embedding_function=embeddings_alt
                        )
                        docs_alt = vectorstore_alt.similarity_search("spiritual", k=3)
                        print(f"   Search with alt embedding: {len(docs_alt)} documents")
                    except Exception as e:
                        print(f"   Alt embedding test failed: {e}")
            else:
                print("   ❌ No documents found in collection")
                
        except Exception as e:
            print(f"   ❌ Query test failed: {e}")
            
    except Exception as e:
        print(f"❌ Langchain inspection failed: {e}")
    
    # 3. Collection Settings Analysis
    print("\n3️⃣ COLLECTION SETTINGS ANALYSIS:")
    print("-" * 40)
    
    try:
        client = chromadb.PersistentClient(path=vector_store_path)
        collection = client.get_collection("spiritual_texts")
        
        print("🔧 Collection configuration:")
        print(f"   - Name: {collection.name}")
        print(f"   - Count: {collection.count()}")
        
        # Check if embeddings exist
        sample = collection.get(limit=1, include=["embeddings", "documents", "metadatas"])
        if sample['embeddings'] and sample['embeddings'][0]:
            print(f"   - Embeddings: ✅ Present (dimension: {len(sample['embeddings'][0])})")
        else:
            print("   - Embeddings: ❌ Missing or empty")
            
        if sample['documents']:
            print(f"   - Documents: ✅ Present")
        else:
            print("   - Documents: ❌ Missing")
            
        print(f"   - Metadata: {'✅ Present' if sample['metadatas'] else '❌ Missing'}")
        
    except Exception as e:
        print(f"❌ Collection analysis failed: {e}")
    
    print("\n" + "=" * 60)

def fix_vector_store_retrieval():
    """Attempt to fix retrieval issues."""
    print("\n🔧 ATTEMPTING TO FIX RETRIEVAL ISSUES...")
    print("=" * 60)
    
    vector_store_path = os.getenv("VECTOR_STORE_DIR", "./vector_store")
    
    try:
        # Reinitialize with explicit collection name
        embeddings = OpenAIEmbeddings()
        
        # Try different initialization approaches
        print("1️⃣ Testing different initialization methods...")
        
        # Method 1: Default initialization
        try:
            vectorstore1 = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embeddings
            )
            docs1 = vectorstore1.similarity_search("spiritual", k=1)
            print(f"   Method 1 (default): {len(docs1)} documents found")
        except Exception as e:
            print(f"   Method 1 failed: {e}")
        
        # Method 2: With explicit collection name
        try:
            vectorstore2 = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embeddings,
                collection_name="spiritual_texts"
            )
            docs2 = vectorstore2.similarity_search("spiritual", k=1)
            print(f"   Method 2 (explicit collection): {len(docs2)} documents found")
        except Exception as e:
            print(f"   Method 2 failed: {e}")
            
        # Method 3: Direct ChromaDB query with embeddings
        try:
            client = chromadb.PersistentClient(path=vector_store_path)
            collection = client.get_collection("spiritual_texts")
            
            # Generate embedding for test query
            test_embedding = embeddings.embed_query("spiritual wisdom")
            
            # Direct similarity search
            results = collection.query(
                query_embeddings=[test_embedding],
                n_results=3
            )
            print(f"   Method 3 (direct ChromaDB): {len(results['ids'][0])} documents found")
            
            if results['ids'][0]:
                print("   ✅ Direct ChromaDB query successful!")
                print(f"      First result: {results['documents'][0][0][:100]}...")
            
        except Exception as e:
            print(f"   Method 3 failed: {e}")
            
    except Exception as e:
        print(f"❌ Fix attempt failed: {e}")

def main():
    """Main diagnostic function."""
    diagnose_vector_store()
    fix_vector_store_retrieval()

if __name__ == "__main__":
    main()
