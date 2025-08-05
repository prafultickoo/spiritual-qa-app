#!/usr/bin/env python3
"""
Script to investigate Sanskrit text encoding issues in ChromaDB vector store.
Specifically looks for Devanagari/Sanskrit text corruption and encoding problems.
"""

import os
import chromadb
import sqlite3
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def investigate_sanskrit_encoding():
    """Investigate Sanskrit text encoding in vector database."""
    print("🔍 INVESTIGATING SANSKRIT TEXT ENCODING IN VECTOR DATABASE...")
    print("=" * 70)
    
    vector_store_path = os.getenv("VECTOR_STORE_DIR", "./vector_store")
    
    # 1. Direct ChromaDB collection inspection
    print("\n1️⃣ CHROMADB COLLECTION INSPECTION:")
    print("-" * 50)
    
    try:
        client = chromadb.PersistentClient(path=vector_store_path)
        collections = client.list_collections()
        
        print(f"📁 Collections found: {len(collections)}")
        for collection in collections:
            print(f"   Collection: {collection.name}")
            
            # Get some documents to check encoding
            results = collection.get(limit=5)  # Get first 5 documents
            
            print(f"   📄 Documents in collection: {collection.count()}")
            print(f"   🔤 Sample documents for encoding check:")
            
            for i, doc in enumerate(results['documents'][:3]):
                print(f"      Document {i+1} (first 200 chars):")
                print(f"      '{doc[:200]}...'")
                
                # Check for potential Sanskrit/Devanagari characters
                if any(ord(char) > 127 for char in doc):
                    print(f"      ✅ Contains non-ASCII characters (possibly Sanskrit)")
                    # Show some Unicode codepoints
                    unicode_chars = [char for char in doc if ord(char) > 127][:10]
                    print(f"      Unicode chars found: {[f'{char}(U+{ord(char):04X})' for char in unicode_chars]}")
                else:
                    print(f"      ❌ Only ASCII characters found")
                print()
                
    except Exception as e:
        print(f"❌ ChromaDB inspection failed: {e}")
    
    # 2. Direct SQLite database inspection
    print("\n2️⃣ DIRECT SQLITE DATABASE INSPECTION:")
    print("-" * 50)
    
    sqlite_path = os.path.join(vector_store_path, "chroma.sqlite3")
    if os.path.exists(sqlite_path):
        try:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.cursor()
            
            # Get table structure
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"📊 Tables in database: {[table[0] for table in tables]}")
            
            # Look for document content in embeddings table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%document%' OR name LIKE '%embedding%'")
            content_tables = cursor.fetchall()
            
            for table in content_tables:
                table_name = table[0]
                print(f"\n🔍 Examining table: {table_name}")
                
                # Get column info
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                print(f"   Columns: {[col[1] for col in columns]}")
                
                # Look for text content columns
                text_columns = [col[1] for col in columns if 'document' in col[1].lower() or 'content' in col[1].lower() or 'text' in col[1].lower()]
                
                if text_columns:
                    for text_col in text_columns:
                        cursor.execute(f"SELECT {text_col} FROM {table_name} LIMIT 3")
                        sample_texts = cursor.fetchall()
                        
                        print(f"\n   📝 Sample content from {text_col}:")
                        for i, text_row in enumerate(sample_texts):
                            if text_row[0]:
                                content = text_row[0][:300]  # First 300 chars
                                print(f"      Sample {i+1}: '{content}...'")
                                
                                # Check encoding
                                if any(ord(char) > 127 for char in content):
                                    print(f"      ✅ Contains non-ASCII (possibly Sanskrit)")
                                    # Look for specific Sanskrit ranges
                                    devanagari_chars = [char for char in content if 0x0900 <= ord(char) <= 0x097F]
                                    if devanagari_chars:
                                        print(f"      🕉️ DEVANAGARI CHARACTERS FOUND: {devanagari_chars[:10]}")
                                    else:
                                        print(f"      ⚠️ Non-ASCII but not Devanagari - might be corrupted")
                                else:
                                    print(f"      ❌ Only ASCII characters")
                                print()
            
            conn.close()
            
        except Exception as e:
            print(f"❌ SQLite inspection failed: {e}")
    else:
        print(f"❌ SQLite database not found at: {sqlite_path}")
    
    # 3. Test Langchain retrieval encoding
    print("\n3️⃣ LANGCHAIN RETRIEVAL ENCODING TEST:")
    print("-" * 50)
    
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            collection_name="spiritual_texts",
            embedding_function=embeddings,
            persist_directory=vector_store_path
        )
        
        # Try to retrieve documents with Sanskrit-related queries
        test_queries = [
            "Krishna",
            "dharma", 
            "Bhagavad Gita verse",
            "Sanskrit shloka"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = vectorstore.similarity_search(query, k=2)
            
            for i, doc in enumerate(results):
                content = doc.page_content[:200]
                print(f"   Result {i+1}: '{content}...'")
                
                # Check for encoding issues
                if any(ord(char) > 127 for char in content):
                    devanagari_chars = [char for char in content if 0x0900 <= ord(char) <= 0x097F]
                    if devanagari_chars:
                        print(f"   ✅ PROPER DEVANAGARI: {devanagari_chars[:5]}")
                    else:
                        print(f"   ⚠️ CORRUPTED ENCODING - Non-ASCII but not Devanagari")
                        # Show the problematic characters
                        problem_chars = [f'{char}(U+{ord(char):04X})' for char in content if ord(char) > 127][:5]
                        print(f"   Problem chars: {problem_chars}")
                print()
                
    except Exception as e:
        print(f"❌ Langchain retrieval test failed: {e}")

    print("\n" + "=" * 70)
    print("🏁 INVESTIGATION COMPLETE")

if __name__ == "__main__":
    investigate_sanskrit_encoding()
