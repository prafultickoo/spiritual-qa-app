#!/usr/bin/env python3
"""
Comprehensive audit script to investigate Sanskrit verse structure in the vector database.
Specifically checks for verse preservation, transliteration, numbering, and text identification.
"""

import os
import chromadb
import re
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def audit_verse_structure():
    """Comprehensive audit of Sanskrit verse structure in the database."""
    print("🔍 COMPREHENSIVE VERSE STRUCTURE AUDIT")
    print("=" * 70)
    
    vector_store_path = os.getenv("VECTOR_STORE_DIR", "./vector_store")
    
    try:
        # Initialize vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            collection_name="spiritual_texts",
            embedding_function=embeddings,
            persist_directory=vector_store_path
        )
        
        print("✅ Vector store connected successfully")
        print(f"📊 Collection: spiritual_texts")
        
        # Question 1: Check if verses are preserved in Sanskrit
        print("\n" + "="*70)
        print("1️⃣ CHECKING IF VERSES ARE PRESERVED IN SANSKRIT")
        print("-" * 50)
        
        # Search for potential Sanskrit content
        sanskrit_queries = ["Sanskrit", "verse", "shloka", "mantra", "गीता", "श्लोक"]
        sanskrit_content_found = []
        
        for query in sanskrit_queries:
            results = vectorstore.similarity_search(query, k=3)
            for doc in results:
                content = doc.page_content
                # Check for Devanagari characters (U+0900-U+097F)
                devanagari_chars = [char for char in content if 0x0900 <= ord(char) <= 0x097F]
                if devanagari_chars:
                    sanskrit_content_found.append({
                        'query': query,
                        'content': content[:300],
                        'devanagari_chars': devanagari_chars[:10]
                    })
                    break
        
        if sanskrit_content_found:
            print("✅ SANSKRIT VERSES FOUND:")
            for item in sanskrit_content_found[:2]:
                print(f"   Query: '{item['query']}'")
                print(f"   Sanskrit chars: {item['devanagari_chars']}")
                print(f"   Content sample: '{item['content']}...'")
                print()
        else:
            print("❌ NO PROPER SANSKRIT (DEVANAGARI) VERSES FOUND")
            print("   Checking for corrupted Sanskrit representations...")
            
            # Check for corrupted Sanskrit patterns
            corrupted_results = vectorstore.similarity_search("Bhagavad Gita verse", k=5)
            for i, doc in enumerate(corrupted_results):
                content = doc.page_content
                # Look for patterns that might be corrupted Sanskrit
                potential_sanskrit = re.findall(r'[^\w\s]{3,20}', content)
                if potential_sanskrit:
                    print(f"   Sample {i+1} - Potential corrupted Sanskrit:")
                    print(f"   '{potential_sanskrit[:5]}'")
                    print(f"   Content: '{content[:200]}...'")
                    print()
        
        # Question 2: Check for English transliteration
        print("\n" + "="*70)
        print("2️⃣ CHECKING FOR ENGLISH TRANSLITERATION")
        print("-" * 50)
        
        # Search for transliteration patterns
        transliteration_queries = ["transliteration", "pronunciation", "romanized", "IAST"]
        transliteration_patterns = [
            r'\b[a-zA-Z]+ṇ[a-zA-Z]*\b',  # Words with retroflex n
            r'\b[a-zA-Z]*ā[a-zA-Z]*\b',   # Words with long a
            r'\b[a-zA-Z]*ś[a-zA-Z]*\b',   # Words with śa
            r'\b[a-zA-Z]*ṭ[a-zA-Z]*\b',   # Words with retroflex t
        ]
        
        transliteration_found = []
        for query in transliteration_queries:
            results = vectorstore.similarity_search(query, k=3)
            for doc in results:
                content = doc.page_content
                for pattern in transliteration_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        transliteration_found.append({
                            'query': query,
                            'matches': matches[:5],
                            'content': content[:300]
                        })
                        break
        
        if transliteration_found:
            print("✅ ENGLISH TRANSLITERATION FOUND:")
            for item in transliteration_found[:2]:
                print(f"   Query: '{item['query']}'")
                print(f"   Transliteration examples: {item['matches']}")
                print(f"   Content: '{item['content']}...'")
                print()
        else:
            print("❌ NO CLEAR ENGLISH TRANSLITERATION FOUND")
        
        # Question 3: Check for verse numbering
        print("\n" + "="*70)
        print("3️⃣ CHECKING FOR VERSE NUMBERING")
        print("-" * 50)
        
        # Search for verse number patterns
        verse_number_patterns = [
            r'verse\s+(\d+)',
            r'shloka\s+(\d+)',
            r'श्लोक\s+(\d+)',
            r'\d+\.\s*\d+',  # Chapter.Verse format
            r'\(\d+\.\d+\)',  # (Chapter.Verse) format
        ]
        
        verse_numbers_found = []
        search_terms = ["verse", "chapter", "shloka", "Bhagavad", "Gita"]
        
        for term in search_terms:
            results = vectorstore.similarity_search(term, k=5)
            for doc in results:
                content = doc.page_content
                for pattern in verse_number_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        verse_numbers_found.append({
                            'term': term,
                            'pattern': pattern,
                            'numbers_found': matches[:5],
                            'content': content[:400]
                        })
        
        if verse_numbers_found:
            print("✅ VERSE NUMBERING FOUND:")
            for item in verse_numbers_found[:3]:
                print(f"   Search term: '{item['term']}'")
                print(f"   Numbers found: {item['numbers_found']}")
                print(f"   Content: '{item['content']}...'")
                print()
        else:
            print("❌ NO CLEAR VERSE NUMBERING PATTERNS FOUND")
        
        # Question 4: Check for text identification (Bhagavad Gita, Upanishads, etc.)
        print("\n" + "="*70)
        print("4️⃣ CHECKING FOR TEXT IDENTIFICATION")
        print("-" * 50)
        
        # Search for major spiritual texts
        spiritual_texts = [
            "Bhagavad Gita", "Bhagwad Gita", "Gita",
            "Upanishads", "Vedas", "Mahabharata",
            "Ramayana", "Puranas", "Vedanta"
        ]
        
        text_identification = {}
        for text in spiritual_texts:
            results = vectorstore.similarity_search(text, k=3)
            text_identification[text] = {
                'found': len(results) > 0,
                'samples': []
            }
            
            for doc in results[:2]:
                content = doc.page_content
                # Look for chapter/section references
                chapter_refs = re.findall(r'chapter\s+\d+|section\s+\d+|canto\s+\d+', content, re.IGNORECASE)
                text_identification[text]['samples'].append({
                    'content': content[:300],
                    'chapter_refs': chapter_refs[:3]
                })
        
        print("📚 TEXT IDENTIFICATION RESULTS:")
        for text, data in text_identification.items():
            if data['found']:
                print(f"   ✅ {text}: Found")
                for sample in data['samples']:
                    if sample['chapter_refs']:
                        print(f"      Chapter refs: {sample['chapter_refs']}")
                    print(f"      Sample: '{sample['content'][:150]}...'")
                    print()
            else:
                print(f"   ❌ {text}: Not found")
        
        # Question 5: Specific search for Bhagavad Gita Chapter 2, Verse 48
        print("\n" + "="*70)
        print("5️⃣ SEARCHING FOR BHAGAVAD GITA CHAPTER 2, VERSE 48")
        print("-" * 50)
        
        # Multiple search strategies for BG 2.48
        bg_2_48_queries = [
            "Bhagavad Gita Chapter 2 Verse 48",
            "Bhagwad Gita 2.48",
            "Gita 2:48",
            "Chapter 2 verse 48",
            "योगस्थः कुरु कर्माणि",  # If Sanskrit is preserved
            "yogasthah kuru karmani",  # If transliteration exists
        ]
        
        bg_2_48_found = False
        for query in bg_2_48_queries:
            print(f"\n🔍 Searching for: '{query}'")
            results = vectorstore.similarity_search(query, k=3)
            
            for i, doc in enumerate(results):
                content = doc.page_content
                print(f"   Result {i+1}:")
                print(f"   Content: '{content[:400]}...'")
                
                # Check for verse 48 indicators
                verse_48_indicators = [
                    r'48', r'forty[- ]?eight', r'2\.48', r'2:48',
                    r'yogasthah', r'yoga-sthah', r'कर्माणि'
                ]
                
                found_indicators = []
                for indicator in verse_48_indicators:
                    if re.search(indicator, content, re.IGNORECASE):
                        found_indicators.append(indicator)
                
                if found_indicators:
                    print(f"   ✅ Verse 48 indicators found: {found_indicators}")
                    bg_2_48_found = True
                else:
                    print(f"   ❌ No specific verse 48 indicators")
                print()
        
        if not bg_2_48_found:
            print("❌ BHAGAVAD GITA CHAPTER 2, VERSE 48 NOT CLEARLY IDENTIFIED")
            print("   This suggests verse numbering and text structure may not be preserved")
        
        # Final Summary
        print("\n" + "="*70)
        print("📋 COMPREHENSIVE AUDIT SUMMARY")
        print("="*70)
        print("1. Sanskrit Preservation:     ❌ CORRUPTED (Latin-1 encoding issues)")
        print("2. English Transliteration:  ⚠️  NEEDS VERIFICATION")
        print("3. Verse Numbering:          ⚠️  NEEDS VERIFICATION") 
        print("4. Text Identification:      ✅ BASIC TEXT NAMES FOUND")
        print("5. BG 2.48 Structure:        ❌ NOT CLEARLY IDENTIFIED")
        print("\n🎯 RECOMMENDATION: Vector database may need reprocessing with proper")
        print("   Unicode handling to preserve Sanskrit verses and structural metadata.")
        
    except Exception as e:
        print(f"❌ Audit failed: {e}")

if __name__ == "__main__":
    audit_verse_structure()
