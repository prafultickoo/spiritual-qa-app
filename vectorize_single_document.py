#!/usr/bin/env python3
"""
Single Document Vectorization Script for Clean Verses
Vectorizes ONLY the new Bhagavad Gita document into a separate collection.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz  # PyMuPDF for better text extraction
import re
import unicodedata

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("single_document_vectorization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SingleDocumentVectorizer:
    """Vectorize a single document with enhanced verse detection."""
    
    def __init__(self, 
                 vector_store_dir: str = "./vector_store",
                 collection_name: str = "clean_verses",
                 embedding_model: str = "openai"):
        """
        Initialize the single document vectorizer.
        
        Args:
            vector_store_dir: Directory for vector storage
            collection_name: Name for the new collection
            embedding_model: Embedding model to use
        """
        self.vector_store_dir = vector_store_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize embeddings - MUST match existing database exactly
        if embedding_model == "openai":
            # Use exact same model as existing database (1536 dimensions)
            self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings()
        
        logger.info(f"Initialized vectorizer for collection: {collection_name}")
    
    def extract_document_with_verses(self, pdf_path: str) -> List[Document]:
        """
        Extract document content with enhanced verse detection.
        
        Args:
            pdf_path: Path to the PDF document
            
        Returns:
            List of Document objects with verse metadata
        """
        logger.info(f"Extracting content from: {pdf_path}")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        documents = []
        
        try:
            doc = fitz.open(str(pdf_path))
            logger.info(f"Processing {len(doc)} pages...")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Unicode normalization
                text = unicodedata.normalize('NFC', text)
                
                if not text.strip():
                    continue
                
                # Enhanced verse detection
                verse_info = self._detect_verse_structure(text, page_num + 1)
                
                # Create document with metadata (filter None values for ChromaDB)
                base_metadata = {
                    'source': str(pdf_path),
                    'page': page_num + 1,
                    'filename': pdf_path.name,
                    'collection_type': 'clean_verses',
                    'loader': 'pymupdf',
                    'unicode_normalized': True
                }
                
                # Add verse_info, filtering None values
                for key, value in verse_info.items():
                    if value is not None:
                        if isinstance(value, list):
                            # Convert lists to strings for ChromaDB
                            base_metadata[key] = str(value) if value else "[]"
                        else:
                            base_metadata[key] = value
                
                metadata = base_metadata
                
                document = Document(
                    page_content=text,
                    metadata=metadata
                )
                documents.append(document)
            
            doc.close()
            logger.info(f"Extracted {len(documents)} pages successfully")
            
        except Exception as e:
            logger.error(f"Error extracting document: {str(e)}")
            raise
        
        return documents
    
    def _detect_verse_structure(self, text: str, page_num: int) -> Dict[str, Any]:
        """
        Detect verse structure and metadata in text.
        
        Args:
            text: Page text content
            page_num: Page number
            
        Returns:
            Dictionary with verse metadata
        """
        verse_info = {
            'has_verses': False,
            'chapter_number': None,
            'verse_numbers': [],
            'contains_sanskrit': False,
            'contains_transliteration': False,
            'verse_count': 0
        }
        
        # Chapter detection
        chapter_patterns = [
            r'chapter\s+(\d+)',
            r'adhyaya\s+(\d+)',
            r'अध्याय\s+(\d+)'
        ]
        
        for pattern in chapter_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                verse_info['chapter_number'] = int(match.group(1))
                break
        
        # Verse detection
        verse_patterns = [
            r'verse\s+(\d+)',
            r'श्लोक\s+(\d+)',
            r'(\d+)\.(\d+)',  # 2.47 format
            r'॥(\d+)॥'  # Sanskrit verse numbering
        ]
        
        verse_numbers = []
        for pattern in verse_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 2:  # Chapter.Verse format
                    verse_numbers.append(f"{match.group(1)}.{match.group(2)}")
                else:
                    verse_numbers.append(match.group(1))
        
        if verse_numbers:
            verse_info['has_verses'] = True
            verse_info['verse_numbers'] = list(set(verse_numbers))
            verse_info['verse_count'] = len(verse_info['verse_numbers'])
        
        # Sanskrit detection (Devanagari)
        sanskrit_pattern = r'[कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह]+'
        if re.search(sanskrit_pattern, text):
            verse_info['contains_sanskrit'] = True
        
        # Transliteration detection
        transliteration_indicators = [
            'karma', 'dharma', 'yoga', 'moksha', 'samsara', 'arjuna', 'krishna',
            'bhagavad', 'gita', 'upanishad', 'vedanta', 'brahman', 'atman'
        ]
        
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in transliteration_indicators):
            verse_info['contains_transliteration'] = True
        
        return verse_info
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents with verse-aware splitting.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        logger.info("Chunking documents with verse-aware splitting...")
        
        # Use smaller chunks for verse content to maintain precision
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for verses
            chunk_overlap=100,
            separators=["\n\n", "\nVERSE", "\n", " ", ""]
        )
        
        chunked_docs = []
        
        for doc in documents:
            # Check if this document contains verses
            has_verses = doc.metadata.get('has_verses', False)
            
            if has_verses:
                # For verse content, use more careful chunking
                chunks = text_splitter.split_documents([doc])
                
                # Add chunk-specific metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata['chunk_id'] = f"{doc.metadata.get('page', 0)}_{i}"
                    chunk.metadata['chunk_type'] = 'verse_content'
                    chunk.metadata['parent_page'] = doc.metadata.get('page')
                    
                    # Re-detect verse info for each chunk
                    chunk_verse_info = self._detect_verse_structure(chunk.page_content, doc.metadata.get('page', 0))
                    
                    # Filter None values before updating metadata
                    for key, value in chunk_verse_info.items():
                        if value is not None:
                            if isinstance(value, list):
                                chunk.metadata[key] = str(value) if value else "[]"
                            else:
                                chunk.metadata[key] = value
                    
                    chunked_docs.append(chunk)
            else:
                # For non-verse content, use standard chunking
                chunks = text_splitter.split_documents([doc])
                for i, chunk in enumerate(chunks):
                    chunk.metadata['chunk_id'] = f"{doc.metadata.get('page', 0)}_{i}"
                    chunk.metadata['chunk_type'] = 'general_content'
                    chunk.metadata['parent_page'] = doc.metadata.get('page')
                    
                    # Ensure no None values in metadata
                    filtered_metadata = {}
                    for key, value in chunk.metadata.items():
                        if value is not None:
                            if isinstance(value, list):
                                filtered_metadata[key] = str(value) if value else "[]"
                            else:
                                filtered_metadata[key] = value
                    chunk.metadata = filtered_metadata
                    
                    chunked_docs.append(chunk)
        
        logger.info(f"Created {len(chunked_docs)} chunks")
        return chunked_docs
    
    def vectorize_and_store(self, chunks: List[Document]) -> bool:
        """
        Vectorize chunks and store in separate collection.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Success status
        """
        logger.info(f"Vectorizing {len(chunks)} chunks into collection: {self.collection_name}")
        
        try:
            # Create vector store with specific collection name
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.vector_store_dir
            )
            
            # Add documents in batches to avoid memory issues
            batch_size = 50
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
                # Extract texts and metadatas
                texts = [chunk.page_content for chunk in batch]
                metadatas = [chunk.metadata for chunk in batch]
                
                # Add to vector store
                vectorstore.add_texts(texts=texts, metadatas=metadatas)
            
            # Persist the vector store
            vectorstore.persist()
            
            logger.info(f"Successfully vectorized and stored {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error during vectorization: {str(e)}")
            return False
    
    def vectorize_document(self, pdf_path: str) -> bool:
        """
        Complete vectorization workflow for a single document.
        
        Args:
            pdf_path: Path to the PDF document
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Starting vectorization of: {pdf_path}")
            
            # Step 1: Extract document
            documents = self.extract_document_with_verses(pdf_path)
            
            # Step 2: Chunk documents
            chunks = self.chunk_documents(documents)
            
            # Step 3: Vectorize and store
            success = self.vectorize_and_store(chunks)
            
            if success:
                logger.info("✅ Single document vectorization completed successfully!")
                
                # Log statistics
                verse_chunks = [c for c in chunks if c.metadata.get('has_verses', False)]
                sanskrit_chunks = [c for c in chunks if c.metadata.get('contains_sanskrit', False)]
                
                logger.info(f"📊 Statistics:")
                logger.info(f"   Total chunks: {len(chunks)}")
                logger.info(f"   Verse chunks: {len(verse_chunks)}")
                logger.info(f"   Sanskrit chunks: {len(sanskrit_chunks)}")
                logger.info(f"   Collection: {self.collection_name}")
                
                return True
            else:
                logger.error("❌ Single document vectorization failed!")
                return False
                
        except Exception as e:
            logger.error(f"Error in vectorization workflow: {str(e)}")
            return False

def main():
    """Main function to run single document vectorization."""
    parser = argparse.ArgumentParser(description="Vectorize a single document for clean verses")
    parser.add_argument("--document", "-d", 
                       default="documents/New_Bhagwad_Gita.pdf",
                       help="Path to the document to vectorize")
    parser.add_argument("--collection", "-c",
                       default="clean_verses", 
                       help="Collection name for the vectorized content")
    parser.add_argument("--vector-dir", "-v",
                       default="./vector_store",
                       help="Vector store directory")
    parser.add_argument("--embedding-model", "-e",
                       default="openai",
                       choices=["openai", "huggingface"],
                       help="Embedding model to use")
    
    args = parser.parse_args()
    
    # Create vectorizer
    vectorizer = SingleDocumentVectorizer(
        vector_store_dir=args.vector_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model
    )
    
    # Vectorize the document
    success = vectorizer.vectorize_document(args.document)
    
    if success:
        print("✅ Single document vectorization completed successfully!")
        sys.exit(0)
    else:
        print("❌ Single document vectorization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
