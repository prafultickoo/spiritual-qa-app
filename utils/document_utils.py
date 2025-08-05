"""
Document utility functions for loading and processing documents.
"""
import os
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
import PyPDF2
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_pdf_document(file_path: str) -> str:
    """
    Load a PDF document and extract its text content.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        logger.info(f"Loading PDF document: {file_path}")
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
            logger.info(f"Successfully extracted text from {file_path}")
            return text
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {str(e)}")
        return ""

def load_documents_from_directory(directory_path: str) -> Dict[str, str]:
    """
    Load all PDF documents from a directory and extract their text content.
    
    Args:
        directory_path (str): Path to the directory containing PDF documents
        
    Returns:
        Dict[str, str]: Dictionary mapping filename to document content
    """
    documents = {}
    try:
        logger.info(f"Loading documents from directory: {directory_path}")
        
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            file_path = os.path.join(directory_path, pdf_file)
            content = load_pdf_document(file_path)
            
            if content:
                documents[pdf_file] = content
        
        return documents
    except Exception as e:
        logger.error(f"Error loading documents from directory {directory_path}: {str(e)}")
        return {}

def preserve_verse_structure(text: str) -> str:
    """
    Preserve verse structure in the extracted text.
    This is important for Sanskrit verses and their translations.
    
    Args:
        text (str): Original text
        
    Returns:
        str: Text with preserved verse structure
    """
    # Simple implementation - maintain line breaks and spacing
    # For more complex cases, this function can be expanded
    return text

def get_document_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF document.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        Dict[str, Any]: Dictionary containing metadata
    """
    try:
        logger.info(f"Extracting metadata from: {file_path}")
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = pdf_reader.metadata
            
            result = {
                "filename": os.path.basename(file_path),
                "pages": len(pdf_reader.pages),
                "title": metadata.get('/Title', ''),
                "author": metadata.get('/Author', ''),
                "creation_date": metadata.get('/CreationDate', ''),
            }
            
            return result
    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
        return {
            "filename": os.path.basename(file_path),
            "error": str(e)
        }
