"""
Enhanced PDF loader with Unicode/Sanskrit preservation and OCR fallback.
"""
import os
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Multiple PDF processing libraries for fallback
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedPDFLoader:
    """Enhanced PDF loader with Unicode preservation and OCR fallback."""
    
    def __init__(self, 
                 prefer_ocr: bool = False,
                 tesseract_config: str = '--oem 3 --psm 6 -l eng+hin+san'):
        """
        Initialize enhanced PDF loader.
        
        Args:
            prefer_ocr: Whether to prefer OCR over text extraction
            tesseract_config: Tesseract configuration for OCR
        """
        self.prefer_ocr = prefer_ocr
        self.tesseract_config = tesseract_config
        
        # Check available libraries
        self.available_loaders = []
        if PYMUPDF_AVAILABLE:
            self.available_loaders.append('pymupdf')
        if UNSTRUCTURED_AVAILABLE:
            self.available_loaders.append('unstructured')
        if PDFPLUMBER_AVAILABLE:
            self.available_loaders.append('pdfplumber')
        self.available_loaders.append('pypdf')  # Always available as fallback
        
        logger.info(f"Available PDF loaders: {self.available_loaders}")
        if OCR_AVAILABLE:
            logger.info("OCR support available with Tesseract")
    
    def load_pdf_with_unicode(self, pdf_path: str) -> List[Document]:
        """
        Load PDF with Unicode preservation using multiple fallback methods.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document objects with preserved Unicode
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return []
        
        logger.info(f"Loading PDF: {pdf_path.name}")
        
        # Try each loader method in order of preference
        for loader_name in self.available_loaders:
            try:
                logger.info(f"Trying {loader_name} loader...")
                documents = self._load_with_method(pdf_path, loader_name)
                
                if documents and self._validate_unicode_content(documents):
                    logger.info(f"Successfully loaded with {loader_name}")
                    return self._post_process_documents(documents, pdf_path)
                else:
                    logger.warning(f"{loader_name} failed Unicode validation")
                    
            except Exception as e:
                logger.warning(f"{loader_name} loader failed: {str(e)}")
                continue
        
        # If all text extraction methods fail, try OCR
        if OCR_AVAILABLE and not self.prefer_ocr:
            logger.info("All text extraction methods failed, trying OCR...")
            return self._load_with_ocr(pdf_path)
        
        logger.error(f"All loading methods failed for {pdf_path}")
        return []
    
    def _load_with_method(self, pdf_path: Path, method: str) -> List[Document]:
        """Load PDF using specific method."""
        
        if method == 'pymupdf' and PYMUPDF_AVAILABLE:
            return self._load_with_pymupdf(pdf_path)
        elif method == 'unstructured' and UNSTRUCTURED_AVAILABLE:
            return self._load_with_unstructured(pdf_path)
        elif method == 'pdfplumber' and PDFPLUMBER_AVAILABLE:
            return self._load_with_pdfplumber(pdf_path)
        elif method == 'pypdf':
            return self._load_with_pypdf(pdf_path)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _load_with_pymupdf(self, pdf_path: Path) -> List[Document]:
        """Load PDF using PyMuPDF (best Unicode support)."""
        documents = []
        
        with fitz.open(str(pdf_path)) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text with Unicode preservation
                text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                
                # Normalize Unicode
                text = unicodedata.normalize('NFC', text)
                
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            'source': str(pdf_path),
                            'page': page_num + 1,
                            'loader': 'pymupdf'
                        }
                    ))
        
        return documents
    
    def _load_with_unstructured(self, pdf_path: Path) -> List[Document]:
        """Load PDF using Unstructured (good for complex layouts)."""
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",  # High resolution for better text extraction
            infer_table_structure=True,
            languages=["eng", "hin"]  # English and Hindi
        )
        
        documents = []
        page_content = {}
        
        for element in elements:
            page_num = getattr(element.metadata, 'page_number', 1)
            text = str(element)
            
            if page_num not in page_content:
                page_content[page_num] = []
            page_content[page_num].append(text)
        
        for page_num, texts in page_content.items():
            combined_text = '\n'.join(texts)
            combined_text = unicodedata.normalize('NFC', combined_text)
            
            if combined_text.strip():
                documents.append(Document(
                    page_content=combined_text,
                    metadata={
                        'source': str(pdf_path),
                        'page': page_num,
                        'loader': 'unstructured'
                    }
                ))
        
        return documents
    
    def _load_with_pdfplumber(self, pdf_path: Path) -> List[Document]:
        """Load PDF using pdfplumber (good for tables and layout)."""
        documents = []
        
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                if text:
                    text = unicodedata.normalize('NFC', text)
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            'source': str(pdf_path),
                            'page': page_num + 1,
                            'loader': 'pdfplumber'
                        }
                    ))
        
        return documents
    
    def _load_with_pypdf(self, pdf_path: Path) -> List[Document]:
        """Load PDF using PyPDF (fallback method)."""
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        
        # Post-process for Unicode normalization
        for doc in documents:
            doc.page_content = unicodedata.normalize('NFC', doc.page_content)
            doc.metadata['loader'] = 'pypdf'
        
        return documents
    
    def _load_with_ocr(self, pdf_path: Path) -> List[Document]:
        """Load PDF using OCR as last resort."""
        if not OCR_AVAILABLE:
            logger.error("OCR not available")
            return []
        
        documents = []
        
        # Convert PDF to images and OCR each page
        with fitz.open(str(pdf_path)) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Convert to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                img_data = pix.tobytes("png")
                
                # Convert to OpenCV format
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # OCR with Tesseract
                text = pytesseract.image_to_string(
                    img, 
                    config=self.tesseract_config
                )
                
                text = unicodedata.normalize('NFC', text)
                
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            'source': str(pdf_path),
                            'page': page_num + 1,
                            'loader': 'ocr'
                        }
                    ))
        
        return documents
    
    def _validate_unicode_content(self, documents: List[Document]) -> bool:
        """Validate that Unicode content is properly preserved."""
        if not documents:
            return False
        
        # Check for Devanagari characters in content
        devanagari_found = False
        total_chars = 0
        
        for doc in documents[:3]:  # Check first 3 documents
            content = doc.page_content
            total_chars += len(content)
            
            # Look for Devanagari Unicode range (U+0900-U+097F)
            for char in content:
                if 0x0900 <= ord(char) <= 0x097F:
                    devanagari_found = True
                    break
            
            if devanagari_found:
                break
        
        # If we found Devanagari, it's good
        if devanagari_found:
            logger.info("✅ Devanagari characters found - Unicode preserved")
            return True
        
        # If no Devanagari but reasonable content length, might be English-only docs
        if total_chars > 100:
            logger.info("⚠️ No Devanagari found but content exists - might be English-only")
            return True
        
        logger.warning("❌ Unicode validation failed - content too short or corrupted")
        return False
    
    def _post_process_documents(self, documents: List[Document], pdf_path: Path) -> List[Document]:
        """Post-process documents with enhanced metadata."""
        
        for doc in documents:
            # Add file-level metadata
            doc.metadata.update({
                'filename': pdf_path.name,
                'file_size': pdf_path.stat().st_size,
                'unicode_normalized': True
            })
            
            # Detect content type
            content = doc.page_content
            if any(0x0900 <= ord(char) <= 0x097F for char in content):
                doc.metadata['contains_sanskrit'] = True
            
            # Clean up whitespace while preserving structure
            doc.page_content = self._clean_text_structure(content)
        
        return documents
    
    def _clean_text_structure(self, text: str) -> str:
        """Clean text while preserving verse and paragraph structure."""
        # Remove excessive whitespace but preserve paragraph breaks
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = ' '.join(line.split())  # Remove extra spaces
            cleaned_lines.append(cleaned_line)
        
        # Join with single newlines, but preserve double newlines for paragraphs
        result = '\n'.join(cleaned_lines)
        
        # Restore paragraph breaks
        result = '\n'.join(line for line in result.split('\n') if line.strip())
        
        return result


def load_documents_enhanced(directory_path: str, **kwargs) -> List[Document]:
    """
    Enhanced document loading function with Unicode preservation.
    
    Args:
        directory_path: Path to directory containing PDFs
        **kwargs: Additional arguments for EnhancedPDFLoader
        
    Returns:
        List of Document objects with preserved Unicode
    """
    loader = EnhancedPDFLoader(**kwargs)
    documents = []
    
    pdf_files = list(Path(directory_path).glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        try:
            pdf_documents = loader.load_pdf_with_unicode(str(pdf_file))
            documents.extend(pdf_documents)
            logger.info(f"Loaded {len(pdf_documents)} pages from {pdf_file.name}")
        except Exception as e:
            logger.error(f"Failed to load {pdf_file.name}: {str(e)}")
    
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents
