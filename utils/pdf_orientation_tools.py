"""
Utilities for detecting and correcting PDF orientation/rotation issues.
"""
import os
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path

# PDF processing libraries
from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader, PdfWriter
import io
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
ROTATION_ANGLES = [0, 90, 180, 270]  # Possible rotation angles to check
MIN_TEXT_CONFIDENCE = 40  # Minimum OCR confidence to consider text valid
MIN_EXPECTED_WORDS = 5   # Minimum expected words per page for a valid orientation


def analyze_pdf_orientation(pdf_path: str) -> Dict[str, Any]:
    """
    Analyzes a PDF to detect if it has orientation/rotation issues.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Dict[str, Any]: Analysis results with rotation data
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return {
            "status": "error",
            "message": "File not found",
            "needs_correction": False
        }
    
    try:
        # First check if we can extract text directly (faster method)
        reader = PdfReader(pdf_path)
        
        # Analysis results
        results = {
            "status": "success",
            "filename": os.path.basename(pdf_path),
            "total_pages": len(reader.pages),
            "pages_with_issues": [],
            "needs_correction": False,
            "correction_method": None
        }
        
        # Check a sample of pages (first, middle, and last page if applicable)
        pages_to_check = get_page_samples(len(reader.pages))
        
        for page_num in pages_to_check:
            page = reader.pages[page_num]
            extracted_text = page.extract_text()
            
            # Check if text extraction looks problematic
            if is_text_extraction_problematic(extracted_text):
                logger.info(f"Page {page_num+1} in {os.path.basename(pdf_path)} might have orientation issues")
                results["pages_with_issues"].append({
                    "page_number": page_num + 1,
                    "issue_type": "possible_orientation",
                    "correction_needed": True
                })
                results["needs_correction"] = True
        
        # If issues found, determine if we need OCR or simple rotation
        if results["needs_correction"]:
            # Check if PDF has extractable text or needs OCR
            if has_extractable_text(reader):
                results["correction_method"] = "rotation"
            else:
                results["correction_method"] = "ocr"
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing PDF orientation for {pdf_path}: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "needs_correction": False
        }


def is_text_extraction_problematic(text: str) -> bool:
    """
    Checks if extracted text appears to have orientation/rotation issues.
    
    Args:
        text (str): Extracted text to analyze
        
    Returns:
        bool: True if text appears problematic
    """
    if not text or len(text.strip()) < 20:
        return True
    
    lines = text.strip().split('\n')
    
    # Check 1: Unusually short lines could indicate vertical text extraction
    avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
    if avg_line_length < 15 and len(lines) > 5:
        return True
    
    # Check 2: Strange character patterns
    strange_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if strange_char_count / max(len(text), 1) > 0.3:
        return True
    
    # Check 3: Few sentences
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    if sentence_count < 2 and len(text) > 200:
        return True
    
    return False


def has_extractable_text(pdf_reader: PdfReader) -> bool:
    """
    Checks if the PDF has properly extractable text or needs OCR.
    
    Args:
        pdf_reader (PdfReader): PyPDF2 reader object
        
    Returns:
        bool: True if PDF has extractable text
    """
    # Sample a few pages
    pages_to_check = get_page_samples(len(pdf_reader.pages))
    
    for page_num in pages_to_check:
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        
        # If we got a reasonable amount of text, consider it extractable
        if text and len(text.strip()) > 100:
            return True
    
    return False


def get_page_samples(total_pages: int) -> List[int]:
    """
    Get a sample of page numbers to check (first, middle, and last if applicable)
    
    Args:
        total_pages (int): Total pages in the document
        
    Returns:
        List[int]: List of page numbers to check
    """
    if total_pages <= 3:
        return list(range(total_pages))
    else:
        return [0, total_pages // 2, total_pages - 1]


def detect_best_orientation(image: Image.Image) -> Tuple[int, float]:
    """
    Detects the best orientation for a page image using OCR.
    
    Args:
        image (Image.Image): Page image to analyze
        
    Returns:
        Tuple[int, float]: Best rotation angle and confidence score
    """
    best_angle = 0
    best_confidence = 0
    best_text = ""
    
    for angle in ROTATION_ANGLES:
        # Rotate image
        rotated_img = image.rotate(angle, expand=True)
        
        # Run OCR with orientation detection
        ocr_data = pytesseract.image_to_data(rotated_img, output_type=pytesseract.Output.DICT)
        
        # Calculate confidence for this orientation
        confidences = [float(conf) for conf in ocr_data['conf'] if conf != '-1']
        words = [word for word in ocr_data['text'] if word.strip()]
        
        if not confidences or not words:
            continue
            
        avg_confidence = sum(confidences) / len(confidences)
        
        # If this orientation gives better results, update best values
        if (len(words) > len(best_text.split()) or 
           (len(words) == len(best_text.split()) and avg_confidence > best_confidence)):
            best_angle = angle
            best_confidence = avg_confidence
            best_text = " ".join(words)
    
    return best_angle, best_confidence


def correct_pdf_orientation(pdf_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Corrects orientation issues in a PDF, saving the corrected version.
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_path (str, optional): Path for corrected PDF output
        
    Returns:
        Dict[str, Any]: Correction results
    """
    if not output_path:
        # Create output path if not specified
        pdf_dir = os.path.dirname(pdf_path)
        pdf_name = os.path.basename(pdf_path)
        output_path = os.path.join(pdf_dir, f"corrected_{pdf_name}")
    
    try:
        # First analyze the PDF
        analysis = analyze_pdf_orientation(pdf_path)
        
        if not analysis["needs_correction"]:
            logger.info(f"No orientation correction needed for {pdf_path}")
            return {
                "status": "success", 
                "message": "No orientation correction needed",
                "output_path": pdf_path
            }
        
        # Choose correction method based on analysis
        if analysis["correction_method"] == "ocr":
            return apply_ocr_with_orientation_correction(pdf_path, output_path)
        else:
            return apply_rotation_correction(pdf_path, output_path, analysis["pages_with_issues"])
        
    except Exception as e:
        logger.error(f"Error correcting PDF orientation for {pdf_path}: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "output_path": None
        }


def apply_rotation_correction(pdf_path: str, output_path: str, 
                             pages_with_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Applies rotation correction to pages with issues.
    
    Args:
        pdf_path (str): Path to the input PDF
        output_path (str): Path for the corrected output PDF
        pages_with_issues (List[Dict]): List of pages with detected issues
        
    Returns:
        Dict[str, Any]: Correction results
    """
    try:
        # Load the PDF
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        
        issue_page_numbers = [p["page_number"] - 1 for p in pages_with_issues]
        
        # Process each page
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            
            if i in issue_page_numbers:
                # Convert page to image to detect orientation
                page_images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
                if not page_images:
                    writer.add_page(page)
                    continue
                
                # Detect best orientation
                best_angle, confidence = detect_best_orientation(page_images[0])
                
                # Only rotate if we're confident
                if confidence > MIN_TEXT_CONFIDENCE:
                    # Apply rotation
                    if best_angle == 90:
                        page.rotate(270)  # PyPDF2 rotates counterclockwise
                    elif best_angle == 180:
                        page.rotate(180)
                    elif best_angle == 270:
                        page.rotate(90)
                    
                    logger.info(f"Rotated page {i+1} by {best_angle} degrees with confidence {confidence}")
            
            writer.add_page(page)
        
        # Save the corrected PDF
        with open(output_path, "wb") as out_file:
            writer.write(out_file)
        
        return {
            "status": "success",
            "message": f"Corrected orientation issues in {len(issue_page_numbers)} pages",
            "output_path": output_path,
            "pages_corrected": len(issue_page_numbers)
        }
        
    except Exception as e:
        logger.error(f"Error applying rotation correction: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "output_path": None
        }


def apply_ocr_with_orientation_correction(pdf_path: str, output_path: str) -> Dict[str, Any]:
    """
    Applies OCR with orientation correction for PDFs with text extraction issues.
    
    Args:
        pdf_path (str): Path to the input PDF
        output_path (str): Path for the corrected output PDF
        
    Returns:
        Dict[str, Any]: OCR results
    """
    try:
        # Convert PDF to images
        pdf_images = convert_from_path(pdf_path)
        
        if not pdf_images:
            return {
                "status": "error",
                "message": "Failed to convert PDF to images",
                "output_path": None
            }
        
        # Create a new PDF with corrected pages
        corrected_pdf = io.BytesIO()
        first_page = True
        
        for i, img in enumerate(pdf_images):
            # Detect best orientation
            best_angle, confidence = detect_best_orientation(img)
            
            # Rotate if needed
            if best_angle != 0:
                img = img.rotate(best_angle, expand=True)
            
            # Convert to PDF page
            if first_page:
                img.save(corrected_pdf, 'PDF')
                first_page = False
            else:
                img_pdf = io.BytesIO()
                img.save(img_pdf, 'PDF')
                img_pdf.seek(0)
                
                # Append to existing PDF
                existing_pdf = PdfReader(corrected_pdf)
                new_page_pdf = PdfReader(img_pdf)
                writer = PdfWriter()
                
                for page in existing_pdf.pages:
                    writer.add_page(page)
                writer.add_page(new_page_pdf.pages[0])
                
                # Save combined PDF
                corrected_pdf = io.BytesIO()
                writer.write(corrected_pdf)
        
        # Save final PDF
        corrected_pdf.seek(0)
        with open(output_path, 'wb') as f:
            f.write(corrected_pdf.getbuffer())
        
        return {
            "status": "success", 
            "message": "Applied OCR with orientation correction", 
            "output_path": output_path,
            "pages_corrected": len(pdf_images)
        }
        
    except Exception as e:
        logger.error(f"Error applying OCR with orientation correction: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "output_path": None
        }
