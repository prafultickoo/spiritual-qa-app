#!/usr/bin/env python3
"""
Diagnostic script to analyze PDF content and determine why Sanskrit is not being preserved.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_pdf_content(pdf_directory: str) -> Dict[str, Any]:
    """
    Diagnose PDF content to understand Sanskrit preservation issues.
    
    Args:
        pdf_directory: Directory containing PDF files
        
    Returns:
        Diagnostic results
    """
    results = {
        'total_pdfs': 0,
        'pdfs_analyzed': 0,
        'text_extraction_methods': {},
        'sanskrit_found': False,
        'sample_content': [],
        'recommendations': []
    }
    
    pdf_files = list(Path(pdf_directory).glob("**/*.pdf"))
    results['total_pdfs'] = len(pdf_files)
    
    logger.info(f"Found {len(pdf_files)} PDF files to analyze")
    
    # Test different extraction methods on first few files
    test_files = pdf_files[:3]  # Test first 3 files
    
    for i, pdf_file in enumerate(test_files):
        logger.info(f"\n📖 Analyzing: {pdf_file.name}")
        
        file_results = {
            'filename': pdf_file.name,
            'methods_tested': {},
            'sanskrit_detected': False,
            'text_samples': []
        }
        
        # Method 1: PyMuPDF
        try:
            import fitz
            logger.info("  🔍 Testing PyMuPDF extraction...")
            
            with fitz.open(str(pdf_file)) as doc:
                # Test first 2 pages
                sample_text = ""
                for page_num in range(min(2, len(doc))):
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    sample_text += text[:500]  # First 500 chars
                
                # Check for Sanskrit
                sanskrit_chars = [char for char in sample_text if 0x0900 <= ord(char) <= 0x097F]
                has_sanskrit = len(sanskrit_chars) > 0
                
                file_results['methods_tested']['pymupdf'] = {
                    'success': True,
                    'text_length': len(sample_text),
                    'sanskrit_chars_found': len(sanskrit_chars),
                    'has_sanskrit': has_sanskrit,
                    'sample_text': sample_text[:200] if sample_text else 'No text extracted'
                }
                
                if has_sanskrit:
                    file_results['sanskrit_detected'] = True
                    results['sanskrit_found'] = True
                    logger.info(f"    ✅ Sanskrit found: {''.join(sanskrit_chars[:10])}")
                
        except Exception as e:
            file_results['methods_tested']['pymupdf'] = {'success': False, 'error': str(e)}
            logger.error(f"    ❌ PyMuPDF failed: {e}")
        
        # Method 2: pdfplumber
        try:
            import pdfplumber
            logger.info("  🔍 Testing pdfplumber extraction...")
            
            with pdfplumber.open(str(pdf_file)) as pdf:
                sample_text = ""
                for page_num in range(min(2, len(pdf.pages))):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text:
                        sample_text += text[:500]
                
                # Check for Sanskrit
                sanskrit_chars = [char for char in sample_text if 0x0900 <= ord(char) <= 0x097F]
                has_sanskrit = len(sanskrit_chars) > 0
                
                file_results['methods_tested']['pdfplumber'] = {
                    'success': True,
                    'text_length': len(sample_text),
                    'sanskrit_chars_found': len(sanskrit_chars),
                    'has_sanskrit': has_sanskrit,
                    'sample_text': sample_text[:200] if sample_text else 'No text extracted'
                }
                
                if has_sanskrit:
                    file_results['sanskrit_detected'] = True
                    results['sanskrit_found'] = True
                    logger.info(f"    ✅ Sanskrit found: {''.join(sanskrit_chars[:10])}")
                
        except Exception as e:
            file_results['methods_tested']['pdfplumber'] = {'success': False, 'error': str(e)}
            logger.error(f"    ❌ pdfplumber failed: {e}")
        
        # Method 3: Check if PDF contains images (might need OCR)
        try:
            import fitz
            logger.info("  🔍 Checking for images (OCR candidates)...")
            
            with fitz.open(str(pdf_file)) as doc:
                total_images = 0
                for page_num in range(min(2, len(doc))):
                    page = doc.load_page(page_num)
                    image_list = page.get_images()
                    total_images += len(image_list)
                
                file_results['methods_tested']['image_check'] = {
                    'success': True,
                    'images_found': total_images,
                    'likely_needs_ocr': total_images > 0 and not file_results['sanskrit_detected']
                }
                
                if total_images > 0:
                    logger.info(f"    📷 Found {total_images} images - might need OCR")
                
        except Exception as e:
            file_results['methods_tested']['image_check'] = {'success': False, 'error': str(e)}
        
        results['sample_content'].append(file_results)
        results['pdfs_analyzed'] += 1
    
    # Generate recommendations
    if not results['sanskrit_found']:
        logger.warning("⚠️ No Sanskrit found in any PDF - analyzing why...")
        
        # Check if any text was extracted at all
        has_text = any(
            method.get('text_length', 0) > 100 
            for file_data in results['sample_content'] 
            for method in file_data['methods_tested'].values()
        )
        
        # Check if images were found
        has_images = any(
            method.get('images_found', 0) > 0 
            for file_data in results['sample_content'] 
            for method in file_data['methods_tested'].values()
        )
        
        if not has_text and has_images:
            results['recommendations'].append("📷 PDFs contain scanned images - OCR processing required")
        elif has_text and not results['sanskrit_found']:
            results['recommendations'].append("📝 Text extracted but no Sanskrit - check if PDFs actually contain Sanskrit")
        else:
            results['recommendations'].append("❓ Unknown issue - manual inspection of PDF files recommended")
    
    return results

def print_diagnostic_results(results: Dict[str, Any]):
    """Print diagnostic results in a readable format."""
    print("\n" + "="*80)
    print("🔍 PDF CONTENT DIAGNOSTIC RESULTS")
    print("="*80)
    
    print(f"Total PDFs Found: {results['total_pdfs']}")
    print(f"PDFs Analyzed: {results['pdfs_analyzed']}")
    print(f"Sanskrit Found: {'✅ YES' if results['sanskrit_found'] else '❌ NO'}")
    
    print(f"\n📊 ANALYSIS BY FILE:")
    for file_data in results['sample_content']:
        print(f"\n📄 {file_data['filename']}")
        print(f"   Sanskrit Detected: {'✅ YES' if file_data['sanskrit_detected'] else '❌ NO'}")
        
        for method_name, method_data in file_data['methods_tested'].items():
            if method_data['success']:
                print(f"   {method_name.upper()}:")
                if 'text_length' in method_data:
                    print(f"     Text Length: {method_data['text_length']}")
                    print(f"     Sanskrit Chars: {method_data['sanskrit_chars_found']}")
                    print(f"     Sample: {method_data['sample_text'][:100]}...")
                if 'images_found' in method_data:
                    print(f"     Images Found: {method_data['images_found']}")
                    print(f"     Needs OCR: {method_data.get('likely_needs_ocr', 'Unknown')}")
            else:
                print(f"   {method_name.upper()}: ❌ Failed - {method_data['error']}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    if results['recommendations']:
        for rec in results['recommendations']:
            print(f"   • {rec}")
    else:
        print("   • No specific recommendations - Sanskrit preservation successful!")
    
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose PDF content for Sanskrit preservation issues")
    parser.add_argument("--pdf-dir", default="./documents", help="Directory containing PDF files")
    
    args = parser.parse_args()
    
    # Run diagnostic
    results = diagnose_pdf_content(args.pdf_dir)
    print_diagnostic_results(results)
