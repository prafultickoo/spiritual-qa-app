"""
Agent tools for PDF orientation detection and correction using CrewAI.
"""
import os
import logging
from typing import List, Dict, Any
from crewai import Tool

# Import our PDF orientation utilities
from utils.pdf_orientation_tools import (
    analyze_pdf_orientation,
    correct_pdf_orientation,
    apply_ocr_with_orientation_correction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Tool functions

def pdf_orientation_analyzer_tool(pdf_path: str) -> Dict[str, Any]:
    """
    Tool for analyzing PDF orientation/rotation issues.
    
    Args:
        pdf_path (str): Path to the PDF file to analyze
        
    Returns:
        Dict[str, Any]: Analysis results with details about orientation issues
    """
    logger.info(f"Analyzing PDF orientation for: {pdf_path}")
    return analyze_pdf_orientation(pdf_path)


def pdf_orientation_corrector_tool(pdf_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Tool for correcting PDF orientation/rotation issues.
    
    Args:
        pdf_path (str): Path to the PDF file to correct
        output_path (str, optional): Path for the corrected PDF output
        
    Returns:
        Dict[str, Any]: Correction results including the path to the corrected PDF
    """
    logger.info(f"Correcting PDF orientation for: {pdf_path}")
    
    # Generate output path if not provided
    if not output_path:
        dir_path = os.path.dirname(os.path.abspath(pdf_path))
        filename = os.path.basename(pdf_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(dir_path, f"{name}_corrected{ext}")
    
    return correct_pdf_orientation(pdf_path, output_path)


def pdf_ocr_tool(pdf_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Tool for applying OCR with orientation correction for problematic PDFs.
    
    Args:
        pdf_path (str): Path to the PDF file to process with OCR
        output_path (str, optional): Path for the OCR-processed PDF output
        
    Returns:
        Dict[str, Any]: OCR results including the path to the processed PDF
    """
    logger.info(f"Applying OCR with orientation correction for: {pdf_path}")
    
    # Generate output path if not provided
    if not output_path:
        dir_path = os.path.dirname(os.path.abspath(pdf_path))
        filename = os.path.basename(pdf_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(dir_path, f"{name}_ocr{ext}")
    
    return apply_ocr_with_orientation_correction(pdf_path, output_path)


def batch_analyze_pdfs_tool(directory_path: str) -> Dict[str, Any]:
    """
    Tool for batch analyzing multiple PDFs in a directory.
    
    Args:
        directory_path (str): Path to directory containing PDF files
        
    Returns:
        Dict[str, Any]: Batch analysis results for all PDFs in the directory
    """
    logger.info(f"Batch analyzing PDFs in directory: {directory_path}")
    
    results = {
        "directory": directory_path,
        "files_analyzed": 0,
        "files_with_issues": 0,
        "analysis_by_file": []
    }
    
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        results["error"] = "Directory not found"
        return results
    
    # Get all PDF files in directory
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        analysis = analyze_pdf_orientation(pdf_path)
        
        results["files_analyzed"] += 1
        if analysis.get("needs_correction", False):
            results["files_with_issues"] += 1
        
        results["analysis_by_file"].append({
            "filename": pdf_file,
            "needs_correction": analysis.get("needs_correction", False),
            "correction_method": analysis.get("correction_method"),
            "pages_with_issues": len(analysis.get("pages_with_issues", []))
        })
    
    logger.info(f"Batch analysis complete. {results['files_with_issues']} out of {results['files_analyzed']} files have orientation issues.")
    return results


# Create CrewAI Tool objects
pdf_orientation_analyzer = Tool(
    name="pdf_orientation_analyzer",
    description="Analyze PDF documents for orientation/rotation issues",
    func=pdf_orientation_analyzer_tool
)

pdf_orientation_corrector = Tool(
    name="pdf_orientation_corrector",
    description="Correct orientation/rotation issues in PDF documents",
    func=pdf_orientation_corrector_tool
)

pdf_ocr_processor = Tool(
    name="pdf_ocr_processor",
    description="Apply OCR with orientation correction for problematic PDFs",
    func=pdf_ocr_tool
)

batch_pdf_analyzer = Tool(
    name="batch_pdf_analyzer",
    description="Batch analyze multiple PDFs in a directory for orientation issues",
    func=batch_analyze_pdfs_tool
)
