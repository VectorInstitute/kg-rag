import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import pytesseract
from pdf2image import convert_from_path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDFs using poppler and tesseract for text extraction."""
    
    def __init__(self, tesseract_path: Optional[str] = None, poppler_path: Optional[str] = None):
        """Initialize PDF processor with optional paths to tesseract and poppler.
        
        Args:
            tesseract_path: Path to tesseract executable
            poppler_path: Path to poppler binaries
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.poppler_path = poppler_path
        
    def _extract_text_poppler(self, pdf_path: Path) -> str:
        """Extract text from PDF using pdftotext (poppler)."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
                # Use pdftotext command line tool
                cmd = ['pdftotext', '-layout', str(pdf_path), tmp.name]
                subprocess.run(cmd, check=True, capture_output=True)
                return tmp.read().decode('utf-8')
        except subprocess.CalledProcessError as e:
            logger.warning(f"pdftotext failed for {pdf_path}: {e}")
            return ""
        
    def _perform_ocr(self, pdf_path: Path) -> str:
        """Perform OCR on PDF using tesseract."""
        try:
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                poppler_path=self.poppler_path,
                fmt='jpeg',
                grayscale=True,
                dpi=300
            )
            
            # Perform OCR on each image
            text_parts = []
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)} with OCR")
                text = pytesseract.image_to_string(image, lang='eng')
                text_parts.append(text)
                
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"OCR failed for {pdf_path}: {e}")
            return ""
        
    def process_pdf(self, pdf_path: Path) -> Tuple[str, str]:
        """Process a PDF file using both text extraction and OCR if needed.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Tuple of (extracted_text, method_used)
        """
        logger.info(f"Processing {pdf_path}")
        
        # Try poppler first
        text = self._extract_text_poppler(pdf_path)
        
        # If we got meaningful text, return it
        if len(text.strip()) > 100:  # Arbitrary threshold for "meaningful" text
            logger.info(f"Successfully extracted text using poppler from {pdf_path}")
            return text, "poppler"
            
        # If poppler didn't get good results, try OCR
        logger.info(f"Poppler extraction insufficient for {pdf_path}, attempting OCR")
        ocr_text = self._perform_ocr(pdf_path)
        
        if len(ocr_text.strip()) > 100:
            logger.info(f"Successfully extracted text using OCR from {pdf_path}")
            return ocr_text, "ocr"
            
        # If both methods failed
        logger.warning(f"Both text extraction methods failed for {pdf_path}")
        return "", "failed"

def process_pdf_directory(
    input_dir: Path,
    output_dir: Path,
    file_pattern: str = "*.pdf"
) -> List[Tuple[Path, str, str]]:
    """Process all PDFs in a directory and save extracted text.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save extracted text files
        file_pattern: Glob pattern for PDF files
        
    Returns:
        List of tuples (pdf_path, text_path, method_used)
    """
    os.makedirs(output_dir, exist_ok=True)
    processor = PDFProcessor()
    results = []
    
    for pdf_path in input_dir.glob(file_pattern):
        try:
            # Process the PDF
            text, method = processor.process_pdf(pdf_path)
            
            if text:
                # Save the extracted text
                text_path = output_dir / f"{pdf_path.stem}.txt"
                text_path.write_text(text)
                
                results.append((pdf_path, text_path, method))
                logger.info(f"Saved extracted text to {text_path}")
            else:
                logger.error(f"No text extracted from {pdf_path}")
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            
    return results

def main():
    """Main function to demonstrate PDF processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PDF files for text extraction')
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory containing PDF files')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory to save extracted text')
    parser.add_argument('--pattern', default="*.pdf", help='File pattern for PDFs')
    
    args = parser.parse_args()
    
    logger.info(f"Processing PDFs from {args.input_dir}")
    results = process_pdf_directory(args.input_dir, args.output_dir, args.pattern)
    
    # Print summary
    logger.info("\nProcessing Summary:")
    methods_used = {}
    for _, _, method in results:
        methods_used[method] = methods_used.get(method, 0) + 1
        
    for method, count in methods_used.items():
        logger.info(f"{method}: {count} files")
        
    logger.info(f"Total files processed: {len(results)}")

if __name__ == "__main__":
    main()