# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing HTML document loader functionality."""

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag

from graphrag.config.models.input_config import InputConfig
from graphrag.index.input.util import load_files, process_data_columns
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.logger.base import ProgressLogger
from graphrag.storage.pipeline_storage import PipelineStorage

log = logging.getLogger(__name__)

# Try to import chardet, but handle the case where it's not installed
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    log.warning("chardet not installed. Automatic encoding detection will be limited.")


async def load_html(
    config: InputConfig,
    progress: Optional[ProgressLogger],
    storage: PipelineStorage,
) -> pd.DataFrame:
    """Load HTML inputs from a directory."""

    async def load_file(path: str, group: Optional[Dict] = None) -> pd.DataFrame:
        if group is None:
            group = {}
        
        # Read the HTML content with encoding detection
        raw_content = await storage.get(path, binary=True)
        html_content, encoding = detect_and_decode_html(raw_content, config.encoding)
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract document structure
        document_structure = extract_document_structure(soup, Path(path).name)
        
        # Create a dataframe with the document information
        new_item = {**group, "text": document_structure["text"]}
        
        # Add metadata if available
        metadata = {
            "filename": document_structure.get("filename", Path(path).name),
            "doc_type": document_structure.get("doc_type"),
            "doc_sequence": document_structure.get("doc_sequence"),
            "pages": len(document_structure.get("pages", [])),
            "paragraphs": len(document_structure.get("paragraphs", [])),
            "encoding": encoding
        }
        
        # Include metadata fields based on config
        if config.metadata:
            for field in config.metadata:
                # Set metadata field if exists in document_structure or default to None
                if field in document_structure:
                    new_item[field] = document_structure[field]
                else:
                    new_item[field] = None
        
        # Always include basic fields
        new_item["id"] = gen_sha512_hash(new_item, new_item.keys())
        new_item["title"] = document_structure.get("title", str(Path(path).name))
        new_item["creation_date"] = await storage.get_creation_date(path)
        
        # Process data columns based on config
        df = pd.DataFrame([new_item])
        return process_data_columns(df, config, path)

    return await load_files(load_file, config, storage, progress)


def detect_and_decode_html(raw_content: bytes, specified_encoding: Optional[str] = None) -> Tuple[str, str]:
    """Detect encoding and decode HTML content."""
    if specified_encoding:
        try:
            return raw_content.decode(specified_encoding), specified_encoding
        except UnicodeDecodeError:
            log.warning(f"Failed to decode with specified encoding {specified_encoding}. Falling back to detection.")
    
    detected_encoding = None
    
    # First, try to detect encoding from content if chardet is available
    if CHARDET_AVAILABLE:
        detection = chardet.detect(raw_content[:10000])  # Use first 10KB for detection
        detected_encoding = detection['encoding']
        confidence = detection['confidence']
        log.debug(f"Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")
    
    # Try to parse with detected encoding to check for meta charset
    if detected_encoding:
        try:
            decoded_sample = raw_content[:10000].decode(detected_encoding)
            soup = BeautifulSoup(decoded_sample, 'html.parser')
            
            # Check for charset in meta tags
            meta_encoding = None
            for meta in soup.find_all('meta'):
                if meta.get('charset'):
                    meta_encoding = meta.get('charset')
                    break
                elif meta.get('content') and 'charset=' in meta.get('content', ''):
                    content = meta.get('content')
                    match = re.search(r'charset=([^;]+)', content)
                    if match:
                        meta_encoding = match.group(1).strip()
                        break
            
            if meta_encoding:
                log.debug(f"Found encoding in meta tag: {meta_encoding}")
                detected_encoding = meta_encoding
        except Exception as e:
            log.debug(f"Error parsing HTML for meta encoding: {e}")
    
    # Default to common encodings if detection failed
    if not detected_encoding or detected_encoding == 'ascii':
        detected_encoding = 'windows-1252'  # Common default for HTML documents
        log.debug(f"Using default encoding: {detected_encoding}")
    
    # Try to decode with the detected encoding
    try:
        decoded_content = raw_content.decode(detected_encoding)
        return decoded_content, detected_encoding
    except UnicodeDecodeError:
        # Try alternate encodings common in HTML documents
        for alt_encoding in ['windows-1252', 'iso-8859-1', 'latin-1', 'utf-8', 'cp1252']:
            if alt_encoding != detected_encoding:
                try:
                    decoded_content = raw_content.decode(alt_encoding)
                    log.debug(f"Successfully decoded with {alt_encoding}")
                    return decoded_content, alt_encoding
                except UnicodeDecodeError:
                    continue
        
        # If all else fails, decode with 'replace' error handling
        log.warning("All encodings failed. Decoding with 'replace' error handling.")
        return raw_content.decode('utf-8', errors='replace'), 'utf-8-replaced'


def extract_document_structure(soup: BeautifulSoup, filename: str) -> Dict[str, Any]:
    """Extract structured content from HTML document."""
    document_structure = {
        'text': '',             # Full text content
        'paragraphs': [],       # List of paragraph elements
        'pages': [],            # List of page markers
        'page_map': {},         # Map of character positions to page IDs
        'filename': filename,   # Default to input filename
    }
    
    # Extract document title
    title_tag = soup.find('title')
    if title_tag:
        document_structure['title'] = title_tag.get_text().strip()
    
    # Extract document metadata
    # Look for type, sequence, and filename tags (common in SEC filings)
    type_tag = soup.find('type')
    if type_tag:
        document_structure['doc_type'] = type_tag.get_text().strip()
    
    sequence_tag = soup.find('sequence')
    if sequence_tag:
        document_structure['doc_sequence'] = sequence_tag.get_text().strip()
    
    filename_tag = soup.find('filename')
    if filename_tag:
        # Get only the direct text content of the tag, not nested elements
        filename_content = ''
        for content in filename_tag.contents:
            if isinstance(content, NavigableString):
                filename_content += content
            else:
                # Stop at the first nested tag - we only want direct text
                break
        
        # Clean up and validate the filename
        filename_content = filename_content.strip()
        
        # Look for common filename patterns
        if re.match(r'^[a-zA-Z0-9_-]+\.(htm|html|txt)$', filename_content):
            document_structure['filename'] = filename_content
        else:
            # If it doesn't look like a valid filename, just use the first 255 chars max
            # and remove any problematic characters
            clean_filename = re.sub(r'[^\w\-\.]', '_', filename_content[:255])
            document_structure['filename'] = clean_filename if clean_filename else filename
    
    # Look for page number patterns in the document
    page_markers = identify_page_markers(soup)
    if page_markers:
        document_structure['pages'] = page_markers
    
    # Extract text content by recursively processing all elements
    char_pos = 0
    
    # Track element positions
    element_positions = {}
    
    def process_element(element, depth=0, parent_path=None):
        nonlocal char_pos
        
        # Skip script, style and other non-visible elements
        if isinstance(element, Tag) and element.name in ['script', 'style', 'meta', 'link', 'head']:
            return
        
        # Initialize element path
        element_path = parent_path or []
        if isinstance(element, Tag) and element.name:
            element_path = element_path + [element.name]
        
        # Save the start position of this element
        if isinstance(element, Tag):
            element_positions[element] = char_pos
        
        # Check if this is an element with text content
        if isinstance(element, NavigableString) and element.strip():
            text_content = element.strip()
            
            # Add text to the full document
            document_structure['text'] += text_content + '\n'
            char_pos += len(text_content) + 1  # +1 for newline
        
        # Process Tag elements
        elif isinstance(element, Tag):
            # Record element start position
            element_start = char_pos
            
            # Process children
            for child in element.children:
                process_element(child, depth + 1, element_path)
            
            # Check if this is a paragraph
            if element.name == 'p':
                element_text = element.get_text().strip()
                if element_text:
                    # Add to paragraphs list
                    para_info = {
                        'type': 'paragraph',
                        'text': element_text,
                        'char_start': element_start,
                        'char_end': char_pos,
                        'para_id': f"p{len(document_structure['paragraphs'])+1}",
                        'para_num': len(document_structure['paragraphs'])+1,
                        'element_path': element_path
                    }
                    document_structure['paragraphs'].append(para_info)
    
    # Process the document
    try:
        process_element(soup)
    except RecursionError:
        log.warning("Document structure too deep for recursive processing.")
        log.warning("Falling back to non-recursive method...")
        # Fall back to a simpler approach
        document_structure['text'] = soup.get_text()
    
    log.debug(f"Processed {len(document_structure['paragraphs'])} paragraphs")
    
    return document_structure


def identify_page_markers(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Identify page markers in the document."""
    page_markers = []
    
    # Regex patterns for different page number formats
    page_patterns = [
        # Standard numbers (e.g., 1, 2, 3)
        r'^\s*(\d+)\s*$',
        
        # Roman numerals (e.g., i, ii, iii, iv, v, vi, vii, viii, ix, x, etc.)
        r'^\s*([ivxlcdmIVXLCDM]+)\s*$',
        
        # Appendix/Section style (e.g., A-1, B-24)
        r'^\s*([A-Za-z]-\d+)\s*$',
        
        # Page indicator with number (e.g., "Page 7")
        r'^\s*[Pp]age\s+(\d+)\s*$',
        
        # Page indicator with roman numeral (e.g., "Page iv")
        r'^\s*[Pp]age\s+([ivxlcdmIVXLCDM]+)\s*$',
        
        # Page indicator with appendix (e.g., "Page A-1")
        r'^\s*[Pp]age\s+([A-Za-z]-\d+)\s*$'
    ]
    
    # Pattern 1: Look for centered paragraphs with page numbers
    for p in soup.find_all('p', align='center'):
        text = p.get_text().strip()
        
        # Try each pattern
        for pattern in page_patterns:
            if re.match(pattern, text):
                # Extract the page identifier
                if "Page" in text or "page" in text:
                    # For "Page X" format, extract just X
                    page_id = re.search(r'[Pp]age\s+(.+)', text).group(1).strip()
                else:
                    # For standalone numbers, use as is
                    page_id = text.strip()
                
                # For traditional numeric page numbers, also store as integer for compatibility
                page_num = None
                if re.match(r'^\s*\d+\s*$', page_id):
                    page_num = int(page_id)
                
                # Remove any decimal points from page_id
                page_id = page_id.replace('.', '')
                
                page_markers.append({
                    'page_id': page_id,
                    'page_num': page_num,
                    'text': text,
                })
                break
    
    # Pattern 2: Look for non-centered paragraphs with "Page" indicators
    for p in soup.find_all('p'):
        if p.get('align') == 'center':
            continue  # Skip centered ones, already processed
            
        text = p.get_text().strip()
        if re.search(r'[Pp]age\s+', text):
            for pattern in [r'[Pp]age\s+(\d+)', r'[Pp]age\s+([ivxlcdmIVXLCDM]+)', r'[Pp]age\s+([A-Za-z]-\d+)']:
                match = re.search(pattern, text)
                if match:
                    page_id = match.group(1).strip()
                    
                    # For traditional numeric page numbers, also store as integer
                    page_num = None
                    if re.match(r'^\d+$', page_id):
                        page_num = int(page_id)
                    
                    # Remove any decimal points from page_id
                    page_id = page_id.replace('.', '')
                    
                    page_markers.append({
                        'page_id': page_id,
                        'page_num': page_num,
                        'text': text,
                    })
                    break
    
    return page_markers