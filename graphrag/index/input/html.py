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
        
        try:
            # Use as_bytes=True to get binary content for encoding detection
            raw_content = await storage.get(path, as_bytes=True)
            
            # Detect encoding if chardet is available
            detected_encoding = None
            if CHARDET_AVAILABLE:
                detection = chardet.detect(raw_content[:10000])
                detected_encoding = detection['encoding']
                log.debug(f"Detected encoding: {detected_encoding} (confidence: {detection['confidence']:.2f})")
            
            # Use the encoding from config, detected encoding, or fallback
            encoding_to_use = config.encoding or detected_encoding or 'windows-1252'
            
            # Get content with the determined encoding
            try:
                html_content = await storage.get(path, encoding=encoding_to_use)
            except UnicodeDecodeError:
                # Fallback to a safe encoding if the detected one fails
                log.warning(f"Failed to decode with {encoding_to_use}, trying fallback encodings")
                for fallback_encoding in ['windows-1252', 'iso-8859-1', 'utf-8', 'latin-1']:
                    try:
                        html_content = await storage.get(path, encoding=fallback_encoding)
                        encoding_to_use = fallback_encoding
                        log.debug(f"Successfully decoded with fallback encoding: {fallback_encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all fallbacks fail, use a replacement strategy
                    html_content = raw_content.decode('utf-8', errors='replace')
                    encoding_to_use = 'utf-8-replaced'
                    log.warning(f"All encodings failed for {path}, using replacement characters")
        except Exception as e:
            log.warning(f"Error reading {path}: {e}")
            raise
            
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract document structure
        document_structure = extract_document_structure(soup, Path(path).name)
        
        # Create document metadata with HTML-specific information
        html_metadata = {
            "filename": document_structure.get("filename", Path(path).name),
            "doc_type": document_structure.get("doc_type"),
            "doc_sequence": document_structure.get("doc_sequence"),
            "encoding": encoding_to_use,
            "html_title": document_structure.get("title"),
            "pages": len(document_structure.get("pages", [])),
            "paragraphs": len(document_structure.get("paragraphs", [])),
            "html_structure": {
                "page_markers": [p.get("page_id") for p in document_structure.get("pages", [])],
                "has_tables": bool(soup.find_all('table')),
                "has_lists": bool(soup.find_all(['ul', 'ol'])),
                "has_images": bool(soup.find_all('img')),
                "has_links": bool(soup.find_all('a')),
                "headings_count": len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
            }
        }
        
        # Create a dataframe with the document information
        new_item = {
            **group, 
            "text": document_structure["text"],
            "metadata": html_metadata,  # Add the HTML metadata
            "html_attributes": {
                "page_info": document_structure.get("pages", []),
                "paragraph_info": document_structure.get("paragraphs", []),
            }
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
                    # Find the current page if available
                    current_page = None
                    current_page_pos = -1
                    for page in document_structure.get('pages', []):
                        page_pos = page.get('char_pos', -1)
                        if page_pos <= element_start and page_pos > current_page_pos:
                            current_page = page.get('page_id')
                            current_page_pos = page_pos
                    
                    # Add to paragraphs list with page information
                    para_info = {
                        'type': 'paragraph',
                        'text': element_text,
                        'char_start': element_start,
                        'char_end': char_pos,
                        'para_id': f"p{len(document_structure['paragraphs'])+1}",
                        'para_num': len(document_structure['paragraphs'])+1,
                        'element_path': element_path,
                        'page_id': current_page,
                        'html_attributes': {
                            'tag': element.name,
                            'class': element.get('class'),
                            'id': element.get('id'),
                            'align': element.get('align'),
                        }
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
    char_pos = 0  # Track character position
    
    # Helper to update character position based on preceding elements
    def update_char_pos(element):
        nonlocal char_pos
        prev = element.previous_elements
        for p in prev:
            if isinstance(p, NavigableString) and p.strip():
                char_pos += len(p.strip()) + 1  # +1 for newline
        return char_pos
    
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
                
                # Calculate and store character position
                current_pos = update_char_pos(p)
                
                page_markers.append({
                    'page_id': page_id,
                    'page_num': page_num,
                    'text': text,
                    'char_pos': current_pos,
                    'html_attributes': {
                        'tag': 'p',
                        'align': 'center',
                        'class': p.get('class'),
                        'id': p.get('id'),
                    }
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
                    
                    # Calculate and store character position
                    current_pos = update_char_pos(p)
                    
                    page_markers.append({
                        'page_id': page_id,
                        'page_num': page_num,
                        'text': text,
                        'char_pos': current_pos,
                        'html_attributes': {
                            'tag': 'p',
                            'align': p.get('align'),
                            'class': p.get('class'),
                            'id': p.get('id'),
                        }
                    })
                    break
    
    # Sort page markers by character position
    page_markers.sort(key=lambda x: x.get('char_pos', 0))
    
    return page_markers