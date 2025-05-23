# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing HTML document loader functionality."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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
            # Use as_bytes parameter for binary reading
            raw_content = await storage.get(path, as_bytes=True)

            # Detect encoding if not specified
            encoding_to_use = config.encoding
            if not encoding_to_use and CHARDET_AVAILABLE:
                detection = chardet.detect(raw_content[:10000])
                encoding_to_use = detection["encoding"]
                log.debug(
                    f"Detected encoding: {encoding_to_use} (confidence: {detection['confidence']:.2f})"
                )

            # Default encoding if none detected
            if not encoding_to_use:
                encoding_to_use = "windows-1252"

            # Read with detected encoding
            try:
                html_content = await storage.get(path, encoding=encoding_to_use)
            except UnicodeDecodeError:
                # Fallback to a safe encoding if detection fails
                log.warning(
                    f"Failed to decode with {encoding_to_use}, falling back to windows-1252"
                )
                html_content = await storage.get(path, encoding="windows-1252")
                encoding_to_use = "windows-1252"
        except Exception as e:
            log.warning(f"Error reading {path}: {e}")
            raise

        # Parse HTML with BeautifulSoup
        log.info(f"Parsing HTML content from {path}")
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract document structure
        log.info(f"Extracting document structure from {path}")
        document_structure = extract_document_structure(soup, Path(path).name)

        # Create HTML metadata with essential information only
        html_info = {
            "doc_type": document_structure.get("doc_type"),
            "filename": document_structure.get("filename"),
            "encoding": encoding_to_use,
            # Only store the essential data for chunking
            "pages": [{"page_id": p["page_id"], "page_num": p["page_num"]} for p in document_structure.get("pages", [])],
            "paragraphs": [{"para_id": p["para_id"], "para_num": p["para_num"], "char_start": p["char_start"], "char_end": p["char_end"]} for p in document_structure.get("paragraphs", [])],
        }

        # Add detailed logging for html_info
        log.info(f"HTML metadata for {path}:")
        log.info(f"  - Document type: {html_info['doc_type']}")
        log.info(f"  - Filename: {html_info['filename']}")
        log.info(f"  - Page count: {len(html_info['pages'])}")
        log.info(f"  - Paragraph count: {len(html_info['paragraphs'])}")
        log.info(f"  - Encoding: {html_info['encoding']}")
        log.debug(
            f"Extracted {len(html_info['pages'])} page markers and {len(html_info['paragraphs'])} paragraphs"
        )

        # Create a dataframe with the document information
        new_item = {**group, "text": document_structure["text"]}

        # Store the complete document structure directly in metadata
        # Don't create a separate html_attributes column
        if isinstance(new_item.get("metadata"), dict):
            existing_metadata = new_item.get("metadata", {})
        elif new_item.get("metadata") is not None:
            existing_metadata = {"original": new_item.get("metadata")}
        else:
            existing_metadata = {}

        # Then construct the full metadata dictionary
        new_item["metadata"] = {"html": html_info, **existing_metadata}

        log.info(f"Stored complete HTML structure in metadata")

        # Add basic fields
        new_item["id"] = gen_sha512_hash(new_item, new_item.keys())
        new_item["title"] = document_structure.get("doc_type", str(Path(path).name))  # Use doc_type instead of title
        new_item["creation_date"] = await storage.get_creation_date(path)

        log.info(
            f"Created document entry with ID: {new_item['id']}, Title: {new_item['title']}"
        )

        # Process data columns based on config
        df = pd.DataFrame([new_item])
        return process_data_columns(df, config, path)

    return await load_files(load_file, config, storage, progress)


def extract_document_structure(soup: BeautifulSoup, filename: str) -> Dict[str, Any]:
    """Extract structured content from HTML document."""
    log.info(f"Beginning document structure extraction for {filename}")

    document_structure = {
        "text": "",  # Full text content
        "paragraphs": [],  # List of paragraph elements
        "pages": [],  # List of page markers
        "filename": filename,  # Default to input filename
    }


    # Extract document metadata
    type_tag = soup.find("type")
    if type_tag:
        document_structure["doc_type"] = type_tag.get_text().strip().split("\n")[0]
        log.info(f"Extracted document type: {document_structure['doc_type']}")
    else:
        log.info("No document type tag found")

    filename_tag = soup.find("filename")
    if filename_tag:
        # Get only the direct text content of the tag, not nested elements
        filename_content = ""
        for content in filename_tag.contents:
            if isinstance(content, NavigableString):
                filename_content += content
            else:
                # Stop at the first nested tag - we only want direct text
                break

        # Clean up and validate the filename
        filename_content = filename_content.strip()
        log.debug(f"Raw filename content: {filename_content}")

        # Look for common filename patterns
        if re.match(r"^[a-zA-Z0-9_-]+\.(htm|html|txt)$", filename_content):
            document_structure["filename"] = filename_content
        else:
            # If it doesn't look like a valid filename, just use the first 255 chars max
            # and remove any problematic characters
            clean_filename = re.sub(r"[^\w\-\.]", "_", filename_content[:255])
            document_structure["filename"] = (
                clean_filename if clean_filename else filename
            )

        log.info(f"Extracted filename: {document_structure['filename']}")
    else:
        log.info(f"No filename tag found, using default: {filename}")

    # Look for page number patterns in the document
    log.info("Identifying page markers in document")
    page_markers = identify_page_markers(soup)
    if page_markers:
        document_structure["pages"] = page_markers
        log.info(f"Found {len(page_markers)} page markers")
        for i, marker in enumerate(page_markers[:5]):  # Log first 5 page markers
            log.debug(
                f"  - Page marker {i + 1}: ID={marker.get('page_id')}, Num={marker.get('page_num')}"
            )
        if len(page_markers) > 5:
            log.debug(f"  - ... and {len(page_markers) - 5} more")
    else:
        log.info("No page markers found in document")

    # Extract text content by recursively processing all elements
    log.info("Extracting text content from document")
    char_pos = 0

    # Track element positions
    element_positions = {}

    def process_element(element, depth=0, parent_path=None):
        nonlocal char_pos

        # Skip script, style and other non-visible elements
        if isinstance(element, Tag) and element.name in [
            "script",
            "style",
            "meta",
            "link",
            "head",
        ]:
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
            document_structure["text"] += text_content + "\n"
            char_pos += len(text_content) + 1  # +1 for newline

        # Process Tag elements
        elif isinstance(element, Tag):
            # Record element start position
            element_start = char_pos

            # Process children
            for child in element.children:
                process_element(child, depth + 1, element_path)

            # Check if this is a paragraph
            if element.name == "p":
                element_text = element.get_text().strip()
                if element_text:
                    # Add to paragraphs list
                    para_info = {
                        "type": "paragraph",
                        "text": element_text,
                        "char_start": element_start,
                        "char_end": char_pos,
                        "para_id": f"p{len(document_structure['paragraphs']) + 1}",
                        "para_num": len(document_structure["paragraphs"]) + 1,
                    }
                    document_structure["paragraphs"].append(para_info)
                    log.debug(
                        f"Added paragraph {para_info['para_id']}: {element_text[:50]}..."
                        if len(element_text) > 50
                        else element_text
                    )

    # Process the document
    try:
        process_element(soup)
    except RecursionError:
        log.warning("Document structure too deep for recursive processing.")
        log.warning("Falling back to non-recursive method...")
        # Fall back to a simpler approach
        document_structure["text"] = soup.get_text()

    log.info(f"Processed {len(document_structure['paragraphs'])} paragraphs")
    for i, para in enumerate(
        document_structure["paragraphs"][:3]
    ):  # Log first 3 paragraphs
        log.debug(
            f"  - Paragraph {i + 1}: ID={para.get('para_id')}, Length={len(para.get('text', ''))}"
        )
        log.debug(f"    Text preview: {para.get('text', '')[:50]}...")
    if len(document_structure["paragraphs"]) > 3:
        log.debug(
            f"  - ... and {len(document_structure['paragraphs']) - 3} more paragraphs"
        )

    log.info(
        f"Total document text length: {len(document_structure['text'])} characters"
    )

    return document_structure


def identify_page_markers(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Identify page markers in the document."""
    log.info("Beginning page marker identification")
    page_markers = []

    # Regex patterns for different page number formats
    page_patterns = [
        # Standard numbers (e.g., 1, 2, 3)
        r"^\s*(\d+)\s*$",
        # Roman numerals (e.g., i, ii, iii, iv, v, vi, vii, viii, ix, x, etc.)
        r"^\s*([ivxlcdmIVXLCDM]+)\s*$",
        # Appendix/Section style (e.g., A-1, B-24)
        r"^\s*([A-Za-z]-\d+)\s*$",
        # Page indicator with number (e.g., "Page 7")
        r"^\s*[Pp]age\s+(\d+)\s*$",
        # Page indicator with roman numeral (e.g., "Page iv")
        r"^\s*[Pp]age\s+([ivxlcdmIVXLCDM]+)\s*$",
        # Page indicator with appendix (e.g., "Page A-1")
        r"^\s*[Pp]age\s+([A-Za-z]-\d+)\s*$",
    ]

    log.debug(f"Using {len(page_patterns)} regex patterns to identify page markers")

    # Pattern 1: Look for centered paragraphs with page numbers
    centered_paragraphs = soup.find_all("p", align="center")
    log.debug(
        f"Found {len(centered_paragraphs)} centered paragraphs to check for page numbers"
    )

    for p in centered_paragraphs:
        text = p.get_text().strip()

        # Try each pattern
        for pattern in page_patterns:
            if re.match(pattern, text):
                # Extract the page identifier
                if "Page" in text or "page" in text:
                    # For "Page X" format, extract just X
                    page_id = re.search(r"[Pp]age\s+(.+)", text).group(1).strip()
                else:
                    # For standalone numbers, use as is
                    page_id = text.strip()

                # For traditional numeric page numbers, also store as integer for compatibility
                page_num = None
                if re.match(r"^\s*\d+\s*$", page_id):
                    page_num = int(page_id)

                # Remove any decimal points from page_id
                page_id = page_id.replace(".", "")

                page_markers.append({
                    "page_id": page_id,
                    "page_num": page_num,
                })
                log.debug(
                    f"Found centered page marker: ID={page_id}, Num={page_num}, Text='{text}'"
                )
                break

    # Pattern 2: Look for non-centered paragraphs with "Page" indicators
    all_paragraphs = soup.find_all("p")
    non_centered_paragraphs = [p for p in all_paragraphs if p.get("align") != "center"]
    log.debug(
        f"Found {len(non_centered_paragraphs)} non-centered paragraphs to check for page indicators"
    )

    for p in non_centered_paragraphs:
        text = p.get_text().strip()
        if re.search(r"[Pp]age\s+", text):
            for pattern in [
                r"[Pp]age\s+(\d+)",
                r"[Pp]age\s+([ivxlcdmIVXLCDM]+)",
                r"[Pp]age\s+([A-Za-z]-\d+)",
            ]:
                match = re.search(pattern, text)
                if match:
                    page_id = match.group(1).strip()

                    # For traditional numeric page numbers, also store as integer
                    page_num = None
                    if re.match(r"^\d+$", page_id):
                        page_num = int(page_id)

                    # Remove any decimal points from page_id
                    page_id = page_id.replace(".", "")

                    page_markers.append({
                        "page_id": page_id,
                        "page_num": page_num,
                    })
                    log.debug(
                        f"Found page indicator in text: ID={page_id}, Num={page_num}, Text='{text}'"
                    )
                    break

    log.info(f"Identified {len(page_markers)} page markers in total")
    return page_markers
