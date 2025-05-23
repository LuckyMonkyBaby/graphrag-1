# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing PDF document loader functionality."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from graphrag.config.models.input_config import InputConfig
from graphrag.index.input.util import load_files, process_data_columns
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.logger.base import ProgressLogger
from graphrag.storage.pipeline_storage import PipelineStorage

log = logging.getLogger(__name__)

# Try to import PDF parsing library
try:
    import pdfplumber
    PDF_PLUMBER_AVAILABLE = True
except ImportError:
    PDF_PLUMBER_AVAILABLE = False
    log.warning("pdfplumber not installed. PDF parsing will not be available. Install with: pip install pdfplumber")

# Fallback to PyPDF2 if pdfplumber is not available
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    log.warning("PyPDF2 not installed. Fallback PDF parsing will not be available. Install with: pip install PyPDF2")


async def load_pdf(
    config: InputConfig,
    progress: Optional[ProgressLogger],
    storage: PipelineStorage,
) -> pd.DataFrame:
    """Load PDF inputs from a directory."""
    
    if not PDF_PLUMBER_AVAILABLE and not PYPDF2_AVAILABLE:
        msg = "No PDF parsing library available. Install pdfplumber or PyPDF2: pip install pdfplumber PyPDF2"
        raise ImportError(msg)

    async def load_file(path: str, group: Optional[Dict] = None) -> pd.DataFrame:
        if group is None:
            group = {}
        
        try:
            # Read PDF content as bytes
            pdf_bytes = await storage.get(path, as_bytes=True)
            
            # Parse PDF with available library
            if PDF_PLUMBER_AVAILABLE:
                document_structure = extract_pdf_structure_pdfplumber(pdf_bytes, Path(path).name)
            elif PYPDF2_AVAILABLE:
                document_structure = extract_pdf_structure_pypdf2(pdf_bytes, Path(path).name)
            else:
                raise ImportError("No PDF parsing library available")
                
        except Exception as e:
            log.warning(f"Error reading PDF {path}: {e}")
            raise
        
        # Create PDF metadata with structured information
        pdf_info = {
            "has_pages": bool(document_structure.get("pages")),
            "has_paragraphs": bool(document_structure.get("paragraphs")), 
            "doc_type": "pdf",
            "filename": document_structure.get("filename"),
            "page_count": len(document_structure.get("pages", [])),
            "paragraph_count": len(document_structure.get("paragraphs", [])),
            # Add direct references to pages and paragraphs for easy access
            "pages": document_structure.get("pages", []),
            "paragraphs": document_structure.get("paragraphs", [])
        }
        
        log.info(f"PDF metadata for {path}:")
        log.info(f"  - Document type: {pdf_info['doc_type']}")
        log.info(f"  - Filename: {pdf_info['filename']}")
        log.info(f"  - Page count: {pdf_info['page_count']}")
        log.info(f"  - Paragraph count: {pdf_info['paragraph_count']}")
        
        # Create a dataframe with the document information
        new_item = {**group, "text": document_structure["text"]}
        
        # Store the complete document structure in metadata
        if isinstance(new_item.get("metadata"), dict):
            existing_metadata = new_item.get("metadata", {})
        elif new_item.get("metadata") is not None:
            existing_metadata = {"original": new_item.get("metadata")}
        else:
            existing_metadata = {}

        new_item["metadata"] = {
            "html": pdf_info,  # Use "html" key for compatibility with existing pipeline
            **existing_metadata
        }
        
        log.info(f"Stored complete PDF structure in metadata")
        
        # Add basic fields
        new_item["id"] = gen_sha512_hash(new_item, new_item.keys())
        new_item["title"] = document_structure.get("title", str(Path(path).name))
        new_item["creation_date"] = await storage.get_creation_date(path)
        
        log.info(f"Created PDF document entry with ID: {new_item['id']}, Title: {new_item['title']}")
        
        # Process data columns based on config
        df = pd.DataFrame([new_item])
        return process_data_columns(df, config, path)

    return await load_files(load_file, config, storage, progress)


def extract_pdf_structure_pdfplumber(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Extract structured content from PDF using pdfplumber."""
    log.info(f"Beginning PDF structure extraction for {filename} using pdfplumber")
    
    document_structure = {
        'text': '',             # Full text content
        'paragraphs': [],       # List of paragraph elements
        'pages': [],            # List of page information
        'filename': filename,   # Document filename
        'title': Path(filename).stem,  # Use filename stem as title
    }
    
    char_pos = 0
    
    try:
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            log.info(f"PDF opened successfully, {len(pdf.pages)} pages found")
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if not page_text:
                    continue
                    
                page_start_char = char_pos
                
                # Add page marker
                page_info = {
                    'page_id': f"page_{page_num}",
                    'page_num': page_num,
                    'text': page_text,
                    'char_start': page_start_char,
                    'char_end': page_start_char + len(page_text),
                }
                document_structure['pages'].append(page_info)
                
                # Split page text into paragraphs (simple heuristic)
                paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                
                for para_idx, para_text in enumerate(paragraphs):
                    para_start_char = char_pos
                    para_end_char = char_pos + len(para_text)
                    
                    para_info = {
                        'type': 'paragraph',
                        'text': para_text,
                        'char_start': para_start_char,
                        'char_end': para_end_char,
                        'para_id': f"page_{page_num}_para_{para_idx + 1}",
                        'para_num': len(document_structure['paragraphs']) + 1,
                        'page_num': page_num,
                        'page_id': f"page_{page_num}",
                    }
                    document_structure['paragraphs'].append(para_info)
                    
                    # Add to full text
                    document_structure['text'] += para_text + '\n'
                    char_pos += len(para_text) + 1
                
                # Add page break
                document_structure['text'] += '\n'
                char_pos += 1
                
    except Exception as e:
        log.error(f"Error extracting PDF structure with pdfplumber: {e}")
        raise
    
    log.info(f"Extracted {len(document_structure['pages'])} pages and {len(document_structure['paragraphs'])} paragraphs")
    return document_structure


def extract_pdf_structure_pypdf2(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Extract structured content from PDF using PyPDF2 (fallback)."""
    log.info(f"Beginning PDF structure extraction for {filename} using PyPDF2")
    
    document_structure = {
        'text': '',             # Full text content
        'paragraphs': [],       # List of paragraph elements
        'pages': [],            # List of page information
        'filename': filename,   # Document filename
        'title': Path(filename).stem,  # Use filename stem as title
    }
    
    char_pos = 0
    
    try:
        import io
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        log.info(f"PDF opened successfully, {len(pdf_reader.pages)} pages found")
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if not page_text:
                continue
                
            page_start_char = char_pos
            
            # Add page marker
            page_info = {
                'page_id': f"page_{page_num}",
                'page_num': page_num,
                'text': page_text,
                'char_start': page_start_char,
                'char_end': page_start_char + len(page_text),
            }
            document_structure['pages'].append(page_info)
            
            # Split page text into paragraphs (simple heuristic)
            paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
            
            for para_idx, para_text in enumerate(paragraphs):
                para_start_char = char_pos
                para_end_char = char_pos + len(para_text)
                
                para_info = {
                    'type': 'paragraph',
                    'text': para_text,
                    'char_start': para_start_char,
                    'char_end': para_end_char,
                    'para_id': f"page_{page_num}_para_{para_idx + 1}",
                    'para_num': len(document_structure['paragraphs']) + 1,
                    'page_num': page_num,
                    'page_id': f"page_{page_num}",
                }
                document_structure['paragraphs'].append(para_info)
                
                # Add to full text
                document_structure['text'] += para_text + '\n'
                char_pos += len(para_text) + 1
            
            # Add page break
            document_structure['text'] += '\n'
            char_pos += 1
                
    except Exception as e:
        log.error(f"Error extracting PDF structure with PyPDF2: {e}")
        raise
    
    log.info(f"Extracted {len(document_structure['pages'])} pages and {len(document_structure['paragraphs'])} paragraphs")
    return document_structure