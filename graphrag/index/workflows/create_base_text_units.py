import json
import logging
from typing import Any, cast

import pandas as pd

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.chunking_config import ChunkStrategyType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import (
    PAGE_ID, PAGE_NUMBER, 
    PARAGRAPH_ID, PARAGRAPH_NUMBER,
    CHAR_POSITION_START, CHAR_POSITION_END
)
from graphrag.index.operations.chunk_text.chunk_text import chunk_text
from graphrag.index.operations.chunk_text.strategies import get_encoding_fn
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.logger.progress import Progress
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

# Add logger
log = logging.getLogger(__name__)

async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform base text_units."""
    log.info("Starting run_workflow process")
    documents = await load_table_from_storage("documents", context.storage)
    log.info(f"Loaded {len(documents)} documents from storage")

    chunks = config.chunks
    log.info(f"Using chunk configuration: size={chunks.size}, overlap={chunks.overlap}, strategy={chunks.strategy}")

    # Create the text units
    log.info("Creating base text units")
    output = create_base_text_units(
        documents,
        context.callbacks,
        chunks.group_by_columns,
        chunks.size,
        chunks.overlap,
        chunks.encoding_model,
        strategy=chunks.strategy,
        prepend_metadata=chunks.prepend_metadata,
        chunk_size_includes_metadata=chunks.chunk_size_includes_metadata,
    )
    log.info(f"Created {len(output)} text units")

    # Write the output
    log.info("Writing output to storage")
    await write_table_to_storage(output, "text_units", context.storage)
    log.info("Successfully wrote text_units to storage")
    
    # Return the result
    return WorkflowFunctionOutput(result=output)


def create_base_text_units(
    documents: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    group_by_columns: list[str],
    size: int,
    overlap: int,
    encoding_model: str,
    strategy: ChunkStrategyType,
    prepend_metadata: bool = False,
    chunk_size_includes_metadata: bool = False,
) -> pd.DataFrame:
    """All the steps to transform base text_units."""
    log.info(f"Creating text units with strategy: {strategy}, size: {size}, overlap: {overlap}")
    
    # Sort documents by ID
    sort = documents.sort_values(by=["id"], ascending=[True])
    
    # Create text_with_ids column for tracking
    sort["text_with_ids"] = list(
        zip(*[sort[col] for col in ["id", "text"]], strict=True)
    )
    
    # Update progress
    callbacks.progress(Progress(percent=0))
    
    # Check for HTML structure in metadata
    has_html_in_metadata = False
    if "metadata" in documents.columns:
        log.info(documents["metadata"])
        try:
            has_html_in_metadata = documents["metadata"].apply(
                lambda m: isinstance(m, dict) and "html" in m 
                or (isinstance(m, str) and "html" in m)
            ).any()
            
            if has_html_in_metadata:
                log.info("Detected HTML structure in document metadata")
        except Exception as e:
            log.warning(f"Error checking for HTML in metadata: {e}")

    # Prepare aggregation dictionary
    agg_dict = {"text_with_ids": list}
    if "metadata" in documents:
        agg_dict["metadata"] = "first"  # type: ignore
    
    # Group documents
    aggregated = (
        (
            sort.groupby(group_by_columns, sort=False)
            if len(group_by_columns) > 0
            else sort.groupby(lambda _x: True)
        )
        .agg(agg_dict)
        .reset_index()
    )
    aggregated.rename(columns={"text_with_ids": "texts"}, inplace=True)
    
    # Define chunker function
    def chunker(row: dict[str, Any]) -> Any:
        line_delimiter = ".\n"
        metadata_str = ""
        metadata_tokens = 0
        
        # Handle metadata prepending if needed
        if prepend_metadata and "metadata" in row:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    # If JSON parsing fails, use as is
                    metadata = {"raw": metadata}
            
            if isinstance(metadata, dict):
                # Remove large HTML arrays if present to save space
                if "html" in metadata and isinstance(metadata["html"], dict):
                    metadata_copy = metadata.copy()
                    html = metadata_copy["html"].copy()
                    
                    # Remove large arrays
                    if "pages" in html:
                        html.pop("pages")
                    if "paragraphs" in html:
                        html.pop("paragraphs")
                    
                    metadata_copy["html"] = html
                    metadata = metadata_copy
                
                # Create metadata string
                metadata_str = (
                    line_delimiter.join(f"{k}: {v}" for k, v in metadata.items())
                    + line_delimiter
                )
            
            # Check if metadata fits in chunk size
            if chunk_size_includes_metadata:
                encode, _ = get_encoding_fn(encoding_model)
                metadata_tokens = len(encode(metadata_str))
                if metadata_tokens >= size:
                    message = "Metadata tokens exceeds the maximum tokens per chunk. Please increase the tokens per chunk."
                    log.error(message)
                    raise ValueError(message)
        
        # Chunk the text
        chunked = chunk_text(
            pd.DataFrame([row]).reset_index(drop=True),
            column="texts",
            size=size - metadata_tokens,
            overlap=overlap,
            encoding_model=encoding_model,
            strategy=strategy,
            callbacks=callbacks,
        )[0]
        
        # Prepend metadata if needed
        if prepend_metadata:
            for index, chunk in enumerate(chunked):
                if isinstance(chunk, str):
                    chunked[index] = metadata_str + chunk
                else:
                    chunked[index] = (
                        (chunk[0], metadata_str + chunk[1], chunk[2]) if chunk else None
                    )
        
        # Add chunk metadata with HTML info
        if "metadata" in row and has_html_in_metadata:
            orig_metadata = row["metadata"]
            if isinstance(orig_metadata, str):
                try:
                    orig_metadata = json.loads(orig_metadata)
                except:
                    orig_metadata = {}
            
            # Extract HTML position info for each chunk
            if isinstance(orig_metadata, dict) and "html" in orig_metadata and isinstance(orig_metadata["html"], dict):
                html_meta = orig_metadata["html"]
                
                # Create enhanced chunks with position info
                for i, chunk in enumerate(chunked):
                    # Skip if chunk is not in expected format
                    if not chunk or not isinstance(chunk, tuple) or len(chunk) < 3:
                        continue
                    
                    # Create base chunk
                    base_chunk = (chunk[0], chunk[1], chunk[2])
                    
                    # Create comprehensive chunk metadata preserving ALL structural information
                    chunk_meta = {}
                    if isinstance(html_meta, dict):
                        # PRESERVE ALL HTML properties including pages and paragraphs
                        chunk_meta = {
                            "html": {
                                "doc_type": html_meta.get("doc_type"),
                                "has_pages": html_meta.get("has_pages", False),
                                "has_paragraphs": html_meta.get("has_paragraphs", False),
                                "page_count": html_meta.get("page_count", 0),
                                "paragraph_count": html_meta.get("paragraph_count", 0),
                                # CRITICAL: PRESERVE primary structural data
                                "pages": html_meta.get("pages", []),
                                "paragraphs": html_meta.get("paragraphs", [])
                            }
                        }
                        
                        # Add chunk-specific positioning if we can determine it
                        chunk_text = chunk[1] if len(chunk) > 1 else ""
                        chunk_position = determine_chunk_position(chunk_text, html_meta)
                        if chunk_position:
                            chunk_meta.update(chunk_position)
                    
                    # Store enhanced chunk
                    chunked[i] = (*base_chunk, chunk_meta)
        
        # Store the chunks
        row["chunks"] = chunked
        return row
    
    # Apply chunker
    log.info("Chunking documents")
    aggregated = aggregated.apply(lambda row: chunker(row), axis=1)
    
    # Keep only necessary columns
    log.info("Processing chunked results")
    aggregated = cast("pd.DataFrame", aggregated[[*group_by_columns, "chunks"]])
    
    # Explode to create one row per chunk
    aggregated = aggregated.explode("chunks")
    
    # Rename chunks column
    aggregated.rename(
        columns={
            "chunks": "chunk",
        },
        inplace=True,
    )
    
    # Generate unique IDs
    aggregated["id"] = aggregated.apply(
        lambda row: gen_sha512_hash(row, ["chunk"]), axis=1
    )
    
    # Extract document IDs, text, and tokens
    try:
        # Convert chunks to DataFrame for extraction
        chunks_df = pd.DataFrame(
            aggregated["chunk"].tolist(), 
            index=aggregated.index
        )
        
        # Set column names based on length
        if len(chunks_df.columns) >= 4:
            # Has metadata column
            chunks_df.columns = ["document_ids", "text", "n_tokens", "metadata"]
            # Convert metadata to string to avoid mixed type issues
            chunk_metadata = chunks_df["metadata"].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else 
                          (x if isinstance(x, str) else None)
            )
            aggregated["chunk_metadata"] = chunk_metadata
        else:
            # Standard columns
            chunks_df.columns = ["document_ids", "text", "n_tokens"]
        
        # Add extracted columns
        aggregated["document_ids"] = chunks_df["document_ids"]
        aggregated["text"] = chunks_df["text"]
        aggregated["n_tokens"] = chunks_df["n_tokens"]
        
    except Exception as e:
        log.error(f"Error extracting chunk components: {e}")
        # Create a simple extraction using original method as fallback
        aggregated[["document_ids", "chunk", "n_tokens"]] = pd.DataFrame(
            aggregated["chunk"].tolist(), index=aggregated.index
        )
        # Rename for downstream consumption
        aggregated.rename(columns={"chunk": "text"}, inplace=True)
    
    # Ensure document_ids is always a list to prevent mixed type errors
    log.info("Normalizing document_ids format")
    aggregated["document_ids"] = aggregated["document_ids"].apply(
        lambda x: [] if x is None else (x if isinstance(x, list) else [x])
    )
    
    # Add HTML structure columns if metadata contains HTML
    if "chunk_metadata" in aggregated.columns and has_html_in_metadata:
        # Initialize structural columns with default None values
        aggregated[PAGE_ID] = None
        aggregated[PAGE_NUMBER] = None
        aggregated[PARAGRAPH_ID] = None
        aggregated[PARAGRAPH_NUMBER] = None
        aggregated[CHAR_POSITION_START] = None
        aggregated[CHAR_POSITION_END] = None
        
        # Create attributes from structured info preserving PRIMARY data
        log.info("Creating attributes column with preserved structural data")
        aggregated["attributes"] = aggregated.apply(
            lambda row: create_attributes(row.get("chunk_metadata", None)), axis=1
        )
        
        # Convert attributes to JSON strings to prevent Parquet issues
        # IMPORTANT: Always store as JSON string, never as dict
        aggregated["attributes"] = aggregated["attributes"].apply(
            lambda x: json.dumps(x) if x is not None else "{}"
        )
        
        # EXTRACT structural information to dedicated columns
        log.info("Extracting structural information to dedicated columns")
        aggregated = extract_structural_columns(aggregated)
        
        # Remove temporary column
        if "chunk_metadata" in aggregated.columns:
            aggregated = aggregated.drop(columns=["chunk_metadata"])
    else:
        # Create empty attributes column as JSON strings and initialize structural columns
        aggregated["attributes"] = ["{}"] * len(aggregated)
        aggregated[PAGE_ID] = None
        aggregated[PAGE_NUMBER] = None
        aggregated[PARAGRAPH_ID] = None
        aggregated[PARAGRAPH_NUMBER] = None
        aggregated[CHAR_POSITION_START] = None
        aggregated[CHAR_POSITION_END] = None
    
    # Filter out rows with no text and reset index
    log.info("Finalizing results")
    result = cast(
        "pd.DataFrame", aggregated[aggregated["text"].notna()].reset_index(drop=True)
    )
    
    return result


def create_attributes(metadata_str) -> dict:
    """Create attributes dictionary preserving primary structural information."""
    log.info("Creating attributes from metadata string")
    
    attributes = {}
    
    # Parse metadata if it's a string
    metadata = {}
    if isinstance(metadata_str, str) and metadata_str:
        try:
            log.debug(f"Parsing JSON metadata string: {metadata_str[:100]}..." if len(metadata_str) > 100 else metadata_str)
            metadata = json.loads(metadata_str)
        except Exception as e:
            log.warning(f"Failed to parse metadata JSON: {e}")
            return attributes
    else:
        log.debug(f"Metadata is not a string: {type(metadata_str)}")
    
    # Log the full metadata structure for debugging
    try:
        metadata_sample = str(metadata)[:500] + "..." if len(str(metadata)) > 500 else str(metadata)
        log.debug(f"Raw metadata structure: {metadata_sample}")
    except Exception as e:
        log.warning(f"Failed to log metadata structure: {e}")
    
    # PRESERVE PRIMARY STRUCTURAL INFORMATION - DO NOT REMOVE
    if isinstance(metadata, dict) and "html" in metadata:
        html = metadata.get("html", {})
        if isinstance(html, dict):
            log.debug(f"HTML metadata keys: {list(html.keys())}")
            
            # Keep ALL HTML properties including pages and paragraphs
            # These are PRIMARY structural attributes, not just metadata
            html_props = {}
            for key, value in html.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    html_props[key] = value
                    log.debug(f"Added HTML property {key}: {value}")
                elif key in ["pages", "paragraphs"]:
                    # CRITICAL: Keep pages and paragraphs as they contain primary structural data
                    if isinstance(value, list):
                        # Preserve the structural information but ensure serializable format
                        html_props[key] = value  # Keep the full list
                        log.info(f"PRESERVED primary structural data '{key}' with {len(value)} elements")
                    else:
                        html_props[key] = value
                        log.debug(f"Added non-list {key}: {type(value)}")
                else:
                    # For other complex objects, convert to string but preserve structure
                    if isinstance(value, (dict, list)):
                        html_props[key] = json.dumps(value) if value else None
                        log.debug(f"Serialized complex HTML property {key}")
                    else:
                        html_props[key] = str(value) if value is not None else None
                        log.debug(f"Converted HTML property {key} to string")
            
            attributes["html"] = html_props
    
    # PRESERVE page, paragraph, and char_position as PRIMARY attributes
    for key in ["page", "paragraph", "char_position"]:
        if key in metadata and metadata[key]:
            log.debug(f"Processing PRIMARY attribute {key}: {metadata[key]}")
            
            if isinstance(metadata[key], dict):
                # Keep the full dictionary structure for primary attributes
                attributes[key] = metadata[key]
                log.info(f"PRESERVED primary attribute {key} with keys: {list(metadata[key].keys())}")
            elif isinstance(metadata[key], (list, str, int, float, bool)):
                # Keep primitive and list values as-is
                attributes[key] = metadata[key]
                log.info(f"PRESERVED primary attribute {key}: {type(metadata[key])}")
            else:
                # Convert complex objects to serializable format but preserve data
                try:
                    attributes[key] = json.loads(json.dumps(metadata[key])) if metadata[key] else None
                    log.info(f"PRESERVED primary attribute {key} after serialization")
                except (TypeError, ValueError):
                    attributes[key] = str(metadata[key])
                    log.warning(f"Converted primary attribute {key} to string due to serialization issues")
    
    # Log the final attributes structure
    log.info(f"Final attributes created with PRIMARY structural data preserved: {list(attributes.keys())}")
    
    return attributes


def extract_structural_columns(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Extract structural information to dedicated columns while preserving in attributes."""
    log.info("Extracting structural information to dedicated columns")
    
    # Extract structural information from attributes for each row
    def extract_row_structure(row):
        try:
            # Parse attributes if it's a string
            attrs = row.get('attributes', '{}')
            if isinstance(attrs, str):
                try:
                    attrs = json.loads(attrs)
                except:
                    attrs = {}
            
            # Extract page information
            if 'page' in attrs and isinstance(attrs['page'], dict):
                page_info = attrs['page']
                row[PAGE_ID] = page_info.get('id') or page_info.get('page_id')
                row[PAGE_NUMBER] = page_info.get('number') or page_info.get('page_num')
                log.debug(f"Extracted page info: ID={row[PAGE_ID]}, Number={row[PAGE_NUMBER]}")
            
            # Extract paragraph information  
            if 'paragraph' in attrs and isinstance(attrs['paragraph'], dict):
                para_info = attrs['paragraph']
                row[PARAGRAPH_ID] = para_info.get('id') or para_info.get('para_id')
                row[PARAGRAPH_NUMBER] = para_info.get('number') or para_info.get('para_num')
                log.debug(f"Extracted paragraph info: ID={row[PARAGRAPH_ID]}, Number={row[PARAGRAPH_NUMBER]}")
            
            # Extract character position information
            if 'char_position' in attrs and isinstance(attrs['char_position'], dict):
                char_info = attrs['char_position']
                row[CHAR_POSITION_START] = char_info.get('start') or char_info.get('char_start')
                row[CHAR_POSITION_END] = char_info.get('end') or char_info.get('char_end')
                log.debug(f"Extracted char positions: Start={row[CHAR_POSITION_START]}, End={row[CHAR_POSITION_END]}")
            
            # Also check HTML structure for additional information
            if 'html' in attrs and isinstance(attrs['html'], dict):
                html_info = attrs['html']
                
                # Look for page/paragraph arrays to extract specific chunk positions
                if 'pages' in html_info and isinstance(html_info['pages'], list):
                    # Extract page information based on chunk content or position
                    chunk_text = row.get('text', '')
                    page_match = find_chunk_page_info(chunk_text, html_info['pages'])
                    if page_match:
                        row[PAGE_ID] = page_match.get('page_id')
                        row[PAGE_NUMBER] = page_match.get('page_num')
                        log.debug(f"Found page match from HTML: ID={page_match.get('page_id')}, Number={page_match.get('page_num')}")
                
                if 'paragraphs' in html_info and isinstance(html_info['paragraphs'], list):
                    # Extract paragraph information based on chunk content or position
                    chunk_text = row.get('text', '')
                    para_match = find_chunk_paragraph_info(chunk_text, html_info['paragraphs'])
                    if para_match:
                        row[PARAGRAPH_ID] = para_match.get('para_id')
                        row[PARAGRAPH_NUMBER] = para_match.get('para_num')
                        row[CHAR_POSITION_START] = para_match.get('char_start')
                        row[CHAR_POSITION_END] = para_match.get('char_end')
                        log.debug(f"Found paragraph match from HTML: ID={para_match.get('para_id')}, Number={para_match.get('para_num')}")
            
        except Exception as e:
            log.warning(f"Error extracting structural information from row: {e}")
        
        return row
    
    # Apply extraction to each row
    log.info("Applying structural extraction to each row")
    aggregated = aggregated.apply(extract_row_structure, axis=1)
    
    # Log summary of extracted information
    structural_columns = [PAGE_ID, PAGE_NUMBER, PARAGRAPH_ID, PARAGRAPH_NUMBER, CHAR_POSITION_START, CHAR_POSITION_END]
    for col in structural_columns:
        non_null_count = aggregated[col].notna().sum()
        log.info(f"Extracted {non_null_count} values for {col}")
    
    return aggregated


def determine_chunk_position(chunk_text: str, html_meta: dict) -> dict:
    """Determine the position of a chunk within the document structure."""
    position_info = {}
    
    try:
        # Try to match chunk text with paragraphs to determine position
        if 'paragraphs' in html_meta and isinstance(html_meta['paragraphs'], list):
            for para in html_meta['paragraphs']:
                if isinstance(para, dict) and 'text' in para:
                    # Check if chunk text matches or is contained in paragraph
                    para_text = para['text']
                    if chunk_text.strip() in para_text or para_text in chunk_text.strip():
                        position_info['paragraph'] = {
                            'para_id': para.get('para_id'),
                            'para_num': para.get('para_num'),
                            'char_start': para.get('char_start'),
                            'char_end': para.get('char_end')
                        }
                        break
        
        # Try to determine page information
        if 'pages' in html_meta and isinstance(html_meta['pages'], list):
            # This would require more sophisticated logic to match chunks to pages
            # For now, we'll rely on the extraction from other metadata
            pass
            
    except Exception as e:
        log.debug(f"Error determining chunk position: {e}")
    
    return position_info


def find_chunk_page_info(chunk_text: str, pages: list) -> dict:
    """Find page information for a chunk based on its text content."""
    try:
        for page in pages:
            if isinstance(page, dict):
                # Check if page has text content that matches
                if 'text' in page and chunk_text.strip() in page['text']:
                    return {
                        'page_id': page.get('page_id'),
                        'page_num': page.get('page_num')
                    }
    except Exception as e:
        log.debug(f"Error finding page info: {e}")
    
    return {}


def find_chunk_paragraph_info(chunk_text: str, paragraphs: list) -> dict:
    """Find paragraph information for a chunk based on its text content."""
    
    try:
        for para in paragraphs:
            if isinstance(para, dict) and 'text' in para:
                # Check for exact or partial matches
                para_text = para['text']
                if chunk_text.strip() in para_text or para_text in chunk_text.strip():
                    return {
                        'para_id': para.get('para_id'),
                        'para_num': para.get('para_num'),
                        'char_start': para.get('char_start'),
                        'char_end': para.get('char_end')
                    }
    except Exception as e:
        log.debug(f"Error finding paragraph info: {e}")
    
    return {}