# Fix to apply to graphrag/index/workflows/create_base_text_units.py

# Modify the create_base_text_units function to ensure consistent types
# and proper serialization of attributes

import json
from typing import Any, Dict, List, Optional, cast

import pandas as pd


from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.chunking_config import ChunkStrategyType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.operations.chunk_text.chunk_text import chunk_text
from graphrag.index.operations.chunk_text.strategies import get_encoding_fn
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.logger.progress import Progress
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform base text_units."""
    documents = await load_table_from_storage("documents", context.storage)

    chunks = config.chunks

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

    # Store the original dataset for later use
    # Make a copy to ensure it's serializable
    documents_copy = documents.copy()
    
    # Convert any problematic columns to JSON strings
    if "html_attributes" in documents_copy.columns:
        documents_copy["html_attributes"] = documents_copy["html_attributes"].apply(
            lambda x: json.dumps(x) if x is not None else None
        )
    
    await write_table_to_storage(documents_copy, "dataset", context.storage)
    await write_table_to_storage(output, "text_units", context.storage)

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
    sort = documents.sort_values(by=["id"], ascending=[True])

    # Preserve HTML attributes if they exist
    if "html_attributes" not in sort.columns:
        sort["html_attributes"] = None
    
    sort["text_with_ids"] = list(
        zip(*[sort[col] for col in ["id", "text"]], strict=True)
    )

    callbacks.progress(Progress(percent=0))

    agg_dict = {"text_with_ids": list}
    if "metadata" in documents:
        agg_dict["metadata"] = "first"  # type: ignore
    
    # Add HTML attributes to aggregation
    if "html_attributes" in documents:
        agg_dict["html_attributes"] = "first"  # type: ignore

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

    def chunker(row: dict[str, Any]) -> Any:
        line_delimiter = ".\n"
        metadata_str = ""
        metadata_tokens = 0

        if prepend_metadata and "metadata" in row:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if isinstance(metadata, dict):
                metadata_str = (
                    line_delimiter.join(f"{k}: {v}" for k, v in metadata.items())
                    + line_delimiter
                )

            if chunk_size_includes_metadata:
                encode, _ = get_encoding_fn(encoding_model)
                metadata_tokens = len(encode(metadata_str))
                if metadata_tokens >= size:
                    message = "Metadata tokens exceeds the maximum tokens per chunk. Please increase the tokens per chunk."
                    raise ValueError(message)

        chunked = chunk_text(
            pd.DataFrame([row]).reset_index(drop=True),
            column="texts",
            size=size - metadata_tokens,
            overlap=overlap,
            encoding_model=encoding_model,
            strategy=strategy,
            callbacks=callbacks,
        )[0]

        if prepend_metadata:
            for index, chunk in enumerate(chunked):
                if isinstance(chunk, str):
                    chunked[index] = metadata_str + chunk
                else:
                    chunked[index] = (
                        (chunk[0], metadata_str + chunk[1], chunk[2]) if chunk else None
                    )

        row["chunks"] = chunked
        return row

    aggregated = aggregated.apply(lambda row: chunker(row), axis=1)

    columns_to_keep = [*group_by_columns, "chunks"]
    
    # Add HTML attributes if they exist
    if "html_attributes" in aggregated.columns:
        columns_to_keep.append("html_attributes")
    
    aggregated = cast("pd.DataFrame", aggregated[columns_to_keep])
    aggregated = aggregated.explode("chunks")
    aggregated.rename(
        columns={
            "chunks": "chunk",
        },
        inplace=True,
    )
    aggregated["id"] = aggregated.apply(
        lambda row: gen_sha512_hash(row, ["chunk"]), axis=1
    )
    
    # Extract document_ids, text, and n_tokens from chunk
    chunk_df = pd.DataFrame(
        aggregated["chunk"].tolist(), 
        index=aggregated.index,
        columns=["document_ids", "text", "n_tokens"]
    )
    
    # Handle case where chunk_df might be empty
    if not chunk_df.empty:
        for col in ["document_ids", "text", "n_tokens"]:
            aggregated[col] = chunk_df[col]
    else:
        aggregated["document_ids"] = None
        aggregated["text"] = None
        aggregated["n_tokens"] = None
    
    # Add HTML structure tracking for chunks
    if "html_attributes" in aggregated.columns:
        # Create attributes column with HTML information
        aggregated["attributes"] = aggregated.apply(
            lambda row: _create_html_attributes(row), axis=1
        )
        
        # Convert attributes to JSON string to ensure it's serializable
        aggregated["attributes"] = aggregated["attributes"].apply(
            lambda x: json.dumps(x) if x is not None else None
        )
    else:
        # Ensure attributes column exists with consistent None values
        aggregated["attributes"] = None
    
    # Filter out rows with no text and reset index
    result = cast(
        "pd.DataFrame", aggregated[aggregated["text"].notna()].reset_index(drop=True)
    )
    
    # Ensure consistent types for list columns to avoid Parquet serialization issues
    # Make sure document_ids is always a list
    result["document_ids"] = result["document_ids"].apply(
        lambda x: [] if x is None else (x if isinstance(x, list) else [x])
    )
    
    return result


def _create_html_attributes(row: pd.Series) -> Optional[Dict[str, Any]]:
    """Create attributes dictionary with HTML information."""
    if row.get("html_attributes") is None:
        return None
    
    # Extract HTML attributes
    html_attrs = row.get("html_attributes", {})
    
    # If html_attrs is a string (JSON), parse it
    if isinstance(html_attrs, str):
        try:
            html_attrs = json.loads(html_attrs)
        except json.JSONDecodeError:
            return None
    
    # Get text content for this chunk
    chunk_text = row.get("text", "")
    if not chunk_text:
        return None
    
    # Try to find the paragraph that matches this chunk
    matching_paragraph = None
    if "paragraph_info" in html_attrs:
        paragraphs = html_attrs.get("paragraph_info", [])
        for para in paragraphs:
            para_text = para.get("text", "")
            if para_text and (para_text in chunk_text or chunk_text in para_text):
                matching_paragraph = para
                break
    
    # Determine the page for this chunk
    page_id = None
    if matching_paragraph:
        page_id = matching_paragraph.get("page_id")
    elif "page_info" in html_attrs:
        pages = html_attrs.get("page_info", [])
        # Find the last page before this chunk
        if pages:
            page_id = pages[-1].get("page_id")
    
    # Create attributes dictionary
    attributes = {
        "html": {
            "page_id": page_id,
        }
    }
    
    # Add paragraph info if available, but ensure it's a simple structure
    if matching_paragraph:
        # Extract only simple attributes to avoid nested complex structures
        attributes["html"]["paragraph_id"] = matching_paragraph.get("para_id")
        attributes["html"]["paragraph_num"] = matching_paragraph.get("para_num")
        
        # Extract HTML tag info
        html_tag_attrs = matching_paragraph.get("html_attributes", {})
        if html_tag_attrs:
            attributes["html"]["tag"] = html_tag_attrs.get("tag")
            attributes["html"]["class"] = html_tag_attrs.get("class")
            attributes["html"]["align"] = html_tag_attrs.get("align")
    
    return attributes