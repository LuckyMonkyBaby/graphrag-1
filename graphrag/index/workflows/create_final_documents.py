# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import pandas as pd

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import DOCUMENTS_FINAL_COLUMNS
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform the documents."""
    base_text_units = await load_table_from_storage("text_units", context.storage)
    
    # Load the original dataset from storage instead of context.pipeline
    input_df = await load_table_from_storage("dataset", context.storage)
    
    output = create_final_documents(input_df, base_text_units)
    
    await write_table_to_storage(output, "documents", context.storage)
    
    return WorkflowFunctionOutput(result=output)


def create_final_documents(input_df: pd.DataFrame, text_units: pd.DataFrame) -> pd.DataFrame:
    """All the steps to transform the documents.
    
    This function prepares the final document table for persistence by:
    1. Mapping document IDs to their corresponding text units
    2. Preserving any document metadata, including HTML-specific metadata 
    3. Ensuring the output schema matches the expected document format
    """
    if "metadata" not in input_df.columns:
        input_df["metadata"] = None
    
    # Process HTML metadata if it exists
    input_df["metadata"] = input_df.apply(
        lambda row: _process_html_metadata(row), axis=1
    )
    
    # Get text unit IDs for each document
    text_units_with_doc_ids = text_units.loc[:, ["id", "document_ids"]]
    text_units_with_doc_ids = text_units_with_doc_ids.explode("document_ids")
    text_units_by_doc = (
        text_units_with_doc_ids.groupby("document_ids", sort=False)
        .agg(text_unit_ids=("id", "unique"))
        .reset_index()
        .rename(columns={"document_ids": "id"})
    )
    
    # Merge document and text unit information
    merged = pd.merge(
        input_df,
        text_units_by_doc,
        on="id",
        how="left",
    )
    
    # Add human readable ID
    merged["human_readable_id"] = [f"doc_{i+1}" for i in range(len(merged))]
    
    # Select and order columns according to the schema
    output = merged.loc[:, DOCUMENTS_FINAL_COLUMNS].copy()
    
    return output


def _process_html_metadata(row):
    """Process and enhance HTML metadata for document preservation."""
    metadata = row.get("metadata", {})
    
    # If the metadata is None, create an empty dict
    if metadata is None:
        metadata = {}
    
    # Add HTML-specific attributes if they exist
    html_attributes = row.get("html_attributes", {})
    if html_attributes:
        # Create a specific HTML section in the metadata
        metadata["html"] = {
            "page_info": html_attributes.get("page_info", []),
            "paragraph_info": html_attributes.get("paragraph_info", []),
        }
    
    return metadata