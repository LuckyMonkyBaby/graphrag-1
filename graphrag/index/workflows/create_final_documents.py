# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import json
import logging
from typing import Any, Dict, Optional

import pandas as pd

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import DOCUMENTS_FINAL_COLUMNS
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage, storage_has_table

log = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform the documents."""
    log.info("Starting create_final_documents workflow")
    
    base_text_units = await load_table_from_storage("text_units", context.storage)
    
    # Load the original dataset from storage - handle both possible names
    input_df = None
    try:
        # First try to load "dataset"
        if await storage_has_table("dataset", context.storage):
            log.info("Loading documents from 'dataset' table")
            input_df = await load_table_from_storage("dataset", context.storage)
        else:
            # Fall back to "documents"
            log.info("'dataset' table not found, loading from 'documents' table")
            input_df = await load_table_from_storage("documents", context.storage)
    except Exception as e:
        log.error(f"Could not load input documents: {e}")
        # Create a minimal input_df from text_units if no documents table exists
        log.info("Creating minimal document structure from text_units")
        unique_doc_ids = []
        for doc_ids in base_text_units["document_ids"]:
            if isinstance(doc_ids, list):
                unique_doc_ids.extend(doc_ids)
            else:
                unique_doc_ids.append(doc_ids)
        
        unique_doc_ids = list(set(unique_doc_ids))
        input_df = pd.DataFrame({
            "id": unique_doc_ids,
            "text": [""] * len(unique_doc_ids),
            "title": [f"Document {i+1}" for i in range(len(unique_doc_ids))],
            "metadata": [None] * len(unique_doc_ids)
        })
        log.info(f"Created minimal document structure with {len(input_df)} documents")

    output = create_final_documents(input_df, base_text_units)

    await write_table_to_storage(output, "documents", context.storage)
    log.info(f"Successfully wrote {len(output)} final documents to storage")

    return WorkflowFunctionOutput(result=output)


def create_final_documents(
    input_df: pd.DataFrame, text_units: pd.DataFrame
) -> pd.DataFrame:
    """All the steps to transform the documents."""
    log.info(f"Creating final documents from {len(input_df)} input documents and {len(text_units)} text units")
    
    if "metadata" not in input_df.columns:
        input_df["metadata"] = None

    # Save a copy of input_df before processing
    result_df = input_df.copy()

    # Process metadata - handle any file type
    log.info("Processing document metadata")
    result_df["metadata"] = result_df["metadata"].apply(
        lambda x: process_existing_metadata(x)
    )

    # Get text unit IDs for each document
    log.info("Mapping text units to documents")
    text_units_with_doc_ids = text_units.loc[:, ["id", "document_ids"]]
    
    # Handle the document_ids column properly
    expanded_rows = []
    for idx, row in text_units_with_doc_ids.iterrows():
        doc_ids = row["document_ids"]
        if isinstance(doc_ids, list):
            for doc_id in doc_ids:
                expanded_rows.append({"id": row["id"], "document_ids": doc_id})
        elif doc_ids is not None:
            expanded_rows.append({"id": row["id"], "document_ids": doc_ids})
    
    if expanded_rows:
        exploded_df = pd.DataFrame(expanded_rows)
        text_units_by_doc = (
            exploded_df.groupby("document_ids", sort=False)
            .agg(text_unit_ids=("id", lambda x: list(x.unique())))
            .reset_index()
            .rename(columns={"document_ids": "id"})
        )
    else:
        # No text units found, create empty mapping
        text_units_by_doc = pd.DataFrame(columns=["id", "text_unit_ids"])
        log.warning("No text unit mappings found")

    # Merge document and text unit information
    log.info("Merging document and text unit information")
    merged = pd.merge(
        result_df,
        text_units_by_doc,
        on="id",
        how="left",
    )
    
    # Fill missing text_unit_ids with empty lists
    merged["text_unit_ids"] = merged["text_unit_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Add human readable ID
    merged["human_readable_id"] = [f"doc_{i + 1}" for i in range(len(merged))]

    # Ensure all required columns are present with sensible defaults
    for col in DOCUMENTS_FINAL_COLUMNS:
        if col not in merged.columns:
            if col == "text_unit_ids":
                merged[col] = [[] for _ in range(len(merged))]
            elif col in ["file_path", "file_type"]:
                # Try to extract from existing data
                if col == "file_path":
                    merged[col] = merged.get("title", None)
                elif col == "file_type":
                    merged[col] = "unknown"  # Default file type
            else:
                merged[col] = None
            log.info(f"Added missing column: {col}")

    # Select and order columns according to the schema
    log.info("Selecting final columns")
    available_columns = [col for col in DOCUMENTS_FINAL_COLUMNS if col in merged.columns]
    output = merged.loc[:, available_columns].copy()

    # Ensure metadata is JSON serializable
    if "metadata" in output.columns:
        output["metadata"] = output["metadata"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else (x if x is not None else "{}")
        )

    log.info(f"Final output: {len(output)} documents with columns: {list(output.columns)}")
    return output


def process_existing_metadata(metadata):
    """Process existing metadata to ensure it's properly formatted for any file type."""
    if metadata is None:
        return {}
    
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except json.JSONDecodeError:
            return {"raw": metadata}
    
    if isinstance(metadata, dict):
        return metadata
    
    return {"raw": str(metadata)}