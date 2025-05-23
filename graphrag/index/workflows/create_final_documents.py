# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import json
from typing import Any, Dict, Optional

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

    # Load the original dataset from storage
    input_df = await load_table_from_storage("dataset", context.storage)

    output = create_final_documents(input_df, base_text_units)

    await write_table_to_storage(output, "documents", context.storage)

    return WorkflowFunctionOutput(result=output)


def create_final_documents(
    input_df: pd.DataFrame, text_units: pd.DataFrame
) -> pd.DataFrame:
    """All the steps to transform the documents."""
    if "metadata" not in input_df.columns:
        input_df["metadata"] = None

    # Save a copy of input_df before processing
    result_df = input_df.copy()

    # Process HTML metadata if available, but keep it simple
    if "html_attributes" in result_df.columns:
        # Extract key HTML info into simplified structure
        result_df["metadata"] = result_df.apply(
            lambda row: extract_html_metadata(row), axis=1
        )

        # Remove the original complex html_attributes column
        result_df = result_df.drop(columns=["html_attributes"])

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
        result_df,
        text_units_by_doc,
        on="id",
        how="left",
    )

    # Add human readable ID
    merged["human_readable_id"] = [f"doc_{i + 1}" for i in range(len(merged))]

    # Select and order columns according to the schema
    output = merged.loc[:, DOCUMENTS_FINAL_COLUMNS].copy()

    return output


def extract_html_metadata(row: pd.Series) -> Dict[str, Any]:
    """Extract simplified HTML metadata."""
    # Start with existing metadata
    metadata = {}
    if row.get("metadata") is not None:
        if isinstance(row.get("metadata"), dict):
            metadata = row.get("metadata")
        else:
            # Don't try to parse strings or complex objects
            metadata = {"original_metadata": str(row.get("metadata"))}

    # Don't process if no html_attributes
    if row.get("html_attributes") is None:
        return metadata

    # Create simple HTML structure info
    html_structure = {
        "has_html_structure": True,
        "doc_type": None,
        "pages": [],
    }

    # Extract basic properties safely
    html_attrs = row.get("html_attributes", {})

    # Add document type if available
    if isinstance(html_attrs, dict) and html_attrs.get("doc_type"):
        html_structure["doc_type"] = str(html_attrs.get("doc_type"))

    # Add page IDs if available, but keep it simple
    if isinstance(html_attrs, dict) and "page_info" in html_attrs:
        pages = html_attrs.get("page_info", [])
        if isinstance(pages, list):
            # Just extract page IDs and numbers
            page_list = []
            for page in pages:
                if isinstance(page, dict):
                    page_id = page.get("page_id")
                    page_num = page.get("page_num")
                    if page_id:
                        page_list.append({"id": str(page_id), "num": page_num})

            html_structure["pages"] = page_list
            html_structure["page_count"] = len(page_list)

    # Add to metadata
    metadata["html"] = html_structure

    return metadata
