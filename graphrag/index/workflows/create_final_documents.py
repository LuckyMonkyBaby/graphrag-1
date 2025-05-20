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


def create_final_documents(input_df: pd.DataFrame, text_units: pd.DataFrame) -> pd.DataFrame:
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
        text_units_with_doc_ids.groupby("document_i