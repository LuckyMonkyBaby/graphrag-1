# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import TEXT_UNITS_FINAL_COLUMNS
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import (
    load_table_from_storage,
    storage_has_table,
    write_table_to_storage,
)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform the text units."""
    text_units = await load_table_from_storage("text_units", context.storage)
    final_entities = await load_table_from_storage("entities", context.storage)
    final_relationships = await load_table_from_storage(
        "relationships", context.storage
    )
    
    final_covariates = None
    if config.extract_claims.enabled and await storage_has_table(
        "covariates", context.storage
    ):
        final_covariates = await load_table_from_storage("covariates", context.storage)
    
    # Try to load original dataset for HTML attributes, but make it optional
    original_dataset = None
    if await storage_has_table("dataset", context.storage):
        original_dataset = await load_table_from_storage("dataset", context.storage)

    output = create_final_text_units(
        text_units,
        final_entities,
        final_relationships,
        final_covariates,
        original_dataset,
    )

    await write_table_to_storage(output, "text_units", context.storage)

    return WorkflowFunctionOutput(result=output)


def create_final_text_units(
    text_units: pd.DataFrame,
    final_entities: pd.DataFrame,
    final_relationships: pd.DataFrame,
    final_covariates: pd.DataFrame | None,
    original_dataset: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """All the steps to transform the text units."""
    selected = text_units.loc[:, ["id", "text", "document_ids", "n_tokens"]]
    selected["human_readable_id"] = selected.index + 1
    
    # If the text units already have attributes, include them
    if "attributes" in text_units.columns:
        # Parse JSON strings back to dictionaries if needed
        selected["attributes"] = text_units["attributes"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    else:
        selected["attributes"] = None

    # Add HTML attributes from original dataset if they exist
    if original_dataset is not None and "html_attributes" in original_dataset.columns:
        # Transform any JSON strings back to dictionaries
        if original_dataset["html_attributes"].dtype == 'object':
            original_dataset["html_attributes"] = original_dataset["html_attributes"].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x is not None else x
            )
        
        # Extract and merge HTML attributes
        html_attributes = _extract_html_attributes(selected, original_dataset)
        
        # Merge with existing attributes
        selected["attributes"] = selected.apply(
            lambda row: _merge_attributes(row["attributes"], html_attributes.get(row.name)), 
            axis=1
        )

    entity_join = _entities(final_entities)
    relationship_join = _relationships(final_relationships)

    entity_joined = _join(selected, entity_join)
    relationship_joined = _join(entity_joined, relationship_join)
    final_joined = relationship_joined

    if final_covariates is not None:
        covariate_join = _covariates(final_covariates)
        final_joined = _join(relationship_joined, covariate_join)
    else:
        final_joined["covariate_ids"] = [[] for i in range(len(final_joined))]

    aggregated = final_joined.groupby("id", sort=False).agg("first").reset_index()

    # If 'attributes' is not in the dataframe, add it as None
    if 'attributes' not in aggregated.columns:
        aggregated['attributes'] = None

    # Ensure all required columns from TEXT_UNITS_FINAL_COLUMNS are present
    for col in TEXT_UNITS_FINAL_COLUMNS:
        if col not in aggregated.columns:
            aggregated[col] = None
    
    # Convert attributes to serializable format
    aggregated["attributes"] = aggregated["attributes"].apply(
        lambda x: json.dumps(x) if x is not None else None
    )

    return aggregated.loc[
        :,
        TEXT_UNITS_FINAL_COLUMNS,
    ]


def _merge_attributes(existing_attrs: Optional[Dict], new_attrs: Optional[Dict]) -> Optional[Dict]:
    """Merge existing attributes with new HTML attributes."""
    if existing_attrs is None and new_attrs is None:
        return None
    
    if existing_attrs is None:
        return new_attrs
    
    if new_attrs is None:
        return existing_attrs
    
    # Create a copy to avoid modifying the original
    result = existing_attrs.copy() if isinstance(existing_attrs, dict) else {}
    
    # If new_attrs has html section, merge it with existing html section
    if "html" in new_attrs:
        if "html" not in result:
            result["html"] = {}
        
        # Update with new HTML attributes
        result["html"].update(new_attrs["html"])
    
    return result


def _extract_html_attributes(text_units: pd.DataFrame, original_dataset: pd.DataFrame) -> Dict[int, Dict]:
    """Extract HTML attributes for text units from the original dataset."""
    # Initialize attributes dict
    attributes = {}
    
    # Map document IDs to their HTML attributes
    doc_to_html = {}
    for _, row in original_dataset.iterrows():
        if "html_attributes" in row and row["html_attributes"] is not None:
            doc_to_html[row["id"]] = row["html_attributes"]
    
    # For each text unit, find its associated document and extract relevant HTML attributes
    for i, row in text_units.iterrows():
        text_attributes = {}
        
        # Get document IDs for this text unit
        doc_ids = row.get("document_ids", [])
        if doc_ids is None:
            continue
            
        # Find HTML attributes for each associated document
        for doc_id in doc_ids:
            if doc_id in doc_to_html:
                html_attrs = doc_to_html[doc_id]
                
                # Extract paragraph info for this text unit
                para_info = _find_paragraph_for_text(row["text"], html_attrs.get("paragraph_info", []))
                if para_info:
                    # Add paragraph and page information
                    text_attributes["html"] = {
                        "paragraph_id": para_info.get("para_id"),
                        "paragraph_num": para_info.get("para_num"),
                        "page_id": para_info.get("page_id"),
                    }
                    
                    # Add HTML tag info if available
                    html_tag_attrs = para_info.get("html_attributes", {})
                    if html_tag_attrs:
                        text_attributes["html"]["tag"] = html_tag_attrs.get("tag")
                        text_attributes["html"]["class"] = html_tag_attrs.get("class")
                        text_attributes["html"]["align"] = html_tag_attrs.get("align")
                    
                    break  # Found a match, no need to check other documents
        
        # If HTML attributes were found, update the attributes dict
        if text_attributes:
            attributes[i] = text_attributes
    
    return attributes


def _find_paragraph_for_text(text: str, paragraphs: list) -> Optional[Dict]:
    """Find the paragraph info that contains the given text."""
    if not text or not paragraphs:
        return None
    
    # Try exact match first
    for para in paragraphs:
        if para.get("text") == text:
            return para
    
    # Try substring matching if exact match fails
    for para in paragraphs:
        para_text = para.get("text", "")
        if para_text and (text in para_text or para_text in text):
            return para
    
    return None


def _entities(df: pd.DataFrame) -> pd.DataFrame:
    selected = df.loc[:, ["id", "text_unit_ids"]]
    unrolled = selected.explode(["text_unit_ids"]).reset_index(drop=True)

    return (
        unrolled.groupby("text_unit_ids", sort=False)
        .agg(entity_ids=("id", "unique"))
        .reset_index()
        .rename(columns={"text_unit_ids": "id"})
    )


def _relationships(df: pd.DataFrame) -> pd.DataFrame:
    selected = df.loc[:, ["id", "text_unit_ids"]]
    unrolled = selected.explode(["text_unit_ids"]).reset_index(drop=True)

    return (
        unrolled.groupby("text_unit_ids", sort=False)
        .agg(relationship_ids=("id", "unique"))
        .reset_index()
        .rename(columns={"text_unit_ids": "id"})
    )


def _covariates(df: pd.DataFrame) -> pd.DataFrame:
    selected = df.loc[:, ["id", "text_unit_id"]]

    return (
        selected.groupby("text_unit_id", sort=False)
        .agg(covariate_ids=("id", "unique"))
        .reset_index()
        .rename(columns={"text_unit_id": "id"})
    )


def _join(left, right):
    return left.merge(
        right,
        on="id",
        how="left",
        suffixes=["_1", "_2"],
    )