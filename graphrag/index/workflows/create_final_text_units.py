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
    
    # Initialize attributes with simple dict if present
    if "attributes" in text_units.columns:
        # Extract only the essential data
        selected["attributes"] = text_units["attributes"].apply(
            lambda x: simplify_attributes(x)
        )
    else:
        selected["attributes"] = None

    # Join with entities and relationships
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

    # Ensure all required columns from TEXT_UNITS_FINAL_COLUMNS are present
    for col in TEXT_UNITS_FINAL_COLUMNS:
        if col not in aggregated.columns:
            aggregated[col] = None

    return aggregated.loc[
        :,
        TEXT_UNITS_FINAL_COLUMNS,
    ]


def simplify_attributes(attributes):
    """Extract only essential data from attributes."""
    if attributes is None:
        return None
    
    # If it's a string, try to parse it, but only extract essential info
    if isinstance(attributes, str):
        try:
            attributes = json.loads(attributes)
        except:
            return None
    
    # If it's not a dictionary after parsing, return None
    if not isinstance(attributes, dict):
        return None
    
    # Create a simple structure with only essential fields
    result = {}
    
    # Extract HTML info if available
    if "html" in attributes:
        html_data = attributes["html"]
        if isinstance(html_data, dict):
            # Only keep simple scalar values
            simple_html = {}
            for key, value in html_data.items():
                # Only include simple fields
                if isinstance(value, (str, int, float, bool)) or value is None:
                    simple_html[key] = value
            
            if simple_html:
                result["html"] = simple_html
    
    return result if result else None


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