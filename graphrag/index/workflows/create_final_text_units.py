# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import json
import logging
from typing import Dict, List, Optional

import pandas as pd

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import (
    TEXT_UNITS_FINAL_COLUMNS,
    PAGE_ID, PAGE_NUMBER, 
    PARAGRAPH_ID, PARAGRAPH_NUMBER,
    CHAR_POSITION_START, CHAR_POSITION_END
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import (
    load_table_from_storage,
    storage_has_table,
    write_table_to_storage,
)

# Add logger
log = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform the text units."""
    log.info("Starting final text units workflow")
    
    # Load required tables
    text_units = await load_table_from_storage("text_units", context.storage)
    final_entities = await load_table_from_storage("entities", context.storage)
    final_relationships = await load_table_from_storage(
        "relationships", context.storage
    )
    
    # Load covariates if available
    final_covariates = None
    if config.extract_claims.enabled and await storage_has_table(
        "covariates", context.storage
    ):
        final_covariates = await load_table_from_storage("covariates", context.storage)

    # Process text units
    output = create_final_text_units(
        text_units,
        final_entities,
        final_relationships,
        final_covariates,
    )

    # Write results
    await write_table_to_storage(output, "text_units", context.storage)

    return WorkflowFunctionOutput(result=output)


def create_final_text_units(
    text_units: pd.DataFrame,
    final_entities: pd.DataFrame,
    final_relationships: pd.DataFrame,
    final_covariates: pd.DataFrame | None,
) -> pd.DataFrame:
    """All the steps to transform the text units."""
    # Select basic columns from text_units
    basic_columns = ["id", "text", "document_ids", "n_tokens"]
    selected = text_units.loc[:, [col for col in basic_columns if col in text_units.columns]].copy()
    
    # Add human_readable_id
    selected["human_readable_id"] = selected.index + 1
    
    # Check for HTML structure columns
    html_columns = [
        PAGE_ID, PAGE_NUMBER, PARAGRAPH_ID, PARAGRAPH_NUMBER,
        CHAR_POSITION_START, CHAR_POSITION_END
    ]
    
    # Add HTML structure columns if they exist in text_units
    for col in html_columns:
        if col in text_units.columns:
            # Ensure values are primitive types, not complex objects
            selected[col] = text_units[col].apply(
                lambda x: str(x) if isinstance(x, (dict, list)) else x
            )
        else:
            selected[col] = None
    
    # Process attributes column if it exists
    if "attributes" in text_units.columns:
        # Parse and simplify attributes - ensure it's always a JSON string
        selected["attributes"] = text_units["attributes"].apply(
            lambda x: json.dumps(simplify_attributes(x))
        )
    else:
        # Create empty attributes as JSON strings
        selected["attributes"] = ["{}"] * len(selected)
    
    # Join with entities and relationships
    entity_join = _entities(final_entities)
    relationship_join = _relationships(final_relationships)

    entity_joined = _join(selected, entity_join)
    relationship_joined = _join(entity_joined, relationship_join)
    final_joined = relationship_joined

    # Join with covariates if available
    if final_covariates is not None and not final_covariates.empty:
        covariate_join = _covariates(final_covariates)
        final_joined = _join(relationship_joined, covariate_join)
    else:
        if "covariate_ids" not in final_joined.columns:
            final_joined["covariate_ids"] = [[] for _ in range(len(final_joined))]

    # Ensure list columns are consistently lists before groupby
    for col in final_joined.columns:
        if col.endswith('_ids'):
            # Ensure list columns are consistently lists
            final_joined[col] = final_joined[col].apply(
                lambda x: [] if x is None else 
                          (x if isinstance(x, list) else [x])
            )
    
    # Group and aggregate results
    try:
        # Group by id and take first value (after ensuring consistent types)
        aggregated = final_joined.groupby("id", sort=False).agg("first").reset_index()
    except Exception as e:
        log.warning(f"Error in groupby operation: {e}. Using original dataframe.")
        aggregated = final_joined.copy()
    
    # Ensure all required columns are present
    for col in TEXT_UNITS_FINAL_COLUMNS:
        if col not in aggregated.columns:
            if col in ["document_ids", "entity_ids", "relationship_ids", "covariate_ids"]:
                aggregated[col] = [[] for _ in range(len(aggregated))]
            else:
                aggregated[col] = None
    
    # Ensure all lists are properly formatted for Parquet compatibility
    for col in ["document_ids", "entity_ids", "relationship_ids", "covariate_ids"]:
        if col in aggregated.columns:
            # Make sure each value is a list, never None or a primitive
            aggregated[col] = aggregated[col].apply(
                lambda x: [] if x is None else (x if isinstance(x, list) else [x])
            )
    
    # Handle complex types in the 'children' column if it exists
    # This appears to be causing one of the errors
    if 'children' in aggregated.columns:
        # Convert any complex objects to strings or remove them
        aggregated['children'] = aggregated['children'].apply(
            lambda x: [] if x is None else 
                     (x if isinstance(x, list) else [str(x)])
        )
    
    # Final check for any complex data types in non-list columns
    for col in aggregated.columns:
        if col not in ["document_ids", "entity_ids", "relationship_ids", "covariate_ids", "children"] and col != "attributes":
            # Ensure non-list columns don't contain complex objects
            if aggregated[col].apply(lambda x: isinstance(x, (dict, list))).any():
                # Convert any dict/list to strings
                aggregated[col] = aggregated[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                )

    # Return only the specified columns to avoid any extra problematic columns
    return aggregated.loc[:, TEXT_UNITS_FINAL_COLUMNS]


def simplify_attributes(attrs):
    """Simplify attributes to ensure proper serialization."""
    if attrs is None:
        return {}
    
    # Parse if string
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except json.JSONDecodeError:
            return {}
    
    # If not a dictionary, return empty dict
    if not isinstance(attrs, dict):
        return {}
    
    # Create simplified version keeping HTML structure fields
    result = {}
    
    # Include only essential fields, removing large arrays and ensuring flat structure
    if "html" in attrs and isinstance(attrs["html"], dict):
        html = attrs["html"]
        # Create a flattened structure with only primitive types
        result["html"] = {
            key: value for key, value in html.items()
            if key not in ["pages", "paragraphs"] and 
               isinstance(value, (str, int, float, bool, type(None)))
        }
    
    # Process page and paragraph data - flatten nested structures
    for key in ["page", "paragraph", "char_position"]:
        if key in attrs and attrs[key]:
            if isinstance(attrs[key], dict):
                # Extract only primitive values
                result[key] = {
                    k: v for k, v in attrs[key].items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                }
            else:
                # If it's already a primitive value, keep it
                if isinstance(attrs[key], (str, int, float, bool)):
                    result[key] = attrs[key]
    
    return result


def _entities(df: pd.DataFrame) -> pd.DataFrame:
    """Extract entity IDs for each text unit."""
    if df is None or df.empty or "text_unit_ids" not in df.columns:
        # Return empty DataFrame with expected structure
        return pd.DataFrame(columns=["id", "entity_ids"])
    
    selected = df.loc[:, ["id", "text_unit_ids"]]
    
    # Handle case where text_unit_ids might contain None/NaN
    selected = selected.dropna(subset=["text_unit_ids"])
    
    # Handle case where text_unit_ids might not be a list
    selected["text_unit_ids"] = selected["text_unit_ids"].apply(
        lambda x: x if isinstance(x, list) else ([x] if x is not None else [])
    )
    
    # Only explode if there are rows
    if selected.empty:
        return pd.DataFrame(columns=["id", "entity_ids"])
        
    unrolled = selected.explode(["text_unit_ids"]).reset_index(drop=True)

    return (
        unrolled.groupby("text_unit_ids", sort=False)
        .agg(entity_ids=("id", lambda x: list(pd.Series(x).unique())))  # Ensure list output
        .reset_index()
        .rename(columns={"text_unit_ids": "id"})
    )


def _relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Extract relationship IDs for each text unit."""
    if df is None or df.empty or "text_unit_ids" not in df.columns:
        # Return empty DataFrame with expected structure
        return pd.DataFrame(columns=["id", "relationship_ids"])
    
    selected = df.loc[:, ["id", "text_unit_ids"]]
    
    # Handle case where text_unit_ids might contain None/NaN
    selected = selected.dropna(subset=["text_unit_ids"])
    
    # Handle case where text_unit_ids might not be a list
    selected["text_unit_ids"] = selected["text_unit_ids"].apply(
        lambda x: x if isinstance(x, list) else ([x] if x is not None else [])
    )
    
    # Only explode if there are rows
    if selected.empty:
        return pd.DataFrame(columns=["id", "relationship_ids"])
        
    unrolled = selected.explode(["text_unit_ids"]).reset_index(drop=True)

    return (
        unrolled.groupby("text_unit_ids", sort=False)
        .agg(relationship_ids=("id", lambda x: list(pd.Series(x).unique())))  # Ensure list output
        .reset_index()
        .rename(columns={"text_unit_ids": "id"})
    )


def _covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Extract covariate IDs for each text unit."""
    if df is None or df.empty or "text_unit_id" not in df.columns:
        # Return empty DataFrame with expected structure
        return pd.DataFrame(columns=["id", "covariate_ids"])
    
    selected = df.loc[:, ["id", "text_unit_id"]]
    
    # Handle case where text_unit_id might contain None/NaN
    selected = selected.dropna(subset=["text_unit_id"])
    
    # Only process if there are rows
    if selected.empty:
        return pd.DataFrame(columns=["id", "covariate_ids"])

    return (
        selected.groupby("text_unit_id", sort=False)
        .agg(covariate_ids=("id", lambda x: list(pd.Series(x).unique())))  # Ensure list output
        .reset_index()
        .rename(columns={"text_unit_id": "id"})
    )


def _join(left, right):
    """Safely join two dataframes on id."""
    # If either dataframe is empty, return a copy of left with expected columns
    if left.empty or right.empty or "id" not in right.columns:
        # Add all columns from right to left with empty values
        result = left.copy()
        for col in right.columns:
            if col != "id" and col not in result.columns:
                if col.endswith("_ids"):
                    # For _ids columns, use empty lists
                    result[col] = [[] for _ in range(len(result))]
                else:
                    result[col] = None
        return result
    
    # Perform the merge
    return left.merge(
        right,
        on="id",
        how="left",
        suffixes=["_1", "_2"],
    )