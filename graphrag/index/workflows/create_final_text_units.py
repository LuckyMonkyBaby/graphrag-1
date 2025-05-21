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
    log.info(f"Creating final text units from {len(text_units) if text_units is not None else 'None'} input rows")
    
    # Select basic columns from text_units
    basic_columns = ["id", "text", "document_ids", "n_tokens"]
    selected = text_units.loc[:, [col for col in basic_columns if col in text_units.columns]].copy()
    log.info(f"Selected {len(selected)} rows with basic columns: {[c for c in basic_columns if c in text_units.columns]}")
    
    # Add human_readable_id
    selected["human_readable_id"] = selected.index + 1
    
    # Check for HTML structure columns
    html_columns = [
        PAGE_ID, PAGE_NUMBER, PARAGRAPH_ID, PARAGRAPH_NUMBER,
        CHAR_POSITION_START, CHAR_POSITION_END
    ]
    
    # Log available HTML structure columns
    present_html_columns = [col for col in html_columns if col in text_units.columns]
    log.info(f"Available HTML structure columns: {present_html_columns}")
    
    # Add HTML structure columns if they exist in text_units
    for col in html_columns:
        if col in text_units.columns:
            # Log a sample of the column values
            sample_values = text_units[col].head(3).tolist()
            log.debug(f"Column {col} sample values: {sample_values}")
            
            # Check for complex objects in the column
            complex_objects = text_units[col].apply(lambda x: isinstance(x, (dict, list))).sum()
            if complex_objects > 0:
                log.warning(f"Found {complex_objects} complex objects in column {col}, converting to strings")
            
            # Ensure values are primitive types, not complex objects
            selected[col] = text_units[col].apply(
                lambda x: str(x) if isinstance(x, (dict, list)) else x
            )
        else:
            selected[col] = None
    
    # Process attributes column if it exists
    if "attributes" in text_units.columns:
        log.info("Processing attributes column")
        
        # Log a sample of the attributes values
        sample_attrs = text_units["attributes"].head(3).tolist()
        log.debug(f"Attributes sample values: {sample_attrs}")
        
        # Parse and simplify attributes - ensure it's always a JSON string
        selected["attributes"] = text_units["attributes"].apply(
            lambda x: json.dumps(simplify_attributes(x))
        )
        
        # Log the parsed results for verification
        sample_parsed = selected["attributes"].head(3).tolist()
        log.debug(f"Parsed and simplified attributes: {sample_parsed}")
    else:
        log.info("No attributes column found, creating empty attributes")
        #def create_final_text_units(
    text_units: pd.DataFrame,
    final_entities: pd.DataFrame,
    final_relationships: pd.DataFrame,
    final_covariates: pd.DataFrame | None,
) -> pd.DataFrame:
    """All the steps to transform the text units."""
    log.info(f"Creating final text units from {len(text_units) if text_units is not None else 'None'} input rows")
    
    # Select basic columns from text_units
    basic_columns = ["id", "text", "document_ids", "n_tokens"]
    selected = text_units.loc[:, [col for col in basic_columns if col in text_units.columns]].copy()
    log.info(f"Selected {len(selected)} rows with basic columns: {[c for c in basic_columns if c in text_units.columns]}")
    
    # Add human_readable_id
    selected["human_readable_id"] = selected.index + 1
    
    # Check for HTML structure columns
    html_columns = [
        PAGE_ID, PAGE_NUMBER, PARAGRAPH_ID, PARAGRAPH_NUMBER,
        CHAR_POSITION_START, CHAR_POSITION_END
    ]
    
    # Log available HTML structure columns
    present_html_columns = [col for col in html_columns if col in text_units.columns]
    log.info(f"Available HTML structure columns: {present_html_columns}")
    
    # Add HTML structure columns if they exist in text_units
    for col in html_columns:
        if col in text_units.columns:
            # Log a sample of the column values
            sample_values = text_units[col].head(3).tolist()
            log.debug(f"Column {col} sample values: {sample_values}")
            
            # Check for complex objects in the column
            complex_objects = text_units[col].apply(lambda x: isinstance(x, (dict, list))).sum()
            if complex_objects > 0:
                log.warning(f"Found {complex_objects} complex objects in column {col}, converting to strings")
            
            # Ensure values are primitive types, not complex objects
            selected[col] = text_units[col].apply(
                lambda x: str(x) if isinstance(x, (dict, list)) else x
            )
        else:
            selected[col] = None
    
    # Process attributes column if it exists
    if "attributes" in text_units.columns:
        log.info("Processing attributes column")
        
        # Log a sample of the attributes values
        sample_attrs = text_units["attributes"].head(3).tolist()
        log.debug(f"Attributes sample values: {sample_attrs}")
        
        # Parse and simplify attributes - ensure it's always a JSON string
        selected["attributes"] = text_units["attributes"].apply(
            lambda x: json.dumps(simplify_attributes(x))
        )
        
        # Log the parsed results for verification
        sample_parsed = selected["attributes"].head(3).tolist()
        log.debug(f"Parsed and simplified attributes: {sample_parsed}")
    else:
        log.info("No attributes column found, creating empty attributes")
        selected["attributes"] = ["{}"] * len(selected)
    
    # Join with entities and relationships
    log.info("Joining with entities and relationships")
    entity_join = _entities(final_entities)
    log.debug(f"Entity join result: {len(entity_join)} rows")
    
    relationship_join = _relationships(final_relationships)
    log.debug(f"Relationship join result: {len(relationship_join)} rows")

    entity_joined = _join(selected, entity_join)
    log.debug(f"After entity join: {len(entity_joined)} rows")
    
    relationship_joined = _join(entity_joined, relationship_join)
    log.debug(f"After relationship join: {len(relationship_joined)} rows")
    
    final_joined = relationship_joined

    # Join with covariates if available
    if final_covariates is not None and not final_covariates.empty:
        log.info(f"Joining with {len(final_covariates)} covariates")
        covariate_join = _covariates(final_covariates)
        log.debug(f"Covariate join result: {len(covariate_join)} rows")
        final_joined = _join(relationship_joined, covariate_join)
        log.debug(f"After covariate join: {len(final_joined)} rows")
    else:
        log.info("No covariates available for joining")
        if "covariate_ids" not in final_joined.columns:
            final_joined["covariate_ids"] = [[] for _ in range(len(final_joined))]

    # Log column types before ensuring consistency
    log.info("Column types before normalization:")
    for col in final_joined.columns:
        # Get the first non-null value to check its type
        sample_value = final_joined[col].dropna().iloc[0] if not final_joined[col].dropna().empty else None
        log.debug(f"Column {col}: dtype={final_joined[col].dtype}, sample value type={type(sample_value)}")

    # Ensure list columns are consistently lists before groupby
    log.info("Normalizing list columns")
    for col in final_joined.columns:
        if col.endswith('_ids'):
            # Check for non-list values
            non_list_count = final_joined[col].apply(
                lambda x: not isinstance(x, list) and x is not None
            ).sum()
            
            if non_list_count > 0:
                log.warning(f"Found {non_list_count} non-list values in column {col}, converting to lists")
            
            # Ensure list columns are consistently lists
            final_joined[col] = final_joined[col].apply(
                lambda x: [] if x is None else 
                          (x if isinstance(x, list) else [x])
            )
    
    # Group and aggregate results
    log.info("Grouping and aggregating results")
    try:
        # Group by id and take first value (after ensuring consistent types)
        aggregated = final_joined.groupby("id", sort=False).agg("first").reset_index()
        log.info(f"Successfully grouped data, resulting in {len(aggregated)} rows")
    except Exception as e:
        log.warning(f"Error in groupby operation: {e}. Using original dataframe.")
        aggregated = final_joined.copy()
    
    # Ensure all required columns are present
    missing_columns = [col for col in TEXT_UNITS_FINAL_COLUMNS if col not in aggregated.columns]
    if missing_columns:
        log.info(f"Adding missing columns: {missing_columns}")
    
    for col in TEXT_UNITS_FINAL_COLUMNS:
        if col not in aggregated.columns:
            if col in ["document_ids", "entity_ids", "relationship_ids", "covariate_ids"]:
                aggregated[col] = [[] for _ in range(len(aggregated))]
                log.debug(f"Added missing list column {col}")
            else:
                aggregated[col] = None
                log.debug(f"Added missing column {col} with None values")
    
    # Ensure all lists are properly formatted for Parquet compatibility
    log.info("Final normalization of list columns for Parquet compatibility")
    for col in ["document_ids", "entity_ids", "relationship_ids", "covariate_ids"]:
        if col in aggregated.columns:
            # Make sure each value is a list, never None or a primitive
            mixed_types = aggregated[col].apply(
                lambda x: not (x is None or isinstance(x, list))
            ).sum()
            
            if mixed_types > 0:
                log.warning(f"Found {mixed_types} mixed type values in {col} during final check")
            
            aggregated[col] = aggregated[col].apply(
                lambda x: [] if x is None else (x if isinstance(x, list) else [x])
            )
            log.debug(f"Normalized column {col} to ensure list values")
    
    # Handle complex types in the 'children' column if it exists
    # This appears to be causing one of the errors
    if 'children' in aggregated.columns:
        log.info("Processing 'children' column which has caused errors")
        # Count complex objects
        complex_objects = aggregated['children'].apply(
            lambda x: not (x is None or isinstance(x, list) or isinstance(x, (str, int, float, bool)))
        ).sum()
        
        if complex_objects > 0:
            log.warning(f"Found {complex_objects} complex objects in 'children' column")
        
        # Convert any complex objects to strings or remove them
        aggregated['children'] = aggregated['children'].apply(
            lambda x: [] if x is None else 
                     (x if isinstance(x, list) else [str(x)])
        )
        log.info(f"Normalized 'children' column to ensure list values")
    
    # Final check for any complex data types in non-list columns
    log.info("Final check for complex data types in non-list columns")
    for col in aggregated.columns:
        if col not in ["document_ids", "entity_ids", "relationship_ids", "covariate_ids", "children"] and col != "attributes":
            # Check for complex objects
            complex_count = aggregated[col].apply(
                lambda x: isinstance(x, (dict, list))
            ).sum()
            
            if complex_count > 0:
                log.warning(f"Found {complex_count} complex objects in column {col}, converting to strings")
                # Convert any dict/list to strings
                aggregated[col] = aggregated[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                )

    # Log final column types
    log.info("Final column types before returning:")
    for col in TEXT_UNITS_FINAL_COLUMNS:
        if col in aggregated.columns:
            sample_value = aggregated[col].dropna().iloc[0] if not aggregated[col].dropna().empty else None
            log.debug(f"Column {col}: dtype={aggregated[col].dtype}, sample value type={type(sample_value)}")

    # Return only the specified columns to avoid any extra problematic columns
    result = aggregated.loc[:, TEXT_UNITS_FINAL_COLUMNS]
    log.info(f"Final result: {len(result)} rows with {len(TEXT_UNITS_FINAL_COLUMNS)} columns")
    
    return result


def simplify_attributes(attrs):
    """Simplify attributes to ensure proper serialization."""
    log.info("Simplifying attributes for Parquet serialization")
    
    # Handle None values
    if attrs is None:
        log.debug("Attributes is None, returning empty dict")
        return {}
    
    # Parse string attributes
    if isinstance(attrs, str):
        try:
            log.debug(f"Parsing attributes JSON string: {attrs[:100]}..." if len(attrs) > 100 else attrs)
            attrs = json.loads(attrs)
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse attributes JSON: {e}")
            return {}
    
    # If not a dictionary, return empty dict
    if not isinstance(attrs, dict):
        log.debug(f"Attributes is not a dict but {type(attrs)}, returning empty dict")
        return {}
    
    # Log original structure
    try:
        attrs_sample = str(attrs)[:500] + "..." if len(str(attrs)) > 500 else str(attrs)
        log.debug(f"Original attributes structure: {attrs_sample}")
    except Exception as e:
        log.warning(f"Error logging attributes: {e}")
    
    # Create simplified version keeping HTML structure fields
    result = {}
    
    # Include only essential fields, removing large arrays and ensuring flat structure
    if "html" in attrs and isinstance(attrs["html"], dict):
        html = attrs["html"]
        log.debug(f"Processing HTML attributes with keys: {list(html.keys())}")
        
        # Create a flattened structure with only primitive types
        html_result = {}
        for key, value in html.items():
            if key not in ["pages", "paragraphs"] and isinstance(value, (str, int, float, bool, type(None))):
                html_result[key] = value
                log.debug(f"Keeping HTML property {key}: {value}")
            else:
                if key in ["pages", "paragraphs"]:
                    array_len = len(value) if isinstance(value, list) else "not a list"
                    log.debug(f"Removing large array '{key}' with {array_len} elements")
                else:
                    log.debug(f"Removing non-primitive HTML property {key}: {type(value)}")
        
        result["html"] = html_result
    
    # Process page and paragraph data - flatten nested structures
    for key in ["page", "paragraph", "char_position"]:
        if key in attrs and attrs[key]:
            log.debug(f"Processing {key} attribute: {attrs[key]}")
            if isinstance(attrs[key], dict):
                # Extract only primitive values
                key_result = {}
                for k, v in attrs[key].items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        key_result[k] = v
                        log.debug(f"Keeping {key}.{k}: {v}")
                    else:
                        log.debug(f"Removing non-primitive {key}.{k}: {type(v)}")
                result[key] = key_result
            else:
                # If it's already a primitive value, keep it
                if isinstance(attrs[key], (str, int, float, bool)):
                    result[key] = attrs[key]
                    log.debug(f"Keeping primitive {key}: {attrs[key]}")
                else:
                    log.debug(f"Removing non-primitive {key}: {type(attrs[key])}")
    
    # Log the final result
    log.info(f"Simplified attributes: {result}")
    
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