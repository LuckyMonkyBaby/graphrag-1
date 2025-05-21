# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import json
from typing import Any, Dict, List, Optional
import logging

import pandas as pd

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import (
    TEXT_UNITS_FINAL_COLUMNS, 
    TEXT_UNITS_BASIC_COLUMNS,
    HTML_STRUCTURE_FIELDS,
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


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform the text units."""
    log = logging.getLogger(__name__)
    
    try:
        text_units = await load_table_from_storage("text_units", context.storage)
        final_entities = await load_table_from_storage("entities", context.storage)
        final_relationships = await load_table_from_storage(
            "relationships", context.storage
        )
        
        final_covariates = None
        if config.extract_claims.enabled and await storage_has_table(
            "covariates", context.storage
        ):
            try:
                final_covariates = await load_table_from_storage("covariates", context.storage)
            except Exception as e:
                log.warning(f"Failed to load covariates table: {e}")
        
        # Try to load original dataset for HTML attributes, but make it optional
        original_dataset = None
        if await storage_has_table("dataset", context.storage):
            try:
                original_dataset = await load_table_from_storage("dataset", context.storage)
            except Exception as e:
                log.warning(f"Failed to load dataset table: {e}")

        # Process text units with HTML structure information
        output = create_final_text_units(
            text_units,
            final_entities,
            final_relationships,
            final_covariates,
            original_dataset,
        )

        # Write the output with both standard columns and HTML structure
        await write_table_to_storage(output, "text_units", context.storage)
        
        return WorkflowFunctionOutput(result=output)
        
    except Exception as e:
        log.error(f"Error in text_units workflow: {e}")
        # In case of error, try to provide at least a minimal valid output
        minimal_output = pd.DataFrame(columns=TEXT_UNITS_FINAL_COLUMNS)
        return WorkflowFunctionOutput(result=minimal_output)


def create_final_text_units(
    text_units: pd.DataFrame,
    final_entities: pd.DataFrame,
    final_relationships: pd.DataFrame,
    final_covariates: pd.DataFrame | None,
    original_dataset: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """All the steps to transform the text units."""
    log = logging.getLogger(__name__)
    
    # Maintain backward compatibility: ensure we have all expected columns
    required_columns = ["id", "text", "document_ids", "n_tokens"]
    selected_columns = [col for col in required_columns if col in text_units.columns]
    
    # Make sure we have at least id and text columns
    if "id" not in selected_columns or "text" not in selected_columns:
        raise ValueError("Text units must have at least 'id' and 'text' columns")
    
    selected = text_units.loc[:, selected_columns].copy()
    
    # Add missing columns if needed
    for col in required_columns:
        if col not in selected.columns:
            selected[col] = None
    
    # Add human_readable_id
    selected["human_readable_id"] = selected.index + 1
    
    # Initialize HTML structure fields 
    for field in HTML_STRUCTURE_FIELDS:
        selected[field] = None
    
    # Process attributes if present
    if "attributes" in text_units.columns:
        # Extract attributes and HTML structure
        for idx, row in text_units.iterrows():
            attrs = row["attributes"]
            
            # Parse attributes if they're in string format
            if isinstance(attrs, str):
                try:
                    attrs = json.loads(attrs)
                    log.debug(f"Successfully parsed attributes for row {idx}")
                except Exception as e:
                    log.warning(f"Failed to parse attributes for row {idx}: {e}")
                    continue
            
            if not isinstance(attrs, dict):
                log.warning(f"Attributes for row {idx} is not a dictionary: {type(attrs)}")
                continue
            
            # Extract page info with better error handling
            if "page" in attrs and attrs["page"] and isinstance(attrs["page"], dict):
                try:
                    # Convert page number to int if it's a numeric string
                    page_number = attrs["page"].get("number")
                    if page_number is not None:
                        try:
                            page_number = int(page_number)
                        except (ValueError, TypeError):
                            pass
                    
                    selected.at[idx, PAGE_ID] = attrs["page"].get("id")
                    selected.at[idx, PAGE_NUMBER] = page_number
                    log.debug(f"Row {idx}: Extracted page {attrs['page'].get('id')}, number {page_number}")
                except Exception as e:
                    log.warning(f"Error extracting page info for row {idx}: {e}")
            
            # Extract paragraph info with better error handling
            if "paragraph" in attrs and attrs["paragraph"] and isinstance(attrs["paragraph"], dict):
                try:
                    # Convert paragraph number to int if it's a numeric string
                    para_number = attrs["paragraph"].get("number")
                    if para_number is not None:
                        try:
                            para_number = int(para_number)
                        except (ValueError, TypeError):
                            pass
                    
                    selected.at[idx, PARAGRAPH_ID] = attrs["paragraph"].get("id")
                    selected.at[idx, PARAGRAPH_NUMBER] = para_number
                    log.debug(f"Row {idx}: Extracted paragraph {attrs['paragraph'].get('id')}, number {para_number}")
                except Exception as e:
                    log.warning(f"Error extracting paragraph info for row {idx}: {e}")
            
            # Extract character position info with better error handling
            if "char_position" in attrs and attrs["char_position"] and isinstance(attrs["char_position"], dict):
                try:
                    # Convert position values to int if they're numeric strings
                    char_start = attrs["char_position"].get("start")
                    char_end = attrs["char_position"].get("end")
                    
                    if char_start is not None:
                        try:
                            char_start = int(char_start)
                        except (ValueError, TypeError):
                            pass
                    
                    if char_end is not None:
                        try:
                            char_end = int(char_end)
                        except (ValueError, TypeError):
                            pass
                    
                    selected.at[idx, CHAR_POSITION_START] = char_start
                    selected.at[idx, CHAR_POSITION_END] = char_end
                    log.debug(f"Row {idx}: Extracted char positions {char_start} to {char_end}")
                except Exception as e:
                    log.warning(f"Error extracting char position for row {idx}: {e}")
        
        # Log attribute extraction statistics
        attribute_counts = {
            'page_ids': selected[PAGE_ID].notna().sum(),
            'paragraph_ids': selected[PARAGRAPH_ID].notna().sum(),
            'char_positions': selected[CHAR_POSITION_START].notna().sum()
        }
        log.info(f"HTML attribute extraction results: {attribute_counts}")
        
        # Keep simplified attributes in the metadata
        selected["attributes"] = text_units["attributes"].apply(
            lambda x: simplify_attributes(x)
        )
    else:
        selected["attributes"] = None

    # Ensure selected has an index before joining
    if selected.empty:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=TEXT_UNITS_FINAL_COLUMNS)

    # Join with entities and relationships
    entity_join = _entities(final_entities)
    relationship_join = _relationships(final_relationships)

    entity_joined = _join(selected, entity_join)
    relationship_joined = _join(entity_joined, relationship_join)
    final_joined = relationship_joined

    if final_covariates is not None and not final_covariates.empty:
        covariate_join = _covariates(final_covariates)
        final_joined = _join(relationship_joined, covariate_join)
    else:
        if "covariate_ids" not in final_joined.columns:
            final_joined["covariate_ids"] = [[] for _ in range(len(final_joined))]

    # Safer groupby that ensures we have data
    if final_joined.empty:
        aggregated = final_joined
    else:
        try:
            # Use the safest approach for groupby
            if "id" in final_joined.columns and not final_joined["id"].isna().any():
                aggregated = final_joined.groupby("id", sort=False, as_index=False).first()
            else:
                # If id column is problematic, just use the dataframe as is
                aggregated = final_joined.copy()
        except Exception as e:
            # Fallback if groupby fails
            log.warning(f"Groupby failed: {e}. Using original dataframe.")
            aggregated = final_joined.copy()

    # Ensure all required columns from TEXT_UNITS_FINAL_COLUMNS are present
    for col in TEXT_UNITS_FINAL_COLUMNS:
        if col not in aggregated.columns:
            if col in ["document_ids", "entity_ids", "relationship_ids", "covariate_ids"]:
                aggregated[col] = [[] for _ in range(len(aggregated))]
            else:
                aggregated[col] = None
    
    # Return the complete dataframe with all columns including HTML structure
    return aggregated[TEXT_UNITS_FINAL_COLUMNS]


def simplify_attributes(attributes):
    """Extract only essential data from attributes with full HTML structure preservation."""
    if attributes is None:
        return None
    
    # If it's a string, try to parse it
    if isinstance(attributes, str):
        try:
            attributes = json.loads(attributes)
        except:
            return None
    
    # If it's not a dictionary after parsing, return None
    if not isinstance(attributes, dict):
        return None
    
    # Create a structure that preserves HTML fields
    result = {}
    
    # Preserve page information
    if "page" in attributes and attributes["page"]:
        result["page"] = attributes["page"]
        
        # Ensure page number is an integer if possible
        if isinstance(result["page"], dict) and "number" in result["page"] and result["page"]["number"] is not None:
            try:
                result["page"]["number"] = int(result["page"]["number"])
            except (ValueError, TypeError):
                pass
    
    # Preserve paragraph information
    if "paragraph" in attributes and attributes["paragraph"]:
        result["paragraph"] = attributes["paragraph"]
        
        # Ensure paragraph number is an integer if possible
        if isinstance(result["paragraph"], dict) and "number" in result["paragraph"] and result["paragraph"]["number"] is not None:
            try:
                result["paragraph"]["number"] = int(result["paragraph"]["number"])
            except (ValueError, TypeError):
                pass
    
    # Preserve character position information
    if "char_position" in attributes and attributes["char_position"]:
        result["char_position"] = attributes["char_position"]
        
        # Ensure character positions are integers if possible
        if isinstance(result["char_position"], dict):
            for pos in ["start", "end"]:
                if pos in result["char_position"] and result["char_position"][pos] is not None:
                    try:
                        result["char_position"][pos] = int(result["char_position"][pos])
                    except (ValueError, TypeError):
                        pass
    
    # Also include any additional HTML metadata for backward compatibility
    if "html" in attributes and isinstance(attributes["html"], dict):
        result["html"] = {}
        for key, value in attributes["html"].items():
            # Only include simple fields
            if isinstance(value, (str, int, float, bool)) or value is None:
                result["html"][key] = value
            else:
                # For complex objects, convert to string and limit length
                try:
                    result["html"][key] = str(value)[:500]  # Limit length
                except:
                    pass
    
    return result if result else None


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
    
    # Skip groupby if dataframe is empty to avoid "cannot set a frame with no defined index" error
    if unrolled.empty:
        return pd.DataFrame(columns=["id", "entity_ids"])
    
    return (
        unrolled.groupby("text_unit_ids", sort=False)
        .agg(entity_ids=("id", "unique"))
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
    
    # Skip groupby if dataframe is empty to avoid "cannot set a frame with no defined index" error
    if unrolled.empty:
        return pd.DataFrame(columns=["id", "relationship_ids"])
    
    return (
        unrolled.groupby("text_unit_ids", sort=False)
        .agg(relationship_ids=("id", "unique"))
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
    
    # Skip groupby if dataframe is empty to avoid "cannot set a frame with no defined index" error
    if selected.empty: