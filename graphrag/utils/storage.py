# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Storage functions for the GraphRAG run module."""

import logging
import json
from io import BytesIO

import pandas as pd

from graphrag.storage.pipeline_storage import PipelineStorage

log = logging.getLogger(__name__)


async def load_table_from_storage(name: str, storage: PipelineStorage) -> pd.DataFrame:
    """Load a parquet from the storage instance."""
    filename = f"{name}.parquet"
    if not await storage.has(filename):
        msg = f"Could not find {filename} in storage!"
        raise ValueError(msg)
    try:
        log.info("reading table from storage: %s", filename)
        df = pd.read_parquet(BytesIO(await storage.get(filename, as_bytes=True)))
        
        # Restore complex data types from JSON strings
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to restore JSON objects/arrays
                try:
                    # Check first non-null value to see if it looks like JSON
                    sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if isinstance(sample, str) and (sample.startswith('{') or sample.startswith('[')):
                        df[col] = df[col].apply(
                            lambda x: json.loads(x) if isinstance(x, str) and x else x
                        )
                except (ValueError, IndexError, AttributeError, TypeError):
                    # If any errors occur during conversion, leave the column as is
                    pass
                    
        return df
    except Exception:
        log.exception("error loading table from storage: %s", filename)
        raise


async def write_table_to_storage(
    table: pd.DataFrame, name: str, storage: PipelineStorage
) -> None:
    """Write a table to storage, converting complex data types to JSON strings."""
    # Create a copy to avoid modifying the original DataFrame
    df_to_save = table.copy()
    
    # Convert complex data types (lists, dicts) to JSON strings
    for col in df_to_save.columns:
        # Check if column contains any complex data types
        if df_to_save[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_to_save[col] = df_to_save[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
    
    # Now save to Parquet
    try:
        parquet_data = df_to_save.to_parquet()
        await storage.set(f"{name}.parquet", parquet_data)
    except Exception as e:
        log.error(f"Failed to convert DataFrame to Parquet: {e}")
        # Fallback approach with explicit handling of problematic columns
        for col in df_to_save.columns:
            # Convert all object columns to strings as a last resort
            if df_to_save[col].dtype == 'object':
                df_to_save[col] = df_to_save[col].astype(str)
        
        # Try again with stringified columns
        parquet_data = df_to_save.to_parquet()
        await storage.set(f"{name}.parquet", parquet_data)


async def delete_table_from_storage(name: str, storage: PipelineStorage) -> None:
    """Delete a table to storage."""
    await storage.delete(f"{name}.parquet")


async def storage_has_table(name: str, storage: PipelineStorage) -> bool:
    """Check if a table exists in storage."""
    return await storage.has(f"{name}.parquet")