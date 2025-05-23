# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Citation and source attribution utilities for GraphRAG search results."""

import json
import logging
from typing import Any

import pandas as pd

from graphrag.data_model.schemas import (
    CHAR_POSITION_END,
    CHAR_POSITION_START,
    DOCUMENT_IDS,
    PAGE_ID,
    PAGE_NUMBER,
    PARAGRAPH_ID,
    PARAGRAPH_NUMBER,
)

log = logging.getLogger(__name__)


def extract_citations_from_context(
    context_records: dict[str, pd.DataFrame],
) -> dict[str, list[str]]:
    """Extract available citation records from context data.

    Args:
        context_records: Dictionary mapping dataset names to DataFrames

    Returns:
        Dictionary mapping dataset names to lists of record IDs
    """
    citations = {}

    for dataset_name, df in context_records.items():
        if df.empty or "id" not in df.columns:
            continue

        # Filter to only records that were actually used in context
        if "in_context" in df.columns:
            used_records = df[df["in_context"] == True]  # noqa: E712
        else:
            used_records = df

        if not used_records.empty:
            citations[dataset_name.title()] = used_records["id"].astype(str).tolist()

    return citations


def extract_source_attributions(
    context_records: dict[str, pd.DataFrame],
    document_metadata: dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    """Extract detailed source attribution information from context records.

    Args:
        context_records: Dictionary mapping dataset names to DataFrames
        document_metadata: Optional mapping of document IDs to metadata (file paths, titles, etc.)

    Returns:
        List of source attribution dictionaries with detailed location info
    """
    attributions = []

    # Process text units (sources) for detailed attribution
    if "sources" in context_records:
        sources_df = context_records["sources"]

        # Filter to only records used in context
        if "in_context" in sources_df.columns:
            used_sources = sources_df[sources_df["in_context"] == True]  # noqa: E712
        else:
            used_sources = sources_df

        for _, source in used_sources.iterrows():
            documents = _safe_list(source.get(DOCUMENT_IDS, []))

            attribution = {
                "source_id": str(source.get("id", "")),
                "text_preview": _truncate_text(source.get("text", ""), 100),
                "documents": documents,
            }

            # Add file information from document metadata
            if document_metadata and documents:
                file_info = []
                for doc_id in documents:
                    if doc_id in document_metadata:
                        doc_meta = document_metadata[doc_id]
                        file_info.append({
                            "document_id": doc_id,
                            "file_path": doc_meta.get("file_path"),
                            "filename": doc_meta.get("filename")
                            or doc_meta.get("title"),
                            "creation_date": doc_meta.get("creation_date"),
                            "file_size": doc_meta.get("file_size"),
                            "doc_type": doc_meta.get("doc_type"),
                        })
                attribution["file_sources"] = file_info

            # Add page information if available
            if pd.notna(source.get(PAGE_ID)) or pd.notna(source.get(PAGE_NUMBER)):
                attribution["page"] = {
                    "page_id": source.get(PAGE_ID),
                    "page_number": source.get(PAGE_NUMBER),
                }

            # Add paragraph information if available
            if pd.notna(source.get(PARAGRAPH_ID)) or pd.notna(
                source.get(PARAGRAPH_NUMBER)
            ):
                attribution["paragraph"] = {
                    "paragraph_id": source.get(PARAGRAPH_ID),
                    "paragraph_number": source.get(PARAGRAPH_NUMBER),
                }

            # Add character position if available
            if pd.notna(source.get(CHAR_POSITION_START)) or pd.notna(
                source.get(CHAR_POSITION_END)
            ):
                attribution["character_position"] = {
                    "start": source.get(CHAR_POSITION_START),
                    "end": source.get(CHAR_POSITION_END),
                }

            # Parse attributes for additional citation info
            attributes = _parse_attributes(source.get("attributes"))
            if attributes:
                attribution["attributes"] = attributes

            attributions.append(attribution)

    return attributions


def format_citation_references(citations: dict[str, list[str]]) -> str:
    """Format citations for display in search responses.

    Args:
        citations: Dictionary mapping dataset names to record ID lists

    Returns:
        Formatted citation string for inclusion in responses
    """
    if not citations:
        return ""

    citation_parts = []
    for dataset, record_ids in citations.items():
        if not record_ids:
            continue

        # Limit displayed IDs to avoid overly long citations
        if len(record_ids) > 5:
            ids_display = record_ids[:5] + [f"+{len(record_ids) - 5} more"]
        else:
            ids_display = record_ids

        citation_parts.append(f"{dataset} ({', '.join(ids_display)})")

    return f"[Data: {'; '.join(citation_parts)}]"


def format_source_attributions(attributions: list[dict[str, Any]]) -> str:
    """Format detailed source attributions for display.

    Args:
        attributions: List of source attribution dictionaries

    Returns:
        Formatted attribution string with detailed source information
    """
    if not attributions:
        return ""

    attribution_lines = []

    for i, attr in enumerate(attributions[:10], 1):  # Limit to first 10 sources
        parts = [f"Source {i}: {attr.get('source_id', 'Unknown')}"]

        # Add page information
        if "page" in attr:
            page_info = attr["page"]
            if page_info.get("page_number"):
                parts.append(f"Page {page_info['page_number']}")
            elif page_info.get("page_id"):
                parts.append(f"Page {page_info['page_id']}")

        # Add paragraph information
        if "paragraph" in attr:
            para_info = attr["paragraph"]
            if para_info.get("paragraph_number"):
                parts.append(f"Paragraph {para_info['paragraph_number']}")

        # Add character positions
        if "character_position" in attr:
            char_info = attr["character_position"]
            if char_info.get("start") is not None and char_info.get("end") is not None:
                parts.append(f"Characters {char_info['start']}-{char_info['end']}")

        attribution_lines.append(" | ".join(parts))

    if len(attributions) > 10:
        attribution_lines.append(f"... and {len(attributions) - 10} more sources")

    return "\n".join(attribution_lines)


def create_citation_metadata(
    citations: dict[str, list[str]],
    attributions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create comprehensive citation metadata for search results.

    Args:
        citations: Citation dictionary from extract_citations_from_context
        attributions: Attribution list from extract_source_attributions

    Returns:
        Comprehensive citation metadata dictionary
    """
    metadata = {
        "total_sources": len(attributions),
        "source_types": list(citations.keys()) if citations else [],
        "has_page_info": any("page" in attr for attr in attributions),
        "has_paragraph_info": any("paragraph" in attr for attr in attributions),
        "has_character_positions": any(
            "character_position" in attr for attr in attributions
        ),
    }

    # Add document statistics
    all_documents = set()
    for attr in attributions:
        all_documents.update(attr.get("documents", []))
    metadata["unique_documents"] = len(all_documents)

    return metadata


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to specified length with ellipsis."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def _safe_list(value: Any) -> list:
    """Safely convert value to list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            # Try to parse as JSON list
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value]
    return [str(value)]


def _parse_attributes(attributes_value: Any) -> dict[str, Any] | None:
    """Parse attributes field which may be JSON string or dict."""
    if not attributes_value:
        return None

    if isinstance(attributes_value, dict):
        return attributes_value

    if isinstance(attributes_value, str):
        try:
            return json.loads(attributes_value)
        except json.JSONDecodeError:
            log.debug(f"Failed to parse attributes JSON: {attributes_value[:100]}")
            return None

    return None


def append_citations_to_response(
    response: str,
    citations: dict[str, list[str]] | None = None,
    attributions: list[dict[str, Any]] | None = None,
    include_detailed_attributions: bool = False,
) -> str:
    """Append citation information to a search response.

    Args:
        response: Original search response text
        citations: Citation dictionary
        attributions: Source attribution list
        include_detailed_attributions: Whether to include detailed source info

    Returns:
        Response text with appended citation information
    """
    if not citations and not attributions:
        return response

    citation_parts = []

    # Add standard citations
    if citations:
        citation_ref = format_citation_references(citations)
        if citation_ref:
            citation_parts.append(citation_ref)

    # Add detailed attributions if requested
    if include_detailed_attributions and attributions:
        detailed_refs = format_source_attributions(attributions)
        if detailed_refs:
            citation_parts.append(f"\n\nDetailed Source References:\n{detailed_refs}")

    if citation_parts:
        return f"{response}\n\n{citation_parts[0]}" + (
            "".join(citation_parts[1:]) if len(citation_parts) > 1 else ""
        )

    return response
