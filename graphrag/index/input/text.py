# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing load method definition."""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from graphrag.config.models.input_config import InputConfig
from graphrag.index.input.util import load_files
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.logger.base import ProgressLogger
from graphrag.storage.pipeline_storage import PipelineStorage

log = logging.getLogger(__name__)


async def load_text(
    config: InputConfig,
    progress: ProgressLogger | None,
    storage: PipelineStorage,
) -> pd.DataFrame:
    """Load text inputs from a directory."""

    async def load_file(path: str, group: dict | None = None) -> pd.DataFrame:
        if group is None:
            group = {}
        text = await storage.get(path, encoding=config.encoding)

        # Create basic document structure for text files
        document_structure = extract_text_structure(text, Path(path).name)

        # Create text metadata with basic structure info
        text_info = {
            "has_pages": False,  # Plain text doesn't have page structure
            "has_paragraphs": bool(document_structure.get("paragraphs")),
            "doc_type": "text",
            "filename": document_structure.get("filename"),
            "page_count": 0,
            "paragraph_count": len(document_structure.get("paragraphs", [])),
            "paragraphs": document_structure.get("paragraphs", []),
        }

        new_item = {**group, "text": document_structure["text"]}

        # Store structure in metadata for consistency with PDF/HTML
        if isinstance(new_item.get("metadata"), dict):
            existing_metadata = new_item.get("metadata", {})
        elif new_item.get("metadata") is not None:
            existing_metadata = {"original": new_item.get("metadata")}
        else:
            existing_metadata = {}

        new_item["metadata"] = {
            "html": text_info,  # Use "html" key for compatibility
            **existing_metadata,
        }

        new_item["id"] = gen_sha512_hash(new_item, new_item.keys())
        new_item["title"] = str(Path(path).name)
        new_item["creation_date"] = await storage.get_creation_date(path)
        return pd.DataFrame([new_item])

    return await load_files(load_file, config, storage, progress)


def extract_text_structure(text: str, filename: str) -> Dict[str, Any]:
    """Extract basic structure from plain text."""
    log.info(f"Extracting structure from text file: {filename}")

    document_structure = {
        "text": text,
        "paragraphs": [],
        "filename": filename,
        "title": Path(filename).stem,
    }

    # Split text into paragraphs (simple heuristic)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    char_pos = 0

    for para_idx, para_text in enumerate(paragraphs, 1):
        para_start_char = char_pos
        para_end_char = char_pos + len(para_text)

        para_info = {
            "type": "paragraph",
            "text": para_text,
            "char_start": para_start_char,
            "char_end": para_end_char,
            "para_id": f"para_{para_idx}",
            "para_num": para_idx,
        }
        document_structure["paragraphs"].append(para_info)

        # Update character position (account for paragraph separator)
        char_pos = para_end_char + 2  # +2 for \n\n

    log.info(
        f"Extracted {len(document_structure['paragraphs'])} paragraphs from text file"
    )
    return document_structure
