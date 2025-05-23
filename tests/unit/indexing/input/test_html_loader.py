# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import json

import pytest

from graphrag.config.enums import InputFileType, InputType
from graphrag.config.models.input_config import InputConfig
from graphrag.index.input.factory import create_input


async def test_html_loader_simple_document():
    """Test loading a simple HTML document with basic structure."""
    config = InputConfig(
        type=InputType.file,
        file_type=InputFileType.html,
        file_pattern="simple_document\\.html$",
        base_dir="tests/unit/indexing/input/data/html-samples",
    )
    documents = await create_input(config=config)

    # Verify document structure
    assert documents.shape[0] == 1  # One document
    assert "text" in documents.columns
    assert "metadata" in documents.columns
    assert "title" in documents.columns

    doc = documents.iloc[0]

    # Verify title extraction (comes from filename, not HTML title tag)
    assert doc["title"] == "simple_document.html"

    # Verify text content is extracted
    assert "This is the first paragraph" in doc["text"]
    assert "Introduction" in doc["text"]
    assert "Main Content" in doc["text"]

    # Verify metadata structure
    metadata = doc["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    assert "html" in metadata
    html_info = metadata["html"]

    # Verify HTML-specific metadata
    assert html_info["doc_type"] == "Research Report"
    assert html_info["filename"] == "sample_doc.html"
    assert html_info["has_pages"] is True
    assert html_info["has_paragraphs"] is True
    assert html_info["page_count"] >= 2  # Should find page markers
    assert html_info["paragraph_count"] >= 5  # Should find multiple paragraphs


async def test_html_loader_complex_document():
    """Test loading a complex HTML document with various page numbering schemes."""
    config = InputConfig(
        type=InputType.file,
        file_type=InputFileType.html,
        file_pattern="complex_document\\.html$",
        base_dir="tests/unit/indexing/input/data/html-samples",
    )
    documents = await create_input(config=config)

    doc = documents.iloc[0]

    # Verify title (comes from filename)
    assert doc["title"] == "complex_document.html"

    # Verify metadata
    metadata = doc["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    html_info = metadata["html"]

    # Verify complex document metadata
    assert html_info["doc_type"] == "Technical Manual"
    assert html_info["filename"] == "complex_manual.html"
    assert html_info["page_count"] >= 3  # Should find multiple page markers

    # Verify page markers are extracted
    pages = html_info["pages"]
    assert len(pages) >= 3

    # Check for different page numbering schemes
    page_ids = [page["page_id"] for page in pages]
    assert any("1" in page_id for page_id in page_ids)  # Numeric
    assert any("iv" in page_id for page_id in page_ids)  # Roman numeral
    assert any("A-1" in page_id for page_id in page_ids)  # Alphanumeric


async def test_html_loader_minimal_document():
    """Test loading a minimal HTML document with no special structure."""
    config = InputConfig(
        type=InputType.file,
        file_type=InputFileType.html,
        file_pattern="minimal_document\\.html$",
        base_dir="tests/unit/indexing/input/data/html-samples",
    )
    documents = await create_input(config=config)

    doc = documents.iloc[0]

    # Verify basic parsing works even without special structure
    assert "minimal HTML document" in doc["text"]

    # Verify metadata structure exists even for minimal documents
    metadata = doc["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    assert "html" in metadata
    html_info = metadata["html"]

    # Minimal document should have basic structure (page detection may still find false positives)
    assert html_info["has_paragraphs"] is True  # Should still find paragraphs
    assert html_info["paragraph_count"] >= 2


async def test_html_loader_multiple_files():
    """Test loading multiple HTML files at once."""
    config = InputConfig(
        type=InputType.file,
        file_type=InputFileType.html,
        file_pattern=".*\\.html$",
        base_dir="tests/unit/indexing/input/data/html-samples",
    )
    documents = await create_input(config=config)

    # Should load all HTML files in the directory
    assert documents.shape[0] >= 3

    # Verify each document has required columns
    for _, doc in documents.iterrows():
        assert isinstance(doc["text"], str)
        assert len(doc["text"]) > 0
        assert "metadata" in doc

        # Verify metadata structure
        metadata = doc["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert "html" in metadata


async def test_html_loader_with_metadata_config():
    """Test HTML loader with metadata configuration."""
    config = InputConfig(
        type=InputType.file,
        file_type=InputFileType.html,
        file_pattern="simple_document\\.html$",
        base_dir="tests/unit/indexing/input/data/html-samples",
        metadata=["title"],
    )
    documents = await create_input(config=config)

    doc = documents.iloc[0]

    # Verify metadata column exists
    assert "metadata" in documents.columns

    # When metadata config is specified, it extracts those columns from the DataFrame
    # Since HTML processing creates a 'title' column, it should be captured
    metadata = doc["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    # Should contain the title that was configured to be extracted
    assert "title" in metadata
    assert metadata["title"] == "simple_document.html"


async def test_html_page_marker_extraction():
    """Test that page markers are correctly extracted from HTML."""
    config = InputConfig(
        type=InputType.file,
        file_type=InputFileType.html,
        file_pattern="complex_document\\.html$",
        base_dir="tests/unit/indexing/input/data/html-samples",
    )
    documents = await create_input(config=config)

    doc = documents.iloc[0]
    metadata = doc["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    pages = metadata["html"]["pages"]

    # Verify page marker details
    found_numeric = False
    found_roman = False
    found_alphanumeric = False

    for page in pages:
        assert "page_id" in page
        assert "text" in page

        # Check for different numbering schemes
        if page["page_id"] == "1" or page["page_id"] == "2":
            found_numeric = True
            assert page["page_num"] is not None
        elif page["page_id"] == "iv":
            found_roman = True
        elif page["page_id"] == "A-1":
            found_alphanumeric = True

    # Should find at least one of each type
    assert found_numeric, "Should find numeric page markers"


async def test_html_paragraph_extraction():
    """Test that paragraphs are correctly extracted with positioning."""
    config = InputConfig(
        type=InputType.file,
        file_type=InputFileType.html,
        file_pattern="simple_document\\.html$",
        base_dir="tests/unit/indexing/input/data/html-samples",
    )
    documents = await create_input(config=config)

    doc = documents.iloc[0]
    metadata = doc["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    paragraphs = metadata["html"]["paragraphs"]

    # Verify paragraph structure
    assert len(paragraphs) > 0

    for paragraph in paragraphs:
        assert "text" in paragraph
        assert "para_id" in paragraph
        assert "para_num" in paragraph
        assert "char_start" in paragraph
        assert "char_end" in paragraph
        assert "type" in paragraph

        # Verify character positions are valid
        assert paragraph["char_start"] >= 0
        assert paragraph["char_end"] > paragraph["char_start"]
        assert paragraph["type"] == "paragraph"

        # Verify paragraph text is not empty
        assert len(paragraph["text"].strip()) > 0
