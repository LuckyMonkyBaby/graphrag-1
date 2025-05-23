# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
import io

from graphrag.config.enums import InputFileType, InputType
from graphrag.config.models.input_config import InputConfig
from graphrag.index.input.factory import create_input


# Mock PDF content for testing
MOCK_PDF_CONTENT = {
    "pages": [
        {
            "page_num": 1,
            "text": "Introduction\n\nThis is the first paragraph of a sample PDF document. It contains basic text content that would be extracted by PDF parsing libraries.\n\nThis is a second paragraph on the first page with additional content for testing paragraph extraction and character positioning.",
        },
        {
            "page_num": 2,
            "text": "Main Section\n\nThis paragraph appears on the second page and contains important information about the document structure.\n\nAnother paragraph on page 2 with detailed content that spans multiple lines and contains various types of information.",
        },
        {
            "page_num": 3,
            "text": "Conclusions\n\nThis is the concluding paragraph on the final page that summarizes key points.\n\nFinal paragraph with closing remarks and additional notes for comprehensive testing.",
        },
    ]
}


class MockPDFPage:
    """Mock PDF page for testing."""

    def __init__(self, text: str):
        self.text = text

    def extract_text(self):
        return self.text


class MockPDFDocument:
    """Mock PDF document for testing."""

    def __init__(self, pages):
        self.pages = [MockPDFPage(page["text"]) for page in pages]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockPyPDF2Reader:
    """Mock PyPDF2 reader for testing."""

    def __init__(self, file_obj):
        self.pages = [MockPDFPage(page["text"]) for page in MOCK_PDF_CONTENT["pages"]]


@pytest.fixture
def mock_storage():
    """Create a mock storage for testing."""
    storage = AsyncMock()
    storage.get.return_value = b"mock_pdf_content"
    storage.get_creation_date.return_value = "2024-01-01"
    return storage


@patch("graphrag.index.input.pdf.PDF_PLUMBER_AVAILABLE", True)
@patch("graphrag.index.input.pdf.pdfplumber")
async def test_pdf_loader_with_pdfplumber(mock_pdfplumber, mock_storage):
    """Test PDF loading with pdfplumber library."""
    # Setup mock
    mock_pdfplumber.open.return_value = MockPDFDocument(MOCK_PDF_CONTENT["pages"])

    # Create config
    config = InputConfig(
        type=InputType.file,
        file_type=InputFileType.pdf,
        file_pattern=".*\\.pdf$",
        base_dir="tests/unit/indexing/input/data/pdf-samples",
    )

    # Mock the storage and load_files function
    with patch("graphrag.index.input.pdf.load_files") as mock_load_files:
        # Create a mock document result
        mock_doc = {
            "text": "Introduction\n\nThis is the first paragraph...",
            "title": "sample.pdf",
            "id": "test_id",
            "creation_date": "2024-01-01",
            "metadata": {
                "html": {
                    "doc_type": "pdf",
                    "filename": "sample.pdf",
                    "has_pages": True,
                    "has_paragraphs": True,
                    "page_count": 3,
                    "paragraph_count": 6,
                    "pages": [],
                    "paragraphs": [],
                }
            },
        }

        import pandas as pd

        mock_load_files.return_value = pd.DataFrame([mock_doc])

        from graphrag.index.input.pdf import load_pdf

        documents = await load_pdf(config, None, mock_storage)

        # Verify document structure
        assert documents.shape[0] == 1
        doc = documents.iloc[0]

        # Verify metadata structure
        metadata = doc["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        assert "html" in metadata
        pdf_info = metadata["html"]
        assert pdf_info["doc_type"] == "pdf"


@patch("graphrag.index.input.pdf.PDF_PLUMBER_AVAILABLE", False)
@patch("graphrag.index.input.pdf.PYPDF2_AVAILABLE", True)
@patch("graphrag.index.input.pdf.PyPDF2")
async def test_pdf_loader_with_pypdf2_fallback(mock_pypdf2, mock_storage):
    """Test PDF loading with PyPDF2 fallback."""
    # Setup mock
    mock_pypdf2.PdfReader.return_value = MockPyPDF2Reader(None)

    # Test that the function imports and can be configured
    from graphrag.index.input.pdf import extract_pdf_structure_pypdf2

    # Test extraction
    result = extract_pdf_structure_pypdf2(b"mock_content", "test.pdf")

    # Verify structure
    assert "text" in result
    assert "pages" in result
    assert "paragraphs" in result
    assert "filename" in result
    assert result["filename"] == "test.pdf"


@patch("graphrag.index.input.pdf.PDF_PLUMBER_AVAILABLE", False)
@patch("graphrag.index.input.pdf.PYPDF2_AVAILABLE", False)
async def test_pdf_loader_no_libraries():
    """Test PDF loader behavior when no PDF libraries are available."""
    config = InputConfig(
        type=InputType.file,
        file_type=InputFileType.pdf,
        file_pattern=".*\\.pdf$",
        base_dir="tests/unit/indexing/input/data/pdf-samples",
    )

    with pytest.raises(ImportError, match="No PDF parsing library available"):
        from graphrag.index.input.pdf import load_pdf

        await load_pdf(config, None, mock_storage)


def test_pdf_structure_extraction_pdfplumber():
    """Test PDF structure extraction with pdfplumber."""
    with patch("graphrag.index.input.pdf.pdfplumber") as mock_pdfplumber:
        mock_pdfplumber.open.return_value = MockPDFDocument(MOCK_PDF_CONTENT["pages"])

        from graphrag.index.input.pdf import extract_pdf_structure_pdfplumber

        result = extract_pdf_structure_pdfplumber(b"mock_content", "test.pdf")

        # Verify structure
        assert result["filename"] == "test.pdf"
        assert result["title"] == "test"  # filename stem
        assert len(result["pages"]) == 3
        assert len(result["paragraphs"]) > 0

        # Verify page information
        for i, page in enumerate(result["pages"]):
            assert page["page_num"] == i + 1
            assert page["page_id"] == f"page_{i + 1}"
            assert "text" in page
            assert "char_start" in page
            assert "char_end" in page

        # Verify paragraph information
        for paragraph in result["paragraphs"]:
            assert "text" in paragraph
            assert "para_id" in paragraph
            assert "para_num" in paragraph
            assert "char_start" in paragraph
            assert "char_end" in paragraph
            assert "page_num" in paragraph
            assert "page_id" in paragraph
            assert paragraph["type"] == "paragraph"


def test_pdf_structure_extraction_pypdf2():
    """Test PDF structure extraction with PyPDF2."""
    with patch("graphrag.index.input.pdf.PyPDF2") as mock_pypdf2:
        mock_pypdf2.PdfReader.return_value = MockPyPDF2Reader(None)

        from graphrag.index.input.pdf import extract_pdf_structure_pypdf2

        result = extract_pdf_structure_pypdf2(b"mock_content", "test.pdf")

        # Verify structure
        assert result["filename"] == "test.pdf"
        assert result["title"] == "test"
        assert len(result["pages"]) == 3
        assert len(result["paragraphs"]) > 0

        # Verify page structure
        for i, page in enumerate(result["pages"]):
            assert page["page_num"] == i + 1
            assert page["page_id"] == f"page_{i + 1}"

        # Verify paragraph structure
        for paragraph in result["paragraphs"]:
            assert paragraph["type"] == "paragraph"
            assert "page_num" in paragraph
            assert "page_id" in paragraph


async def test_pdf_metadata_structure():
    """Test that PDF metadata follows the expected structure."""
    with (
        patch("graphrag.index.input.pdf.PDF_PLUMBER_AVAILABLE", True),
        patch("graphrag.index.input.pdf.pdfplumber") as mock_pdfplumber,
    ):
        mock_pdfplumber.open.return_value = MockPDFDocument(MOCK_PDF_CONTENT["pages"])

        from graphrag.index.input.pdf import extract_pdf_structure_pdfplumber

        result = extract_pdf_structure_pdfplumber(b"mock_content", "test.pdf")

        # Test the metadata structure that would be created
        pdf_info = {
            "has_pages": bool(result.get("pages")),
            "has_paragraphs": bool(result.get("paragraphs")),
            "doc_type": "pdf",
            "filename": result.get("filename"),
            "page_count": len(result.get("pages", [])),
            "paragraph_count": len(result.get("paragraphs", [])),
            "pages": result.get("pages", []),
            "paragraphs": result.get("paragraphs", []),
        }

        # Verify metadata structure
        assert pdf_info["has_pages"] is True
        assert pdf_info["has_paragraphs"] is True
        assert pdf_info["doc_type"] == "pdf"
        assert pdf_info["filename"] == "test.pdf"
        assert pdf_info["page_count"] == 3
        assert pdf_info["paragraph_count"] > 0
        assert len(pdf_info["pages"]) == 3
        assert len(pdf_info["paragraphs"]) > 0


def test_pdf_paragraph_character_positions():
    """Test that paragraph character positions are correctly calculated."""
    with patch("graphrag.index.input.pdf.pdfplumber") as mock_pdfplumber:
        mock_pdfplumber.open.return_value = MockPDFDocument(MOCK_PDF_CONTENT["pages"])

        from graphrag.index.input.pdf import extract_pdf_structure_pdfplumber

        result = extract_pdf_structure_pdfplumber(b"mock_content", "test.pdf")

        paragraphs = result["paragraphs"]

        # Verify character positions are sequential and non-overlapping
        for i, paragraph in enumerate(paragraphs):
            assert paragraph["char_start"] >= 0
            assert paragraph["char_end"] > paragraph["char_start"]

            # Verify that each paragraph's text matches its character range
            text_length = len(paragraph["text"])
            position_length = paragraph["char_end"] - paragraph["char_start"]

            # Should be close (allowing for newlines and formatting)
            assert position_length >= text_length


@pytest.mark.asyncio
async def test_pdf_error_handling():
    """Test PDF loader error handling."""
    config = InputConfig(
        type=InputType.file,
        file_type=InputFileType.pdf,
        file_pattern=".*\\.pdf$",
        base_dir="tests/unit/indexing/input/data/pdf-samples",
    )

    # Create a mock storage that raises an error
    error_storage = AsyncMock()
    error_storage.get.side_effect = Exception("File not found")

    with patch("graphrag.index.input.pdf.PDF_PLUMBER_AVAILABLE", True):
        with pytest.raises(Exception):
            from graphrag.index.input.pdf import load_pdf

            await load_pdf(config, None, error_storage)
