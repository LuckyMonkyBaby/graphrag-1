# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

```bash
# Install dependencies
poetry install

# Core CLI operations  
poetry run poe init        # Initialize project configuration
poetry run poe index       # Build knowledge graph index
poetry run poe update      # Update existing index with new data
poetry run poe query       # Query the knowledge graph
poetry run poe prompt_tune # Auto-generate custom prompts

# Development workflow
poetry run poe test        # Run all tests with coverage
poetry run poe check       # Static analysis (format, lint, type-check)
poetry run poe format      # Auto-format code
poetry run poe fix         # Auto-fix linting issues

# Specific test suites
poetry run poe test_unit        # Unit tests only
poetry run poe test_integration # Integration tests only
poetry run poe test_smoke       # Smoke tests only
poetry run poe test_only "pattern" # Run specific test pattern
```

## High-Level Architecture

GraphRAG is a **data pipeline system** that transforms unstructured text into knowledge graphs using LLMs, enabling graph-based retrieval-augmented generation.

### Core Pipeline Flow
1. **Text Processing** → Chunk documents into text units
2. **Graph Extraction** → Extract entities/relationships using LLMs or NLP
3. **Community Detection** → Cluster entities into hierarchical communities  
4. **Report Generation** → Create summaries for each community
5. **Embedding Creation** → Generate vector embeddings for search

### Key Architectural Patterns

**Factory Pattern**: Most components use factories (`factory.py` files) for dependency injection and configuration-based instantiation.

**Pipeline Pattern**: The `index/workflows/` directory contains discrete pipeline steps that can be composed into indexing workflows.

**Strategy Pattern**: Multiple implementations for core operations (e.g., `extract_graph` supports both LLM and NLP-based extraction).

### Module Responsibilities

- **`config/`** - Pydantic models for type-safe configuration management
- **`data_model/`** - Core domain objects (Entity, Community, Relationship, etc.)
- **`index/`** - Indexing pipeline engine with pluggable operations
- **`query/`** - Query engine supporting local, global, drift, and basic search
- **`storage/`** - Abstracted storage backends (file, blob, CosmosDB, memory)
- **`vector_stores/`** - Vector database integrations (LanceDB, Azure AI Search)
- **`language_model/`** - LLM provider abstractions built on `fnllm`

### Indexing Methods

**Standard Pipeline**: Uses LLMs for high-quality graph extraction (expensive but accurate).

**Fast Pipeline**: Uses NLP libraries (spaCy, NLTK) for faster, lower-cost extraction with reduced quality.

### Query Strategies

- **Local Search**: Context-aware search within specific entity communities
- **Global Search**: Hierarchical map-reduce over community reports  
- **Drift Search**: Multi-step conversational reasoning
- **Basic Search**: Simple vector similarity search

## Development Notes

- Uses **Poetry** for dependency management
- **Ruff** for linting/formatting, **Pyright** for type checking
- Configuration uses **Pydantic v2** models with environment variable support
- Test coverage required for new functionality
- Always run `poetry run poe check` before committing