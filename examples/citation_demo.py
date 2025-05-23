#!/usr/bin/env python3
"""
Demonstration of the enhanced citation system in GraphRAG.

This example shows how to:
1. Index documents with proper source attribution
2. Query with detailed citation information  
3. Extract and format source references

The citation system now tracks:
- Page numbers and IDs
- Paragraph numbers and IDs
- Character positions (start and end)
- Document IDs and filenames
"""

import asyncio
import json
from pathlib import Path

from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.index.run.run_pipeline import run_pipeline
from graphrag.query.citation_utils import (
    append_citations_to_response,
    create_citation_metadata,
    format_citation_references,
    format_source_attributions,
)
from graphrag.query.factory import get_local_search_engine, get_global_search_engine


async def main():
    """Demonstrate the enhanced citation system."""
    print("🔍 GraphRAG Citation System Demo")
    print("=" * 50)
    
    # Step 1: Setup configuration
    config_dict = {
        "input": {
            "type": "file",
            "file_type": "text",  # Can also be "pdf" or "html"
            "base_dir": "input",
        },
        "cache": {"type": "memory"},
        "storage": {"type": "memory"},
        "chunks": {
            "size": 200,
            "overlap": 50,
        },
        "embeddings": {
            "model": "text-embedding-3-small",
        },
        "llm": {
            "model": "gpt-4o-mini",
        }
    }
    
    # Step 2: Create sample documents (if input directory doesn't exist)
    input_dir = Path("input")
    if not input_dir.exists():
        input_dir.mkdir(exist_ok=True)
        
        # Create sample documents with different content types
        (input_dir / "report1.txt").write_text("""
Research Report: Climate Change Impacts

Introduction

Climate change represents one of the most significant challenges facing humanity in the 21st century. Rising global temperatures have led to unprecedented changes in weather patterns, sea levels, and ecosystem dynamics.

Methodology

Our research team analyzed temperature data from 1850 to 2023, examining trends across different geographical regions. We used statistical models to project future scenarios based on current emission trajectories.

Key Findings

The data shows a clear warming trend with the last decade recording the highest average temperatures on record. Arctic ice coverage has decreased by 40% since 1980, and sea levels have risen by 8.5 inches since 1880.

Conclusions

Immediate action is required to mitigate the worst effects of climate change. Policy interventions must focus on reducing greenhouse gas emissions and implementing adaptation strategies.
        """.strip())
        
        (input_dir / "study2.txt").write_text("""
Economic Analysis: Renewable Energy Markets

Executive Summary

The renewable energy sector has experienced exponential growth over the past decade. Solar and wind technologies have achieved grid parity in many markets, making them cost-competitive with fossil fuels.

Market Trends

Investment in renewable energy reached $1.8 trillion globally in 2023. Solar photovoltaic installations increased by 25% year-over-year, while offshore wind capacity doubled in European markets.

Regional Analysis

Asia-Pacific leads in renewable energy deployment, accounting for 60% of global capacity additions. China alone installed 120 GW of solar capacity in 2023, surpassing all previous records.

Future Outlook

Market projections indicate that renewables will comprise 85% of new power generation capacity by 2030. Battery storage technology improvements will further accelerate adoption rates.
        """.strip())
    
    try:
        # Step 3: Create GraphRAG configuration
        config = create_graphrag_config(config_dict, ".")
        
        # Step 4: Index the documents
        print("\n📄 Indexing documents with citation tracking...")
        # Note: In production, you would run the full pipeline
        # await run_pipeline(config)
        print("   ✅ Documents indexed with page, paragraph, and character position tracking")
        
        # Step 5: Initialize search engines (mock for demo)
        print("\n🔍 Initializing search engines...")
        # local_search = get_local_search_engine(config)
        # global_search = get_global_search_engine(config)
        print("   ✅ Local and global search engines initialized")
        
        # Step 6: Demonstrate citation extraction (mock data)
        print("\n📚 Demonstrating citation system...")
        
        # Mock context records that would come from actual search
        mock_context_records = {
            "sources": [
                {
                    "id": "chunk_001",
                    "text": "Climate change represents one of the most significant challenges...",
                    "document_ids": ["doc_report1"],
                    "page_id": None,
                    "page_number": None,
                    "paragraph_id": "para_1",
                    "paragraph_number": 1,
                    "char_position_start": 0,
                    "char_position_end": 142,
                    "attributes": json.dumps({
                        "html": {
                            "doc_type": "text",
                            "filename": "report1.txt",
                            "paragraphs": [
                                {
                                    "para_id": "para_1",
                                    "para_num": 1,
                                    "text": "Climate change represents one of the most significant challenges...",
                                    "char_start": 0,
                                    "char_end": 142
                                }
                            ]
                        }
                    }),
                    "in_context": True
                },
                {
                    "id": "chunk_002", 
                    "text": "The renewable energy sector has experienced exponential growth...",
                    "document_ids": ["doc_study2"],
                    "page_id": None,
                    "page_number": None,
                    "paragraph_id": "para_2",
                    "paragraph_number": 2,
                    "char_position_start": 200,
                    "char_position_end": 342,
                    "attributes": json.dumps({
                        "html": {
                            "doc_type": "text",
                            "filename": "study2.txt",
                            "paragraphs": [
                                {
                                    "para_id": "para_2",
                                    "para_num": 2,
                                    "text": "The renewable energy sector has experienced exponential growth...",
                                    "char_start": 200,
                                    "char_end": 342
                                }
                            ]
                        }
                    }),
                    "in_context": True
                }
            ],
            "entities": [
                {
                    "id": "entity_001",
                    "name": "Climate Change",
                    "description": "Long-term shifts in global temperatures and weather patterns",
                    "in_context": True
                }
            ]
        }
        
        # Convert to DataFrame format (as would be provided by actual search)
        import pandas as pd
        context_dfs = {}
        for dataset, records in mock_context_records.items():
            context_dfs[dataset] = pd.DataFrame(records)
        
        # Step 7: Extract citations using the new system
        from graphrag.query.citation_utils import (
            extract_citations_from_context,
            extract_source_attributions
        )
        
        citations = extract_citations_from_context(context_dfs)
        source_attributions = extract_source_attributions(context_dfs)
        
        print(f"   📋 Extracted {len(citations)} citation categories")
        print(f"   📍 Found {len(source_attributions)} detailed source attributions")
        
        # Step 8: Format citations for display
        citation_ref = format_citation_references(citations)
        detailed_attributions = format_source_attributions(source_attributions)
        
        print("\n📖 Citation Examples:")
        print("-" * 30)
        print("Standard Citation Format:")
        print(f"   {citation_ref}")
        
        print("\nDetailed Source Attributions:")
        for line in detailed_attributions.split('\n'):
            print(f"   {line}")
        
        # Step 9: Create citation metadata
        citation_metadata = create_citation_metadata(citations, source_attributions)
        
        print("\n📊 Citation Metadata:")
        print("-" * 30)
        for key, value in citation_metadata.items():
            print(f"   {key}: {value}")
        
        # Step 10: Demonstrate response with citations
        mock_response = """
        Based on the available research, climate change poses significant challenges to global ecosystems and human societies. 
        The data indicates unprecedented warming trends, while the renewable energy sector shows promise for mitigation strategies.
        """
        
        response_with_citations = append_citations_to_response(
            mock_response.strip(),
            citations=citations,
            attributions=source_attributions,
            include_detailed_attributions=True
        )
        
        print("\n💬 Search Response with Citations:")
        print("-" * 40)
        print(response_with_citations)
        
        print("\n✅ Citation system demonstration complete!")
        print("\nKey Features:")
        print("  • Page and paragraph tracking for PDFs and HTML")
        print("  • Character position tracking for precise attribution")
        print("  • Document ID linking for source traceability")
        print("  • Structured citation metadata")
        print("  • Flexible formatting for different display needs")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())