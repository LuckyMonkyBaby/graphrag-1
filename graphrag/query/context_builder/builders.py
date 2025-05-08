# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Base classes for global and local context builders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any

import pandas as pd

from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)


@dataclass
class SourceReference:
    """Source reference information for text chunks."""
    file_path: str
    start_line: int
    end_line: int
    start_char: int = 0
    end_char: int = 0
    text: str = ""


@dataclass
class ContextBuilderResult:
    """A class to hold the results of the build_context."""

    context_chunks: str | list[str]
    context_records: dict[str, pd.DataFrame]
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    source_references: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_source_references_text(self) -> str:
        """Format source references as text for the prompt."""
        if not self.source_references:
            return "No source file information available."
        
        sources_text = []
        for ref in self.source_references:
            file_path = ref.get('source_file', 'Unknown')
            start_line = ref.get('source_line_start', 0)
            end_line = ref.get('source_line_end', 0)
            
            if start_line and end_line:
                sources_text.append(f"{file_path} (lines {start_line}-{end_line})")
            else:
                sources_text.append(file_path)
        
        if sources_text:
            return "Source references:\n" + "\n".join(sources_text)
        else:
            return "No source file information available."


class GlobalContextBuilder(ABC):
    """Base class for global-search context builders."""

    @abstractmethod
    async def build_context(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> ContextBuilderResult:
        """Build the context for the global search mode."""


class LocalContextBuilder(ABC):
    """Base class for local-search context builders."""

    @abstractmethod
    def build_context(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> ContextBuilderResult:
        """Build the context for the local search mode."""


class DRIFTContextBuilder(ABC):
    """Base class for DRIFT-search context builders."""

    @abstractmethod
    async def build_context(
        self,
        query: str,
        **kwargs,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Build the context for the primer search actions."""


class BasicContextBuilder(ABC):
    """Base class for basic-search context builders."""

    @abstractmethod
    def build_context(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> ContextBuilderResult:
        """Build the context for the basic search mode."""