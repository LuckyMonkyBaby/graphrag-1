# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing source location tracking models."""

from dataclasses import dataclass


@dataclass
class SourceLocation:
    """Source location information for text chunks."""
    file_path: str
    start_line: int
    end_line: int
    start_char: int = 0
    end_char: int = 0