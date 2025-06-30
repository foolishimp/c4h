"""
Messages package for standardized inter-team communication.

This package provides MCP-based data models for consistent data exchange
between teams in the C4H workflow system.
"""

from .models import (
    TextContentBlock,
    ContentBlock,
    Message,
    TeamHandoff,
    TeamResult
)

__all__ = [
    "TextContentBlock",
    "ContentBlock", 
    "Message",
    "TeamHandoff",
    "TeamResult"
]