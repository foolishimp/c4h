"""
MCP-based Pydantic models for standardized inter-team communication.

This module defines data models that mirror the Anthropic Messages API structure
to provide a consistent contract for data exchange between teams.
"""

from pydantic import BaseModel, Field, field_serializer
from typing import List, Literal, Union, Optional, Dict, Any
from datetime import datetime


class TextContentBlock(BaseModel):
    """A text content block in a message."""
    type: Literal["text"] = "text"
    text: str


# Union type for content blocks - can be extended with other block types
ContentBlock = Union[TextContentBlock]


class Message(BaseModel):
    """
    A message in the conversation format.
    
    This follows the Anthropic Messages API structure for consistency.
    """
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]
    
    # Optional metadata for tracking
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, timestamp: Optional[str], _info) -> Optional[str]:
        """Ensure timestamp is always a string."""
        if isinstance(timestamp, datetime):
            return timestamp.isoformat()
        return timestamp


class TeamHandoff(BaseModel):
    """
    Standardized data structure for team-to-team handoffs.
    
    This ensures consistent data contracts between teams and prevents
    context bleeding by explicitly defining what data is passed.
    """
    # The conversation history up to this point
    messages: List[Message]
    
    # The specific output from the current team
    team_output: Optional[Dict[str, Any]] = None
    
    # Metadata about the handoff
    source_team: str
    target_team: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Workflow context (immutable reference data)
    workflow_context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Immutable workflow context like project info, intent, etc."
    )
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, timestamp: Union[str, datetime], _info) -> str:
        """Ensure timestamp is always a string."""
        if isinstance(timestamp, datetime):
            return timestamp.isoformat()
        return timestamp


class TeamResult(BaseModel):
    """
    Standardized result structure from a team execution.
    
    This replaces the ad-hoc dictionary returns with a typed structure.
    """
    success: bool
    team_id: str
    
    # The data produced by this team
    output_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Next team routing
    next_team: Optional[str] = None
    
    # Error information if failed
    error: Optional[str] = None
    
    # Execution metadata
    duration_seconds: Optional[float] = None
    agents_executed: Optional[int] = None