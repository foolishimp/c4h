"""
Base class for all type-based agents.

This module defines the abstract base class for all type-based agents
in the new architecture. It provides common functionality and interface
definitions.
"""

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ...context.execution_context import ExecutionContext


class AgentResponse:
    """Structured response from agent processing."""
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        context_updates: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.metadata = metadata or {}
        self.context_updates = context_updates or {}


class BaseTypeAgent(ABC):
    """Base class for all type-based agents."""
    
    def __init__(self, persona_config: Dict[str, Any], context: Dict[str, Any], skill_registry: Any):
        """
        Initialize a type-based agent.
        
        Args:
            persona_config: Configuration for this agent's persona
            context: Initial execution context
            skill_registry: Registry for skill discovery and invocation
        """
        self.agent_id = str(uuid.uuid4())
        self.agent_type = self.__class__.__name__
        self.persona_config = persona_config
        self.context = ExecutionContext(context)
        self.skill_registry = skill_registry
    
    @abstractmethod
    def process(self, **kwargs) -> AgentResponse:
        """
        Process input according to the agent's behavior.
        
        Args:
            **kwargs: Input parameters for processing
            
        Returns:
            AgentResponse: Structured response from processing
        """
        pass
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "persona": self.persona_config.get("persona_key", "unknown")
        }