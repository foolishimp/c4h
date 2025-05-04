"""
Type-based agent implementations for the new architecture.

This package contains agent implementations for the type-based architecture,
including:
- BaseTypeAgent: Abstract base class for all type-based agents
- GenericLLMAgent: Generic agent that processes input using an LLM
- GenericOrchestratorAgent: Orchestrates execution of skills according to a plan
- AgentFactory: Factory for creating agent instances based on configuration
"""

from .type_base_agent import BaseTypeAgent, AgentResponse
from .type_generic import GenericLLMAgent, GenericOrchestratorAgent
from .factory import AgentFactory

__all__ = ['BaseTypeAgent', 'AgentResponse', 'GenericLLMAgent', 
          'GenericOrchestratorAgent', 'AgentFactory']