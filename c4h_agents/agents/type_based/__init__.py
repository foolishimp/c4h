"""
Type-based agent implementations for the new architecture.

This package contains agent implementations for the type-based architecture,
including:
- BaseTypeAgent: Abstract base class for all type-based agents
- GenericLLMAgent: Generic agent that processes input using an LLM
- GenericOrchestratorAgent: Orchestrates execution of skills according to a plan
"""

from .type_base_agent import BaseTypeAgent, AgentResponse

__all__ = ['BaseTypeAgent', 'AgentResponse']