"""
Agent package initialization.
Provides both legacy and type-based agent imports.
"""

# Legacy agent classes
from c4h_agents.agents.discovery import DiscoveryAgent
from c4h_agents.agents.solution_designer import SolutionDesigner
from c4h_agents.agents.coder import Coder

# Type-based agents
from c4h_agents.agents.generic import (
    GenericLLMAgent,
    GenericSingleShotAgent,
    GenericOrchestratorAgent,
    GenericSkillAgent,
    GenericFallbackAgent
)