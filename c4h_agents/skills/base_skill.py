"""
Base skill contract and protocol definitions.
Path: c4h_agents/skills/base_skill.py
"""

from typing import Dict, Any, Optional, Protocol, TypeVar, runtime_checkable
from c4h_agents.agents.types import SkillResult
from c4h_agents.agents.base_config import BaseConfig

T = TypeVar('T')

@runtime_checkable
class SkillProtocol(Protocol):
    """Protocol defining the contract all skills must implement"""
    
    def execute(self, **kwargs) -> SkillResult:
        """
        Execute the skill with the provided parameters
        
        Args:
            **kwargs: Arbitrary keyword arguments for skill execution
            
        Returns:
            SkillResult containing the operation outcome
        """
        ...

class BaseSkill(BaseConfig):
    """Base class for all skill implementations"""
    
    def __init__(self, config: Dict[str, Any], skill_name: str):
        """
        Initialize skill with configuration and name
        
        Args:
            config: Configuration dictionary
            skill_name: Unique name for this skill
        """
        super().__init__(config)
        self.skill_name = skill_name
        
    def execute(self, **kwargs) -> SkillResult:
        """
        Execute the skill with provided parameters
        
        Args:
            **kwargs: Arbitrary keyword arguments for skill execution
            
        Returns:
            SkillResult containing the operation outcome
        """
        raise NotImplementedError("Skills must implement execute method")
        
    def _handle_errors(self, operation: callable, *args, **kwargs) -> SkillResult:
        """
        Error handling wrapper for skill operations
        
        Args:
            operation: Callable to execute with error handling
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation
            
        Returns:
            SkillResult with error information if operation fails
        """
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}"
            )
