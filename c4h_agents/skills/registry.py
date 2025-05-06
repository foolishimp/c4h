"""
Skill registry for dynamic skill discovery and invocation.

This module provides the functionality for registering, discovering,
and invoking skills in the type-based architecture.
"""

import importlib
import logging
from typing import Any, Dict, Optional, Union
from c4h_agents.agents.types import SkillResult

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Registry for skills with dynamic loading."""
    
    def __init__(self, skill_configs: Dict[str, Dict[str, Any]]):
        """
        Initialize the skill registry.
        
        Args:
            skill_configs: Dictionary mapping skill keys to configurations
        """
        self.skill_configs = skill_configs
        self.skill_instances = {}
    
    def invoke_skill(self, skill_key: str, **kwargs) -> SkillResult:
        """
        Invoke a skill by key with parameters.
        
        Args:
            skill_key: Key of the skill to invoke
            **kwargs: Parameters to pass to the skill
            
        Returns:
            SkillResult[Any]: Standardized result from skill execution
        """
        try:
            if skill_key not in self.skill_configs:
                return SkillResult(
                    success=False,
                    error=f"Skill '{skill_key}' not found in registry"
                )
            
            # Get skill configuration
            skill_config = self.skill_configs[skill_key]
            
            # Get or create skill instance
            skill_instance = self._get_skill_instance(skill_key, skill_config)
            
            # Get method to invoke
            method_name = skill_config.get('method', 'execute')
            if not hasattr(skill_instance, method_name):
                return SkillResult(
                    success=False,
                    error=f"Skill '{skill_key}' has no method '{method_name}'"
                )
            
            method = getattr(skill_instance, method_name)
            
            # Invoke skill
            logger.info(f"Invoking skill '{skill_key}.{method_name}'")
            result = method(**kwargs)
        except Exception as e:
            logger.error(f"Error invoking skill '{skill_key}': {str(e)}")
            return SkillResult(
                success=False,
                error=f"Skill invocation error: {str(e)}"
            )
        
        # Process result
        if isinstance(result, SkillResult):
            # Already a standardized SkillResult, return as-is
            return result
        elif isinstance(result, tuple) and len(result) == 2:
            # Assume (result, outputs) tuple
            return SkillResult(
                success=True,
                value=result[0],
                metrics=result[1]
            )
        elif isinstance(result, dict) and 'result' in result and 'outputs' in result:
            # Assume result dict with 'result' and 'outputs' keys
            return SkillResult(
                success=True,
                value=result['result'],
                metrics=result['outputs']
            )
        else:
            # Assume result only
            return SkillResult(
                success=True,
                value=result
            )
    
    def _get_skill_instance(self, skill_key: str, skill_config: Dict[str, Any]) -> Optional[Any]:
        """
        Get or create a skill instance.
        
        Args:
            skill_key: Key of the skill
            skill_config: Configuration for the skill
            
        Returns:
            Optional[Any]: Skill instance or None if creation fails
        """
        if skill_key in self.skill_instances:
            return self.skill_instances[skill_key]
        
        # Load module and class
        module_name = skill_config.get('module')
        class_name = skill_config.get('class')
        
        if not module_name or not class_name:
            logger.error(f"Skill '{skill_key}' missing module or class in configuration")
            return None
        
        try:
            module = importlib.import_module(module_name)
            skill_class = getattr(module, class_name)
            
            # Instantiate skill
            skill_instance = skill_class()
            
            # Cache instance
            self.skill_instances[skill_key] = skill_instance
            
            return skill_instance
        except Exception as e:
            logger.error(f"Failed to instantiate skill '{skill_key}': {e}")
            return None
    
    def list_skills(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available skills.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping skill keys to metadata
        """
        return {
            key: {
                'module': config.get('module', ''),
                'class': config.get('class', ''),
                'method': config.get('method', 'execute'),
                'description': config.get('description', '')
            }
            for key, config in self.skill_configs.items()
        }