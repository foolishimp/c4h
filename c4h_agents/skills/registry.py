"""
Skill registry for dynamic skill discovery and invocation.

This module provides the functionality for registering, discovering,
and invoking skills in the type-based architecture.
"""

import importlib
import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class SkillResult:
    """Result from a skill execution."""
    
    def __init__(self, result: Any, outputs: Optional[Dict[str, Any]] = None):
        self.result = result
        self.outputs = outputs or {}


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
            SkillResult: Result from skill execution
        """
        if skill_key not in self.skill_configs:
            raise ValueError(f"Skill '{skill_key}' not found in registry")
        
        # Get skill configuration
        skill_config = self.skill_configs[skill_key]
        
        # Get or create skill instance
        skill_instance = self._get_skill_instance(skill_key, skill_config)
        
        # Get method to invoke
        method_name = skill_config.get('method', 'execute')
        if not hasattr(skill_instance, method_name):
            raise ValueError(f"Skill '{skill_key}' has no method '{method_name}'")
        
        method = getattr(skill_instance, method_name)
        
        # Invoke skill
        logger.info(f"Invoking skill '{skill_key}.{method_name}'")
        result = method(**kwargs)
        
        # Process result
        if isinstance(result, tuple) and len(result) == 2:
            # Assume (result, outputs) tuple
            return SkillResult(result[0], result[1])
        elif isinstance(result, dict) and 'result' in result and 'outputs' in result:
            # Assume result dict with 'result' and 'outputs' keys
            return SkillResult(result['result'], result['outputs'])
        else:
            # Assume result only
            return SkillResult(result)
    
    def _get_skill_instance(self, skill_key: str, skill_config: Dict[str, Any]) -> Any:
        """
        Get or create a skill instance.
        
        Args:
            skill_key: Key of the skill
            skill_config: Configuration for the skill
            
        Returns:
            Any: Skill instance
        """
        if skill_key in self.skill_instances:
            return self.skill_instances[skill_key]
        
        # Load module and class
        module_name = skill_config.get('module')
        class_name = skill_config.get('class')
        
        if not module_name or not class_name:
            raise ValueError(f"Skill '{skill_key}' missing module or class in configuration")
        
        try:
            module = importlib.import_module(module_name)
            skill_class = getattr(module, class_name)
            
            # Instantiate skill
            skill_instance = skill_class()
            
            # Cache instance
            self.skill_instances[skill_key] = skill_instance
            
            return skill_instance
        except Exception as e:
            raise ValueError(f"Failed to instantiate skill '{skill_key}': {e}")
    
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