"""
Compatibility factory for supporting both legacy and type-based agents.
This module provides a compatibility layer between the original class-based
approach and the new type-based agent architecture.

Path: c4h_services/src/orchestration/compat_factory.py
"""

from typing import Dict, Any, Optional, Union, Type
import importlib
import structlog
from pathlib import Path
import os

from c4h_agents.agents.base_agent import BaseAgent
from c4h_agents.agents.generic import (
    GenericLLMAgent,
    GenericSingleShotAgent,
    GenericOrchestratorAgent,
    GenericSkillAgent,
    GenericFallbackAgent
)
from c4h_services.src.orchestration.factory import AgentFactory
from c4h_services.src.intent.impl.prefect.models import AgentTaskConfig

logger = structlog.get_logger()

class CompatAgentFactory:
    """
    Enhanced agent factory with backward compatibility support.
    
    This factory bridges between:
    1. The original factory that creates agents from agent_class strings
    2. The new type-based factory that creates agents from agent_type strings
    
    It intelligently selects the right approach based on the task configuration.
    """
    
    def __init__(self, effective_config_snapshot: Dict[str, Any], config_path: Optional[Path] = None):
        """
        Initialize with effective configuration snapshot.
        
        Args:
            effective_config_snapshot: Complete effective configuration
            config_path: Optional path to the config file for reference
        """
        self.config = effective_config_snapshot
        self.config_path = config_path
        
        # Initialize the new type-based factory (only needs the config)
        self.type_factory = AgentFactory(effective_config_snapshot)
        
        # Extract compatibility mappings if present
        self.agent_mappings = self.config.get('orchestration', {}).get('agent_mappings', {})
        self.persona_mappings = self.config.get('orchestration', {}).get('persona_mappings', {})
        
        # Map of agent class name -> class object
        self._agent_classes = {}
        
        logger.info("compat_factory.initialized", 
                   mappings_count=len(self.agent_mappings),
                   config_path=str(config_path) if config_path else None)
    
    def _resolve_agent_class(self, agent_class_path: str) -> Type[BaseAgent]:
        """
        Resolve an agent class from its string path.
        
        Args:
            agent_class_path: Full path to agent class, e.g., "c4h_agents.agents.discovery.DiscoveryAgent"
            
        Returns:
            BaseAgent subclass
            
        Raises:
            ImportError: If the module cannot be imported
            AttributeError: If the class does not exist in the module
        """
        # Check if we've already resolved this class
        if agent_class_path in self._agent_classes:
            return self._agent_classes[agent_class_path]
        
        try:
            # Split into module path and class name
            module_path, class_name = agent_class_path.rsplit('.', 1)
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the class
            agent_class = getattr(module, class_name)
            
            # Store for future use
            self._agent_classes[agent_class_path] = agent_class
            
            return agent_class
        except (ImportError, AttributeError) as e:
            logger.error("compat_factory.class_resolution_failed", 
                       agent_class=agent_class_path, 
                       error=str(e))
            raise
    
    def _get_agent_type_for_class(self, agent_class_path: str) -> str:
        """
        Map a legacy agent class to the corresponding agent type.
        
        Args:
            agent_class_path: Full path to agent class
            
        Returns:
            Corresponding agent type or 'GenericLLMAgent' as fallback
        """
        agent_type = self.agent_mappings.get(agent_class_path, 'GenericLLMAgent')
        logger.debug("compat_factory.mapped_class_to_type", 
                   agent_class=agent_class_path, 
                   agent_type=agent_type)
        return agent_type
    
    def _get_persona_for_class(self, agent_class_path: str) -> Optional[str]:
        """
        Infer persona key from agent class path.
        
        Args:
            agent_class_path: Full path to agent class
            
        Returns:
            Corresponding persona key or None
        """
        # Extract the class name portion
        try:
            class_name = agent_class_path.rsplit('.', 1)[1]
            
            # Check if we have a direct mapping
            for legacy_name, persona_key in self.persona_mappings.items():
                if legacy_name.lower() in class_name.lower():
                    return persona_key
            
            # Try to infer from class name
            if 'Discovery' in class_name:
                return 'discovery'
            elif 'Solution' in class_name:
                return 'solution_designer'
            elif 'Coder' in class_name:
                return 'coder'
            
            return None
        except Exception:
            return None
    
    def create_agent(self, task_config: Union[Dict[str, Any], AgentTaskConfig]) -> BaseAgent:
        """
        Create an agent instance based on task configuration.
        
        This method handles both legacy and new agent configurations:
        - If agent_class is specified, it uses the legacy approach
        - If agent_type is specified, it uses the new type-based approach
        
        Args:
            task_config: Agent task configuration (dict or AgentTaskConfig)
            
        Returns:
            Instantiated agent
            
        Raises:
            ValueError: If neither agent_class nor agent_type is specified
        """
        # Convert to dict if AgentTaskConfig is provided
        if isinstance(task_config, AgentTaskConfig):
            task_config_dict = task_config.dict()
        else:
            task_config_dict = dict(task_config)
        
        agent_class_path = task_config_dict.get('agent_class')
        agent_type = task_config_dict.get('agent_type')
        agent_name = task_config_dict.get('name', 'unknown_agent')
        
        # Check if both agent_class and agent_type are specified
        if agent_class_path and agent_type:
            logger.warning("compat_factory.both_class_and_type_specified",
                        agent_class=agent_class_path,
                        agent_type=agent_type,
                        using="agent_class")
        
        # Legacy approach - class-based
        if agent_class_path:
            logger.info("compat_factory.using_legacy_approach", 
                      agent_class=agent_class_path, 
                      agent_name=agent_name)
            
            try:
                # Check if we should map to a new agent type
                if agent_class_path in self.agent_mappings:
                    # Get the mapped agent type
                    mapped_agent_type = self._get_agent_type_for_class(agent_class_path)
                    
                    # Get the persona key if not specified
                    persona_key = task_config_dict.get('persona_key')
                    if not persona_key:
                        persona_key = self._get_persona_for_class(agent_class_path)
                        if persona_key:
                            task_config_dict['persona_key'] = persona_key
                    
                    logger.info("compat_factory.mapped_to_new_type", 
                              agent_class=agent_class_path,
                              agent_type=mapped_agent_type,
                              persona_key=persona_key)
                    
                    # Create using the type-based factory with the mapped type
                    task_config_dict['agent_type'] = mapped_agent_type
                    return self.type_factory.create_agent(task_config_dict)
                
                # Fall back to direct class instantiation
                agent_class = self._resolve_agent_class(agent_class_path)
                return agent_class(self.config, agent_name)
                
            except Exception as e:
                logger.error("compat_factory.legacy_agent_creation_failed", 
                           agent_class=agent_class_path,
                           error=str(e))
                raise
        
        # New approach - type-based
        elif agent_type:
            logger.info("compat_factory.using_new_approach", 
                      agent_type=agent_type, 
                      agent_name=agent_name)
            return self.type_factory.create_agent(task_config_dict)
        
        # Neither approach specified
        else:
            error_msg = f"Neither agent_class nor agent_type specified for agent {agent_name}"
            logger.error("compat_factory.missing_agent_specification", 
                       agent_name=agent_name,
                       task_config=task_config_dict)
            raise ValueError(error_msg)
    
    def get_config(self) -> Dict[str, Any]:
        """Return the effective configuration snapshot."""
        return self.config