"""
AgentFactory implementation for creating agent instances.
Path: c4h_services/src/orchestration/factory.py
"""

from typing import Dict, Any, Type
import importlib
from c4h_services.src.utils.logging import get_logger
from c4h_agents.agents.base_agent import BaseAgent

logger = get_logger()

class AgentFactory:
    """
    Factory responsible for creating agent instances based on configuration.
    Uses the effective configuration snapshot for instantiation.
    """
    
    def __init__(self, effective_config_snapshot: Dict[str, Any]):
        """
        Initialize the factory with the full effective configuration snapshot.
        
        Args:
            effective_config_snapshot: The complete merged and resolved configuration
        """
        self.effective_config_snapshot = effective_config_snapshot
    
    def _get_agent_class(self, agent_type: str) -> Type[BaseAgent]:
        """
        Map agent_type string to the corresponding agent class.
        
        Args:
            agent_type: The type of agent to instantiate (e.g., "generic_single_shot")
            
        Returns:
            The agent class that inherits from BaseAgent
        """
        try:
            # Hardcoded mapping instead of dynamic resolution to ensure we get the right classes
            agent_type_map = {
                "generic_single_shot": "c4h_agents.agents.generic.GenericSingleShotAgent",
                "generic_orchestrating": "c4h_agents.agents.generic.GenericOrchestratingAgent"
            }
            
            # Check if we have a known mapping for this agent type
            if agent_type not in agent_type_map:
                # Fall back to the original dynamic resolution
                parts = agent_type.split('_')
                agent_category = parts[0]  # e.g. 'generic'
                
                # Build class name (e.g. 'GenericSingleShotAgent')
                class_name = ''.join(part.capitalize() for part in parts) + 'Agent'
                
                # Build import path (e.g. 'c4h_agents.agents.generic')
                module_path = f"c4h_agents.agents.{agent_category}"
                
                logger.info("factory.resolving_agent_class_dynamic", 
                            agent_type=agent_type,
                            module_path=module_path, 
                            class_name=class_name)
                
                # Import the class dynamically
                return getattr(importlib.import_module(module_path), class_name)
            
            # Use the hardcoded mapping
            class_path = agent_type_map[agent_type]
            module_path, class_name = class_path.rsplit(".", 1)
            
            logger.info("factory.resolving_agent_class_mapped", 
                        agent_type=agent_type,
                        module_path=module_path, 
                        class_name=class_name)
            
            # Import the class using the mapping
            return getattr(importlib.import_module(module_path), class_name)
        
        except (ValueError, ImportError, AttributeError) as e:
            logger.error("factory.agent_class_loading_failed", agent_type=agent_type, error=str(e))
            raise ValueError(f"Failed to load agent class for type {agent_type}: {str(e)}")
    
    def create_agent(self, task_config: Dict[str, Any]) -> BaseAgent:
        """
        Create an agent instance based on the task configuration.
        
        Args:
            task_config: Configuration for the task including agent_type and name
            
        Returns:
            An instantiated agent object
        """
        try:
            # Extract agent_type and unique_name from task_config
            agent_type = task_config.get("agent_type")
            unique_name = task_config.get("name")
            
            if not agent_type:
                raise ValueError("Missing required field 'agent_type' in task_config")
            if not unique_name:
                raise ValueError("Missing required field 'name' in task_config")
            
            # Get the agent class
            agent_class = self._get_agent_class(agent_type)
            
            # Instantiate the agent with full effective config and unique name
            # Pass the effective_config_snapshot as the first positional parameter (full_effective_config)
            # and unique_name as the second positional parameter
            logger.info("factory.creating_agent", agent_type=agent_type, unique_name=unique_name)
            
            # Add debug print to see what's being called
            print(f"DEBUG - Creating agent: {agent_class.__name__}({type(self.effective_config_snapshot).__name__}, {unique_name})")
            
            # Explicitly use positional arguments
            agent = agent_class(self.effective_config_snapshot, unique_name)
            return agent
            
        except Exception as e:
            logger.error("factory.agent_creation_failed", error=str(e))
            raise