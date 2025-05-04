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
            agent_type: The type of agent to instantiate 
                   (e.g., "generic_single_shot", "GenericLLMAgent", "GenericOrchestratorAgent")
            
        Returns:
            The agent class that inherits from BaseAgent
        """
        try:
            # Enhanced hardcoded mapping for both legacy and new agent types
            agent_type_map = {
                # Legacy agent types
                "generic_single_shot": "c4h_agents.agents.generic.GenericSingleShotAgent",
                "generic_orchestrating": "c4h_agents.agents.generic.GenericOrchestratorAgent", # Updated for compatibility
                
                # New type-based agent types
                "GenericLLMAgent": "c4h_agents.agents.generic.GenericLLMAgent",
                "GenericOrchestratorAgent": "c4h_agents.agents.generic.GenericOrchestratorAgent",
                "GenericSkillAgent": "c4h_agents.agents.generic.GenericSkillAgent",
                "GenericFallbackAgent": "c4h_agents.agents.generic.GenericFallbackAgent",
                
                # Lowercase versions for convenience
                "generic_llm": "c4h_agents.agents.generic.GenericLLMAgent",
                "generic_orchestrator": "c4h_agents.agents.generic.GenericOrchestratorAgent",
                "generic_skill": "c4h_agents.agents.generic.GenericSkillAgent",
                "generic_fallback": "c4h_agents.agents.generic.GenericFallbackAgent"
            }
            
            # Check if we have a known mapping for this agent type
            if agent_type in agent_type_map:
                # Use the hardcoded mapping
                class_path = agent_type_map[agent_type]
                module_path, class_name = class_path.rsplit(".", 1)
                
                logger.info("factory.resolving_agent_class_mapped", 
                            agent_type=agent_type,
                            module_path=module_path, 
                            class_name=class_name)
                
                # Import the class using the mapping
                return getattr(importlib.import_module(module_path), class_name)
            
            # Try handling AgentType enum values directly
            # First check if it looks like an enum value (all caps with underscores)
            if agent_type.isupper() and "_" in agent_type:
                # Try to resolve from AgentType
                try:
                    from c4h_agents.agents.types import AgentType
                    # Convert to AgentType enum value
                    agent_enum = AgentType[agent_type]
                    # Use the value from the enum (e.g., "GenericLLMAgent")
                    enum_value = agent_enum.value
                    
                    logger.info("factory.resolving_agent_class_from_enum", 
                                enum_name=agent_type,
                                enum_value=enum_value)
                    
                    # Use recursive call with the enum value
                    return self._get_agent_class(enum_value)
                except (ImportError, KeyError) as e:
                    logger.warning("factory.enum_resolution_failed", 
                                  agent_type=agent_type,
                                  error=str(e))
                    # Continue to dynamic resolution
            
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
            # Extract agent_type, unique_name, and persona_key from task_config
            agent_type = task_config.get("agent_type")
            unique_name = task_config.get("name")
            persona_key = task_config.get("persona_key")
            
            if not agent_type:
                raise ValueError("Missing required field 'agent_type' in task_config")
            if not unique_name:
                raise ValueError("Missing required field 'name' in task_config")
            
            # Log agent creation with details
            log_data = {
                "agent_type": agent_type,
                "unique_name": unique_name
            }
            
            if persona_key:
                log_data["persona_key"] = persona_key
                
            logger.info("factory.creating_agent", **log_data)
            
            # Get the agent class
            agent_class = self._get_agent_class(agent_type)
            
            # Ensure agent configuration exists in effective config for this agent
            agent_config = self.effective_config_snapshot.get("llm_config", {}).get("agents", {})
            if unique_name not in agent_config:
                # Create the config entry if it doesn't exist
                if "llm_config" not in self.effective_config_snapshot:
                    self.effective_config_snapshot["llm_config"] = {}
                if "agents" not in self.effective_config_snapshot["llm_config"]:
                    self.effective_config_snapshot["llm_config"]["agents"] = {}
                    
                # Add a minimal agent config entry
                self.effective_config_snapshot["llm_config"]["agents"][unique_name] = {
                    "agent_type": agent_type
                }
                
                # Add persona_key if provided
                if persona_key:
                    self.effective_config_snapshot["llm_config"]["agents"][unique_name]["persona_key"] = persona_key
                    
                logger.info("factory.created_agent_config", 
                           unique_name=unique_name, 
                           agent_type=agent_type,
                           persona_key=persona_key)
            
            # Instantiate the agent with full effective config and unique name
            agent = agent_class(self.effective_config_snapshot, unique_name)
            
            return agent
            
        except Exception as e:
            logger.error("factory.agent_creation_failed", 
                        error=str(e), 
                        agent_type=task_config.get("agent_type", "unknown"),
                        unique_name=task_config.get("name", "unknown"))
            raise