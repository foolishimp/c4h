"""
Factory for creating type-based agent instances.

This module provides functionality for instantiating agents based on
their type, as defined in configuration.
"""

import json
import os
import uuid
import logging
from typing import Any, Dict, Optional, Type

from c4h_agents.context.execution_context import ExecutionContext
from c4h_agents.skills.registry import SkillRegistry
from c4h_agents.agents.type_based.type_base_agent import BaseTypeAgent
from c4h_agents.agents.type_based.type_generic import GenericLLMAgent, GenericOrchestratorAgent

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating agent instances based on configuration."""
    
    # Registry of agent types
    _agent_types = {
        'GenericLLMAgent': GenericLLMAgent,
        'GenericOrchestratorAgent': GenericOrchestratorAgent
    }
    
    def __init__(self, config_path: str):
        """
        Initialize the agent factory.
        
        Args:
            config_path: Path to system configuration file
        """
        self.config_path = config_path
        self.system_config = self._load_config(config_path)
        
        # Create skill registry
        self.skill_registry = SkillRegistry(self.system_config.get('skills', {}))
        
        # Load persona configurations
        self.persona_configs = self._load_persona_configs()
    
    def create_agent(self, agent_key: str, context: Dict[str, Any]) -> BaseTypeAgent:
        """
        Create an agent instance based on configuration.
        
        Args:
            agent_key: Key of the agent configuration to use
            context: Initial execution context
            
        Returns:
            BaseTypeAgent: Instance of the agent
        """
        # Get agent configuration
        agent_config = self.system_config.get('agents', {}).get(agent_key)
        if not agent_config:
            raise ValueError(f"Agent '{agent_key}' not found in configuration")
        
        # Get agent type
        agent_type = agent_config.get('agent_type')
        if not agent_type:
            raise ValueError(f"Agent '{agent_key}' missing type in configuration")
        
        # Get agent class
        agent_class = self._get_agent_class(agent_type)
        
        # Get persona configuration
        persona_key = agent_config.get('persona_key')
        if not persona_key:
            raise ValueError(f"Agent '{agent_key}' missing persona_key in configuration")
        
        persona_config = self.persona_configs.get(persona_key)
        if not persona_config:
            raise ValueError(f"Persona '{persona_key}' not found")
        
        # Create execution context with proper lineage structure
        exec_context = {
            'system': self.system_config,
            'agent': {
                'key': agent_key,
                'config': agent_config
            }
        }
        
        # Merge with provided context
        exec_context.update(context)
        
        # Ensure workflow run ID is set up for lineage tracking
        run_id = exec_context.get('workflow_run_id')
        if not run_id:
            # Check in system section
            if 'system' in exec_context and isinstance(exec_context['system'], dict):
                run_id = exec_context['system'].get('runid')
            
            # Generate if not found
            if not run_id:
                import datetime
                time_str = datetime.datetime.now().strftime('%H%M')
                run_id = f"wf_{time_str}_{uuid.uuid4()}"
                logger.info(f"Generated new workflow run ID: {run_id}")
        
        # Ensure run_id is set in all required locations
        exec_context['workflow_run_id'] = run_id
        if 'system' not in exec_context:
            exec_context['system'] = {}
        exec_context['system']['runid'] = run_id
        
        # Set up runtime section if missing
        if 'runtime' not in exec_context:
            exec_context['runtime'] = {}
        # Add run_id to runtime for compatibility
        exec_context['runtime']['workflow_run_id'] = run_id
        
        # Configure lineage in BOTH locations where BaseLineage looks for it
        # Use absolute path for lineage to ensure consistency
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
        lineage_path = os.path.join(repo_root, 'workspaces/lineage')
        
        lineage_config = {
            'enabled': True,
            'namespace': 'c4h_agents.type_based',
            'backends': {
                'file': {
                    'enabled': True,
                    'path': lineage_path
                }
            }
        }
        
        # Set in runtime.lineage (first fallback location)
        if 'runtime' not in exec_context:
            exec_context['runtime'] = {}
        exec_context['runtime']['lineage'] = lineage_config
        
        # Set in llm_config.agents.lineage (primary location)
        if 'llm_config' not in exec_context:
            exec_context['llm_config'] = {}
        if 'agents' not in exec_context['llm_config']:
            exec_context['llm_config']['agents'] = {}
        exec_context['llm_config']['agents']['lineage'] = lineage_config
        
        # Create agent instance
        logger.info(f"Creating agent '{agent_key}' of type '{agent_type}' with run_id {run_id}")
        agent = agent_class(persona_config, exec_context, self.skill_registry)
        
        return agent
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a file."""
        if not os.path.exists(config_path):
            raise ValueError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    return json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    return yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported configuration format: {config_path}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def _load_persona_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all persona configurations."""
        personas = {}
        
        # Get persona directory from system config
        persona_dir = self.system_config.get('persona_directory', 'personas')
        
        # Ensure absolute path
        if not os.path.isabs(persona_dir):
            # Assume relative to config directory
            config_dir = os.path.dirname(self.config_path)
            persona_dir = os.path.join(config_dir, persona_dir)
        
        # Load all persona files
        if os.path.exists(persona_dir):
            for filename in os.listdir(persona_dir):
                if filename.endswith(('.yaml', '.yml', '.json')):
                    try:
                        file_path = os.path.join(persona_dir, filename)
                        persona_config = self._load_config(file_path)
                        
                        # Get persona key
                        persona_key = persona_config.get('persona_key')
                        if persona_key:
                            personas[persona_key] = persona_config
                    except Exception as e:
                        logger.warning(f"Failed to load persona from {filename}: {e}")
        
        return personas
    
    def _get_agent_class(self, agent_type: str) -> Type[BaseTypeAgent]:
        """Get agent class by type."""
        if agent_type not in self._agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return self._agent_types[agent_type]
    
    def register_agent_type(self, type_name: str, agent_class: Type[BaseTypeAgent]) -> None:
        """
        Register a new agent type.
        
        Args:
            type_name: Name of the agent type
            agent_class: Agent class to register
        """
        self._agent_types[type_name] = agent_class