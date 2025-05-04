#!/usr/bin/env python3
"""
Configuration converter for migrating from legacy class-based to type-based architecture.

This script converts legacy configurations to the new type-based format, allowing
for a smooth transition between architectures.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class ConfigurationConverter:
    """Converter for migrating legacy configurations to type-based format."""
    
    def __init__(self, schema_dir: str):
        """
        Initialize converter with schema directory.
        
        Args:
            schema_dir: Directory containing JSON schemas
        """
        self.schema_dir = schema_dir
        self.type_system_schema = self._load_json_schema('type_system.json')
        self.type_persona_schema = self._load_json_schema('type_persona.json')
    
    def convert_system_config(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert legacy system configuration to type-based format.
        
        Args:
            legacy_config: Legacy system configuration
            
        Returns:
            Dict[str, Any]: Type-based system configuration
        """
        # Create new system config structure
        new_config = {
            "version": "1.0",
            "agents": {},
            "skills": {},
            "orchestration": {
                "default_entry_point": None,
                "agents": []
            }
        }
        
        # Convert LLM config
        if 'llm' in legacy_config:
            new_config['llm_config'] = {
                "provider": legacy_config['llm'].get('provider', 'anthropic'),
                "model": legacy_config['llm'].get('model', 'claude-3-opus-20240229'),
                "default_parameters": {
                    "temperature": legacy_config['llm'].get('temperature', 0.7),
                    "max_tokens": legacy_config['llm'].get('max_tokens', 4000)
                }
            }
        
        # Convert agents
        if 'agents' in legacy_config:
            for agent_key, agent_config in legacy_config['agents'].items():
                agent_class = agent_config.get('class')
                persona_key = agent_config.get('persona')
                
                # Determine agent type based on class
                agent_type = self._determine_agent_type(agent_class)
                
                new_config['agents'][agent_key] = {
                    "agent_type": agent_type,
                    "persona_key": persona_key or agent_key
                }
                
                # Copy LLM overrides if present
                if 'llm_provider' in agent_config:
                    new_config['agents'][agent_key]['llm_provider'] = agent_config['llm_provider']
                if 'llm_model' in agent_config:
                    new_config['agents'][agent_key]['llm_model'] = agent_config['llm_model']
        
        # Convert skills
        if 'skills' in legacy_config:
            for skill_key, skill_config in legacy_config['skills'].items():
                skill_class = skill_config.get('class')
                skill_module = skill_config.get('module')
                
                if skill_class and skill_module:
                    new_config['skills'][skill_key] = {
                        "module": skill_module,
                        "class": skill_class,
                        "method": skill_config.get('method', 'execute'),
                        "description": skill_config.get('description', f"{skill_key} skill")
                    }
        
        # Convert orchestration
        if 'orchestration' in legacy_config:
            old_orch = legacy_config['orchestration']
            
            # Set default entry point
            new_config['orchestration']['default_entry_point'] = old_orch.get('entry_point')
            
            # Convert error handling
            if 'error_handling' in old_orch:
                new_config['orchestration']['error_handling'] = {
                    "fallback": old_orch['error_handling'].get('strategy', 'fail'),
                    "max_retries": old_orch['error_handling'].get('max_retries', 3)
                }
            
            # Convert lineage tracking
            if 'lineage' in old_orch:
                new_config['orchestration']['lineage_tracking'] = {
                    "enabled": old_orch['lineage'].get('enabled', True),
                    "storage_path": old_orch['lineage'].get('storage', 'lineage')
                }
            
            # Convert agent sequence
            if 'agents' in old_orch:
                for i, agent_item in enumerate(old_orch['agents']):
                    agent_key = agent_item.get('key', f"step_{i}")
                    agent_ref = agent_item.get('agent')
                    
                    if agent_ref:
                        # Create agent step
                        agent_step = {
                            "key": agent_key,
                            "agent": agent_ref,
                            "inputs": agent_item.get('inputs', {})
                        }
                        
                        # Add outputs if present
                        if 'outputs' in agent_item:
                            agent_step['outputs'] = agent_item['outputs']
                        
                        # Add next if present
                        if 'next' in agent_item:
                            agent_step['next'] = {"agent": agent_item['next']}
                        
                        new_config['orchestration']['agents'].append(agent_step)
        
        return new_config
    
    def convert_persona_config(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert legacy persona configuration to type-based format.
        
        Args:
            legacy_config: Legacy persona configuration
            
        Returns:
            Dict[str, Any]: Type-based persona configuration
        """
        persona_key = legacy_config.get('name', 'unknown')
        
        # Create new persona config structure
        new_config = {
            "version": "1.0",
            "persona_key": persona_key,
            "description": legacy_config.get('description', f"{persona_key} persona"),
            "llm_configuration": {
                "system_prompt": legacy_config.get('system_prompt', ''),
                "user_prompt_template": legacy_config.get('user_prompt_template', '{input}')
            }
        }
        
        # Add prompt parameters if present
        if 'llm_parameters' in legacy_config:
            new_config['llm_configuration']['prompt_parameters'] = legacy_config['llm_parameters']
        
        # Convert skill bindings
        if 'skills' in legacy_config:
            new_config['skill_bindings'] = {}
            for skill_key, skill_config in legacy_config['skills'].items():
                if isinstance(skill_config, dict):
                    new_config['skill_bindings'][skill_key] = {
                        "skill_key": skill_config.get('ref', skill_key),
                        "required": skill_config.get('required', False)
                    }
                else:
                    # Simple string reference
                    new_config['skill_bindings'][skill_key] = {
                        "skill_key": skill_config if isinstance(skill_config, str) else skill_key,
                        "required": False
                    }
        
        # Convert execution plan
        if 'execution_plan' in legacy_config:
            old_plan = legacy_config['execution_plan']
            
            new_config['execution_plan'] = {
                "enabled": True,
                "steps": []
            }
            
            if 'steps' in old_plan:
                for step in old_plan['steps']:
                    step_type = step.get('type', 'skill')
                    step_name = step.get('name', 'unnamed')
                    
                    new_step = {
                        "name": step_name,
                        "type": step_type,
                        "is_output": step.get('is_output', False)
                    }
                    
                    # Add step-specific properties based on type
                    if step_type == 'skill':
                        new_step['skill'] = step.get('skill', '')
                        if 'parameters' in step:
                            new_step['parameters'] = step['parameters']
                        if 'outputs' in step:
                            new_step['outputs'] = step['outputs']
                    elif step_type == 'loop':
                        new_step['iterate_on'] = step.get('iterate_on', '')
                        new_step['as_variable'] = step.get('as_variable', 'item')
                        new_step['steps'] = step.get('steps', [])
                        new_step['collect_results'] = step.get('collect_results', False)
                        if 'results_variable' in step:
                            new_step['results_variable'] = step['results_variable']
                    elif step_type == 'conditional':
                        new_step['condition'] = step.get('condition', '')
                        if 'then' in step:
                            new_step['then'] = step['then']
                        if 'else' in step:
                            new_step['else'] = step['else']
                    
                    new_config['execution_plan']['steps'].append(new_step)
        
        return new_config
    
    def _determine_agent_type(self, class_name: str) -> str:
        """
        Determine agent type based on legacy class name.
        
        Args:
            class_name: Legacy agent class name
            
        Returns:
            str: Type-based agent type
        """
        class_mapping = {
            # Generic LLM agents
            'GenericAgent': 'GenericLLMAgent',
            'CoderAgent': 'GenericLLMAgent',
            'DiscoveryAgent': 'GenericLLMAgent',
            'SolutionDesignerAgent': 'GenericLLMAgent',
            
            # Orchestrator agents
            'OrchestratorAgent': 'GenericOrchestratorAgent',
            'WorkflowAgent': 'GenericOrchestratorAgent'
        }
        
        return class_mapping.get(class_name, 'GenericLLMAgent')
    
    def _load_json_schema(self, filename: str) -> Dict[str, Any]:
        """
        Load JSON schema from schema directory.
        
        Args:
            filename: Schema filename
            
        Returns:
            Dict[str, Any]: JSON schema
        """
        schema_path = os.path.join(self.schema_dir, filename)
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load schema {filename}: {e}")
            return {}


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration
    """
    with open(file_path, 'r') as f:
        if file_path.endswith('.json'):
            return json.load(f)
        elif file_path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")


def save_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration to save
        file_path: Path to save configuration to
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        if file_path.endswith('.json'):
            json.dump(config, f, indent=2)
        elif file_path.endswith(('.yaml', '.yml')):
            yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Convert legacy configurations to type-based format')
    parser.add_argument('input', help='Input configuration file')
    parser.add_argument('output', help='Output configuration file')
    parser.add_argument('--type', choices=['system', 'persona'], default='system',
                      help='Type of configuration to convert')
    parser.add_argument('--schema-dir', default='config/schemas',
                      help='Directory containing JSON schemas')
    
    args = parser.parse_args()
    
    # Load input configuration
    try:
        legacy_config = load_config(args.input)
    except Exception as e:
        print(f"Error loading input configuration: {e}")
        return 1
    
    # Create converter
    converter = ConfigurationConverter(args.schema_dir)
    
    # Convert configuration
    try:
        if args.type == 'system':
            new_config = converter.convert_system_config(legacy_config)
        else:
            new_config = converter.convert_persona_config(legacy_config)
    except Exception as e:
        print(f"Error converting configuration: {e}")
        return 1
    
    # Save output configuration
    try:
        save_config(new_config, args.output)
        print(f"Successfully converted {args.type} configuration to {args.output}")
    except Exception as e:
        print(f"Error saving output configuration: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())