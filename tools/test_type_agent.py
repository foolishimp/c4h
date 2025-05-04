#!/usr/bin/env python3
"""
Test script for the type-based architecture.

This script provides a simple way to test the type-based architecture
with a configuration file.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import type-based architecture components
from c4h_agents.agents.type_based import AgentFactory, GenericLLMAgent, GenericOrchestratorAgent
from c4h_agents.context.execution_context import ExecutionContext
from c4h_agents.skills.registry import SkillRegistry


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


def test_agent_factory(config_path: str) -> None:
    """
    Test agent factory with a configuration file.
    
    Args:
        config_path: Path to system configuration file
    """
    print(f"Loading configuration from {config_path}")
    factory = AgentFactory(config_path)
    
    # Print available agents
    print("\nAvailable agents:")
    for agent_key in factory.system_config.get('agents', {}):
        agent_config = factory.system_config['agents'][agent_key]
        agent_type = agent_config.get('agent_type', 'unknown')
        persona_key = agent_config.get('persona_key', 'unknown')
        print(f"  - {agent_key} ({agent_type}, persona: {persona_key})")
    
    # Print available skills
    print("\nAvailable skills:")
    for skill_key, skill_meta in factory.skill_registry.list_skills().items():
        print(f"  - {skill_key} ({skill_meta['module']}.{skill_meta['class']}.{skill_meta['method']})")
        
    # Test entry point
    entry_point = factory.system_config.get('orchestration', {}).get('default_entry_point')
    if entry_point:
        print(f"\nDefault entry point: {entry_point}")


def test_agent_execution(config_path: str, agent_key: str, input_text: str) -> None:
    """
    Test execution of an agent.
    
    Args:
        config_path: Path to system configuration file
        agent_key: Key of the agent to test
        input_text: Input text for the agent
    """
    print(f"Testing agent '{agent_key}' with input: {input_text}")
    
    # Create agent factory
    factory = AgentFactory(config_path)
    
    # Create agent
    agent = factory.create_agent(agent_key, {
        'runtime': {
            'input': input_text
        }
    })
    
    # Print agent info
    print("\nAgent info:")
    for key, value in agent.get_agent_info().items():
        print(f"  - {key}: {value}")
    
    # Execute agent
    print("\nExecuting agent...")
    try:
        response = agent.process(input=input_text)
        
        # Print response
        print("\nAgent response:")
        print(f"  Content: {response.content}")
        print("\n  Metadata:")
        for key, value in response.metadata.items():
            if isinstance(value, dict) and len(value) > 100:
                print(f"    {key}: <large dict>")
            else:
                print(f"    {key}: {value}")
        
        if response.context_updates:
            print("\n  Context updates:")
            for key, value in response.context_updates.items():
                print(f"    {key}: {value}")
    except Exception as e:
        print(f"Error executing agent: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test type-based architecture')
    parser.add_argument('--config', default='config/system_config.yml',
                      help='Path to system configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Factory test command
    factory_parser = subparsers.add_parser('factory', help='Test agent factory')
    
    # Agent execution test command
    agent_parser = subparsers.add_parser('agent', help='Test agent execution')
    agent_parser.add_argument('agent', help='Key of the agent to test')
    agent_parser.add_argument('input', help='Input text for the agent')
    
    args = parser.parse_args()
    
    if args.command == 'factory':
        test_agent_factory(args.config)
    elif args.command == 'agent':
        test_agent_execution(args.config, args.agent, args.input)
    else:
        parser.print_help()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())