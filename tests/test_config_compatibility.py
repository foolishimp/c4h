#!/usr/bin/env python3
"""
Test script to validate the compatibility configuration approach.
This script tests that the AgentFactory can handle both legacy class-based
and new type-based agent configurations without requiring new code.
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from c4h_services.src.orchestration.factory import AgentFactory
from c4h_agents.config import load_config
from c4h_agents.agents.base_agent import BaseAgent


def test_config_compatibility():
    """Test that the pure configuration-based compatibility approach works."""
    print("\n--- Testing Configuration-Based Compatibility (Agent Class Resolution) ---\n")
    
    # Load the compatibility configuration
    config_path = project_root / "config" / "compat_system_config.yml"
    if not config_path.exists():
        print(f"ERROR: Compatibility config not found at {config_path}")
        return False
    
    config = load_config(config_path)
    print(f"Loaded compatibility config from {config_path}")
    
    # Create the agent factory
    agent_factory = AgentFactory(config)
    print("Created AgentFactory with compatibility config")
    
    # Define test cases for different agent specifications
    test_cases = [
        {
            "name": "Legacy DiscoveryAgent",
            "spec": {
                "agent_class": "c4h_agents.agents.discovery.DiscoveryAgent",
                "persona_key": "discovery"
            }
        },
        {
            "name": "Legacy SolutionDesigner", 
            "spec": {
                "agent_class": "c4h_agents.agents.solution_designer.SolutionDesigner",
                "persona_key": "solution_designer"
            }
        },
        {
            "name": "Legacy Coder",
            "spec": {
                "agent_class": "c4h_agents.agents.coder.Coder",
                "persona_key": "coder"
            }
        },
        {
            "name": "New GenericLLMAgent (discovery)",
            "spec": {
                "agent_type": "GenericLLMAgent",
                "persona_key": "discovery"
            }
        },
        {
            "name": "New GenericLLMAgent (solution_designer)",
            "spec": {
                "agent_type": "GenericLLMAgent",
                "persona_key": "solution_designer"
            }
        },
        {
            "name": "New GenericLLMAgent (coder)",
            "spec": {
                "agent_type": "GenericLLMAgent",
                "persona_key": "coder"
            }
        },
        {
            "name": "Legacy GenericSingleShotAgent",
            "spec": {
                "agent_class": "c4h_agents.agents.generic.GenericSingleShotAgent",
                "persona_key": "coder"
            }
        }
    ]
    
    # Run the tests for class resolution
    class_success_count = 0
    for test_case in test_cases:
        print(f"\nTesting {test_case['name']}...")
        try:
            # For agent_class based tests
            if "agent_class" in test_case['spec']:
                agent_class_path = test_case['spec']["agent_class"]
                # Check if we have a mapping for this agent class
                if "agent_mappings" in config and agent_class_path in config["agent_mappings"]:
                    mapped_type = config["agent_mappings"][agent_class_path]
                    print(f"  Found mapping: {agent_class_path} → {mapped_type}")
                    # Resolve the agent class using the mapped type
                    agent_class = agent_factory._get_agent_class(mapped_type)
                    print(f"  Resolved to agent class: {agent_class.__name__}")
                    class_success_count += 1
                    print(f"  ✅ SUCCESS: {test_case['name']} resolved via mapping")
                else:
                    print(f"  ❌ FAILURE: No mapping found for {agent_class_path}")
            
            # For agent_type based tests
            elif "agent_type" in test_case['spec']:
                agent_type = test_case['spec']["agent_type"]
                # Get the agent class directly using the type
                agent_class = agent_factory._get_agent_class(agent_type)
                print(f"  Resolved to agent class: {agent_class.__name__}")
                class_success_count += 1
                print(f"  ✅ SUCCESS: {test_case['name']} resolved directly")
            
            else:
                print(f"  ❌ FAILURE: Invalid spec for {test_case['name']}")
                
        except Exception as e:
            print(f"  ❌ FAILURE: Error resolving {test_case['name']}: {str(e)}")
    
    # Print summary for class resolution
    print(f"\n=== Agent Class Resolution: {class_success_count}/{len(test_cases)} tests passed ===")
    
    # Now test agent instance creation
    print("\n--- Testing Configuration-Based Compatibility (Agent Instance Creation) ---\n")
    
    # Define task configurations for agent creation
    task_configs = [
        {
            "name": "Legacy Discovery Task",
            "task_config": {
                "name": "discovery_legacy_test",
                "agent_class": "c4h_agents.agents.discovery.DiscoveryAgent",
                "agent_type": config["agent_mappings"]["c4h_agents.agents.discovery.DiscoveryAgent"],
                "persona_key": "discovery"
            }
        },
        {
            "name": "New Discovery Task",
            "task_config": {
                "name": "discovery_new_test",
                "agent_type": "GenericLLMAgent",
                "persona_key": "discovery"
            }
        }
    ]
    
    # Run tests for agent instance creation
    instance_success_count = 0
    for task_test in task_configs:
        print(f"\nTesting instance creation: {task_test['name']}...")
        try:
            # Create the agent instance
            agent_instance = agent_factory.create_agent(task_test['task_config'])
            
            # Check that we got a valid agent instance
            if isinstance(agent_instance, BaseAgent):
                print(f"  Created agent instance: {agent_instance.__class__.__name__}")
                instance_success_count += 1
                print(f"  ✅ SUCCESS: {task_test['name']} agent instance created successfully")
            else:
                print(f"  ❌ FAILURE: Invalid agent instance for {task_test['name']}")
        
        except Exception as e:
            print(f"  ❌ FAILURE: Error creating agent instance for {task_test['name']}: {str(e)}")
    
    # Print summary for instance creation
    print(f"\n=== Agent Instance Creation: {instance_success_count}/{len(task_configs)} tests passed ===")
    
    # Overall success
    total_success = class_success_count == len(test_cases) and instance_success_count == len(task_configs)
    print(f"\n=== Overall Test Result: {'SUCCESS' if total_success else 'FAILURE'} ===")
    
    return total_success


if __name__ == "__main__":
    success = test_config_compatibility()
    sys.exit(0 if success else 1)