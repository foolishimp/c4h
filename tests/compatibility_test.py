#!/usr/bin/env python3
"""
Compatibility test script for testing the legacy-to-new agent mapping.
This script tests both the original class-based approach and the new type-based approach.

Usage:
    python compatibility_test.py
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("compatibility_test")

# Import the compatibility factory
from c4h_services.src.orchestration.compat_factory import CompatAgentFactory

def load_config(config_path: str) -> dict:
    """Load config from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def test_agent_creation(config_path: str):
    """Test agent creation with both approaches"""
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    # Create the compatibility factory
    factory = CompatAgentFactory(config, Path(config_path))
    
    # Test cases to try
    test_tasks = [
        # Legacy class-based approach
        {
            "name": "legacy_discovery",
            "agent_class": "c4h_agents.agents.discovery.DiscoveryAgent",
            "requires_approval": False
        },
        {
            "name": "legacy_solution_designer",
            "agent_class": "c4h_agents.agents.solution_designer.SolutionDesigner",
            "requires_approval": True
        },
        {
            "name": "legacy_coder",
            "agent_class": "c4h_agents.agents.coder.Coder",
            "requires_approval": True
        },
        
        # New type-based approach
        {
            "name": "new_discovery",
            "agent_type": "GenericLLMAgent",
            "persona_key": "discovery",
            "description": "Analyze project structure and files",
            "requires_approval": False
        },
        {
            "name": "new_solution_designer",
            "agent_type": "GenericLLMAgent",
            "persona_key": "solution_designer",
            "description": "Create solution design and code changes",
            "requires_approval": True
        },
        {
            "name": "new_coder",
            "agent_type": "GenericLLMAgent",
            "persona_key": "coder",
            "description": "Implement code changes based on solution design",
            "requires_approval": True
        },
        {
            "name": "new_fallback",
            "agent_type": "GenericFallbackAgent",
            "persona_key": "coder",
            "description": "Implement code changes with conservative approach",
            "config": {
                "temperature": 0,
                "max_retries": 2,
                "persona_key": "coder"  # This is necessary at the config level too
            }
        }
    ]
    
    # Try to create each agent
    results = []
    for task in test_tasks:
        try:
            logger.info(f"Creating agent: {task['name']}")
            agent = factory.create_agent(task)
            
            # Extract type information
            agent_type = getattr(agent, 'agent_type', None)
            agent_class = agent.__class__.__name__
            persona_key = getattr(agent, 'persona_key', None)
            
            result = {
                "task_name": task["name"],
                "success": True,
                "agent_type": str(agent_type) if agent_type else None,
                "agent_class": agent_class,
                "persona_key": persona_key
            }
            
            logger.info(f"Success: {task['name']} -> {agent_class} (Type: {agent_type}, Persona: {persona_key})")
            
        except Exception as e:
            logger.error(f"Failed to create agent {task['name']}: {str(e)}")
            result = {
                "task_name": task["name"],
                "success": False,
                "error": str(e)
            }
        
        results.append(result)
    
    # Print summary
    logger.info("\n" + "="*40)
    logger.info("RESULTS SUMMARY")
    logger.info("="*40)
    
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(results) - success_count}")
    
    # Print details for successful tests
    logger.info("\nSuccessful tests:")
    for result in results:
        if result["success"]:
            logger.info(f"  - {result['task_name']}: {result['agent_class']} (Type: {result['agent_type']}, Persona: {result['persona_key']})")
    
    # Print details for failed tests
    if len(results) != success_count:
        logger.info("\nFailed tests:")
        for result in results:
            if not result["success"]:
                logger.info(f"  - {result['task_name']}: {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    # Path to compatibility config
    config_path = os.path.join(project_root, "config", "compat_system_config.yml")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Run the tests
    results = test_agent_creation(config_path)
    
    # Exit with error code if any tests failed
    success_count = sum(1 for r in results if r["success"])
    sys.exit(0 if success_count == len(results) else 1)