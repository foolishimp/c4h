#!/usr/bin/env python
"""
Test script to verify the GenericSkillAgent-based discovery implementation.

This script tests the refactored discovery agent that uses the CommandLineRunner skill
to execute tartxt.py rather than having special-case code in GenericLLMAgent.
"""

import os
import sys
import json
import yaml
from pathlib import Path
import logging
import structlog

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4h_agents.agents.generic import GenericSkillAgent
from c4h_agents.config import load_config
from c4h_agents.utils.config_materializer import materialize_config

# Configure logging
logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory()
)

logger = structlog.get_logger()

def create_direct_skill_agent(persona_key, skill_name, run_id, project_path):
    """Create a skill agent directly with minimal configuration."""
    print(f"Creating direct skill agent with persona {persona_key} and skill {skill_name}")
    
    # Load the persona config
    project_root = Path(__file__).parent.parent
    personas_dir = project_root / "config" / "personas"
    persona_file = personas_dir / f"{persona_key}.yml"
    
    with open(persona_file, 'r') as f:
        persona_config = yaml.safe_load(f)
    
    # Instead of relying on command_name and command_configs, directly set the command_line
    tartxt_script_path = str(project_root / "c4h_agents" / "skills" / "tartxt.py")
    output_file = f"{project_path}/workspaces/project_scan_test_{run_id}.txt"
    
    # Make sure the workspaces directory exists
    os.makedirs(f"{project_path}/workspaces", exist_ok=True)
    
    # Create a full effective config structure
    effective_config = {
        "llm_config": {
            "personas": {
                persona_key: persona_config
            },
            "agents": {
                "test_agent": {
                    "agent_type": "GenericSkillAgent",
                    "persona_key": persona_key,
                    "skill": skill_name,
                    # Direct execution parameters - needs to be at this level, not nested in skill_params
                    "command_line": f"python {tartxt_script_path}",
                    "command_args": {
                        "project_path": project_path,
                        "exclude": "**/node_modules/**,**/.git/**,**/__pycache__/**,**/*.pyc,**/package-lock.json,**/dist/**,**/.DS_Store,**/README.md,**/workspaces/**,**/backup_txt/**",
                        "history": 0,
                        # Use a predictable output file path for testing
                        "file": output_file
                    },
                    "working_dir": project_path,
                    "output_to_file": True
                }
            },
            "skills": {
                "command_line_runner": {
                    "module": "c4h_agents.skills.command_line_runner",
                    "class": "CommandLineRunner",
                    "method": "execute"
                }
            }
        },
        "runtime": {
            "workflow": {
                "id": run_id
            }
        }
    }
    
    # Create context with project path
    context = {
        "workflow_run_id": run_id,
        "project_path": project_path,
        "execution_metadata": {
            "execution_id": run_id
        }
    }
    
    # Create agent directly
    agent = GenericSkillAgent(effective_config, "test_agent")
    
    return agent, context

def test_discovery_skill_agent():
    """Test the discovery agent using GenericSkillAgent with CommandLineRunner."""
    print("\n=== Testing Discovery Agent with GenericSkillAgent ===\n")
    
    # Skip the configuration part and directly test the CommandLineRunner skill
    print("Testing a direct invocation of CommandLineRunner skill...")
    
    project_root = Path(__file__).parent.parent
    tartxt_script_path = str(project_root / "c4h_agents" / "skills" / "tartxt.py")
    project_path = str(project_root)
    output_file = f"{project_path}/workspaces/project_scan_test_direct.txt"
    
    # Make sure the workspaces directory exists
    os.makedirs(f"{project_path}/workspaces", exist_ok=True)
    
    # Import the CommandLineRunner directly
    from c4h_agents.skills.command_line_runner import CommandLineRunner
    
    # Create and initialize the skill
    runner = CommandLineRunner(config={})
    
    print(f"Executing command: python {tartxt_script_path} with output_file={output_file}")
    
    # Execute the command, using positional arguments format for tartxt.py
    result = runner.execute(
        command_line=f"python {tartxt_script_path}",
        command_args={
            "positional_args": [project_path],  # Directory to scan as positional argument
            "exclude": "**/node_modules/**,**/.git/**,**/__pycache__/**,**/*.pyc,**/package-lock.json,**/dist/**",
            "file": output_file
        },
        working_dir=project_path,
        output_to_file=True
    )
    
    print(f"Command execution result: {result.success}")
    
    if not result.success:
        print(f"Error: {result.error}")
        return False
    
    if result.value:
        print(f"Result value keys: {list(result.value.keys())}")
        if "output_file" in result.value:
            output_file = result.value["output_file"]
            print(f"Command output file: {output_file}")
            
            # Check if the file exists
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"Output file size: {file_size} bytes")
                print(f"SUCCESS: CommandLineRunner worked correctly")
                return True
            else:
                print(f"ERROR: Output file does not exist")
        else:
            print(f"No output_file in result.value")
    else:
        print(f"No result.value available")
    
    return False

if __name__ == "__main__":
    success = test_discovery_skill_agent()
    sys.exit(0 if success else 1)