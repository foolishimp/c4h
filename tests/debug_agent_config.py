#!/usr/bin/env python
"""
Debug script to inspect the configuration used by the discovery agent.
"""

import os
import sys
from pathlib import Path
import yaml
import json
from pprint import pprint
import logging
import structlog

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def debug_agent_config():
    """Debug the agent configuration loading process."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "system_config.yml"
    
    # 1. Load the raw config file
    raw_config = load_config(config_path)
    
    # Check specific sections in the raw config
    print("Checking raw config for discovery agent configuration:")
    
    # Navigate through the orchestration section
    teams = raw_config.get("orchestration", {}).get("teams", {})
    discovery_team = teams.get("discovery", {})
    discovery_tasks = discovery_team.get("tasks", [])
    
    print(f"Found {len(discovery_tasks)} tasks in discovery team")
    for i, task in enumerate(discovery_tasks):
        print(f"Task {i+1}: {task.get('name')}")
        print(f"  agent_type: {task.get('agent_type')}")
        print(f"  persona_key: {task.get('persona_key')}")
    
    # 2. Materialize the config
    print("\nMaterializing config...")
    effective_config = materialize_config(raw_config, run_id="debug_config")
    
    # 3. Check the effective config
    print("\nChecking effective config for discovery agent configuration:")
    
    # Check agent instances in effective config
    agents = effective_config.get("llm_config", {}).get("agents", {})
    
    # Check for discovery_phase specifically
    if "discovery_phase" in agents:
        discovery_agent = agents["discovery_phase"]
        print("Found discovery_phase in effective config:")
        print(f"  agent_type: {discovery_agent.get('agent_type')}")
        print(f"  persona_key: {discovery_agent.get('persona_key')}")
        
        # This is important - check if discovery_phase was updated to use GenericSkillAgent
        if discovery_agent.get("agent_type") != "GenericSkillAgent":
            print("WARNING: agent_type in effective config doesn't match system_config.yml!")
            
        if discovery_agent.get("persona_key") != "discovery_by_skill_v1":
            print("WARNING: persona_key in effective config doesn't match system_config.yml!")
    
    # 4. Check the orchestration section of effective config
    eff_teams = effective_config.get("orchestration", {}).get("teams", {})
    eff_discovery_team = eff_teams.get("discovery", {})
    eff_discovery_tasks = eff_discovery_team.get("tasks", [])
    
    print(f"\nFound {len(eff_discovery_tasks)} tasks in discovery team in effective config")
    for i, task in enumerate(eff_discovery_tasks):
        print(f"Task {i+1}: {task.get('name')}")
        print(f"  agent_type: {task.get('agent_type')}")
        print(f"  persona_key: {task.get('persona_key')}")
    
    # 5. Check the personas section of effective config
    personas = effective_config.get("llm_config", {}).get("personas", {})
    
    print("\nPersonas in effective config:")
    for persona_key, persona_config in personas.items():
        print(f"Persona: {persona_key}")
        if "skill" in persona_config:
            print(f"  skill: {persona_config.get('skill')}")
        if "agent_type" in persona_config:
            print(f"  agent_type: {persona_config.get('agent_type')}")
            
    # Check if discovery_by_skill_v1 is in the personas
    if "discovery_by_skill_v1" in personas:
        print("\nFound discovery_by_skill_v1 persona in effective config:")
        dbs_persona = personas["discovery_by_skill_v1"]
        print(f"  Skill: {dbs_persona.get('skill')}")
        print(f"  Agent Type: {dbs_persona.get('agent_type')}")
    else:
        print("\nWARNING: discovery_by_skill_v1 persona not found in effective config!")
    
    return True

if __name__ == "__main__":
    debug_agent_config()