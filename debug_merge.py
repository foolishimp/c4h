#!/usr/bin/env python3
"""Debug script to check configuration merging"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import copy

def debug_merge():
    """Debug configuration merging"""
    
    # Clear any existing instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    # Initialize Hydra
    config_path = project_root / "conf"
    relative_path = os.path.relpath(str(config_path), start=os.path.dirname(__file__))
    
    with initialize(version_base=None, config_path=relative_path):
        cfg = compose(config_name="config")
        
        # Simulate what happens in the task
        agent_name = "discovery_phase"
        persona_key = "discovery_by_skill_v1"
        
        # Get persona config
        persona_config = OmegaConf.select(cfg, f"llm_config.personas.{persona_key}")
        print(f"=== Persona config for {persona_key} ===")
        if persona_config:
            print(f"Keys: {list(persona_config.keys())}")
            if "execution_plan" in persona_config:
                print("Has execution_plan in persona: YES")
            else:
                print("Has execution_plan in persona: NO")
        
        # Get agent config
        agent_config = OmegaConf.select(cfg, f"llm_config.agents.{agent_name}")
        print(f"\n=== Agent config for {agent_name} ===")
        if agent_config:
            print(f"Keys: {list(agent_config.keys())}")
            if "execution_plan" in agent_config:
                print("Has execution_plan in agent: YES")
                print(f"Execution plan enabled: {agent_config.execution_plan.get('enabled', 'NOT SET')}")
            else:
                print("Has execution_plan in agent: NO")
        
        # Simulate the merge
        merged_config = {}
        if persona_config:
            merged_config.update(copy.deepcopy(persona_config))
            print(f"\nAfter persona merge, keys: {list(merged_config.keys())}")
        
        if agent_config:
            merged_config.update(copy.deepcopy(agent_config))
            print(f"After agent merge, keys: {list(merged_config.keys())}")
        
        # Check execution plan
        print(f"\n=== Final merged config ===")
        print(f"Keys: {list(merged_config.keys())}")
        execution_plan = merged_config.get("execution_plan")
        
        if execution_plan is None:
            print("execution_plan is None")
        elif not isinstance(execution_plan, dict):
            print(f"execution_plan is not a dict, it's: {type(execution_plan)}")
        elif not execution_plan.get("enabled", True):
            print(f"execution_plan.enabled is: {execution_plan.get('enabled')}")
        else:
            print("execution_plan is valid!")
            print(f"Steps: {len(execution_plan.get('steps', []))}")

if __name__ == "__main__":
    debug_merge()