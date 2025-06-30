#!/usr/bin/env python3
"""Test script to check how Hydra loads the configuration"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

def test_hydra_config():
    """Test loading Hydra configuration"""
    
    # Clear any existing instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    # Initialize Hydra
    config_path = project_root / "conf"
    relative_path = os.path.relpath(str(config_path), start=os.path.dirname(__file__))
    
    with initialize(version_base=None, config_path=relative_path):
        cfg = compose(config_name="config")
        
        # Print the structure
        print("=== Top level keys ===")
        print(list(cfg.keys()))
        
        print("\n=== Persona location ===")
        # Check if personas are under llm_config
        if "llm_config" in cfg and "personas" in cfg.llm_config:
            print("Personas found under llm_config.personas:")
            print(list(cfg.llm_config.personas.keys()))
        else:
            print("Personas NOT found under llm_config.personas")
            
        # Check if persona is at root
        if "persona" in cfg:
            print("\nPersona found at root level:")
            print(OmegaConf.to_yaml(cfg.persona))
            
        # Check orchestration.teams.discovery
        if "orchestration" in cfg and "teams" in cfg.orchestration:
            print("\n=== Discovery team config ===")
            discovery = cfg.orchestration.teams.discovery
            print(f"Tasks: {len(discovery.tasks)}")
            print(f"First task persona_key: {discovery.tasks[0].get('persona_key', 'NOT SET')}")
            
        # Check if agents are loaded
        if "llm_config" in cfg and "agents" in cfg.llm_config:
            print("\n=== Agents loaded ===")
            print(list(cfg.llm_config.agents.keys()))
            if "discovery_phase" in cfg.llm_config.agents:
                print("Discovery phase agent config found!")
                if "execution_plan" in cfg.llm_config.agents.discovery_phase:
                    print("Has execution_plan: YES")
                else:
                    print("Has execution_plan: NO")

if __name__ == "__main__":
    test_hydra_config()