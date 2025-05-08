#!/usr/bin/env python
"""
Debug script to test persona loading for discovery_by_skill_v1.yml.
"""

import os
import sys
from pathlib import Path
import yaml
import logging
import structlog

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4h_agents.config import load_config, load_persona_config, get_available_personas

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

def test_persona_loading():
    """Test persona loading mechanism."""
    project_root = Path(__file__).parent.parent
    personas_dir = project_root / "config" / "personas"
    
    print(f"Project root: {project_root}")
    print(f"Personas directory: {personas_dir}")
    print(f"Directory exists: {personas_dir.exists()}")
    print("\nListing persona files:")
    for ext in ['.yml', '.yaml']:
        for p in personas_dir.glob(f'*{ext}'):
            print(f"  - {p.name} (size: {p.stat().st_size} bytes)")
    
    print("\nScanning available personas:")
    available_personas = get_available_personas(personas_dir)
    print(f"Found {len(available_personas)} personas:")
    for key, path in available_personas.items():
        print(f"  - {key}: {path}")
    
    print("\nTesting direct loading of discovery_by_skill_v1 persona:")
    persona_file = personas_dir / "discovery_by_skill_v1.yml"
    print(f"File exists: {persona_file.exists()}")
    
    if persona_file.exists():
        print(f"File content:")
        with open(persona_file, 'r') as f:
            content = f.read()
            print(content)
        
        print("\nTrying to load with YAML:")
        try:
            with open(persona_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
                print(f"YAML loaded successfully: {yaml_data is not None}")
                print(f"Top-level keys: {list(yaml_data.keys())}")
        except Exception as e:
            print(f"Error loading YAML: {e}")
    
    print("\nTesting load_persona_config:")
    persona_config = load_persona_config("discovery_by_skill_v1", personas_dir)
    print(f"Loaded persona config: {bool(persona_config)}")
    print(f"Content: {persona_config}")
    
    return True

if __name__ == "__main__":
    success = test_persona_loading()
    sys.exit(0 if success else 1)