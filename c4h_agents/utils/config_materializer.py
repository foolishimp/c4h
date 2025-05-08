"""
Configuration materialization for C4H Unified Architecture.
Path: c4h_agents/utils/config_materializer.py
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import structlog
from copy import deepcopy
import hashlib
import yaml
import os
import json

from c4h_agents.config import (
    deep_merge, expand_env_vars, ConfigNode, create_config_node
)
from c4h_agents.utils.config_validation import ConfigValidator
from c4h_agents.skills.registry import SkillRegistry

logger = structlog.get_logger()

def materialize_config(
    config: Dict[str, Any],
    run_id: str,
    workdir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Materialize configuration with execution plans, agent types, and concrete skills.
    
    This function:
    1. Resolves all agent types to concrete agent configurations
    2. Validates execution plans at all levels
    3. Ensures skill references are valid
    4. Expands environment variables
    
    Args:
        config: Base configuration to materialize
        run_id: Run identifier for logging
        workdir: Optional working directory for output
        
    Returns:
        Materialized effective configuration
    """
    logger.info("config.materialize.starting", run_id=run_id)
    
    # Deep copy to avoid modifying original
    effective_config = deepcopy(config)
    
    # Create config node for easier access
    config_node = create_config_node(effective_config)
    
    # 1. Initialize the skill registry with all available skills
    try:
        registry = SkillRegistry()
        registry.register_builtin_skills()
        registry.load_skills_from_config(effective_config)
        registry._auto_discover_skills()
        logger.info("config.materialize.skill_registry.initialized", 
                 skills_count=len(registry.list_skills()),
                 skill_names=registry.list_skills())
    except Exception as e:
        logger.error("config.materialize.skill_registry.failed", error=str(e))
    
    # 2. Materialize agent types into concrete agent configurations
    try:
        materialize_agent_types(effective_config)
    except Exception as e:
        logger.error("config.materialize.agent_types.failed", error=str(e))
    
    # 3. Validate and normalize execution plans
    try:
        effective_config = ConfigValidator.validate_execution_plans(effective_config)
    except Exception as e:
        logger.error("config.materialize.validation.failed", error=str(e))
    
    # 4. Expand environment variables
    effective_config = expand_env_vars(effective_config)
    
    # 5. Add runtime configuration
    effective_config["runtime"] = effective_config.get("runtime", {})
    effective_config["runtime"]["run_id"] = run_id
    
    # 6. Persist to disk if workdir is provided
    if workdir:
        save_effective_config(effective_config, run_id, workdir)
    
    logger.info("config.materialize.complete", run_id=run_id)
    return effective_config

def materialize_agent_types(config: Dict[str, Any]) -> None:
    """
    Resolve agent_type references to concrete agent configurations.
    
    Modifies the config in place.
    
    Args:
        config: Configuration dictionary to modify
    """
    # Skip if there's no llm_config section
    if "llm_config" not in config:
        return
    
    # Get agent types and agents
    agent_types = config.get("llm_config", {}).get("agent_types", {})
    agents = config.get("llm_config", {}).get("agents", {})
    
    if not agent_types or not agents:
        logger.debug("config.materialize.agent_types.no_types_or_agents")
        return
    
    # Process each agent
    resolved_count = 0
    for agent_name, agent_config in agents.items():
        # Skip if no agent_type defined
        if "agent_type" not in agent_config:
            continue
            
        agent_type = agent_config["agent_type"]
        
        # Resolve agent type if it exists
        if agent_type in agent_types:
            type_config = agent_types[agent_type]
            
            # Deep merge type config with agent config (agent config takes precedence)
            merged_config = deep_merge(type_config, agent_config)
            
            # Remove the now-resolved agent_type reference
            if "agent_type" in merged_config:
                del merged_config["agent_type"]
                
            # Update the agent config
            config["llm_config"]["agents"][agent_name] = merged_config
            resolved_count += 1
            
            logger.debug("config.materialize.agent_types.resolved", 
                       agent_name=agent_name, 
                       agent_type=agent_type)
    
    logger.info("config.materialize.agent_types.complete", 
              resolved_count=resolved_count, 
              total_agents=len(agents))

def save_effective_config(
    config: Dict[str, Any],
    run_id: str,
    workdir: Path
) -> Path:
    """
    Save the effective configuration to disk.
    
    Args:
        config: Effective configuration to save
        run_id: Run identifier
        workdir: Working directory to save in
        
    Returns:
        Path to the saved configuration file
    """
    # Create output directory
    output_dir = workdir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Serialize to YAML and calculate hash
    yaml_content = yaml.safe_dump(config, sort_keys=True, default_flow_style=False)
    config_hash = hashlib.sha256(yaml_content.encode()).hexdigest()[:8]
    output_path = output_dir / f"effective_config_{config_hash}.yml"
    
    # Write to file
    with open(output_path, "w") as f:
        f.write(yaml_content)
    
    logger.info("config.materialize.saved", 
              path=str(output_path), 
              hash=config_hash)
              
    return output_path