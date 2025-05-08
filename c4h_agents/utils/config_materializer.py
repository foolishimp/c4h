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
    2. Handles personas with explicit agent_type declarations
    3. Processes team structures with execution plans and agent lists
    4. Validates execution plans at all levels
    5. Ensures skill references are valid
    6. Expands environment variables
    
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
    
    # 3. Process team structures with execution plans and agent lists
    try:
        process_team_structures(effective_config)
    except Exception as e:
        logger.error("config.materialize.team_structures.failed", error=str(e))
    
    # 4. Validate and normalize execution plans
    try:
        effective_config = ConfigValidator.validate_execution_plans(effective_config)
    except Exception as e:
        logger.error("config.materialize.validation.failed", error=str(e))
    
    # 5. Expand environment variables
    effective_config = expand_env_vars(effective_config)
    
    # 6. Add runtime configuration
    effective_config["runtime"] = effective_config.get("runtime", {})
    effective_config["runtime"]["run_id"] = run_id
    
    # 7. Persist to disk if workdir is provided
    if workdir:
        save_effective_config(effective_config, run_id, workdir)
    
    logger.info("config.materialize.complete", run_id=run_id)
    return effective_config

def materialize_agent_types(config: Dict[str, Any]) -> None:
    """
    Resolve agent_type references to concrete agent configurations.
    
    Modifies the config in place. Handles both top-level agents and team-level agents.
    Also ensures personas explicitly declare their agent_type.
    
    Args:
        config: Configuration dictionary to modify
    """
    # Skip if there's no llm_config section
    if "llm_config" not in config:
        return
    
    # Get agent types and agents
    agent_types = config.get("llm_config", {}).get("agent_types", {})
    agents = config.get("llm_config", {}).get("agents", {})
    personas = config.get("llm_config", {}).get("personas", {})
    teams = config.get("orchestration", {}).get("teams", {})
    
    if not agent_types:
        logger.debug("config.materialize.agent_types.no_agent_types")
        
    # Track total resolved references
    resolved_count = 0
    
    # 1. Process top-level agents
    if agents:
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
                
                # Leave the agent_type intact for reference per ArchDocV3
                
                # Update the agent config
                config["llm_config"]["agents"][agent_name] = merged_config
                resolved_count += 1
                
                logger.debug("config.materialize.agent_types.top_level_resolved", 
                           agent_name=agent_name, 
                           agent_type=agent_type)
    
    # 2. Process personas with agent_type declarations
    if personas:
        for persona_name, persona_config in personas.items():
            # Add default agent_type if not present based on presence of execution_plan
            if "agent_type" not in persona_config:
                if "execution_plan" in persona_config and persona_config.get("execution_plan", {}).get("enabled", False):
                    persona_config["agent_type"] = "GenericOrchestratorAgent"
                    logger.debug("config.materialize.personas.inferred_orchestrator_type", 
                               persona_name=persona_name)
                else:
                    persona_config["agent_type"] = "GenericLLMAgent"
                    logger.debug("config.materialize.personas.inferred_llm_type", 
                               persona_name=persona_name)
            
            # Apply agent type configurations if available
            agent_type = persona_config["agent_type"]
            if agent_type in agent_types:
                type_config = agent_types[agent_type]
                merged_config = deep_merge(type_config, persona_config)
                # Keep the agent_type reference per ArchDocV3
                config["llm_config"]["personas"][persona_name] = merged_config
                resolved_count += 1
                logger.debug("config.materialize.personas.type_applied", 
                           persona_name=persona_name, 
                           agent_type=agent_type)
    
    # 3. Process team-level agents (tasks)
    if teams:
        team_agent_count = 0
        
        for team_name, team_config in teams.items():
            tasks = team_config.get("tasks", [])
            
            for i, task in enumerate(tasks):
                if "agent_type" not in task:
                    continue
                    
                agent_type = task["agent_type"]
                if agent_type in agent_types:
                    type_config = agent_types[agent_type]
                    # Deep merge preserving task specifics
                    merged_task = deep_merge(type_config, task)
                    # Keep the agent_type reference per ArchDocV3
                    # Update the task in the team config
                    team_config["tasks"][i] = merged_task
                    team_agent_count += 1
                    resolved_count += 1
                    
                    logger.debug("config.materialize.team_task.type_applied", 
                               team_name=team_name, 
                               task_index=i,
                               agent_type=agent_type)
        
        logger.debug("config.materialize.team_tasks.processed", count=team_agent_count)
    
    logger.info("config.materialize.agent_types.complete", 
              resolved_count=resolved_count, 
              top_level_agents=len(agents) if agents else 0,
              personas=len(personas) if personas else 0,
              teams=len(teams) if teams else 0)

def process_team_structures(config: Dict[str, Any]) -> None:
    """
    Process team structures with internal execution plans and agent lists.
    
    This function implements TASK-CORE-003 by ensuring teams have properly
    structured execution_plans and agents lists according to ArchDocV3.
    
    It handles:
    1. Teams with internal execution_plans
    2. Teams with agents lists that reference personas
    3. Validation of all execution plans at team level
    
    Args:
        config: Configuration dictionary to modify
    """
    # Skip if there's no orchestration section
    if "orchestration" not in config:
        logger.debug("config.materialize.team_structures.no_orchestration")
        return
        
    # Get teams and personas
    teams = config.get("orchestration", {}).get("teams", {})
    personas = config.get("llm_config", {}).get("personas", {})
    
    if not teams:
        logger.debug("config.materialize.team_structures.no_teams")
        return
        
    processed_teams = 0
    processed_exec_plans = 0
    processed_agent_lists = 0
    
    for team_name, team_config in teams.items():
        # Check for presence of execution_plan
        if "execution_plan" in team_config:
            exec_plan = team_config["execution_plan"]
            # Apply default template if needed
            if "steps" not in exec_plan:
                exec_plan["steps"] = []
            if "enabled" not in exec_plan:
                exec_plan["enabled"] = True
                
            processed_exec_plans += 1
            logger.debug("config.materialize.team_structures.exec_plan_processed", 
                       team_name=team_name, 
                       step_count=len(exec_plan.get("steps", [])))
                       
        # Check for presence of agents list
        if "agents" in team_config:
            agents_list = team_config["agents"]
            
            # Process each agent in the list
            for i, agent_config in enumerate(agents_list):
                # Check if agent has persona_key
                if "persona_key" in agent_config:
                    persona_key = agent_config["persona_key"]
                    
                    # Apply persona if available
                    if persona_key in personas:
                        persona_config = personas[persona_key]
                        
                        # Get agent_type from persona if not in agent config
                        if "agent_type" not in agent_config and "agent_type" in persona_config:
                            agent_config["agent_type"] = persona_config["agent_type"]
                            
                        logger.debug("config.materialize.team_structures.agent_persona_linked", 
                                   team_name=team_name, 
                                   agent_index=i,
                                   persona_key=persona_key)
                    else:
                        logger.warning("config.materialize.team_structures.persona_not_found", 
                                     team_name=team_name, 
                                     agent_index=i,
                                     persona_key=persona_key)
            
            processed_agent_lists += 1
            logger.debug("config.materialize.team_structures.agents_list_processed", 
                       team_name=team_name, 
                       agent_count=len(agents_list))
                       
        # Convert tasks to agents if using old format
        if "tasks" in team_config and "agents" not in team_config:
            # Auto-convert tasks to agents list for backward compatibility
            tasks = team_config["tasks"]
            team_config["agents"] = tasks
            logger.info("config.materialize.team_structures.tasks_converted_to_agents", 
                      team_name=team_name, 
                      task_count=len(tasks))
                      
        # Create default execution_plan if not present but agents are
        if "execution_plan" not in team_config and "agents" in team_config:
            # Create a simple sequential execution plan that calls each agent in order
            steps = []
            agents_list = team_config["agents"]
            
            for i, agent_config in enumerate(agents_list):
                agent_name = agent_config.get("name", f"agent_{i}")
                steps.append({
                    "name": f"call_{agent_name}",
                    "type": "agent_call",
                    "node": agent_name,
                    "description": f"Call {agent_name}",
                    "input_params": {},
                    "output_field": f"results.{agent_name}"
                })
                
            team_config["execution_plan"] = {
                "enabled": True,
                "description": f"Auto-generated sequential execution plan for team {team_name}",
                "steps": steps
            }
            
            logger.info("config.materialize.team_structures.default_exec_plan_created", 
                      team_name=team_name, 
                      step_count=len(steps))
            processed_exec_plans += 1
            
        processed_teams += 1
    
    logger.info("config.materialize.team_structures.complete", 
              processed_teams=processed_teams,
              processed_exec_plans=processed_exec_plans,
              processed_agent_lists=processed_agent_lists)


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