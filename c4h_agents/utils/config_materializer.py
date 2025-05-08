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
from datetime import datetime, timezone

from c4h_agents.config import (
    deep_merge, expand_env_vars, ConfigNode, create_config_node
)
from c4h_agents.utils.config_validation import ConfigValidator
from c4h_agents.skills.registry import SkillRegistry

logger = structlog.get_logger()

def materialize_config(
    config: Dict[str, Any],
    run_id: str,
    workdir: Optional[Path] = None,
    arch_doc_version: str = "v3"
) -> Dict[str, Any]:
    """
    Materialize configuration with execution plans, agent types, and concrete skills.
    
    This function processes the configuration according to the architecture document
    version specified (default: v3), ensuring all components are properly resolved,
    validated, and ready for use in the execution engine.
    
    ArchDocV3 processing includes:
    1. Resolves all agent types to concrete agent configurations
    2. Handles personas with explicit agent_type declarations
    3. Processes team structures with execution plans and agent lists
    4. Validates execution plans at all levels (teams, agents, personas)
    5. Processes execution_plan embedded in agent and persona configurations
    6. Ensures skill references are valid across all contexts
    7. Expands environment variables in all configuration sections
    8. Processes template resolution in execution plans
    9. Adds runtime metadata for execution tracing
    
    Args:
        config: Base configuration to materialize
        run_id: Run identifier for logging
        workdir: Optional working directory for output
        arch_doc_version: Architecture document version to conform to (v2, v3)
        
    Returns:
        Materialized effective configuration
    """
    logger.info("config.materialize.starting", 
             run_id=run_id, 
             arch_doc_version=arch_doc_version)
    
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
                 
        # Store available skills in the config for reference
        effective_config["available_skills"] = registry.list_skills()
    except Exception as e:
        logger.error("config.materialize.skill_registry.failed", error=str(e))
    
    # 2. Materialize agent types into concrete agent configurations
    try:
        materialize_agent_types(effective_config, arch_doc_version)
    except Exception as e:
        logger.error("config.materialize.agent_types.failed", error=str(e))
    
    # 3. Process team structures with execution plans and agent lists
    try:
        process_team_structures(effective_config, arch_doc_version)
    except Exception as e:
        logger.error("config.materialize.team_structures.failed", error=str(e))
    
    # 4. ArchDocV3: Process embedded execution plans in agents and personas
    if arch_doc_version.lower() == "v3":
        try:
            process_embedded_execution_plans(effective_config)
        except Exception as e:
            logger.error("config.materialize.embedded_plans.failed", error=str(e))
    
    # 5. Validate and normalize execution plans
    try:
        effective_config = ConfigValidator.validate_execution_plans(effective_config)
    except Exception as e:
        logger.error("config.materialize.validation.failed", error=str(e))
    
    # 6. Expand environment variables
    effective_config = expand_env_vars(effective_config)
    
    # 7. Add runtime configuration
    effective_config["runtime"] = effective_config.get("runtime", {})
    effective_config["runtime"]["run_id"] = run_id
    effective_config["runtime"]["arch_doc_version"] = arch_doc_version
    effective_config["runtime"]["materialized_at"] = datetime.now().isoformat()
    
    # 8. Persist to disk if workdir is provided
    if workdir:
        save_effective_config(effective_config, run_id, workdir)
    
    logger.info("config.materialize.complete", 
              run_id=run_id, 
              arch_doc_version=arch_doc_version)
              
    return effective_config

def materialize_agent_types(config: Dict[str, Any], arch_doc_version: str = "v3") -> None:
    """
    Resolve agent_type references to concrete agent configurations.
    
    Modifies the config in place. Handles both top-level agents and team-level agents.
    Also ensures personas explicitly declare their agent_type.
    
    ArchDocV3 enhancements:
    - Supports embedded execution_plan in agent configurations
    - Ensures consistent agent_type declarations across the system
    - Handles agent capability declarations (skills, permissions)
    - Processes typed parameters in agent_config
    
    Args:
        config: Configuration dictionary to modify
        arch_doc_version: Architecture document version (v2, v3)
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
                # ArchDocV3: Infer agent_type if execution_plan is present
                if arch_doc_version.lower() == "v3" and "execution_plan" in agent_config:
                    agent_config["agent_type"] = "GenericOrchestratorAgent"
                    logger.debug("config.materialize.agents.inferred_orchestrator_type", 
                               agent_name=agent_name)
                else:
                    agent_config["agent_type"] = "GenericLLMAgent"
                    logger.debug("config.materialize.agents.inferred_llm_type", 
                               agent_name=agent_name)
                
            agent_type = agent_config["agent_type"]
            
            # Resolve agent type if it exists
            if agent_type in agent_types:
                type_config = agent_types[agent_type]
                
                # Deep merge type config with agent config (agent config takes precedence)
                merged_config = deep_merge(type_config, agent_config)
                
                # ArchDocV3: Add agent name to config for reference
                if arch_doc_version.lower() == "v3":
                    merged_config["name"] = agent_name
                    
                    # Ensure agent_capabilities section is present
                    if "agent_capabilities" not in merged_config:
                        merged_config["agent_capabilities"] = {}
                
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
                
                # ArchDocV3: Add persona name to config for reference
                if arch_doc_version.lower() == "v3":
                    merged_config["persona_name"] = persona_name
                    
                    # Ensure agent_capabilities section is present
                    if "agent_capabilities" not in merged_config:
                        merged_config["agent_capabilities"] = {}
                
                # Keep the agent_type reference per ArchDocV3
                config["llm_config"]["personas"][persona_name] = merged_config
                resolved_count += 1
                logger.debug("config.materialize.personas.type_applied", 
                           persona_name=persona_name, 
                           agent_type=agent_type)
    
    # 3. Process team-level agents
    if teams:
        team_agent_count = 0
        
        for team_name, team_config in teams.items():
            # Process both "tasks" (legacy) and "agents" (ArchDocV3)
            for key in ["tasks", "agents"]:
                if key not in team_config:
                    continue
                    
                agent_list = team_config[key]
                
                for i, agent in enumerate(agent_list):
                    # Set default agent_type if not present
                    if "agent_type" not in agent:
                        # Infer from execution_plan if present
                        if arch_doc_version.lower() == "v3" and "execution_plan" in agent:
                            agent["agent_type"] = "GenericOrchestratorAgent"
                            logger.debug(f"config.materialize.team_{key}.inferred_orchestrator_type", 
                                       team_name=team_name,
                                       agent_index=i)
                        else:
                            agent["agent_type"] = "GenericLLMAgent"
                            logger.debug(f"config.materialize.team_{key}.inferred_llm_type", 
                                       team_name=team_name,
                                       agent_index=i)
                    
                    agent_type = agent["agent_type"]
                    if agent_type in agent_types:
                        type_config = agent_types[agent_type]
                        # Deep merge preserving agent specifics
                        merged_agent = deep_merge(type_config, agent)
                        
                        # ArchDocV3: Add team context to the agent config
                        if arch_doc_version.lower() == "v3":
                            if "context" not in merged_agent:
                                merged_agent["context"] = {}
                            
                            merged_agent["context"]["team_name"] = team_name
                            merged_agent["context"]["team_index"] = i
                            
                            # Ensure agent_capabilities section is present
                            if "agent_capabilities" not in merged_agent:
                                merged_agent["agent_capabilities"] = {}
                        
                        # Update the agent in the team config
                        team_config[key][i] = merged_agent
                        team_agent_count += 1
                        resolved_count += 1
                        
                        logger.debug(f"config.materialize.team_{key}.type_applied", 
                                   team_name=team_name, 
                                   agent_index=i,
                                   agent_type=agent_type)
        
        logger.debug("config.materialize.team_agents.processed", count=team_agent_count)
    
    logger.info("config.materialize.agent_types.complete", 
              resolved_count=resolved_count, 
              top_level_agents=len(agents) if agents else 0,
              personas=len(personas) if personas else 0,
              teams=len(teams) if teams else 0,
              arch_doc_version=arch_doc_version)

def process_team_structures(config: Dict[str, Any], arch_doc_version: str = "v3") -> None:
    """
    Process team structures with internal execution plans and agent lists.
    
    This function implements TASK-CORE-003 by ensuring teams have properly
    structured execution_plans and agents lists according to ArchDocV3.
    
    It handles:
    1. Teams with internal execution_plans
    2. Teams with agents lists that reference personas
    3. Validation of all execution plans at team level
    4. Enhanced team metadata for ArchDocV3
    
    Args:
        config: Configuration dictionary to modify
        arch_doc_version: Architecture document version (v2, v3)
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
                
            # ArchDocV3: Add team context to the execution plan
            if arch_doc_version.lower() == "v3":
                if "metadata" not in exec_plan:
                    exec_plan["metadata"] = {}
                    
                exec_plan["metadata"]["team_id"] = team_name
                exec_plan["metadata"]["team_name"] = team_config.get("name", team_name)
                
                # Add description if not present
                if "description" not in exec_plan:
                    exec_plan["description"] = f"Execution plan for team {team_name}"
                
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
                            
                        # ArchDocV3: Deep merge persona into agent, with agent taking precedence
                        if arch_doc_version.lower() == "v3":
                            # Start with persona as base
                            merged_config = deepcopy(persona_config)
                            # Override with agent-specific config
                            for key, value in agent_config.items():
                                if key != "persona_key":  # Keep the persona_key reference
                                    if key in merged_config and isinstance(value, dict) and isinstance(merged_config[key], dict):
                                        # Deep merge nested dictionaries
                                        merged_config[key] = deep_merge(merged_config[key], value)
                                    else:
                                        # Direct override for non-dict values
                                        merged_config[key] = value
                            
                            # Keep the persona_key reference
                            merged_config["persona_key"] = persona_key
                            
                            # Add team context 
                            if "context" not in merged_config:
                                merged_config["context"] = {}
                            merged_config["context"]["team_id"] = team_name
                            merged_config["context"]["team_index"] = i
                            
                            # Update the agent config with the merged result
                            for key, value in merged_config.items():
                                if key != "persona_key":  # We already have this
                                    agent_config[key] = value
                        
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
                       
        # Convert tasks to agents if using old format (for backward compatibility)
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
                "steps": steps,
                "metadata": {
                    "team_id": team_name,
                    "team_name": team_config.get("name", team_name),
                    "auto_generated": True,
                    "generator": "process_team_structures"
                }
            }
            
            logger.info("config.materialize.team_structures.default_exec_plan_created", 
                      team_name=team_name, 
                      step_count=len(steps))
            processed_exec_plans += 1
            
        # ArchDocV3: Add team metadata
        if arch_doc_version.lower() == "v3":
            # Ensure metadata section
            if "metadata" not in team_config:
                team_config["metadata"] = {}
                
            # Add team ID and name
            team_config["metadata"]["team_id"] = team_name
            if "name" not in team_config:
                team_config["name"] = team_name
                
            # Add agent count
            agent_count = len(team_config.get("agents", [])) if "agents" in team_config else 0
            team_config["metadata"]["agent_count"] = agent_count
            
            # Add execution plan flag
            team_config["metadata"]["has_execution_plan"] = "execution_plan" in team_config
            
        processed_teams += 1
    
    logger.info("config.materialize.team_structures.complete", 
              processed_teams=processed_teams,
              processed_exec_plans=processed_exec_plans,
              processed_agent_lists=processed_agent_lists,
              arch_doc_version=arch_doc_version)


def process_embedded_execution_plans(config: Dict[str, Any]) -> None:
    """
    Process embedded execution plans in agent and persona configurations.
    
    This function specifically handles the ArchDocV3 feature of embedding execution plans
    directly in agent and persona configurations. It ensures all embedded plans are
    properly validated, normalized, and enriched with metadata.
    
    Args:
        config: Configuration dictionary to modify
    """
    # Initialize counters
    processed_agent_plans = 0
    processed_persona_plans = 0
    
    # 1. Process agent execution plans
    agents = config.get("llm_config", {}).get("agents", {})
    for agent_name, agent_config in agents.items():
        if "execution_plan" in agent_config:
            exec_plan = agent_config["execution_plan"]
            
            # Apply default template if needed
            if "steps" not in exec_plan:
                exec_plan["steps"] = []
            if "enabled" not in exec_plan:
                exec_plan["enabled"] = True
                
            # Add metadata
            if "metadata" not in exec_plan:
                exec_plan["metadata"] = {}
                
            exec_plan["metadata"]["agent_name"] = agent_name
            exec_plan["metadata"]["plan_type"] = "agent_embedded"
            
            # Add description if not present
            if "description" not in exec_plan:
                exec_plan["description"] = f"Embedded execution plan for agent {agent_name}"
                
            # Process skill references
            processed_steps = process_skill_references(exec_plan.get("steps", []), config)
            exec_plan["steps"] = processed_steps
            
            processed_agent_plans += 1
            
            logger.debug("config.materialize.embedded_plans.agent_processed",
                       agent_name=agent_name,
                       step_count=len(exec_plan.get("steps", [])))
    
    # 2. Process persona execution plans
    personas = config.get("llm_config", {}).get("personas", {})
    for persona_name, persona_config in personas.items():
        if "execution_plan" in persona_config:
            exec_plan = persona_config["execution_plan"]
            
            # Apply default template if needed
            if "steps" not in exec_plan:
                exec_plan["steps"] = []
            if "enabled" not in exec_plan:
                exec_plan["enabled"] = True
                
            # Add metadata
            if "metadata" not in exec_plan:
                exec_plan["metadata"] = {}
                
            exec_plan["metadata"]["persona_name"] = persona_name
            exec_plan["metadata"]["plan_type"] = "persona_embedded"
            
            # Add description if not present
            if "description" not in exec_plan:
                exec_plan["description"] = f"Embedded execution plan for persona {persona_name}"
                
            # Process skill references
            processed_steps = process_skill_references(exec_plan.get("steps", []), config)
            exec_plan["steps"] = processed_steps
            
            processed_persona_plans += 1
            
            logger.debug("config.materialize.embedded_plans.persona_processed",
                       persona_name=persona_name,
                       step_count=len(exec_plan.get("steps", [])))
    
    logger.info("config.materialize.embedded_plans.complete",
              processed_agent_plans=processed_agent_plans,
              processed_persona_plans=processed_persona_plans)

def process_skill_references(steps: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process skill references in execution plan steps.
    
    Validates skill names, adds metadata, and ensures proper step configuration.
    
    Args:
        steps: List of execution plan steps
        config: Configuration dictionary for skill lookup
        
    Returns:
        Updated list of steps with processed skill references
    """
    available_skills = config.get("available_skills", [])
    processed_steps = []
    
    for step in steps:
        # Make a copy to avoid modifying the original
        processed_step = deepcopy(step)
        
        # Check for skill references in skill_call steps
        if processed_step.get("type") == "skill_call" and "skill" in processed_step:
            skill_name = processed_step["skill"]
            
            # Check if the skill exists
            if available_skills and skill_name not in available_skills:
                logger.warning("process_skill_references.unknown_skill", 
                            skill_name=skill_name,
                            step_name=processed_step.get("name", "unnamed_step"))
                
            # Add skill_call metadata
            if "metadata" not in processed_step:
                processed_step["metadata"] = {}
                
            processed_step["metadata"]["skill_name"] = skill_name
        
        # Check for nested steps in loop bodies
        if processed_step.get("type") == "loop" and "body" in processed_step:
            # Process the body steps recursively
            processed_body = process_skill_references(processed_step["body"], config)
            processed_step["body"] = processed_body
        
        # Check for nested steps in branch bodies
        if processed_step.get("type") == "branch":
            # Process the branch bodies if present
            if "branches" in processed_step:
                for branch in processed_step["branches"]:
                    if "body" in branch:
                        processed_body = process_skill_references(branch["body"], config)
                        branch["body"] = processed_body
        
        processed_steps.append(processed_step)
    
    return processed_steps

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