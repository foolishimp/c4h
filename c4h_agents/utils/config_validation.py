"""
Configuration validation utilities for C4H Unified Architecture.
Path: c4h_agents/utils/config_validation.py
"""

from typing import Dict, Any, List, Optional, Union, Set, Tuple
from pathlib import Path
import os
import structlog
from copy import deepcopy

from c4h_agents.utils.schema_validation import SchemaValidator
from c4h_agents.skills.registry import SkillRegistry

logger = structlog.get_logger()

class ConfigValidator:
    """
    Utility class for validating and normalizing configuration structures.
    
    Provides methods to validate execution plans, agent configurations,
    skill configurations, and ensure proper integration between components.
    """
    
    @classmethod
    def validate_execution_plans(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all execution plans in the configuration.
        
        Checks execution plans at all levels:
        - Workflows (top-level execution plans)
        - Teams (team-level execution plans - both legacy format and ArchDocV3 format)
        - Personas (persona-level execution plans in ArchDocV3)
        - Agents (agent-level execution plans)
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Configuration with validated execution plans
        """
        validated_config = deepcopy(config)
        
        # Find the schema path
        module_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_dir = module_dir.parent.parent
        schema_path = project_dir / "config" / "schemas" / "execution_plan_v1.json"
        
        # Count for logging
        total_plans = 0
        valid_plans = 0
        
        # Validate top-level execution plans
        if "execution_plans" in validated_config:
            for plan_id, plan in validated_config["execution_plans"].items():
                total_plans += 1
                try:
                    validated_config["execution_plans"][plan_id] = SchemaValidator.validate_execution_plan(
                        plan, schema_path
                    )
                    valid_plans += 1
                except Exception as e:
                    logger.error("config.validation.workflow_plan_invalid",
                                plan_id=plan_id,
                                error=str(e))
        
        # Validate team execution plans (legacy location)
        if "llm_config" in validated_config and "teams" in validated_config["llm_config"]:
            for team_id, team_config in validated_config["llm_config"]["teams"].items():
                if "execution_plan" in team_config:
                    total_plans += 1
                    try:
                        validated_config["llm_config"]["teams"][team_id]["execution_plan"] = SchemaValidator.validate_execution_plan(
                            team_config["execution_plan"], schema_path
                        )
                        valid_plans += 1
                    except Exception as e:
                        logger.error("config.validation.team_plan_invalid",
                                    team_id=team_id,
                                    error=str(e))
        
        # Validate team execution plans (ArchDocV3 location)
        if "orchestration" in validated_config and "teams" in validated_config["orchestration"]:
            for team_id, team_config in validated_config["orchestration"]["teams"].items():
                if "execution_plan" in team_config:
                    total_plans += 1
                    try:
                        validated_config["orchestration"]["teams"][team_id]["execution_plan"] = SchemaValidator.validate_execution_plan(
                            team_config["execution_plan"], schema_path
                        )
                        valid_plans += 1
                    except Exception as e:
                        logger.error("config.validation.archv3_team_plan_invalid",
                                    team_id=team_id,
                                    error=str(e))
        
        # Validate persona-level execution plans (ArchDocV3)
        if "llm_config" in validated_config and "personas" in validated_config["llm_config"]:
            for persona_id, persona_config in validated_config["llm_config"]["personas"].items():
                if "execution_plan" in persona_config:
                    total_plans += 1
                    try:
                        validated_config["llm_config"]["personas"][persona_id]["execution_plan"] = SchemaValidator.validate_execution_plan(
                            persona_config["execution_plan"], schema_path
                        )
                        valid_plans += 1
                        
                        # If agent_type not specified but has execution_plan, infer GenericOrchestratorAgent
                        if "agent_type" not in persona_config:
                            validated_config["llm_config"]["personas"][persona_id]["agent_type"] = "GenericOrchestratorAgent"
                            logger.debug("config.validation.inferred_orchestrator_type", 
                                      persona_id=persona_id)
                    except Exception as e:
                        logger.error("config.validation.persona_plan_invalid",
                                    persona_id=persona_id,
                                    error=str(e))
        
        # Validate agent execution plans (legacy location)
        if "llm_config" in validated_config and "agents" in validated_config["llm_config"]:
            for agent_id, agent_config in validated_config["llm_config"]["agents"].items():
                if "execution_plan" in agent_config:
                    total_plans += 1
                    try:
                        validated_config["llm_config"]["agents"][agent_id]["execution_plan"] = SchemaValidator.validate_execution_plan(
                            agent_config["execution_plan"], schema_path
                        )
                        valid_plans += 1
                    except Exception as e:
                        logger.error("config.validation.agent_plan_invalid",
                                    agent_id=agent_id,
                                    error=str(e))
        
        logger.info("config.validation.execution_plans_complete",
                   total_plans=total_plans,
                   valid_plans=valid_plans,
                   invalid_plans=total_plans - valid_plans)
        
        return validated_config
    
    @classmethod
    def validate_skill_references(cls, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Set[str]]:
        """
        Validate that all skill references in execution plans exist in the skill registry.
        
        This method handles all locations of execution plans in both legacy and
        ArchDocV3 configuration structures.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Tuple of (validated_config, missing_skills)
        """
        validated_config = deepcopy(config)
        registry = SkillRegistry()
        
        # Get all registered skill names
        registered_skills = set(registry.get_all_skill_names())
        
        # Track missing skills for reporting
        missing_skills = set()
        
        # Function to check skill references in an execution plan
        def check_skills_in_plan(plan: Dict[str, Any]) -> None:
            if not isinstance(plan, dict) or "steps" not in plan:
                return
                
            for step in plan.get("steps", []):
                if isinstance(step, dict) and step.get("type") == "skill_call":
                    skill_name = step.get("skill")
                    if skill_name and skill_name not in registered_skills:
                        missing_skills.add(skill_name)
                        logger.warning("config.validation.missing_skill",
                                      skill_name=skill_name,
                                      step_name=step.get("name", "unknown"))
                
                # Check skills in loops
                if isinstance(step, dict) and step.get("type") == "loop" and "body" in step:
                    for body_step in step.get("body", []):
                        if isinstance(body_step, dict) and body_step.get("type") == "skill_call":
                            skill_name = body_step.get("skill")
                            if skill_name and skill_name not in registered_skills:
                                missing_skills.add(skill_name)
                                logger.warning("config.validation.missing_skill",
                                            skill_name=skill_name,
                                            step_name=body_step.get("name", "unknown"))
        
        # Check top-level execution plans
        if "execution_plans" in validated_config:
            for plan_id, plan in validated_config["execution_plans"].items():
                check_skills_in_plan(plan)
        
        # Check team execution plans (legacy location)
        if "llm_config" in validated_config and "teams" in validated_config["llm_config"]:
            for team_id, team_config in validated_config["llm_config"]["teams"].items():
                if "execution_plan" in team_config:
                    check_skills_in_plan(team_config["execution_plan"])
        
        # Check team execution plans (ArchDocV3 location)
        if "orchestration" in validated_config and "teams" in validated_config["orchestration"]:
            for team_id, team_config in validated_config["orchestration"]["teams"].items():
                if "execution_plan" in team_config:
                    check_skills_in_plan(team_config["execution_plan"])
        
        # Check persona execution plans (ArchDocV3)
        if "llm_config" in validated_config and "personas" in validated_config["llm_config"]:
            for persona_id, persona_config in validated_config["llm_config"]["personas"].items():
                if "execution_plan" in persona_config:
                    check_skills_in_plan(persona_config["execution_plan"])
        
        # Check agent execution plans (legacy location)
        if "llm_config" in validated_config and "agents" in validated_config["llm_config"]:
            for agent_id, agent_config in validated_config["llm_config"]["agents"].items():
                if "execution_plan" in agent_config:
                    check_skills_in_plan(agent_config["execution_plan"])
        
        # Report detailed information about registered skills
        total_registered = len(registered_skills)
        checked_skills = total_registered - len(missing_skills)
        
        logger.info("config.validation.skill_references_complete",
                   total_missing_skills=len(missing_skills),
                   missing_skills=list(missing_skills) if missing_skills else None,
                   total_registered_skills=total_registered,
                   valid_skills_count=checked_skills,
                   archv3_structure=bool("orchestration" in validated_config))
        
        return validated_config, missing_skills
    
    @classmethod
    def validate_agent_references(cls, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Set[str]]:
        """
        Validate that all agent references in execution plans exist in the configuration.
        
        This method handles both legacy and ArchDocV3 agent definitions.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Tuple of (validated_config, missing_agents)
        """
        validated_config = deepcopy(config)
        
        # Get all defined agent names
        defined_agents = set()
        
        # 1. Legacy agent definitions
        if "llm_config" in validated_config and "agents" in validated_config["llm_config"]:
            defined_agents.update(validated_config["llm_config"]["agents"].keys())
        
        # 2. ArchDocV3: Team-level agent definitions (agents list in teams)
        if "orchestration" in validated_config and "teams" in validated_config["orchestration"]:
            for team_id, team_config in validated_config["orchestration"]["teams"].items():
                if "agents" in team_config and isinstance(team_config["agents"], list):
                    for agent in team_config["agents"]:
                        if isinstance(agent, dict) and "name" in agent:
                            defined_agents.add(agent["name"])
                        
                # Legacy format - tasks as agents
                if "tasks" in team_config and isinstance(team_config["tasks"], list):
                    for task in team_config["tasks"]:
                        if isinstance(task, dict) and "name" in task:
                            defined_agents.add(task["name"])
        
        # Track missing agents for reporting
        missing_agents = set()
        
        # Function to check agent references in an execution plan
        def check_agents_in_plan(plan: Dict[str, Any]) -> None:
            if not isinstance(plan, dict) or "steps" not in plan:
                return
                
            for step in plan.get("steps", []):
                if isinstance(step, dict) and step.get("type") == "agent_call":
                    agent_name = step.get("node")
                    if agent_name and agent_name not in defined_agents:
                        missing_agents.add(agent_name)
                        logger.warning("config.validation.missing_agent",
                                      agent_name=agent_name,
                                      step_name=step.get("name", "unknown"))
                
                # Check agents in loops
                if isinstance(step, dict) and step.get("type") == "loop" and "body" in step:
                    for body_step in step.get("body", []):
                        if isinstance(body_step, dict) and body_step.get("type") == "agent_call":
                            agent_name = body_step.get("node")
                            if agent_name and agent_name not in defined_agents:
                                missing_agents.add(agent_name)
                                logger.warning("config.validation.missing_agent",
                                            agent_name=agent_name,
                                            step_name=body_step.get("name", "unknown"))
        
        # Check top-level execution plans
        if "execution_plans" in validated_config:
            for plan_id, plan in validated_config["execution_plans"].items():
                check_agents_in_plan(plan)
        
        # Check team execution plans (legacy format)
        if "llm_config" in validated_config and "teams" in validated_config["llm_config"]:
            for team_id, team_config in validated_config["llm_config"]["teams"].items():
                if "execution_plan" in team_config:
                    check_agents_in_plan(team_config["execution_plan"])
        
        # Check team execution plans (ArchDocV3 format)
        if "orchestration" in validated_config and "teams" in validated_config["orchestration"]:
            for team_id, team_config in validated_config["orchestration"]["teams"].items():
                if "execution_plan" in team_config:
                    check_agents_in_plan(team_config["execution_plan"])
        
        # Check persona execution plans (ArchDocV3 format)
        if "llm_config" in validated_config and "personas" in validated_config["llm_config"]:
            for persona_id, persona_config in validated_config["llm_config"]["personas"].items():
                if "execution_plan" in persona_config:
                    check_agents_in_plan(persona_config["execution_plan"])
        
        # Check agent execution plans (legacy format)
        if "llm_config" in validated_config and "agents" in validated_config["llm_config"]:
            for agent_id, agent_config in validated_config["llm_config"]["agents"].items():
                if "execution_plan" in agent_config:
                    check_agents_in_plan(agent_config["execution_plan"])
        
        logger.info("config.validation.agent_references_complete",
                   total_missing_agents=len(missing_agents),
                   missing_agents=list(missing_agents) if missing_agents else None,
                   defined_agents_count=len(defined_agents),
                   archv3_structure=bool("orchestration" in validated_config and "teams" in validated_config["orchestration"]))
        
        return validated_config, missing_agents
    
    @classmethod
    def validate_team_references(cls, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Set[str]]:
        """
        Validate that all team references in execution plans exist in the configuration.
        
        This method handles both legacy and ArchDocV3 team structures.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Tuple of (validated_config, missing_teams)
        """
        validated_config = deepcopy(config)
        
        # Get all defined team names from both legacy and ArchDocV3 locations
        defined_teams = set()
        if "llm_config" in validated_config and "teams" in validated_config["llm_config"]:
            # Legacy teams
            defined_teams.update(validated_config["llm_config"]["teams"].keys())
        
        if "orchestration" in validated_config and "teams" in validated_config["orchestration"]:
            # ArchDocV3 teams
            defined_teams.update(validated_config["orchestration"]["teams"].keys())
        
        # Track missing teams for reporting
        missing_teams = set()
        
        # Function to check team references in an execution plan
        def check_teams_in_plan(plan: Dict[str, Any]) -> None:
            if not isinstance(plan, dict) or "steps" not in plan:
                return
                
            for step in plan.get("steps", []):
                if isinstance(step, dict) and step.get("type") == "team_call":
                    team_name = step.get("target_team")
                    if team_name and team_name not in defined_teams:
                        missing_teams.add(team_name)
                        logger.warning("config.validation.missing_team",
                                      team_name=team_name,
                                      step_name=step.get("name", "unknown"))
                
                # Check teams in loops
                if isinstance(step, dict) and step.get("type") == "loop" and "body" in step:
                    for body_step in step.get("body", []):
                        if isinstance(body_step, dict) and body_step.get("type") == "team_call":
                            team_name = body_step.get("target_team")
                            if team_name and team_name not in defined_teams:
                                missing_teams.add(team_name)
                                logger.warning("config.validation.missing_team",
                                            team_name=team_name,
                                            step_name=body_step.get("name", "unknown"))
        
        # Check top-level execution plans
        if "execution_plans" in validated_config:
            for plan_id, plan in validated_config["execution_plans"].items():
                check_teams_in_plan(plan)
        
        # Check team execution plans - legacy location
        if "llm_config" in validated_config and "teams" in validated_config["llm_config"]:
            for team_id, team_config in validated_config["llm_config"]["teams"].items():
                if "execution_plan" in team_config:
                    check_teams_in_plan(team_config["execution_plan"])
        
        # Check team execution plans - ArchDocV3 location
        if "orchestration" in validated_config and "teams" in validated_config["orchestration"]:
            for team_id, team_config in validated_config["orchestration"]["teams"].items():
                if "execution_plan" in team_config:
                    check_teams_in_plan(team_config["execution_plan"])
        
        # Check persona execution plans
        if "llm_config" in validated_config and "personas" in validated_config["llm_config"]:
            for persona_id, persona_config in validated_config["llm_config"]["personas"].items():
                if "execution_plan" in persona_config:
                    check_teams_in_plan(persona_config["execution_plan"])
        
        # Check agent execution plans
        if "llm_config" in validated_config and "agents" in validated_config["llm_config"]:
            for agent_id, agent_config in validated_config["llm_config"]["agents"].items():
                if "execution_plan" in agent_config:
                    check_teams_in_plan(agent_config["execution_plan"])
        
        logger.info("config.validation.team_references_complete",
                   total_missing_teams=len(missing_teams),
                   missing_teams=list(missing_teams) if missing_teams else None,
                   defined_teams_count=len(defined_teams),
                   archv3_structure=bool("orchestration" in validated_config and "teams" in validated_config["orchestration"]))
        
        return validated_config, missing_teams
    
    @classmethod
    def validate_complete_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete configuration with all cross-references.
        
        This method handles both legacy and ArchDocV3 structures for comprehensive validation.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Validated and normalized configuration
        """
        # Detect configuration structure
        is_archv3 = "orchestration" in config and "teams" in config.get("orchestration", {})
        has_personas = "llm_config" in config and "personas" in config.get("llm_config", {})
        
        # Validate execution plans first
        validated_config = cls.validate_execution_plans(config)
        
        # Then validate cross-references
        validated_config, missing_skills = cls.validate_skill_references(validated_config)
        validated_config, missing_agents = cls.validate_agent_references(validated_config)
        validated_config, missing_teams = cls.validate_team_references(validated_config)
        
        # Additional ArchDocV3-specific validations
        if is_archv3:
            # Check that referenced personas exist
            defined_personas = set()
            missing_personas = set()
            
            if has_personas:
                defined_personas = set(config["llm_config"]["personas"].keys())
            
            # Check for persona references in team agents
            if "orchestration" in config and "teams" in config["orchestration"]:
                for team_id, team_config in config["orchestration"]["teams"].items():
                    # Check agents list
                    if "agents" in team_config and isinstance(team_config["agents"], list):
                        for agent_config in team_config["agents"]:
                            if isinstance(agent_config, dict) and "persona_key" in agent_config:
                                persona_key = agent_config["persona_key"]
                                if persona_key not in defined_personas:
                                    missing_personas.add(persona_key)
                                    logger.warning("config.validation.missing_persona",
                                                  team_id=team_id,
                                                  persona_key=persona_key)
                    
                    # Check tasks list for backward compatibility
                    if "tasks" in team_config and isinstance(team_config["tasks"], list):
                        for task_config in team_config["tasks"]:
                            if isinstance(task_config, dict) and "persona_key" in task_config:
                                persona_key = task_config["persona_key"]
                                if persona_key not in defined_personas:
                                    missing_personas.add(persona_key)
                                    logger.warning("config.validation.missing_persona_in_task",
                                                  team_id=team_id,
                                                  persona_key=persona_key)
            
            # Log persona validation results
            if missing_personas:
                logger.warning("config.validation.missing_personas",
                              count=len(missing_personas),
                              missing_personas=list(missing_personas))
        
        # Log overall results
        validation_status = "success" if not (missing_skills or missing_agents or missing_teams) else "warnings"
        logger.info("config.validation.complete",
                   status=validation_status,
                   architecture="ArchDocV3" if is_archv3 else "Legacy",
                   missing_skills_count=len(missing_skills),
                   missing_agents_count=len(missing_agents),
                   missing_teams_count=len(missing_teams),
                   has_personas=has_personas)
        
        return validated_config