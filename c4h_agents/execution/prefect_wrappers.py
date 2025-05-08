"""
Prefect wrappers for C4H execution components.
Path: c4h_agents/execution/prefect_wrappers.py

This module provides Prefect task and flow wrappers for executing C4H components:
- Execution plans (both top-level and team/agent plans)
- Skills
- Agent instances
- Teams

These wrappers enable orchestration of C4H components with Prefect's workflow engine,
ensuring proper execution, error handling, and lineage tracking.
"""

from typing import Dict, Any, Optional, List, Union, Callable, Type
import inspect
from copy import deepcopy
import structlog
import traceback
import time
import importlib
from datetime import datetime, timedelta

# Import conditionally to handle cases when Prefect is not installed
try:
    from prefect import task, flow
    PREFECT_AVAILABLE = True
except ImportError:
    # Create dummy decorators when Prefect is not available
    def task(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
        
    def flow(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
        
    PREFECT_AVAILABLE = False

from c4h_agents.execution.executor import ExecutionPlanExecutor, ExecutionResult
from c4h_agents.skills.registry import SkillRegistry
from c4h_agents.lineage import EventLogger
from c4h_agents.utils.logging import get_logger

logger = get_logger()

@task(name="execute_plan_task", 
     description="Execute a C4H execution plan with context",
     retries=1,
     retry_delay_seconds=5)
def execute_plan_task(
    execution_plan: Dict[str, Any],
    context: Dict[str, Any],
    effective_config: Dict[str, Any],
    event_logger_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prefect task wrapper for executing a C4H execution plan.
    
    Args:
        execution_plan: The execution plan to execute
        context: Initial context for execution
        effective_config: The effective configuration for the system
        event_logger_config: Optional configuration for the event logger
        
    Returns:
        Execution result dictionary
    """
    start_time = time.time()
    
    try:
        # Initialize skill registry
        registry = SkillRegistry()
        registry.register_builtin_skills()
        registry.load_skills_from_config(effective_config)
        
        # Initialize event logger if configured
        event_logger = None
        if event_logger_config:
            try:
                event_logger = EventLogger(**event_logger_config)
            except Exception as e:
                logger.error("prefect.task.event_logger_init_failed", 
                           error=str(e),
                           traceback=traceback.format_exc())
        
        # Create and run executor
        executor = ExecutionPlanExecutor(
            effective_config=effective_config,
            skill_registry=registry,
            event_logger=event_logger
        )
        
        # Execute the plan
        result = executor.execute_plan(execution_plan, context)
        
        # Log execution time
        duration = time.time() - start_time
        logger.info("prefect.task.execute_plan.complete",
                  duration=duration,
                  steps_executed=result.steps_executed,
                  success=result.success)
        
        # Convert ExecutionResult to dict for Prefect
        return {
            "context": result.context,
            "output": result.output,
            "success": result.success,
            "error": result.error,
            "steps_executed": result.steps_executed,
            "duration": duration
        }
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("prefect.task.execute_plan.failed",
                   error=str(e),
                   traceback=traceback.format_exc(),
                   duration=duration)
        
        # Return error result
        return {
            "context": context,  # Return original context
            "output": None,
            "success": False,
            "error": str(e),
            "steps_executed": 0,
            "duration": duration
        }

@flow(name="execution_plan_flow", 
     description="Flow for executing a C4H execution plan")
def execution_plan_flow(
    execution_plan: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    effective_config: Optional[Dict[str, Any]] = None,
    event_logger_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prefect flow wrapper for executing a C4H execution plan.
    
    Args:
        execution_plan: The execution plan to execute
        context: Initial context for execution (defaults to empty dict)
        effective_config: The effective configuration (defaults to empty dict)
        event_logger_config: Optional configuration for the event logger
        
    Returns:
        Execution result dictionary
    """
    # Set defaults
    context = context or {}
    effective_config = effective_config or {}
    
    # Add flow start time to context
    context["flow_start_time"] = datetime.utcnow().isoformat()
    
    # Run the execution plan task
    result = execute_plan_task(
        execution_plan=execution_plan,
        context=context,
        effective_config=effective_config,
        event_logger_config=event_logger_config
    )
    
    return result

@task(name="execute_skill_task", 
     description="Execute a C4H skill from the registry",
     retries=1,
     retry_delay_seconds=5)
def execute_skill_task(
    skill_name: str,
    parameters: Dict[str, Any],
    effective_config: Dict[str, Any],
    event_logger_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prefect task wrapper for executing a C4H skill.
    
    Args:
        skill_name: Name of the skill to execute
        parameters: Parameters for the skill execution
        effective_config: The effective configuration
        event_logger_config: Optional configuration for the event logger
        
    Returns:
        Skill execution result
    """
    start_time = time.time()
    
    try:
        # Initialize skill registry
        registry = SkillRegistry()
        registry.register_builtin_skills()
        registry.load_skills_from_config(effective_config)
        
        # Initialize event logger if configured
        event_logger = None
        if event_logger_config:
            try:
                event_logger = EventLogger(**event_logger_config)
            except Exception as e:
                logger.error("prefect.task.event_logger_init_failed", 
                           error=str(e),
                           traceback=traceback.format_exc())
        
        # Log skill execution start
        logger.info("prefect.task.execute_skill.starting",
                  skill_name=skill_name,
                  parameter_keys=list(parameters.keys()))
                  
        if event_logger:
            event_logger.log_event(
                event_type="skill_execution_start",
                data={
                    "skill_name": skill_name,
                    "parameters": parameters
                }
            )
        
        # Instantiate and execute the skill
        skill_instance = registry.instantiate_skill(skill_name, effective_config)
        skill_method = getattr(skill_instance, "execute")
        
        # Check if config should be included in parameters
        sig = inspect.signature(skill_method)
        if "config" in sig.parameters and "config" not in parameters:
            parameters["config"] = effective_config
        
        # Execute the skill
        skill_result = skill_method(**parameters)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log skill execution complete
        logger.info("prefect.task.execute_skill.complete",
                  skill_name=skill_name,
                  success=getattr(skill_result, 'success', True),
                  duration=duration)
                  
        if event_logger:
            event_logger.log_event(
                event_type="skill_execution_complete",
                data={
                    "skill_name": skill_name,
                    "success": getattr(skill_result, 'success', True),
                    "duration": duration
                }
            )
        
        # Convert result to dict for Prefect
        if hasattr(skill_result, 'to_dict'):
            return skill_result.to_dict()
        elif isinstance(skill_result, dict):
            return skill_result
        else:
            return {"result": skill_result, "success": True, "duration": duration}
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("prefect.task.execute_skill.failed",
                   skill_name=skill_name,
                   error=str(e),
                   traceback=traceback.format_exc(),
                   duration=duration)
                   
        if event_logger:
            event_logger.log_event(
                event_type="skill_execution_error",
                data={
                    "skill_name": skill_name,
                    "error": str(e),
                    "duration": duration
                }
            )
        
        # Return error result
        return {
            "result": None,
            "success": False,
            "error": str(e),
            "duration": duration
        }

@flow(name="skill_execution_flow", 
     description="Flow for executing a C4H skill")
def skill_execution_flow(
    skill_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    effective_config: Optional[Dict[str, Any]] = None,
    event_logger_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prefect flow wrapper for executing a C4H skill.
    
    Args:
        skill_name: Name of the skill to execute
        parameters: Parameters for the skill execution (defaults to empty dict)
        effective_config: The effective configuration (defaults to empty dict)
        event_logger_config: Optional configuration for the event logger
        
    Returns:
        Skill execution result
    """
    # Set defaults
    parameters = parameters or {}
    effective_config = effective_config or {}
    
    # Add flow start time to parameters
    parameters["flow_start_time"] = datetime.utcnow().isoformat()
    
    # Run the skill execution task
    result = execute_skill_task(
        skill_name=skill_name,
        parameters=parameters,
        effective_config=effective_config,
        event_logger_config=event_logger_config
    )
    
    return result


@task(name="execute_agent_task", 
     description="Execute a C4H agent with context",
     retries=1,
     retry_delay_seconds=5)
def execute_agent_task(
    agent_type: str,
    persona_config: Dict[str, Any],
    input_context: Dict[str, Any],
    effective_config: Dict[str, Any],
    event_logger_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prefect task wrapper for executing a C4H agent with execution plan.
    
    This task executes an agent by:
    1. Checking if the agent has an execution_plan in its configuration
    2. Using the ExecutionPlanExecutor to run the plan
    
    Args:
        agent_type: Type of agent to instantiate
        persona_config: Configuration for the agent's persona
        input_context: Input context for agent processing
        effective_config: The effective configuration for the system
        event_logger_config: Optional configuration for the event logger
        
    Returns:
        Agent execution result dictionary
    """
    start_time = time.time()
    
    try:
        # Initialize event logger if configured
        event_logger = None
        if event_logger_config:
            try:
                event_logger = EventLogger(**event_logger_config)
            except Exception as e:
                logger.error("prefect.task.event_logger_init_failed", 
                           error=str(e),
                           traceback=traceback.format_exc())
        
        # Log agent execution start
        logger.info("prefect.task.execute_agent.starting",
                  agent_type=agent_type,
                  context_keys=list(input_context.keys()))
                  
        if event_logger:
            event_logger.log_event(
                event_type="agent_execution_start",
                data={
                    "agent_type": agent_type,
                    "input_context": {k: str(v)[:100] for k, v in input_context.items()}
                }
            )
        
        # Check for execution plan in the configuration
        if "execution_plan" not in persona_config or not persona_config["execution_plan"].get("enabled", True):
            error_msg = f"Agent does not have a valid execution_plan in its configuration"
            logger.error(error_msg)
            return {
                "result": None,
                "success": False,
                "error": error_msg,
                "duration": time.time() - start_time
            }
            
        # Get the execution plan
        execution_plan = persona_config["execution_plan"]
        
        # Create executor
        executor = ExecutionPlanExecutor(
            effective_config=effective_config,
            event_logger=event_logger
        )
        
        # Execute the plan
        execution_result = executor.execute_plan(execution_plan, input_context)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log agent execution complete
        logger.info("prefect.task.execute_agent.complete",
                  agent_type=agent_type,
                  duration=duration,
                  success=execution_result.success)
                  
        if event_logger:
            event_logger.log_event(
                event_type="agent_execution_complete",
                data={
                    "agent_type": agent_type,
                    "duration": duration,
                    "success": execution_result.success
                }
            )
        
        # Convert ExecutionResult to dict
        result_dict = {
            "context": execution_result.context,
            "result": execution_result.output,
            "success": execution_result.success,
            "error": execution_result.error,
            "steps_executed": execution_result.steps_executed,
            "duration": duration
        }
        
        return result_dict
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("prefect.task.execute_agent.failed",
                   agent_type=agent_type,
                   error=str(e),
                   traceback=traceback.format_exc(),
                   duration=duration)
                   
        if event_logger:
            event_logger.log_event(
                event_type="agent_execution_error",
                data={
                    "agent_type": agent_type,
                    "error": str(e),
                    "duration": duration
                }
            )
        
        # Return error result
        return {
            "result": None,
            "success": False,
            "error": str(e),
            "duration": duration
        }


@flow(name="agent_execution_flow", 
     description="Flow for executing a C4H agent")
def agent_execution_flow(
    agent_type: str,
    persona_config: Dict[str, Any],
    input_context: Optional[Dict[str, Any]] = None,
    effective_config: Optional[Dict[str, Any]] = None,
    event_logger_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prefect flow wrapper for executing a C4H agent.
    
    Args:
        agent_type: Type of agent to instantiate ("GenericLLMAgent", "GenericOrchestratorAgent", etc.)
        persona_config: Configuration for the agent's persona
        input_context: Input context for agent processing (defaults to empty dict)
        effective_config: The effective configuration (defaults to empty dict)
        event_logger_config: Optional configuration for the event logger
        
    Returns:
        Agent execution result dictionary
    """
    # Set defaults
    input_context = input_context or {}
    effective_config = effective_config or {}
    
    # Add flow start time to context
    input_context["flow_start_time"] = datetime.utcnow().isoformat()
    
    # Run the agent execution task
    result = execute_agent_task(
        agent_type=agent_type,
        persona_config=persona_config,
        input_context=input_context,
        effective_config=effective_config,
        event_logger_config=event_logger_config
    )
    
    return result


@task(name="execute_team_task", 
     description="Execute a C4H team with context",
     retries=1,
     retry_delay_seconds=5)
def execute_team_task(
    team_config: Dict[str, Any],
    team_name: str,
    input_context: Dict[str, Any],
    effective_config: Dict[str, Any],
    event_logger_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prefect task wrapper for executing a C4H team.
    
    This task executes a team by using the ExecutionPlanExecutor with the team's execution plan.
    All teams must have an execution_plan in their configuration.
    
    Args:
        team_config: Team configuration dictionary
        team_name: Name of the team (for logging)
        input_context: Input context for team processing
        effective_config: The effective configuration for the system
        event_logger_config: Optional configuration for the event logger
        
    Returns:
        Team execution result dictionary
    """
    start_time = time.time()
    
    try:
        # Initialize event logger if configured
        event_logger = None
        if event_logger_config:
            try:
                event_logger = EventLogger(**event_logger_config)
            except Exception as e:
                logger.error("prefect.task.event_logger_init_failed", 
                           error=str(e),
                           traceback=traceback.format_exc())
        
        # Log team execution start
        logger.info("prefect.task.execute_team.starting",
                  team_name=team_name,
                  context_keys=list(input_context.keys()))
                  
        if event_logger:
            event_logger.log_event(
                event_type="team_execution_start",
                data={
                    "team_name": team_name,
                    "input_context": {k: str(v)[:100] for k, v in input_context.items()}
                }
            )
        
        # Check for execution plan in the team config
        if "execution_plan" not in team_config or not team_config["execution_plan"].get("enabled", True):
            error_msg = f"Team '{team_name}' does not have a valid execution_plan in its configuration"
            logger.error(error_msg)
            return {
                "context": input_context,
                "output": None,
                "success": False,
                "error": error_msg,
                "duration": time.time() - start_time
            }
            
        # Team has an execution plan, use ExecutionPlanExecutor
        execution_plan = team_config["execution_plan"]
        
        # Create executor
        executor = ExecutionPlanExecutor(
            effective_config=effective_config,
            event_logger=event_logger
        )
        
        # Execute the plan
        execution_result = executor.execute_plan(execution_plan, input_context)
        
        # Convert ExecutionResult to dict
        result = {
            "context": execution_result.context,
            "output": execution_result.output,
            "success": execution_result.success,
            "error": execution_result.error,
            "steps_executed": execution_result.steps_executed
        }
        
        # Calculate duration
        duration = time.time() - start_time
        result["duration"] = duration
        
        # Log team execution complete
        logger.info("prefect.task.execute_team.complete",
                  team_name=team_name,
                  duration=duration,
                  success=result["success"])
                  
        if event_logger:
            event_logger.log_event(
                event_type="team_execution_complete",
                data={
                    "team_name": team_name,
                    "duration": duration,
                    "success": result["success"]
                }
            )
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("prefect.task.execute_team.failed",
                   team_name=team_name,
                   error=str(e),
                   traceback=traceback.format_exc(),
                   duration=duration)
                   
        if event_logger:
            event_logger.log_event(
                event_type="team_execution_error",
                data={
                    "team_name": team_name,
                    "error": str(e),
                    "duration": duration
                }
            )
        
        # Return error result
        return {
            "context": input_context,  # Return original context
            "output": None,
            "success": False,
            "error": str(e),
            "duration": duration
        }


@flow(name="team_execution_flow", 
     description="Flow for executing a C4H team")
def team_execution_flow(
    team_config: Dict[str, Any],
    team_name: str,
    input_context: Optional[Dict[str, Any]] = None,
    effective_config: Optional[Dict[str, Any]] = None,
    event_logger_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prefect flow wrapper for executing a C4H team.
    
    Args:
        team_config: Team configuration dictionary
        team_name: Name of the team (for logging)
        input_context: Input context for team processing (defaults to empty dict)
        effective_config: The effective configuration (defaults to empty dict)
        event_logger_config: Optional configuration for the event logger
        
    Returns:
        Team execution result dictionary
    """
    # Set defaults
    input_context = input_context or {}
    effective_config = effective_config or {}
    
    # Add flow start time to context
    input_context["flow_start_time"] = datetime.utcnow().isoformat()
    
    # Run the team execution task
    result = execute_team_task(
        team_config=team_config,
        team_name=team_name,
        input_context=input_context,
        effective_config=effective_config,
        event_logger_config=event_logger_config
    )
    
    return result