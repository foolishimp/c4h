"""
Prefect wrappers for C4H execution components.
Path: c4h_agents/execution/prefect_wrappers.py
"""

from typing import Dict, Any, Optional, List, Union, Callable
import inspect
from copy import deepcopy
import structlog
import traceback
import time
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