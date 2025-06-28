from typing import Dict, Any, Optional, List
from c4h_services.src.utils.logging import get_logger
from prefect import task, get_run_logger, flow
from prefect import flow
from prefect.runtime import flow_run
from prefect.runtime.flow_run import FlowRunContext
import importlib
import yaml
import operator
import os
import re
from datetime import datetime, timezone, timedelta
import time
import copy

from c4h_services.src.orchestration.factory import AgentFactory
from c4h_agents.agents.base_agent import BaseAgent
from c4h_agents.skills.semantic_iterator import SemanticIterator
from c4h_agents.skills.shared.types import ExtractConfig
from c4h_agents.lineage.event_logger import EventLogger, EventType
from omegaconf import OmegaConf
from c4h_services.src.utils.config_utils import validate_config_fragment
from .models import AgentTaskConfig

logger = get_logger()

@task(retries=2, retry_delay_seconds=10)
def run_agent_task(
    task_config: Dict[str, Any],
    context: Dict[str, Any],
    effective_config: Dict[str, Any],
    task_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prefect task wrapper for agent execution using ExecutionPlanExecutor.
    Requires agents to have embedded execution plans.

    Context Structure Conventions:
    The context follows these conventions for separation of concerns:
    
    1. data_context: Contains the evolving payload/results of the workflow
       - Project-specific data, usually modified by agents
       - Content produced during workflow execution
       - Input/output data exchanged between teams
    
    2. execution_metadata: Contains information about the workflow execution itself
       - workflow_run_id: Unique identifier for this workflow run
       - agent_execution_id: Identifiers for specific agent executions
       - step: Current step in the workflow
       - execution_path: Array of executed team IDs
       - timestamps: Execution timestamps
    
    3. config: Reference to the effective configuration snapshot
       - Contains all configuration needed for the workflow
    
    IMPORTANT: Context is treated as immutable. This function does not modify the 
    input context directly. Any updates to context state should be returned in the 
    result object for the orchestrator to handle.
    
    Args:
        task_config: Configuration for the specific task
        context: Execution context (treated as immutable read-only input)
        effective_config: Complete effective configuration snapshot
        task_name: Optional name override
        
    Returns:
        Dict with agent execution results, including any context updates
    """
    prefect_logger = get_run_logger()
    execution_start_time = datetime.now(timezone.utc)
    
    # Error response template with default fields
    error_response_template = {
        "success": False,
        "result_data": {},
        "error": "",
        "input": {},
        "raw_output": None,
        "metrics": None,
        "task_name": task_name or "unknown_task",
        "execution_type": "error",
        "duration_seconds": 0
    }
    
    try:
        # Validate required inputs
        if task_config is None:
            error_msg = "task_config cannot be None"
            prefect_logger.error(f"Task failed: {error_msg}")
            error_response = dict(error_response_template)
            error_response["error"] = error_msg
            return error_response
            
        if context is None:
            context = {}  # Use empty dict as default
            prefect_logger.warning("Empty context provided, using empty dict")
            
        if effective_config is None:
            error_msg = "effective_config cannot be None"
            prefect_logger.error(f"Task failed: {error_msg}")
            error_response = dict(error_response_template)
            error_response["error"] = error_msg
            return error_response
        
        # Create deep copies to ensure immutability
        task_config_copy = copy.deepcopy(task_config)
        context_copy = copy.deepcopy(context)
        effective_config_copy = copy.deepcopy(effective_config)
        
        # Get task name from task config or parameter
        task_name = task_name or task_config_copy.get("name", "unnamed_task")
        prefect_logger.info(f"Running agent task: {task_name}")
        
        # Create configuration node for context
        try:
            context_node = OmegaConf.create(context_copy)
        except Exception as e:
            error_msg = f"Failed to create context node: {str(e)}"
            prefect_logger.error(error_msg)
            return {
                "success": False,
                "result_data": {},
                "error": error_msg,
                "execution_type": "error",
                "run_id": "unknown",
                "task_name": task_config.get("name", task_name)
            }
        
        # Get run ID for tracking and lineage
        try:
            run_id = OmegaConf.select(context_node, "workflow_run_id") or str(flow_run.get_id())
        except Exception as e:
            prefect_logger.error(f"Failed to get run_id: {str(e)}")
            run_id = str(flow_run.get_id())
        
        # Validate required task configuration - both agent_type and name are required now
        if not task_config_copy.get("agent_type") or not task_config_copy.get("name"):
            error_msg = "Missing required fields in task_config. Both 'agent_type' and 'name' are required."
            raise ValueError(error_msg)
            
        if not task_config_copy.get("name"):
            # Set name from parameter or use a fallback
            task_config_copy["name"] = task_name or f"unnamed_task_{str(flow_run.get_id())[-6:]}"
            prefect_logger.warning(f"Missing 'name' in task_config, using: {task_config_copy['name']}")
        
        # Update error template with task info
        error_response_template["task_name"] = task_config_copy.get("name", task_name)
        
        # Log task configuration for transparency
        persona_key = task_config_copy.get("persona_key")
        agent_info = f"agent_type={task_config_copy['agent_type']}, name={task_config_copy['name']}"
        if persona_key:
            agent_info += f", persona_key={persona_key}"
        prefect_logger.info(f"Task configuration: {agent_info}")
        
        # Enhance context with task metadata, run ID, and configuration snapshot info
        enhanced_context = {
            **context_copy,
            'workflow_run_id': run_id,
            'system': {'runid': run_id},  # Explicitly include system namespace
            'task_name': task_name,
        }
        
        # Add configuration snapshot information if available in effective_config_copy
        if "runtime" in effective_config_copy and "config_metadata" in effective_config_copy["runtime"]:
            config_metadata = effective_config_copy["runtime"]["config_metadata"]
            if isinstance(config_metadata, dict):
                # Create a deep copy to avoid modifying the original
                enhanced_context["config_metadata"] = copy.deepcopy(config_metadata)
                
                # Also add top-level snapshot path for backward compatibility
                if "snapshot_path" in config_metadata:
                    enhanced_context["config_snapshot_path"] = config_metadata["snapshot_path"]
        
        # Initialize event logger if configured
        event_logger = None
        try:
            # Create config node for effective config - not context!
            config_node = OmegaConf.create(effective_config_copy)
            lineage_config = OmegaConf.select(config_node, "llm_config.agents.lineage") or {}
            if lineage_config and lineage_config.get("enabled", True):
                try:
                    # Initialize event logger
                    event_logger = EventLogger(
                        lineage_config,
                        parent_id=run_id,
                        namespace=f"agent_{task_config_copy.get('name')}"
                    )
                    prefect_logger.debug("Initialized event logger for agent", agent_name=task_config_copy.get('name'))
                except Exception as e:
                    prefect_logger.error("Failed to initialize event logger", error=str(e))
        except Exception as e:
            prefect_logger.error("Failed to get lineage config", error=str(e))
        
        # Get agent configuration by merging persona and agent-specific configs
        agent_config = None
        persona_config = None
        merged_config = {}
        
        # First check if we have a persona key
        if persona_key:
            # config_node already created above for effective_config_copy
            persona_config = OmegaConf.select(config_node, f"llm_config.personas.{persona_key}")
            if persona_config:
                prefect_logger.info(f"Found persona config for persona key: {persona_key}")
                merged_config.update(copy.deepcopy(persona_config))
        
        # Then update with specific agent config if available
        agent_name = task_config_copy.get("name")
        if agent_name:
            # config_node already created above for effective_config_copy
            agent_config = OmegaConf.select(config_node, f"llm_config.agents.{agent_name}")
            if agent_config:
                prefect_logger.info(f"Found agent config for agent: {agent_name}")
                merged_config.update(copy.deepcopy(agent_config))
        
        # Always merge in the task_config as it may have overrides
        merged_config.update(copy.deepcopy(task_config_copy))
        
        # Check for execution_plan in the merged config
        execution_plan = merged_config.get("execution_plan")
        if execution_plan is None or not isinstance(execution_plan, dict) or not execution_plan.get("enabled", True):
            error_msg = f"Agent '{agent_name}' does not have a valid execution_plan in its configuration"
            prefect_logger.error(error_msg)
            return {
                "success": False,
                "result_data": {},
                "error": error_msg,
                "execution_type": "error",
                "run_id": run_id,
                "task_name": task_config.get("name", task_name)
            }
        
        # Execute using ExecutionPlanExecutor
        logger.info("agent.using_execution_plan_executor", agent_name=task_name)
        prefect_logger.info(f"Agent has execution_plan, using ExecutionPlanExecutor for {task_name}")
        try:
            # Import the ExecutionPlanExecutor
            from c4h_agents.execution.executor import ExecutionPlanExecutor
            
            # Initialize skill registry for the executor
            from c4h_agents.skills.registry import SkillRegistry
            registry = SkillRegistry()
            registry.register_builtin_skills()
            registry.load_skills_from_config(effective_config)
            
            # Initialize the executor with the effective config
            executor = ExecutionPlanExecutor(
                effective_config=effective_config,
                skill_registry=registry,
                event_logger=event_logger
            )
            
            # Get the execution plan from the merged config
            execution_plan = merged_config["execution_plan"]
            
            # Log execution plan details
            logger.info("agent.execution_plan.starting", 
                        agent_name=task_name,
                        step_count=len(execution_plan.get("steps", [])),
                        executor_id=executor.execution_id)
            prefect_logger.info(f"Executing agent's execution plan for {task_name}")
                          
            # Execute the plan
            execution_result = executor.execute_plan(execution_plan, enhanced_context)
            
            # Calculate execution duration
            execution_end_time = datetime.now(timezone.utc)
            duration_seconds = (execution_end_time - execution_start_time).total_seconds()
            
            # Convert ExecutionResult to agent result format
            response = {
                "success": execution_result.success,
                "result_data": execution_result.output or {},
                "context": execution_result.context,
                "error": execution_result.error,
                "execution_type": "execution_plan",
                "duration_seconds": duration_seconds,
                "execution_id": executor.execution_id,
                "steps_executed": execution_result.steps_executed,
                "run_id": run_id,
                "task_name": task_config.get("name", task_name)
            }
            
            logger.info("agent.execution_plan.completed", 
                        agent_name=task_name,
                        success=execution_result.success,
                        steps_executed=execution_result.steps_executed,
                        duration_seconds=duration_seconds)
            prefect_logger.info(f"Agent execution plan completed for {task_name}")
            
            return response
            
        except Exception as e:
            prefect_logger.error("Execution plan execution failed", 
                           agent_name=task_name,
                           error=str(e),
                           exc_info=True)
            return {
                "success": False,
                "result_data": {},
                "error": f"Execution plan execution failed: {str(e)}",
                "execution_type": "execution_plan_error",
                "run_id": run_id,
                "task_name": task_config.get("name", task_name)
            }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Task failed: {error_msg}")
        
        # Calculate duration even for errors
        execution_end_time = datetime.now(timezone.utc)
        duration_seconds = (execution_end_time - execution_start_time).total_seconds() 
        
        # Create consistent error response
        error_response = dict(error_response_template)  # Use template for consistency
        error_response.update({
            "error": error_msg,
            "input": {"context": context},  # Preserve original context reference
            "task_name": task_name or "unknown_task",
            "duration_seconds": duration_seconds
        })
        
        return error_response


@task(name="evaluate_routing")
def evaluate_routing_task(
    team_results: Dict[str, Any], 
    current_context: Dict[str, Any], 
    effective_config: Dict[str, Any], 
    team_id: str
) -> Dict[str, Any]:
    """
    Evaluate routing rules to determine next team and context updates.
    
    IMPORTANT: This function treats current_context as immutable/read-only.
    It does not modify the input context directly but instead returns context_updates
    which will be merged immutably by the orchestrator to create the next context.
    
    This follows the convention that context state progression is managed by the orchestrator
    
    Args:
        team_results: Results from team execution
        current_context: Current workflow context
        effective_config: Complete effective configuration (from snapshot)
        team_id: Current team ID
        
    Returns:
        Dict with next_team_id and context_updates
        (context_updates will be merged immutably with current_context by the orchestrator)
    """
    prefect_logger = get_run_logger()
    
    try:
        # Create config node for accessing the effective config
        config_node = OmegaConf.create(effective_config)
        
        # Look up routing configuration for the team
        routing_config = OmegaConf.select(config_node, f"orchestration.teams.{team_id}.routing")
        if not routing_config:
            prefect_logger.warning(f"No routing configuration found for team: {team_id}")
            return {"next_team_id": None, "context_updates": {}}
        
        # Get rules list
        rules = routing_config.get("rules", [])
        
        # Define operator functions with enhanced capabilities
        ops = {
            # Basic comparison operators
            "equals": operator.eq,
            "eq": operator.eq,  # Alias for equals
            "not_equals": operator.ne,
            "ne": operator.ne,  # Alias for not_equals
            "contains": lambda a, b: b in a if a is not None else False,
            "contains_any": lambda a, b: any(item in a for item in b) if a is not None and isinstance(b, (list, tuple)) else False,
            "contains_all": lambda a, b: all(item in a for item in b) if a is not None and isinstance(b, (list, tuple)) else False,
            "greater_than": operator.gt,
            "gt": operator.gt,  # Alias
            "less_than": operator.lt, 
            "lt": operator.lt,  # Alias
            "greater_equal": operator.ge,
            "ge": operator.ge,  # Alias
            "less_equal": operator.le,
            "le": operator.le,  # Alias
            
            # Existence checks
            "exists": lambda a, b: a is not None,
            "is_empty": lambda a, b: not a if a is not None else True,
            "is_null": lambda a, b: a is None,
            "not_null": lambda a, b: a is not None,
            
            # Type check operators
            "is_type": lambda a, b: isinstance(a, eval(b)) if isinstance(b, str) else False,
            "has_length": lambda a, b: len(a) == b if hasattr(a, '__len__') else False,
            "min_length": lambda a, b: len(a) >= b if hasattr(a, '__len__') else False,
            "max_length": lambda a, b: len(a) <= b if hasattr(a, '__len__') else False,
            
            # String operators
            "starts_with": lambda a, b: a.startswith(b) if isinstance(a, str) else False,
            "ends_with": lambda a, b: a.endswith(b) if isinstance(a, str) else False,
            "matches": lambda a, b: bool(re.search(b, a)) if isinstance(a, str) and isinstance(b, str) else False,
            
            # Numeric operators
            "in_range": lambda a, b: b[0] <= a <= b[1] if isinstance(b, (list, tuple)) and len(b) == 2 else False,
        }
        
        # Helper function to get value from dotted path
        def get_value_by_path(data, path):
            """Extract value from nested dictionary using dot notation"""
            if not data or not path:
                return None
            
            if isinstance(path, str):
                parts = path.split('.')
            else:
                parts = path  # Assume it's already a list
                
            current = data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif isinstance(current, (list, tuple)) and part.isdigit():
                    index = int(part)
                    if 0 <= index < len(current):
                        current = current[index]
                    else:
                        return None
                else:
                    return None
            return current
            
        # Enhanced function to evaluate a single condition with support for complex nested conditions
        def evaluate_condition(condition):
            """
            Evaluate a condition structure against the current state.
            Supports nested logical operators (AND, OR, NOT) and field references.
            """
            # Handle logical operators first
            if "type" in condition:
                condition_type = condition["type"].lower()
                
                # AND condition - all subconditions must be true
                if condition_type == "and":
                    subconditions = condition.get("conditions", [])
                    return all(evaluate_condition(c) for c in subconditions)
                    
                # OR condition - any subcondition must be true
                elif condition_type == "or":
                    subconditions = condition.get("conditions", [])
                    return any(evaluate_condition(c) for c in subconditions)
                    
                # NOT condition - negate the subcondition
                elif condition_type == "not":
                    subcondition = condition.get("condition", {})
                    return not evaluate_condition(subcondition)
                    
                # ALL_OF condition - same as AND but different syntax 
                elif condition_type == "all_of":
                    subconditions = condition.get("conditions", [])
                    return all(evaluate_condition(c) for c in subconditions)
                    
                # ANY_OF condition - same as OR but different syntax
                elif condition_type == "any_of":
                    subconditions = condition.get("conditions", [])
                    return any(evaluate_condition(c) for c in subconditions)
                
                # Legacy syntax support
                elif condition_type == "simple" and "field" in condition:
                    # Legacy simple field comparison
                    field = condition.get("field")
                    operator_name = condition.get("operator", "equals")
                    expected_value = condition.get("value")
                    
                    # Extract actual value
                    actual_value = get_value_by_path(current_context, field)
                    
                    # Apply operator
                    op_func = ops.get(operator_name, operator.eq)
                    return op_func(actual_value, expected_value)
                
            # Handle task output conditions
            if "task" in condition:
                # Find task result by name
                task_name = condition["task"]
                task_result = next((r for r in team_results.get("results", []) 
                                   if r.get("task_name") == task_name), None)
                
                if not task_result:
                    prefect_logger.warning(f"Task '{task_name}' not found in team results")
                    return False
                
                # Check status condition
                if "status" in condition:
                    status_match = task_result.get("success") == (condition["status"] == "success")
                    prefect_logger.debug(f"Status condition for task '{task_name}': {status_match}")
                    return status_match
                
                # Check output field condition
                if "output_field" in condition:
                    field_path = condition["output_field"]
                    field_value = get_value_by_path(task_result.get("result_data", {}), field_path)
                    
                    operator_name = condition.get("operator", "equals")
                    expected_value = condition.get("value")
                    
                    op_func = ops.get(operator_name, operator.eq)
                    result = op_func(field_value, expected_value)
                    
                    prefect_logger.debug(f"Field condition for task '{task_name}', field '{field_path}': {result}, "
                                       f"actual value: {field_value}, expected: {expected_value}, operator: {operator_name}")
                    return result
            
            # Handle context field conditions
            elif "context_field" in condition:
                field_path = condition["context_field"]
                field_value = get_value_by_path(current_context, field_path)
                
                operator_name = condition.get("operator", "equals")
                expected_value = condition.get("value")
                
                op_func = ops.get(operator_name, operator.eq)
                result = op_func(field_value, expected_value)
                
                prefect_logger.debug(f"Context field condition for '{field_path}': {result}, "
                                   f"actual value: {field_value}, expected: {expected_value}, operator: {operator_name}")
                return result
                
            # Handle config field conditions
            elif "config_field" in condition:
                field_path = condition["config_field"]
                field_value = get_value_by_path(effective_config, field_path)
                
                operator_name = condition.get("operator", "equals")
                expected_value = condition.get("value")
                
                op_func = ops.get(operator_name, operator.eq)
                result = op_func(field_value, expected_value)
                
                prefect_logger.debug(f"Config field condition for '{field_path}': {result}, "
                                   f"actual value: {field_value}, expected: {expected_value}, operator: {operator_name}")
                return result
                
            # Legacy support for simplified conditions as direct field=value checks
            elif isinstance(condition, dict) and not any(k in ("type", "task", "context_field", "config_field") for k in condition.keys()):
                # Simple comparison of context fields
                # Example: {"status": "success", "complete": true}
                return all(current_context.get(k) == v for k, v in condition.items())
                
            # Default for unrecognized condition format
            prefect_logger.warning(f"Unrecognized condition format: {condition}")
            return False
        
        # Process routing rules in order with enhanced logging
        for i, rule in enumerate(rules):
            condition = rule.get("condition", {})
            
            # Log rule we're evaluating
            prefect_logger.debug(f"Evaluating routing rule #{i+1}/{len(rules)} for team: {team_id}")
            
            try:
                # Handle list of conditions (AND logic) - legacy support
                if isinstance(condition, list):
                    all_true = all(evaluate_condition(c) for c in condition)
                    if all_true:
                        prefect_logger.info(f"Rule #{i+1} matched (list conditions): {rule.get('next_team')}")
                        # Include recursion strategy for legacy conditions as well
                        recursion_strategy = rule.get("recursion_strategy", "default")
                        
                        return {
                            "next_team_id": rule.get("next_team"),
                            "context_updates": rule.get("context_updates", {}),
                            "matched_rule": i+1,
                            "recursion_strategy": recursion_strategy
                        }
                    else:
                        prefect_logger.debug(f"Rule #{i+1} did not match (list conditions)")
                        
                # Handle structured condition (enhanced DSL)
                elif isinstance(condition, dict):
                    # Enhanced condition evaluation
                    result = evaluate_condition(condition)
                    if result:
                        prefect_logger.info(f"Rule #{i+1} matched: navigating to {rule.get('next_team')}")
                        
                        # Get context updates with metadata for tracking
                        context_updates = rule.get("context_updates", {})
                        
                        # Add routing metadata to context updates for lineage tracking
                        if "routing_info" not in context_updates:
                            context_updates["routing_info"] = {}
                            
                        context_updates["routing_info"] = {
                            "team_id": team_id,
                            "rule_index": i,
                            "matched_rule": i+1,
                            "next_team": rule.get("next_team"),
                            # Add temporal data to execution_metadata namespace
                            # following context structure conventions:
                            # - data_context: workflow payload data
                            # - execution_metadata: workflow execution information
                            "execution_metadata": {
                                "prior_team": team_id,
                                "routing_evaluation_time": datetime.now(timezone.utc).isoformat()
                            },
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "condition_type": "structured" if "type" in condition else "simple"
                        }
                        
                        # Check for recursion_strategy in the rule
                        recursion_strategy = rule.get("recursion_strategy", "default")
                             
                        # Enhanced return value with recursion support
                        return {
                            # The next_team field can now be either a string (single team) 
                            # or a list of strings (multiple teams for fan-out)
                            "next_team_id": rule.get("next_team"),
                            "context_updates": context_updates,
                            "matched_rule": i+1,
                            "matched_condition": condition,
                            "recursion_strategy": recursion_strategy
                        }
                    else:
                        prefect_logger.debug(f"Rule #{i+1} did not match")
                        
                # Handle string conditions (legacy support)
                elif isinstance(condition, str):
                    prefect_logger.warning(f"String conditions deprecated but attempting to evaluate: {condition}")
                    # Support minimal legacy behavior
                    if condition == "all_success" and all(r.get("success", False) for r in team_results.get("results", [])):
                        prefect_logger.info(f"Rule #{i+1} matched (all_success): {rule.get('next_team')}")
                        # Will support next_team being a list for fan-out
                        # Include recursion strategy for legacy conditions as well
                        recursion_strategy = rule.get("recursion_strategy", "default")
                        
                        return {
                            "next_team_id": rule.get("next_team"),
                            "context_updates": rule.get("context_updates", {}),
                            "matched_rule": i+1,
                            "recursion_strategy": recursion_strategy
                        }
                    elif condition == "any_failure" and any(not r.get("success", True) for r in team_results.get("results", [])):
                        prefect_logger.info(f"Rule #{i+1} matched (any_failure): {rule.get('next_team')}")
                        # Will support next_team being a list for fan-out
                        # Include recursion strategy for legacy conditions as well
                        recursion_strategy = rule.get("recursion_strategy", "default")
                        
                        return {
                            "next_team_id": rule.get("next_team"),
                            "context_updates": rule.get("context_updates", {}),
                            "matched_rule": i+1,
                            "recursion_strategy": recursion_strategy
                        }
                
            except Exception as e:
                # Log error but continue to next rule
                prefect_logger.error(f"Error evaluating rule #{i+1}: {str(e)}")
        
        # No rules matched, use default
        prefect_logger.info(f"No rules matched, using default: {routing_config.get('default')}")
        
        # Note: default can also be a list of teams for fan-out
        # Check for default recursion strategy
        default_recursion_strategy = routing_config.get("default_recursion_strategy", "default")
        
        return {
            "next_team_id": routing_config.get("default"),
            "context_updates": routing_config.get("default_context_updates", {}),
            "matched_rule": "default",
            "recursion_strategy": default_recursion_strategy
        }
        
    except Exception as e:
        prefect_logger.error(f"Routing evaluation failed: {str(e)}")
        return {
            "next_team_id": None, 
            "context_updates": {},
            "recursion_strategy": "default",
            "error": str(e)
        }