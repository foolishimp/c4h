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
import json
import re
from pathlib import Path
from datetime import datetime, timezone

from c4h_services.src.orchestration.factory import AgentFactory
from c4h_agents.agents.base_agent import BaseAgent
from c4h_agents.skills.semantic_iterator import SemanticIterator
from c4h_agents.skills.shared.types import ExtractConfig
from c4h_agents.config import create_config_node, deep_merge, load_config, load_persona_config, render_config, get_available_personas
from c4h_services.src.utils.config_utils import validate_config_fragment, validate_config_fragments, ConfigValidationError
from .models import AgentTaskConfig, EffectiveConfigInfo

logger = get_logger()

@task(retries=2, retry_delay_seconds=10)
def run_agent_task(
    task_config: Dict[str, Any],
    context: Dict[str, Any],
    effective_config: Dict[str, Any],
    task_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prefect task wrapper for agent execution with factory-based instantiation.
    Uses only the factory pattern for agent creation.

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
    
    try:
        # Get task name from task config or parameter
        task_name = task_name or task_config.get("name", "unnamed_task")
        prefect_logger.info(f"Running agent task: {task_name}")
        
        # Create configuration node for context
        context_node = create_config_node(context)
        
        # Get run ID for tracking and lineage
        run_id = context_node.get_value("workflow_run_id") or str(flow_run.get_id())
        
        # Validate required task configuration - both agent_type and name are required now
        if not task_config.get("agent_type") or not task_config.get("name"):
            error_msg = "Missing required fields in task_config. Both 'agent_type' and 'name' are required."
            raise ValueError(error_msg)
            
        if not task_config.get("name"):
            # Set name from parameter or use a fallback
            task_config["name"] = task_name or f"unnamed_task_{str(flow_run.get_id())[-6:]}"
            prefect_logger.warning(f"Missing 'name' in task_config, using: {task_config['name']}")
        
        # Log task configuration for transparency
        persona_key = task_config.get("persona_key")
        agent_info = f"agent_type={task_config['agent_type']}, name={task_config['name']}"
        if persona_key:
            agent_info += f", persona_key={persona_key}"
        prefect_logger.info(f"Task configuration: {agent_info}")
        
        # Log that we're using the full effective config for agent creation
        prefect_logger.info(f"Using effective config snapshot for agent creation")
            
        # Create agent using factory - ONLY supported pattern
        factory = AgentFactory(effective_config)
        
        prefect_logger.info(f"Creating agent using factory: {task_config['agent_type']}, "
                           f"name={task_config['name']}")
        
        # Create agent using factory
        # Only pass the necessary fields needed by the factory
        agent = factory.create_agent(task_config)
            
        # Enhance context with task metadata, run ID, and configuration snapshot info
        enhanced_context = {
            **context,
            'workflow_run_id': run_id,
            'system': {'runid': run_id},  # Explicitly include system namespace
            'task_name': task_name,
        }
        
        # Add configuration snapshot information if available in effective_config
        if "runtime" in effective_config and "config_metadata" in effective_config["runtime"]:
            config_metadata = effective_config["runtime"]["config_metadata"]
            if isinstance(config_metadata, dict):
                enhanced_context["config_metadata"] = config_metadata
                
                # Also add top-level snapshot path for backward compatibility
                if "snapshot_path" in config_metadata:
                    enhanced_context["config_snapshot_path"] = config_metadata["snapshot_path"]
        
        # Special handling for iterator
        if isinstance(agent, SemanticIterator):
            # Get extract config
            extract_config = None
            if 'config' in context:
                extract_config = ExtractConfig(**context['config'])
            
            # Configure iterator if config provided
            if extract_config:
                agent.configure(
                    content=context.get('content', ''),
                    config=extract_config
                )
                
            # Process with configured iterator
            result = agent.process(enhanced_context)
            
        else:
            # Standard agent execution
            result = agent.process(enhanced_context)
        
        # Capture complete agent response 
        response = {
            "success": result.success,
            "result_data": result.data,
            "error": result.error,
            "input": {
                "messages": result.messages.to_dict() if result.messages else None,
                "context": enhanced_context
            },
            "raw_output": result.raw_output,
            "metrics": result.metrics,
            "run_id": run_id,
            "task_name": task_config.get("name", task_name)
        }

        return response

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Task failed: {error_msg}")
        return {
            "success": False,
            "result_data": {},
            "error": error_msg,
            "input": {"context": context},  # Preserve original context
            "raw_output": None,
            "metrics": None,
            "task_name": task_name
        }

@task(retries=1, retry_delay_seconds=5)
def materialise_config(
    context: Dict[str, Any],
    system_config_path: Path,
    workspace_dir: Optional[Path] = None,
    strict_validation: bool = False
) -> EffectiveConfigInfo:
    """
    Generate and persist an effective configuration snapshot after merging all fragments.
    Validates each configuration fragment against its corresponding schema.
    
    Args:
        context: Job context that may contain config fragments and persona keys
        system_config_path: Path to base system configuration
        workspace_dir: Optional directory to store the snapshot (defaults to workspaces/{run_id})
        strict_validation: If True, validation errors will halt processing; 
                          if False, validation errors will be logged but processing will continue
    
    Returns:
        Information about the effective configuration including snapshot path
    """
    prefect_logger = get_run_logger()
    try:
        # Determine workspace directory
        if workspace_dir is None:
            run_id = str(flow_run.get_id())
            workspace_dir = Path("workspaces") / run_id
        
        # Create directory if it doesn't exist
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Load system config as base
        prefect_logger.info(f"Loading system config from: {system_config_path}")
        system_config = load_config(system_config_path)

        # Determine the personas directory path from system_config
        personas_base_path = None
        if "config_locations" in system_config and "personas_dir" in system_config["config_locations"]:
            # Get the relative personas path from config
            relative_personas_dir = system_config["config_locations"]["personas_dir"]
            # Calculate the absolute path based on system_config location
            personas_base_path = system_config_path.parent / relative_personas_dir
            prefect_logger.info(f"Using personas directory from config: {personas_base_path}")
        else:
            # Fallback to default location
            personas_base_path = system_config_path.parent / "personas"
            prefect_logger.warning(f"config_locations.personas_dir not found in system config, using default: {personas_base_path}")
            
        # Extract all config fragments including job-specific ones
        config_fragments = [system_config]
        fragment_sources = ["system"]
        
        # Scan for all available personas first to enable reporting on what's available
        available_personas = get_available_personas(personas_base_path)
        prefect_logger.info(f"Found {len(available_personas)} personas available", 
                          persona_keys=list(available_personas.keys()))
        
        # Create a set to track loaded personas (avoid duplicate loading)
        loaded_personas = set()
        persona_configs = {}
        
        # Check for team-level persona specifications (one level up from tasks)
        if "teams" in context:
            for team_id, team_config in context.get("teams", {}).items():
                if isinstance(team_config, dict):
                    # Check for team-level persona_key
                    if "persona_key" in team_config:
                        persona_key = team_config["persona_key"]
                        prefect_logger.info(f"Loading team-level persona config for team {team_id}: {persona_key}")
                        
                        if persona_key not in loaded_personas:
                            persona_config = load_persona_config(persona_key, personas_base_path)
                            if persona_config:
                                persona_configs[persona_key] = persona_config
                                loaded_personas.add(persona_key)
                    
                    # Look for task-level persona_keys in this team
                    if "tasks" in team_config and isinstance(team_config["tasks"], list):
                        for task in team_config["tasks"]:
                            if isinstance(task, dict) and "persona_key" in task:
                                persona_key = task["persona_key"]
                                prefect_logger.info(f"Loading task-level persona config for task in team {team_id}: {persona_key}")
                                
                                if persona_key not in loaded_personas:
                                    persona_config = load_persona_config(persona_key, personas_base_path)
                                    if persona_config:
                                        persona_configs[persona_key] = persona_config
                                        loaded_personas.add(persona_key)
        
        # Check task-level persona_keys in the context (flat list)
        if "tasks" in context:
            for task in context.get("tasks", []):
                if isinstance(task, dict) and "persona_key" in task:
                    persona_key = task["persona_key"]
                    prefect_logger.info(f"Loading persona config for task: {persona_key}")
                    
                    if persona_key not in loaded_personas:
                        persona_config = load_persona_config(persona_key, personas_base_path)
                        if persona_config:
                            persona_configs[persona_key] = persona_config
                            loaded_personas.add(persona_key)
                            
        # Check for explicit persona override in the context
        if "persona_key" in context:
            persona_key = context["persona_key"]
            prefect_logger.info(f"Loading explicit persona config from context: {persona_key}")
            
            if persona_key not in loaded_personas:
                persona_config = load_persona_config(persona_key, personas_base_path)
                if persona_config:
                    persona_configs[persona_key] = persona_config
                    loaded_personas.add(persona_key)
                    
        # Add all loaded personas to the fragments list
        for persona_key, persona_config in persona_configs.items():
            config_fragments.append(persona_config)
            fragment_sources.append(f"persona_{persona_key}")
            prefect_logger.info(f"Added persona {persona_key} to configuration fragments", 
                              persona_keys=list(persona_config.keys()))
        
        # Add job-specific config as final override
        if "config" in context:
            config_fragments.append(context["config"])
            fragment_sources.append("job")
        
        # Create schema mapping for validation
        schema_map = {}
        for i, source in enumerate(fragment_sources):
            if source == "system":
                schema_map[i] = "system"
            elif source.startswith("persona_"):
                schema_map[i] = "persona"
            elif source == "job":
                schema_map[i] = "job"
        
        # Validate all config fragments against their respective schemas
        prefect_logger.info(f"Validating {len(config_fragments)} configuration fragments")
        validation_results = validate_config_fragments(
            config_fragments, 
            schema_map,
            strict=strict_validation
        )
        
        # Determine overall validation success
        schema_validated = all(success for success, _ in validation_results.values())
        if not schema_validated:
            # Log all validation errors
            error_count = sum(1 for success, _ in validation_results.values() if not success)
            failures = [(idx, error) for idx, (success, error) in validation_results.items() if not success]
            prefect_logger.warning(f"Configuration validation had {error_count} failures", failures=failures)
            
            # In strict mode, this would have already raised an exception
            if strict_validation:
                raise RuntimeError("Configuration validation failed in strict mode")
                
            # In non-strict mode, we continue despite validation errors
            prefect_logger.info("Continuing with snapshot generation despite validation errors")
        
        # Get Prefect run ID for snapshot path
        run_id = str(flow_run.get_id())
        prefect_logger.info(f"Creating config snapshot for run: {run_id}")
        
        # Generate the effective configuration snapshot
        snapshot_path = render_config(
            fragments=config_fragments,
            run_id=run_id,
            workdir=workspace_dir
        )
        
        prefect_logger.info(f"Effective configuration snapshot saved to: {snapshot_path}")
        
        # Create fragment metadata for lineage tracking
        fragment_metadata = [
            {
                "source": source,
                "schema": schema_map.get(i),
                "validated": validation_results.get(i, (False, "Not validated"))[0] if i in validation_results else False,
                "size": len(json.dumps(fragment)) if fragment else 0
            }
            for i, (source, fragment) in enumerate(zip(fragment_sources, config_fragments))
        ]
        
        # Return detailed information about the effective configuration
        return EffectiveConfigInfo(
            snapshot_path=snapshot_path,
            fragments_count=len(config_fragments),
            run_id=run_id,
            schema_validated=schema_validated,
            fragment_metadata=fragment_metadata
        )
        
    except Exception as e:
        error_msg = f"Failed to materialize config: {str(e)}"
        prefect_logger.error(error_msg)
        raise RuntimeError(error_msg)

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
        config_node = create_config_node(effective_config)
        
        # Look up routing configuration for the team
        routing_config = config_node.get_value(f"orchestration.teams.{team_id}.routing")
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