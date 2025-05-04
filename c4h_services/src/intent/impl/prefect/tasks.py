"""
Path: c4h_services/src/intent/impl/prefect/tasks.py 
Task wrapper implementation with enhanced configuration handling.
"""

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
from pathlib import Path

from c4h_services.src.orchestration.factory import AgentFactory
from c4h_agents.agents.base_agent import BaseAgent
from c4h_agents.skills.semantic_iterator import SemanticIterator
from c4h_agents.skills.shared.types import ExtractConfig
from c4h_agents.config import create_config_node, deep_merge, load_config, load_persona_config, render_config
from c4h_services.src.utils.config_utils import validate_config_fragment, ConfigValidationError
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
    
    Args:
        task_config: Configuration for the specific task
        context: Execution context
        effective_config: Complete effective configuration snapshot
        task_name: Optional name override
        
    Returns:
        Dict with agent execution results
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
            
        # Enhance context with task metadata and ensure run ID is set
        enhanced_context = {
            **context,
            'workflow_run_id': run_id,
            'system': {'runid': run_id},  # Explicitly include system namespace
            'task_name': task_name,
        }
        
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
    workspace_dir: Optional[Path] = None
) -> EffectiveConfigInfo:
    """
    Generate and persist an effective configuration snapshot after merging all fragments.
    
    Args:
        context: Job context that may contain config fragments and persona keys
        system_config_path: Path to base system configuration
        workspace_dir: Optional directory to store the snapshot (defaults to workspaces/{run_id})
    
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
        
        # If tasks are defined with persona_key, load and inject those configs
        if "tasks" in context:
            for task in context.get("tasks", []):
                if "persona_key" in task:
                    persona_key = task["persona_key"]
                    prefect_logger.info(f"Loading persona config for: {persona_key}")
                    persona_config = load_persona_config(persona_key, personas_base_path)
                    
                    if persona_config:
                        # Add persona config as a fragment before any task-specific overrides
                        config_fragments.append(persona_config)
        
        # Add job-specific config as final override
        if "config" in context:
            config_fragments.append(context["config"])
        
        # Validate config fragments against schemas
        schema_validated = True
        try:
            # Validate system config
            validate_config_fragment(system_config, "system")
            
            # Validate persona fragments
            for i, fragment in enumerate(config_fragments[1:-1]):
                validate_config_fragment(fragment, "persona")
                
            # Validate job config if present
            if "config" in context:
                validate_config_fragment(context["config"], "job")
                
        except ConfigValidationError as e:
            schema_validated = False
            prefect_logger.error(f"Configuration validation failed: {str(e)}")
            # Continue with snapshot generation despite validation errors
        
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
        
        # Return information about the effective configuration
        return EffectiveConfigInfo(
            snapshot_path=snapshot_path,
            fragments_count=len(config_fragments),
            run_id=run_id,
            schema_validated=schema_validated
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
    
    Args:
        team_results: Results from team execution
        current_context: Current workflow context
        effective_config: Complete effective configuration (from snapshot)
        team_id: Current team ID
        
    Returns:
        Dict with next_team_id and context_updates
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
        
        # Define operator functions
        ops = {
            "equals": operator.eq,
            "not_equals": operator.ne,
            "contains": lambda a, b: b in a,
            "greater_than": operator.gt,
            "less_than": operator.lt,
            "exists": lambda a, b: a is not None,
            "is_empty": lambda a, b: not a
        }
        
        # Helper function to evaluate a single condition
        def evaluate_condition(condition):
            if "task" in condition:
                # Find task result by name
                task_name = condition["task"]
                task_result = next((r for r in team_results.get("results", []) 
                                   if r.get("task_name") == task_name), None)
                
                if not task_result:
                    return False
                
                # Check status condition
                if "status" in condition:
                    return task_result.get("success") == (condition["status"] == "success")
                
                # Check output field condition
                if "output_field" in condition:
                    field_value = task_result.get("result_data", {})
                    for part in condition["output_field"].split("."):
                        field_value = field_value.get(part, {})
                    
                    op = ops.get(condition.get("operator", "equals"), operator.eq)
                    return op(field_value, condition.get("value"))
            
            elif "context_field" in condition:
                # Evaluate against context
                field_path = condition["context_field"]
                field_value = current_context
                for part in field_path.split("."):
                    field_value = field_value.get(part, {})
                
                op = ops.get(condition.get("operator", "equals"), operator.eq)
                return op(field_value, condition.get("value"))
            
            return False
        
        # Process routing rules in order
        for rule in rules:
            condition = rule.get("condition", {})
            
            # Handle list of conditions (AND logic)
            if isinstance(condition, list):
                if all(evaluate_condition(c) for c in condition):
                    return {
                        "next_team_id": rule.get("next_team"),
                        "context_updates": rule.get("context_updates", {})
                    }
            # Handle single condition
            elif evaluate_condition(condition):
                return {
                    "next_team_id": rule.get("next_team"),
                    "context_updates": rule.get("context_updates", {})
                }
        
        # No rules matched, use default
        return {
            "next_team_id": routing_config.get("default"),
            "context_updates": {}
        }
        
    except Exception as e:
        prefect_logger.error(f"Routing evaluation failed: {str(e)}")
        return {"next_team_id": None, "context_updates": {}}