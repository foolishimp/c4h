from prefect import flow, get_run_logger
from prefect.context import get_run_context
from typing import Dict, Any, Optional, List
import asyncio
import concurrent.futures
# Import get_logger from the shared utility path
from c4h_services.src.utils.logging import get_logger, truncate_log_string
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import uuid
import time
import os
import json
import io
import yaml

from c4h_agents.lineage.event_logger import EventLogger, EventType

from c4h_agents.config import create_config_node
from c4h_agents.agents.lineage_context import LineageContext
# Import evaluate_routing_task from the tasks module
from .tasks import run_agent_task, materialise_config, evaluate_routing_task
# Remove unused factory imports if run_basic_workflow was deleted
# from .factories import (...)

# Import the LineageContext utility
from c4h_agents.agents.lineage_context import LineageContext

# Use the imported get_logger
logger = get_logger()

# --- prepare_workflow_config function ---
def prepare_workflow_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare workflow configuration with proper run ID and context.
    Uses hierarchical configuration access.
    """
    # Use get_run_logger() inside the flow/task for Prefect context awareness
    run_logger = get_run_logger() if get_run_context() else logger # Fallback to global logger if no context
    try:
        # Get workflow run ID from Prefect context
        workflow_id = None
        ctx = get_run_context()
        if ctx and hasattr(ctx, "flow_run") and ctx.flow_run and hasattr(ctx.flow_run, "id"):
            workflow_id = str(ctx.flow_run.id)

        # If no Prefect context, check base_config or generate new ID
        if not workflow_id:
             config_node_temp = create_config_node(base_config)
             workflow_id = config_node_temp.get_value("workflow_run_id") or \
                           config_node_temp.get_value("system.runid")

        if not workflow_id:
            workflow_id = f"wf_standalone_{str(uuid.uuid4())[:8]}" # Generate ID if still missing
            run_logger.warning("workflow.missing_prefect_context_and_config_id", generated_workflow_id=workflow_id)

        # Deep copy to avoid mutations
        config = deepcopy(base_config)

        # Ensure system namespace exists and set runid
        if 'system' not in config: config['system'] = {}
        config['system']['runid'] = workflow_id

        # Ensure runtime namespace exists and set workflow IDs
        if 'runtime' not in config: config['runtime'] = {}
        config['runtime'].update({
            'workflow_run_id': workflow_id,
            'run_id': workflow_id,
        })
        if 'workflow' not in config['runtime']: config['runtime']['workflow'] = {}
        config['runtime']['workflow']['id'] = workflow_id
        if 'start_time' not in config['runtime']['workflow']:
             config['runtime']['workflow']['start_time'] = datetime.now(timezone.utc).isoformat()


        # Also set workflow_run_id at top level for direct access
        config['workflow_run_id'] = workflow_id

        # Ensure lineage tracking configuration exists and is enabled
        if 'llm_config' not in config: config['llm_config'] = {}
        if 'agents' not in config['llm_config']: config['llm_config']['agents'] = {}
        if 'lineage' not in config['llm_config']['agents']:
            config['llm_config']['agents']['lineage'] = {}

        # Set defaults if not present, but don't override existing values unless necessary
        lineage_defaults = {
            'enabled': True,
            'namespace': 'c4h_agents',
            'event_detail_level': 'full',
            'separate_input_output': False,
            'backend': { 'type': 'file', 'path': 'workspaces/lineage' }
        }
        # Merge defaults without overwriting existing keys
        current_lineage_config = config['llm_config']['agents']['lineage']
        for key, value in lineage_defaults.items():
            if key not in current_lineage_config:
                current_lineage_config[key] = value
            elif key == 'backend' and isinstance(value, dict):
                 # Merge backend defaults carefully
                 current_backend = current_lineage_config.get('backend', {})
                 if not isinstance(current_backend, dict): current_backend = {}
                 for bk, bv in value.items():
                      if bk not in current_backend:
                           current_backend[bk] = bv
                 current_lineage_config['backend'] = current_backend


        run_logger.debug("workflow.config_prepared",
            workflow_id=workflow_id,
            config_keys=list(config.keys()))

        return config

    except Exception as e:
        run_logger.error("workflow.config_prep_failed", error=str(e), exc_info=True) # Log traceback
        raise

# --- run_declarative_workflow function ---
@flow(name="declarative_workflow")
def run_declarative_workflow(
    initial_context: Dict[str, Any],
    system_config_path: Path = Path("config/system_config.yml"),
    max_total_teams: Optional[int] = None, # Allow None to use config default
    max_recursion_depth: Optional[int] = None # Allow None to use config default
) -> Dict[str, Any]:
    """
    Execute a workflow defined declaratively in configuration.
    Uses the effective configuration snapshot for all execution.
    
    Context Structure Conventions:
    The workflow context follows these conventions for separation of concerns:
    
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
       (Often managed by LineageContext)
    
    3. config: Reference to the effective configuration snapshot
       - Immutable snapshot reference that doesn't change during execution
       - Contains all configuration needed for the workflow
    
    IMPORTANT: Context is treated as immutable. Context updates create new context
    objects rather than modifying existing ones. This ensures reproducible state
    transitions and prevents side effects.

    Args:
        initial_context: Initial context including config fragments
        system_config_path: Path to base system configuration
        max_total_teams: Safety limit for total team executions (overrides config if set)
        max_recursion_depth: Safety limit for recursive team executions (overrides config if set)

    Returns:
        Dict with workflow execution results
    """
    run_logger = get_run_logger()
    config_info = None # Initialize for potential error handling

    try:
        # Step 1: Generate the effective configuration snapshot with schema validation
        config_info = materialise_config(
            context=initial_context,
            system_config_path=system_config_path,
            workspace_dir=None,  # Use default workspace dir based on run ID
            strict_validation=False  # Continue even with validation errors, just log them
        )

        # Log validation results
        if config_info.schema_validated:
            run_logger.info("Configuration validation succeeded for all fragments")
        else:
            run_logger.warning("Configuration validation had failures, using snapshot anyway")
            
        # Log metadata about fragments for lineage tracking
        if config_info.fragment_metadata:
            run_logger.info("Configuration fragment information", 
                          fragments_count=len(config_info.fragment_metadata),
                          metadata=config_info.fragment_metadata)

        # Load the effective configuration FROM THE SNAPSHOT
        run_logger.info(f"Loading effective configuration from: {config_info.snapshot_path}")
        with open(config_info.snapshot_path, 'r') as f:
            effective_config = yaml.safe_load(f) or {} # Ensure it's a dict

        # Store config metadata in the config itself for lineage tracking
        if 'runtime' not in effective_config:
            effective_config['runtime'] = {}
            
        # Get lineage configuration from effective config
        lineage_config = config_node.get_value("llm_config.agents.lineage", {})
        if not lineage_config.get("enabled", True):
            run_logger.info("Lineage tracking is disabled in configuration")

        # Initialize event logger for orchestration-level event sourcing
        event_logger = EventLogger(lineage_config, workflow_id)
        
        # Log workflow start event with initial context metadata
        event_logger.log_event(
            event_type=EventType.WORKFLOW_START,
            payload={"initial_context": initial_context},
            config_snapshot_path=str(config_info.snapshot_path),
            config_hash=config_info.config_hash)
        if 'config_metadata' not in effective_config['runtime']:
            effective_config['runtime']['config_metadata'] = {}
            
        # Store config snapshot info in runtime metadata
        effective_config['runtime']['config_metadata'] = {
            'snapshot_path': str(config_info.snapshot_path),
            'config_hash': config_info.config_hash,
            'schema_validated': config_info.schema_validated,
            'fragments_count': config_info.fragments_count,
            'timestamp': config_info.timestamp
        }

        # Create config node for path-based access
        config_node = create_config_node(effective_config)

        # Get workflow run ID for tracking (should be embedded in snapshot)
        workflow_id = config_node.get_value("system.runid") or str(config_info.run_id)
        run_logger.info(f"Workflow Run ID: {workflow_id}") # Log the ID being used

        # Get global safety limits from effective config, overridden by args if provided
        config_max_total = config_node.get_value("orchestration.max_total_teams", default=30)
        final_max_total_teams = max_total_teams if max_total_teams is not None else config_max_total

        config_max_recursion = config_node.get_value("orchestration.max_recursion_depth", default=5)
        final_max_recursion_depth = max_recursion_depth if max_recursion_depth is not None else config_max_recursion

        run_logger.info("Workflow limits", max_total_teams=final_max_total_teams, max_recursion_depth=final_max_recursion_depth)

        # Initialize safety counters
        total_teams_executed = 0
        team_execution_counts = {}  # Track execution count per team

        # Get entry team from config or use default
        entry_team_id = config_node.get_value("orchestration.entry_team") or "discovery"
        run_logger.info(f"Starting workflow with entry team: {entry_team_id}")

        # Initialize current state
        current_team_id = entry_team_id
        # Start context with initial context, but ensure config points to the loaded effective_config
        current_context = initial_context.copy()
        current_context["config"] = effective_config # Ensure context uses the loaded snapshot
        current_context["config_snapshot_path"] = str(config_info.snapshot_path)
        # Ensure workflow_run_id is consistently set in context
        current_context["workflow_run_id"] = workflow_id
        if "system" not in current_context: current_context["system"] = {}
        current_context["system"]["runid"] = workflow_id


        # Track execution results
        team_results = {}
        execution_path = []
        # Track the workflow status
        final_workflow_status = "success" # Assume success initially
        final_workflow_error = None

        # Helper function to execute a team and handle its result
        def execute_single_team(team_id, team_context):
            """Execute a single team with the given context and return its result"""
            logger = get_run_logger()
            logger.info(f"Executing team: {team_id} as part of workflow")
            
            try:
                # Get per-team recursion depth limit (if specified in config)
                team_max_depth = config_node.get_value(f"orchestration.teams.{team_id}.max_recursion_depth", default=final_max_recursion_depth)
                
                # Check team-specific recursion depth
                current_team_count = team_execution_counts.get(team_id, 0)
                if current_team_count >= team_max_depth:
                    logger.warning(f"Maximum recursion depth ({team_max_depth}) reached for team {team_id}")
                    return {
                        "success": False,
                        "error": f"Exceeded maximum recursion depth ({team_max_depth}) for team {team_id}",
                        "team_id": team_id,
                        "execution_path": team_context.get("execution_path", [])
                    }
                
                # Update team execution counter
                team_execution_counts[team_id] = current_team_count + 1
                nonlocal total_teams_executed
                total_teams_executed += 1
                
                # Add to execution path for tracking
                execution_path_update = list(team_context.get("execution_path", [])) + [team_id]
                team_context["execution_path"] = execution_path_update
                
                logger.info(f"Executing team: {team_id} "
                            f"(execution {total_teams_executed}, depth {current_team_count + 1})")
                
                # Execute the team as a subflow
                team_result = execute_team_subflow(
                    team_id=team_id,
                    effective_config=effective_config,
                    current_context=team_context
                )
                
                # Add execution path information to the result
                team_result["execution_path"] = execution_path_update
                return team_result
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to execute team {team_id}: {error_msg}", exc_info=True)
                return {
                    "success": False,
                    "error": error_msg,
                    "team_id": team_id,
                    "execution_path": team_context.get("execution_path", [])
                }

        # Prepare concentrator tracking (for fan-in)
        concentrator_data = {
            "active": False,
            "team_id": None,
            "source_teams": [],
            "completion_condition": {},
            "aggregation_strategy": {},
            "pending_results": {},
            "received_results": []
        }
        
        # Main workflow loop
        while current_team_id:
            # Check global execution limits
            # For fan-out/parallel execution we need to account for potential parallel teams
            remaining_capacity = final_max_total_teams - total_teams_executed
            if remaining_capacity <= 0:
                run_logger.warning(f"Maximum total team executions ({final_max_total_teams}) reached")
                # Log error event for max team executions reached
                event_logger.log_event(
                    event_type=EventType.ERROR_EVENT,
                    payload={
                        "error_type": "max_teams_exceeded",
                        "max_total_teams": final_max_total_teams,
                        "teams_executed": total_teams_executed,
                        "current_team_id": current_team_id
                    },
                    step_name="safety_check",
                    execution_path=execution_path,
                    config_snapshot_path=str(config_info.snapshot_path),
                    config_hash=config_info.config_hash
                )
                final_workflow_status = "error"
                final_workflow_error = f"Exceeded maximum total team executions ({final_max_total_teams})"
                # Log workflow end with error status
                event_logger.log_event(
                    event_type=EventType.WORKFLOW_END,
                    payload={"status": final_workflow_status, "error": final_workflow_error}
                )
                break

            # Get per-team recursion depth limit (if specified in config)
            team_max_depth = config_node.get_value(f"orchestration.teams.{current_team_id}.max_recursion_depth", default=final_max_recursion_depth)

            # Check if the current team is a concentrator
            team_type = config_node.get_value(f"orchestration.teams.{current_team_id}.type", default="standard")
            
            if team_type == "concentrator":
                run_logger.info(f"Executing concentrator team: {current_team_id}")
                
                # Get concentrator configuration
                concentrator_config = config_node.get_value(f"orchestration.teams.{current_team_id}")
                
                # Extract concentrator parameters
                source_teams = concentrator_config.get("source_teams", [])
                source_step = concentrator_config.get("source_step")
                completion_condition = concentrator_config.get("completion_condition", {"type": "count", "value": len(source_teams)})
                aggregation_strategy = concentrator_config.get("aggregation_strategy", {"type": "list", "output_field": "aggregated_results"})
                
                run_logger.info(f"Concentrator setup: source_teams={source_teams}, completion_condition={completion_condition}")
                
                # Initialize concentrator state
                concentrator_data = {
                    "active": True,
                    "team_id": current_team_id,
                    "source_teams": source_teams,
                    "source_step": source_step,
                    "completion_condition": completion_condition,
                    "aggregation_strategy": aggregation_strategy,
                    "pending_results": {team: None for team in source_teams},
                    "received_results": []
                }
                
                # If we don't have any source teams defined, we can't concentrate
                if not source_teams:
                    run_logger.error(f"Concentrator {current_team_id} has no source_teams defined")
                    team_result = {
                        "success": False,
                        "team_id": current_team_id,
                        "error": "Concentrator has no source_teams defined",
                        "type": "concentrator_error"
                    }
                    
                    team_results[f"{current_team_id}_concentrator"] = team_result

                    # Log event for concentrator error
                    event_logger.log_event(
                        event_type=EventType.ERROR_EVENT,
                        payload={
                            "team_id": current_team_id,
                            "error": "Concentrator has no source_teams defined",
                            "type": "concentrator_error"
                        },
                        step_name=current_team_id,
                        execution_path=execution_path,
                    )
                    
                    # Evaluate routing to determine next team
                    routing_result = evaluate_routing_task(
                        team_results=team_result, # Pass the concentrator error result
                        current_context=current_context,
                        effective_config=effective_config,
                        team_id=current_team_id
                    )
                    
                    current_team_id = routing_result.get("next_team_id")
                    continue
                
                # Check if we're waiting for a completion count
                if completion_condition.get("type") == "count":
                    required_count = completion_condition.get("value", len(source_teams))
                    run_logger.info(f"Concentrator waiting for {required_count} results")
                    
                    # Wait for results to arrive - initially just wait for all (future: implement timeout)
                    received_results = []
                    
                    # Create a deep copy of the context for each team to ensure immutability
                    team_contexts = {team_id: deepcopy(current_context) for team_id in source_teams}
                    
                    # Add concentrator ID to each team's context
                    for team_id, ctx in team_contexts.items():
                        if "concentrator" not in ctx:
                            ctx["concentrator"] = {}
                        ctx["concentrator"]["id"] = current_team_id
                        ctx["concentrator"]["role"] = "source"
                    
                # Log concentrator start event
                event_logger.log_event(
                    event_type=EventType.CONCENTRATOR_START,
                    payload={
                        "source_teams": source_teams,
                        "completion_condition": completion_condition
                    },
                    step_name=current_team_id)
                # Add concentrator metadata to the main context
                concentrator_ctx = deepcopy(current_context)
                if "concentrator" not in concentrator_ctx:
                    concentrator_ctx["concentrator"] = {}
                concentrator_ctx["concentrator"]["id"] = current_team_id
                concentrator_ctx["concentrator"]["role"] = "aggregator"
                concentrator_ctx["concentrator"]["expected_teams"] = source_teams
                
                # Execute all source teams in parallel
                run_logger.info(f"Executing concentrator source teams in parallel: {source_teams}")
                
                try:
                    # Execute all source teams concurrently using Prefect's concurrent execution
                    concurrent_results = []
                    
                    # Use a thread pool for parallelism (simple approach)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(source_teams)) as executor:
                        # Submit all team executions
                        future_to_team = {
                            executor.submit(execute_single_team, team_id, team_contexts[team_id]): team_id
                            for team_id in source_teams
                        }
                        
                        # As futures complete, we get their results
                        for future in concurrent.futures.as_completed(future_to_team):
                            team_id = future_to_team[future]
                            try:
                                result = future.result()
                                run_logger.info(f"Received result from team {team_id} for concentrator {current_team_id}")
                                
                                # Add to received results
                                received_results.append({
                                    "team_id": team_id,
                                    "result": result
                                })
                                
                                # Store in team results with a unique key
                                team_results[f"{team_id}_source_for_{current_team_id}"] = result

                                # Log event for concentrator input received
                                event_logger.log_event(
                                    event_type=EventType.CONCENTRATOR_INPUT_RECEIVED,
                                    payload={
                                        "team_id": team_id,
                                        "source_for": current_team_id,
                                        "result_summary": {"success": result.get("success", False)}
                                    },
                                    step_name=team_id,
                                )
                            
                            except Exception as exc:
                                run_logger.error(f"Source team {team_id} generated an exception: {exc}")
                                # Add a failed result
                                received_results.append({
                                    "team_id": team_id,
                                    "result": {
                                        "success": False,
                                        "error": f"Exception in team execution: {str(exc)}",
                                        "team_id": team_id
                                    }
                                })

                                # Log error event for source team
                                event_logger.log_event(
                                    event_type=EventType.ERROR_EVENT,
                                    payload={
                                        "team_id": team_id,
                                        "error": f"Exception in team execution: {str(exc)}",
                                        "source_for": current_team_id
                                    },
                                    step_name=team_id,
                                )
                
                except Exception as e:
                    run_logger.error(f"Error during parallel execution of source teams: {str(e)}", exc_info=True)
                    # If we failed to execute parallel teams, treat as concentrator failure
                    team_result = {
                        "success": False,
                        "team_id": current_team_id,
                        "error": f"Failed to execute source teams in parallel: {str(e)}",
                        "type": "concentrator_error"
                    }

                    # Log error event for concentrator
                    event_logger.log_event(
                        event_type=EventType.ERROR_EVENT,
                        payload={
                            "team_id": current_team_id,
                            "error": f"Failed to execute source teams in parallel: {str(e)}",
                            "type": "concentrator_error"
                        },
                        step_name=current_team_id,
                    )
                    team_results[f"{current_team_id}_concentrator"] = team_result
                    
                    # Evaluate routing with failure result
                    routing_result = evaluate_routing_task(
                        team_results=team_result,
                        current_context=current_context,
                        effective_config=effective_config,
                        team_id=current_team_id
                    )
                    
                    current_team_id = routing_result.get("next_team_id")
                    continue
                
                # Check if we received the required number of results
                if len(received_results) >= required_count:
                    run_logger.info(f"Concentrator received required {required_count} results")
                    
                    # Apply aggregation strategy
                    aggregation_type = aggregation_strategy.get("type", "list")
                    output_field = aggregation_strategy.get("output_field", "aggregated_results")
                    
                    if aggregation_type == "list":
                        # Simple list aggregation - collect all results
                        aggregated_data = {
                            output_field: [item["result"] for item in received_results]
                        }
                        run_logger.info(f"Applied 'list' aggregation strategy, collected {len(aggregated_data[output_field])} results")
                        
                    elif aggregation_type == "merge_dict":
                        # Merge dictionaries from all results
                        merged_dict = {}
                        for item in received_results:
                            result_data = item.get("result", {}).get("data", {})
                            if isinstance(result_data, dict):
                                merged_dict.update(result_data)
                        
                        aggregated_data = {output_field: merged_dict}
                        run_logger.info(f"Applied 'merge_dict' aggregation strategy, merged keys: {list(merged_dict.keys())}")
                        
                    elif aggregation_type == "custom_skill":
                        # TODO: Implement custom skill aggregation in future
                        run_logger.warning(f"Custom skill aggregation not yet implemented, falling back to list")
                        aggregated_data = {
                            output_field: [item["result"] for item in received_results]
                        }
                    
                    else:
                        # Unknown aggregation type, default to list
                        run_logger.warning(f"Unknown aggregation type '{aggregation_type}', falling back to list")
                        aggregated_data = {
                            output_field: [item["result"] for item in received_results]
                        }
                    
                    # Create concentrator result
                    team_result = {
                        "success": True,
                        "team_id": current_team_id,
                        "type": "concentrator",
                        "aggregated_count": len(received_results),
                        "source_teams": [item["team_id"] for item in received_results],
                        "data": aggregated_data
                    }
                    
                    # Store in team results
                    team_results[f"{current_team_id}_concentrator"] = team_result
                    
                    # Log concentrator end event with aggregation results
                    event_logger.log_event(
                        event_type=EventType.CONCENTRATOR_END,
                        payload={
                            "team_id": current_team_id,
                            "aggregated_count": len(received_results),
                            "source_teams": [item["team_id"] for item in received_results],
                            "aggregation_strategy": aggregation_type
                        },
                        step_name=current_team_id,
                    )
                    # Evaluate routing based on concentrator result
                    routing_result = evaluate_routing_task(
                        team_results=team_result,
                        current_context=current_context,
                        effective_config=effective_config,
                        team_id=current_team_id
                    )
                    
                    # Update the context with aggregated data
                    context_updates = routing_result.get("context_updates", {})
                    # Merge in the aggregated data if not already in context updates
                    for key, value in aggregated_data.items():
                        if key not in context_updates:
                            context_updates[key] = value
                    
                    routing_result["context_updates"] = context_updates
                    
                    # Get next team
                    current_team_id = routing_result.get("next_team_id")
                else:
                    # We didn't receive enough results, treat as error
                    run_logger.error(f"Concentrator did not receive enough results: got {len(received_results)}, needed {required_count}")
                    # Log error event for concentrator incomplete
                    event_logger.log_event(
                        event_type=EventType.ERROR_EVENT,
                        payload={
                            "team_id": current_team_id,
                            "error": f"Concentrator did not receive enough results: got {len(received_results)}, needed {required_count}",
                            "received_count": len(received_results),
                            "required_count": required_count
                        },
                        step_name=current_team_id,
                    )
                    team_result = {
                        "success": False,
                        "team_id": current_team_id,
                        "error": f"Concentrator did not receive enough results: got {len(received_results)}, needed {required_count}",
                        "type": "concentrator_error"
                    }
                    
                    # Evaluate routing with failure result
                    routing_result = evaluate_routing_task(
                        team_results=team_result,
                        current_context=current_context,
                        effective_config=effective_config,
                        team_id=current_team_id
                    )
                    
                    current_team_id = routing_result.get("next_team_id")
                
                # For other completion condition types, implement similar logic
                # (For now we only support 'count')
                
                continue  # Skip to next iteration to process the next team
            
            else:
                # Standard team execution (not concentrator)
                # Check if the next_team_id from previous routing is a list (fan-out)
                if isinstance(current_team_id, list):
                    run_logger.info(f"Executing fan-out to {len(current_team_id)} teams: {current_team_id}")
                    
                    # Log fan-out dispatch event
                    event_logger.log_event(
                        event_type=EventType.FAN_OUT_DISPATCH,
                        payload={
                            "target_teams": current_team_id,
                            "count": len(current_team_id)
                        },
                        step_name="fan_out_dispatch",
                        execution_path=execution_path
                    )
                    # Check if we have enough capacity
                    if len(current_team_id) > remaining_capacity:
                        run_logger.warning(f"Fan-out exceeds remaining capacity. Need {len(current_team_id)}, have {remaining_capacity}.")
                        current_team_id = current_team_id[:remaining_capacity]  # Truncate to remaining capacity
                        run_logger.warning(f"Truncated fan-out to: {current_team_id}")
                    
                    # Create contexts for each team to maintain immutability
                    team_contexts = {team_id: deepcopy(current_context) for team_id in current_team_id}
                    
                    # Add fan-out metadata to each context
                    for team_id, ctx in team_contexts.items():
                        if "fan_out" not in ctx:
                            ctx["fan_out"] = {}
                        ctx["fan_out"]["source"] = "fan_out_dispatch"
                        ctx["fan_out"]["target_teams"] = current_team_id
                    
                    try:
                        # Execute all teams in parallel using a thread pool
                        with concurrent.futures.ThreadPoolExecutor(max_workers=len(current_team_id)) as executor:
                            # Submit all team executions
                            future_to_team = {
                                executor.submit(execute_single_team, team_id, team_contexts[team_id]): team_id
                                for team_id in current_team_id
                            }
                            
                            # Process results as they complete
                            parallel_results = []
                            for future in concurrent.futures.as_completed(future_to_team):
                                team_id = future_to_team[future]
                                try:
                                    result = future.result()
                                    run_logger.info(f"Received result from fan-out team {team_id}")
                                    parallel_results.append({
                                        "team_id": team_id,
                                        "result": result
                                    })
                                    
                                    # Store in team results
                                    team_results[f"{team_id}_fanout"] = result
                                    
                                    # Track execution paths
                                    if "execution_path" in result:
                                        for path_item in result["execution_path"]:
                                            if path_item not in execution_path:
                                                execution_path.append(path_item)
                                    else:
                                        if team_id not in execution_path:
                                            execution_path.append(team_id)
                                    
                                except Exception as exc:
                                    run_logger.error(f"Fan-out team {team_id} generated an exception: {exc}")
                                    # Add failed result
                                    parallel_results.append({
                                        "team_id": team_id,
                                        "result": {
                                            "success": False,
                                            "error": f"Exception in team execution: {str(exc)}",
                                            "team_id": team_id
                                        }
                                    })
                        
                        # After all parallel executions complete:
                        # - For fan-out without explicit concentrator, we need to determine next step
                        # - The simplest approach is to check if any team specified a next_team explicitly,
                        #   otherwise end the workflow (return None)
                        
                        # Look for explicit next team in any result
                        next_teams = set()
                        for item in parallel_results:
                            result = item["result"]
                            # Check if the team explicitly specified next_team in its routing
                            if "routing_result" in result and "next_team_id" in result["routing_result"]:
                                next_team = result["routing_result"]["next_team_id"]
                                if next_team:  # Only add non-None values
                                    if isinstance(next_team, list):
                                        for t in next_team:
                                            next_teams.add(t)
                                    else:
                                        next_teams.add(next_team)
                        
                        if next_teams:
                            # Found explicit next teams, use them
                            next_teams_list = list(next_teams)
                            run_logger.info(f"Fan-out continuing to explicit next teams: {next_teams_list}")
                            # If just one team, set directly; if multiple, pass as list for another fan-out
                            current_team_id = next_teams_list[0] if len(next_teams_list) == 1 else next_teams_list
                        else:
                            # No explicit next teams, end workflow
                            run_logger.info("Fan-out complete, no explicit next teams specified; ending workflow")
                            current_team_id = None
                    
                    except Exception as e:
                        run_logger.error(f"Error during fan-out execution: {str(e)}", exc_info=True)
                        final_workflow_status = "error"
                        # Log error event for fan-out failure
                        event_logger.log_event(
                            event_type=EventType.ERROR_EVENT,
                            payload={
                                "error_type": "fan_out_execution_failed",
                                "error": str(e),
                                "target_teams": current_team_id
                            },
                            step_name="fan_out_execution",
                            execution_path=execution_path,
                        )
                        # Log workflow end with error
                        event_logger.log_event(
                            event_type=EventType.WORKFLOW_END,
                            payload={"status": "error", "error": f"Fan-out execution failed: {str(e)}"}
                        )
                        final_workflow_error = f"Fan-out execution failed: {str(e)}"
                        break
                
                else:
                    # Regular single team execution
                    run_logger.info(f"Executing single team: {current_team_id}")
                    
                    # Check team-specific recursion depth
                    current_team_count = team_execution_counts.get(current_team_id, 0)
                    if current_team_count >= team_max_depth:
                        run_logger.warning(f"Maximum recursion depth ({team_max_depth}) reached for team {current_team_id}")
                        final_workflow_status = "error"
                        # Log error event for max recursion depth
                        event_logger.log_event(
                            event_type=EventType.ERROR_EVENT,
                            payload={
                                "error_type": "max_recursion_depth",
                                "max_depth": team_max_depth,
                                "team_id": current_team_id
                            },
                            step_name=current_team_id,
                            execution_path=execution_path
                        )
                        final_workflow_error = f"Exceeded maximum recursion depth ({team_max_depth}) for team {current_team_id}"
                        break

                    # Update team execution counter
                    team_execution_counts[current_team_id] = current_team_count + 1
                    total_teams_executed += 1

                    # Add to execution path for tracking
                    execution_path.append(current_team_id)

                    run_logger.info(f"Executing team: {current_team_id} "
                                    f"(execution {total_teams_executed}, depth {current_team_count + 1})")

                    # Log step start event
                    event_logger.log_event(
                        event_type=EventType.STEP_START,
                        payload={"team_id": current_team_id, "execution_count": total_teams_executed, "depth": current_team_count + 1},
                        step_name=current_team_id,
                        execution_path=execution_path,
                        config_snapshot_path=str(config_info.snapshot_path),
                        config_hash=config_info.config_hash
                    )
                    # Execute the team as a subflow
                    # Ensure the effective_config dictionary is passed correctly
                    team_result = execute_team_subflow(
                        team_id=current_team_id,
                        effective_config=effective_config,
                        current_context=current_context
                    )

                    # Store the team result with a unique key per execution
                    team_results[f"{current_team_id}_{current_team_count}"] = team_result

                    # Log step end event with team result details
                    event_logger.log_event(
                        event_type=EventType.STEP_END,
                        payload={
                            "team_id": current_team_id,
                            "success": team_result.get("success", False),
                            "task_count": team_result.get("task_count", 0),
                            "results_summary": [
                                {
                                    "task_name": r.get("task_name", "unknown"),
                                    "success": r.get("success", False),
                                    "error": r.get("error")
                                } for r in team_result.get("results", [])
                            ],
                            "error": team_result.get("error")
                        },
                        step_name=current_team_id,
                        execution_path=execution_path,
                        config_snapshot_path=str(config_info.snapshot_path),
                        config_hash=config_info.config_hash
                    )
                    # Check for team execution failure
                    if not team_result.get("success", False):
                        error_detail = team_result.get('error', 'Unknown team error')
                        run_logger.error(f"Team {current_team_id} execution failed: {error_detail}")
                        # Check if workflow should stop on failure for this team
                        stop_on_fail = config_node.get_value(f"orchestration.teams.{current_team_id}.stop_on_failure", default=True)
                        # Log error event if team failed
                        event_logger.log_event(
                            event_type=EventType.ERROR_EVENT,
                            payload={
                                "team_id": current_team_id,
                                "error": error_detail
                            },
                            step_name=current_team_id)
                        if stop_on_fail:
                            final_workflow_status = "error"
                            final_workflow_error = f"Team {current_team_id} failed: {error_detail}"
                            break # Stop the workflow
                        else:
                            run_logger.warning(f"Team {current_team_id} failed but stop_on_failure is false, continuing workflow.")

                    # Evaluate routing to determine next team
                    routing_result = evaluate_routing_task(
                        team_results=team_result, # Pass the result of the *current* team execution
                        current_context=current_context,
                        effective_config=effective_config,
                        team_id=current_team_id
                    )
                    
                    # Log routing evaluation event
                    event_logger.log_event(
                        event_type=EventType.ROUTING_EVALUATION,
                        payload={
                            "team_id": current_team_id,
                            "next_team_id": routing_result.get("next_team_id"),
                            "matched_rule": routing_result.get("matched_rule"),
                            "recursion_strategy": routing_result.get("recursion_strategy"),
                            "has_context_updates": bool(routing_result.get("context_updates"))
                        },
                        step_name=current_team_id,
                        execution_path=execution_path
                    )
                    
                    # Store routing result in team result for potential use in fan-out
                    team_result["routing_result"] = routing_result

            # Update context with any context updates returned by routing
            if "context_updates" in routing_result and routing_result["context_updates"]:
                context_updates = routing_result["context_updates"]
                if isinstance(context_updates, dict): # Ensure it's a dict                         
                     # Check if config_fragments were supplied for a new snapshot
                     create_new_snapshot = False
                     new_config_fragments = None
                     
                     if "config_fragments" in context_updates:
                         # Extract config_fragments without modifying the original context_updates 
                         new_config_fragments = context_updates["config_fragments"]
                         # Create a new context_updates without config_fragments for regular merging
                         context_updates = {k: v for k, v in context_updates.items() if k != "config_fragments"}
                         run_logger.info(f"Extracted config_fragments for new snapshot, remaining context updates: {list(context_updates.keys())}")
                     else:
                         run_logger.info(f"Applying context updates from routing: {list(context_updates.keys())}")
                         
                         if isinstance(new_config_fragments, list) and len(new_config_fragments) > 0:
                             create_new_snapshot = True
                             run_logger.info(f"Found config fragments for new snapshot: {len(new_config_fragments)} fragments")
                             
                     # Check for special recursion flags
                     force_new_snapshot = context_updates.pop("force_new_snapshot", False)
                     if force_new_snapshot and not create_new_snapshot:
                         # Force a new snapshot with current context as a fragment
                         new_config_fragments = [effective_config, {"context_override": current_context}]
                         create_new_snapshot = True
                         run_logger.info("Forcing new snapshot creation based on force_new_snapshot flag")
                     
                     # Generate a new snapshot if requested
                     if create_new_snapshot:
                         run_logger.info("Creating new effective configuration snapshot for recursion")
                         
                         # Create a new snapshot
                         new_config_info = materialise_config(
                             context={"config_fragments": new_config_fragments},
                             system_config_path=system_config_path,
                             workspace_dir=None  # Use default based on run ID
                         )
                         
                         # Load the new effective configuration
                         with open(new_config_info.snapshot_path, 'r') as f:
                             new_effective_config = yaml.safe_load(f) or {}
                         
                         # Update our effective configuration
                         effective_config = new_effective_config
                         
                         # Create a new config node
                         config_node = create_config_node(effective_config)
                         
                         # Add snapshot metadata to context
                         context_updates["config_snapshot"] = {
                             "path": str(new_config_info.snapshot_path),
                             "hash": new_config_info.config_hash,
                             "timestamp": new_config_info.timestamp,
                             "schema_validated": new_config_info.schema_validated
                         }
                         
                         run_logger.info(f"Created new snapshot at: {new_config_info.snapshot_path}")
                     
                     # Create a new context dictionary by unpacking the current context and then
                     # the updates, ensuring immutability (AR4, REQ-DET-05)
                     
                     # IMPORTANT: For proper immutability, we need to handle nested dictionaries
                     # by creating deep copies rather than shallow references
                     from copy import deepcopy
                     
                     # First create a deep copy of the current context to avoid modifying shared references
                     next_context = deepcopy(current_context)
                     
                     # Then apply updates, ensuring each update is also a deep copy to prevent shared references
                     for key, value in context_updates.items():
                         next_context[key] = deepcopy(value)
                         
                     run_logger.debug("Created new deeply immutable context with updates applied")
                     current_context = next_context
                else:
                     run_logger.warning("Routing returned non-dict context_updates, ignoring.", updates=context_updates)

            # Get next team ID
            current_team_id = routing_result.get("next_team_id")
            
            # Check if there's a recursion strategy specified
            recursion_strategy = routing_result.get("recursion_strategy", "default")
            if recursion_strategy != "default" and current_team_id:
                run_logger.info(f"Using recursion strategy: {recursion_strategy}")
                
                # Handle different recursion strategies
                if recursion_strategy == "restart_with_context":
                    # The execution continues but with awareness that this is a recursive call
                    current_context["recursion"] = {                        "parent_team": current_team_id,
                        "depth": current_team_count + 1,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    run_logger.info(f"Recursion strategy 'restart_with_context' applied for team: {current_team_id}")
                
                elif recursion_strategy == "recursive_snapshot":
                    # Force a new snapshot for the next team execution
                    if "config_snapshot" not in current_context:
                        # Generate a new snapshot with current context
                        run_logger.info("Creating recursive snapshot with current context")
                        
                        # We'll create a new fragment from the current context
                        context_fragment = {"context_override": current_context}
                        
                        # Generate a new snapshot
                        new_config_info = materialise_config(
                            context={"config": context_fragment},
                            system_config_path=system_config_path,
                            workspace_dir=None
                        )
                        
                        # Load the new effective configuration
                        with open(new_config_info.snapshot_path, 'r') as f:
                            new_effective_config = yaml.safe_load(f) or {}
                        
                        # Update our effective configuration
                        effective_config = new_effective_config
                        
                        # Create a new config node
                        config_node = create_config_node(effective_config)
                        
                        # Add snapshot metadata to context
                        current_context["config_snapshot"] = {
                            "path": str(new_config_info.snapshot_path),
                            "hash": new_config_info.config_hash,
                            "timestamp": new_config_info.timestamp,
                            "schema_validated": new_config_info.schema_validated
                        }
                        
                        run_logger.info(f"Created recursive snapshot at: {new_config_info.snapshot_path}")
            
            # Log next team for visibility
            if current_team_id:
                if isinstance(current_team_id, list):
                    run_logger.info(f"Next step: fan-out to {len(current_team_id)} teams: {current_team_id}")
                else:
                    run_logger.info(f"Next team: {current_team_id}")
            else:
                run_logger.info("No next team, workflow will terminate")

            # Check if we've reached the end
            if not current_team_id:
                run_logger.info("Workflow completed - no next team specified by routing.")
                break

        # Final workflow metrics
        metrics = {
            "total_teams_executed": total_teams_executed,
            "unique_teams_executed": len(team_execution_counts),
            "max_team_recursion": max(team_execution_counts.values()) if team_execution_counts else 0
        }

        # Return comprehensive workflow results with enhanced config metadata
        final_result_data = {
            "status": final_workflow_status,
            "workflow_run_id": workflow_id,
            "execution_metrics": metrics,
            "execution_path": execution_path,
            "team_results": team_results,
            "config_metadata": {
                "snapshot_path": str(config_info.snapshot_path),
                "schema_validated": config_info.schema_validated,
                "config_hash": config_info.config_hash,
                "fragments_count": config_info.fragments_count,
                "timestamp": config_info.timestamp,
                "fragment_sources": [item.get("source") for item in config_info.fragment_metadata] if config_info.fragment_metadata else []
            }
        }
        if final_workflow_error:
             final_result_data["error"] = final_workflow_error

        # Log workflow end event
        event_logger.log_event(
            event_type=EventType.WORKFLOW_END,
            payload={
                "status": final_workflow_status,
                "error": final_workflow_error,
                "metrics": metrics,
                "execution_path": execution_path
            },
            config_snapshot_path=str(config_info.snapshot_path),
            config_hash=config_info.config_hash
        )

        run_logger.info("Declarative workflow finished.", status=final_workflow_status, teams_executed=total_teams_executed)
        return final_result_data

    except Exception as e:
        error_msg = str(e)
        run_logger.error(f"Declarative workflow execution failed unexpectedly: {error_msg}", exc_info=True) # Log traceback
        # Attempt to get run_id for error reporting
        run_id_for_error = "unknown"
        if config_info and hasattr(config_info, 'run_id'): run_id_for_error = config_info.run_id
        elif 'initial_context' in locals() and isinstance(initial_context, dict):
             run_id_for_error = initial_context.get('workflow_run_id', run_id_for_error)

        # Log error event for unexpected workflow failure
        try:
            if 'event_logger' in locals():
                event_logger.log_event(
                    event_type=EventType.ERROR_EVENT,
                    payload={
                        "error_type": "unexpected_workflow_failure",
                        "error": error_msg,
                        "workflow_run_id": run_id_for_error
                    },
                    step_name="workflow_execution"
                )
                
                # Log workflow end with error status
                event_logger.log_event(
                    event_type=EventType.WORKFLOW_END,
                    payload={
                        "status": "error",
                        "error": error_msg,
                        "workflow_run_id": run_id_for_error
                    }
                )
        except Exception as log_err:
            run_logger.error(f"Failed to log workflow error event: {str(log_err)}")

        # Create error result with as much config metadata as we have
        error_result = {
            "status": "error",
            "error": error_msg,
            "workflow_run_id": run_id_for_error,
            "config_metadata": {
                "snapshot_path": str(getattr(config_info, "snapshot_path", "unknown")) if config_info else "unknown",
                "schema_validated": getattr(config_info, "schema_validated", False) if config_info else False,
                "config_hash": getattr(config_info, "config_hash", None) if config_info else None,
                "fragments_count": getattr(config_info, "fragments_count", 0) if config_info else 0,
                "timestamp": getattr(config_info, "timestamp", None) if config_info else None
            }
        }
        
        return error_result

# --- execute_team_subflow function ---
@flow(name="execute_team")
def execute_team_subflow(
    team_id: str,
    effective_config: Dict[str, Any],
    current_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a team's execution plan as defined in the effective configuration.
    
    This function uses the ExecutionPlanExecutor to execute teams with execution plans.
    All teams must have an execution_plan in their configuration.
    
    Key features:
    - Context immutability: current_context is treated as read-only
    - Support for execution plans embedded in teams
    - Comprehensive logging and error handling
    
    Args:
        team_id: ID of the team to execute
        effective_config: The complete effective configuration snapshot
        current_context: Current execution context (treated as immutable read-only input)

    Returns:
        Dict with team execution results
    """
    run_logger = get_run_logger()
    execution_start_time = datetime.now(timezone.utc)

    try:
        # Create config node for path-based access
        config_node = create_config_node(effective_config)

        # Look up team configuration in the effective config
        team_config = config_node.get_value(f"orchestration.teams.{team_id}")
        if not team_config or not isinstance(team_config, dict):
            run_logger.error(f"Team configuration not found or invalid for team: {team_id}")
            return {
                "success": False,
                "error": f"Team configuration not found or invalid: {team_id}",
                "team_id": team_id,
                "results": []
            }

        # Add team_id and execution time to context for lineage tracking
        updated_context = deepcopy(current_context)
        if "execution_metadata" not in updated_context:
            updated_context["execution_metadata"] = {}
        updated_context["execution_metadata"]["team_id"] = team_id
        updated_context["execution_metadata"]["start_time"] = execution_start_time.isoformat()
            
        # Initialize event logger if configured
        lineage_config = config_node.get_value("llm_config.agents.lineage", {})
        event_logger = None
        if lineage_config.get("enabled", True):
            try:
                # Get workflow run ID from context
                workflow_run_id = updated_context.get("workflow_run_id")
                
                # Initialize event logger
                event_logger = EventLogger(
                    lineage_config, 
                    parent_id=workflow_run_id,
                    namespace=f"team_{team_id}"
                )
                run_logger.debug("Initialized event logger for team", team_id=team_id)
            except Exception as e:
                run_logger.error("Failed to initialize event logger", error=str(e))
                
        # Check if this team has an execution plan
        if "execution_plan" not in team_config:
            error_msg = f"Team '{team_id}' does not have an execution_plan in its configuration"
            run_logger.error(error_msg)
            return {
                "success": False,
                "team_id": team_id,
                "error": error_msg,
                "execution_type": "error"
            }
            
        run_logger.info(f"Executing team '{team_id}' with ExecutionPlanExecutor")
        try:
            # Import the ExecutionPlanExecutor
            from c4h_agents.execution.executor import ExecutionPlanExecutor, ExecutionResult
            
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
            
            # Get the execution plan from the team config
            execution_plan = team_config["execution_plan"]
            
            # Log execution plan details
            run_logger.info(f"Executing team's execution plan - Team: {team_id}, Steps: {len(execution_plan.get('steps', []))}, Executor: {executor.execution_id}")
                          
            # Execute the plan
            execution_result = executor.execute_plan(execution_plan, updated_context)
            
            # Calculate execution duration
            execution_end_time = datetime.now(timezone.utc)
            duration_seconds = (execution_end_time - execution_start_time).total_seconds()
            
            # Convert ExecutionResult to team result format
            team_result = {
                "success": execution_result.success,
                "team_id": team_id,
                "output": execution_result.output,
                "context": execution_result.context,
                "steps_executed": execution_result.steps_executed,
                "error": execution_result.error,
                "execution_type": "execution_plan",
                "duration_seconds": duration_seconds,
                "execution_id": executor.execution_id
            }
            
            run_logger.info(f"Team execution plan completed - Team: {team_id}, Success: {execution_result.success}, Steps executed: {execution_result.steps_executed}, Duration: {duration_seconds:.2f}s")
            
            return team_result
            
        except Exception as e:
            run_logger.error(f"Execution plan execution failed for team {team_id}: {str(e)}", exc_info=True)
            return {
                "success": False,
                "team_id": team_id,
                "error": f"Execution plan execution failed: {str(e)}",
                "execution_type": "execution_plan_error"
            }

    except Exception as e:
        run_logger.error(f"Team execution failed unexpectedly for team {team_id}: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Unexpected error in team {team_id}: {str(e)}",
            "team_id": team_id,
            "execution_type": "error"
        }

