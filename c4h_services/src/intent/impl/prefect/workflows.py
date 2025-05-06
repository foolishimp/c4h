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
                final_workflow_status = "error"
                final_workflow_error = f"Exceeded maximum total team executions ({final_max_total_teams})"
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
                    
                    except Exception as e:
                        run_logger.error(f"Error during parallel execution of source teams: {str(e)}", exc_info=True)
                        # If we failed to execute parallel teams, treat as concentrator failure
                        team_result = {
                            "success": False,
                            "team_id": current_team_id,
                            "error": f"Failed to execute source teams in parallel: {str(e)}",
                            "type": "concentrator_error"
                        }
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
                        final_workflow_error = f"Exceeded maximum recursion depth ({team_max_depth}) for team {current_team_id}"
                        break

                    # Update team execution counter
                    team_execution_counts[current_team_id] = current_team_count + 1
                    total_teams_executed += 1

                    # Add to execution path for tracking
                    execution_path.append(current_team_id)

                    run_logger.info(f"Executing team: {current_team_id} "
                                    f"(execution {total_teams_executed}, depth {current_team_count + 1})")

                    # Execute the team as a subflow
                    # Ensure the effective_config dictionary is passed correctly
                    team_result = execute_team_subflow(
                        team_id=current_team_id,
                        effective_config=effective_config,
                        current_context=current_context
                    )

                    # Store the team result with a unique key per execution
                    team_results[f"{current_team_id}_{current_team_count}"] = team_result

                    # Check for team execution failure
                    if not team_result.get("success", False):
                        error_detail = team_result.get('error', 'Unknown team error')
                        run_logger.error(f"Team {current_team_id} execution failed: {error_detail}")
                        # Check if workflow should stop on failure for this team
                        stop_on_fail = config_node.get_value(f"orchestration.teams.{current_team_id}.stop_on_failure", default=True)
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
                    current_context["recursion"] = {
                        "parent_team": current_team_id,
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
    Execute a team's tasks as defined in the effective configuration.
    Supports regular team execution and loop-based iteration.
    
    IMPORTANT: This function treats current_context as immutable/read-only.
    It does not modify the input context directly. Any required context
    updates are returned in the result object for the orchestrator to handle
    according to REQ-DET-05.

    Args:
        team_id: ID of the team to execute
        effective_config: The complete effective configuration snapshot
        current_context: Current execution context (treated as immutable read-only input)

    Returns:
        Dict with team execution results
    """
    run_logger = get_run_logger()
    task_results = [] # Initialize outside try block

    try:
        # Create config node for path-based access
        config_node = create_config_node(effective_config)

        # Look up team configuration in the effective config
        team_config = config_node.get_value(f"orchestration.teams.{team_id}")
        if not team_config or not isinstance(team_config, dict): # Check if dict
            run_logger.error(f"Team configuration not found or invalid for team: {team_id}")
            return {
                "success": False,
                "error": f"Team configuration not found or invalid: {team_id}",
                "team_id": team_id,
                "results": []
            }

        # Check if this is a loop team
        team_type = team_config.get("type", "standard")
        if team_type == "loop":
            # Execute as a loop team
            return execute_loop_team(
                team_id=team_id,
                team_config=team_config,
                effective_config=effective_config,
                current_context=current_context
            )

        # Standard team execution flow - get the team's tasks list
        tasks = team_config.get("tasks", [])
        if not tasks:
            run_logger.warning(f"No tasks defined for team: {team_id}")
            # Return success if no tasks, as the team technically completed
            return { "success": True, "team_id": team_id, "results": [], "task_count": 0 }

        # Execute each task in sequence
        for i, task_config_def in enumerate(tasks):
            # Ensure task_config_def is a dictionary
            if not isinstance(task_config_def, dict):
                 run_logger.error(f"Invalid task definition format in team {team_id}, task index {i}", task_def=task_config_def)
                 # Decide how to handle: skip task or fail team? Failing team seems safer.
                 return { "success": False, "error": f"Invalid task definition at index {i} for team {team_id}", "team_id": team_id, "results": task_results }

            task_name = task_config_def.get('name', f'unnamed_task_{i}')
            run_logger.info(f"Executing task {i+1}/{len(tasks)}: {task_name}")

            # Add task_config to context for factory creation
            # Use deepcopy to ensure true immutability by creating new copies of nested objects
            from copy import deepcopy
            task_context = deepcopy(current_context)
            task_context["task_config"] = deepcopy(task_config_def)

            # Execute the task using run_agent_task
            # Pass the full effective_config snapshot
            result = run_agent_task(
                task_config=task_config_def, # Pass the specific task definition
                context=task_context,
                effective_config=effective_config
            )

            task_results.append(result)

            # Stop on failure if configured to do so
            stop_on_fail = team_config.get("stop_on_failure", True)
            if not result.get("success", False) and stop_on_fail:
                run_logger.warning(f"Task {task_name} failed, stopping team sequence based on stop_on_failure=True.")
                break # Stop processing further tasks in this team

        # Determine overall team success (all tasks succeeded OR failures were allowed)
        all_succeeded = all(r.get("success", False) for r in task_results)
        team_success = all_succeeded or not team_config.get("stop_on_failure", True)

        # Return comprehensive team results
        return {
            "success": team_success,
            "team_id": team_id,
            "results": task_results, # Contains results for each task executed
            "task_count": len(tasks)
        }

    except Exception as e:
        run_logger.error(f"Team execution failed unexpectedly for team {team_id}", error=str(e), exc_info=True) # Log traceback
        return {
            "success": False,
            "error": f"Unexpected error in team {team_id}: {str(e)}",
            "team_id": team_id,
            "results": task_results # Include results up to the point of failure
        }

@flow(name="execute_loop_team")
def execute_loop_team(
    team_id: str,
    team_config: Dict[str, Any],
    effective_config: Dict[str, Any],
    current_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a team's tasks in a loop over items in a collection.
    
    IMPORTANT: This function treats current_context as immutable/read-only.
    It does not modify the input context directly. For each loop iteration,
    a new iteration context is created by copying the current context and adding
    iteration-specific variables. This follows the immutability principle required
    by REQ-DET-05.

    Args:
        team_id: ID of the loop team
        team_config: Configuration for the specific team
        effective_config: The complete effective configuration
        current_context: Current workflow context (treated as immutable read-only input)

    Returns:
        Dict with aggregated loop execution results
    """
    run_logger = get_run_logger()
    iteration_results = []  # Track results for each iteration
    
    try:
        # Validate and extract required loop configuration
        iterate_on = team_config.get("iterate_on")
        if not iterate_on:
            error_msg = f"Missing required 'iterate_on' parameter for loop team: {team_id}"
            run_logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "team_id": team_id,
                "results": []
            }
        
        # Get iteration collection using path notation
        context_node = create_config_node(current_context)
        collection = context_node.get_value(iterate_on)
        
        # Validate collection
        if not collection or not isinstance(collection, (list, dict)):
            error_msg = f"Invalid or empty collection at path '{iterate_on}' for loop team: {team_id}"
            run_logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "team_id": team_id,
                "results": []
            }
        
        # Get loop variable name (default to "item")
        loop_variable = team_config.get("loop_variable", "item")
        
        # Get loop body teams or tasks (data_context convention)
        # Note: This section operates on the provided context as read-only input
        # Any modifications are returned in the result, not made to the input context
        body = team_config.get("body", []) 
        if not body:
            error_msg = f"Empty loop body for loop team: {team_id}"
            run_logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "team_id": team_id,
                "results": []
            }
        
        # Get stop_on_failure config (default to True)
        stop_on_fail = team_config.get("stop_on_failure", True)
        
        # Convert body to list if it's not
        if not isinstance(body, list):
            body = [body]
        
        # Log loop execution start
        run_logger.info(f"Executing loop team: {team_id} over {len(collection)} items")
        
        # Track index for debugging
        for idx, item in enumerate(collection):
            # Create iteration context with loop variable
            # Use deepcopy to ensure true immutability by creating new copies of nested objects
            from copy import deepcopy
            iter_context = deepcopy(current_context)
            
            # Add iteration-specific variables
            iter_context[loop_variable] = item
            iter_context["loop_index"] = idx
            iter_context["loop_count"] = len(collection)
            
            run_logger.info(f"Loop {team_id}: iteration {idx+1}/{len(collection)}")
            
            # Initialize iteration result
            iter_result = {
                "index": idx,
                "item_key": loop_variable,
                "success": True,
                "results": []
            }
            
            # Execute each body item (task or embedded team) with a new iteration context
            # This follows the data_context convention by preserving the original context
            # and creating new contexts for each iteration
            for body_item in body:
                # Check if this is a task or a team reference 
                # (using read-only context pattern)
                if isinstance(body_item, dict) and "agent_type" in body_item:
                    # This is a task configuration
                    task_name = body_item.get("name", f"loop_task_{idx}")
                    run_logger.info(f"Executing loop task: {task_name} (iteration {idx+1})")
                    
                    # Create task context
                    task_context = {**iter_context, "task_config": body_item}
                    
                    # Execute task
                    result = run_agent_task(
                        task_config=body_item,
                        context=task_context,
                        effective_config=effective_config
                    )
                    
                    # Add to iteration results
                    iter_result["results"].append(result)
                    
                    # Check for failure
                    if not result.get("success", False) and stop_on_fail:
                        iter_result["success"] = False
                        iter_result["error"] = f"Task {task_name} failed in iteration {idx}"
                        run_logger.warning(f"Loop {team_id}: task {task_name} failed in iteration {idx+1}, stopping iteration.")
                        break
                    
                elif isinstance(body_item, str):
                    # This is a reference to another team
                    embedded_team_id = body_item
                    run_logger.info(f"Executing embedded team: {embedded_team_id} (iteration {idx+1})")
                    
                    # Execute the referenced team (recursive)
                    team_result = execute_team_subflow(
                        team_id=embedded_team_id,
                        effective_config=effective_config,
                        current_context=iter_context
                    )
                    
                    # Add to iteration results
                    iter_result["results"].append(team_result)
                    
                    # Check for failure
                    if not team_result.get("success", False) and stop_on_fail:
                        iter_result["success"] = False
                        iter_result["error"] = f"Team {embedded_team_id} failed in iteration {idx}"
                        run_logger.warning(f"Loop {team_id}: embedded team {embedded_team_id} failed in iteration {idx+1}, stopping iteration.")
                        break
                    
                else:
                    # Invalid body item
                    error_msg = f"Invalid body item in loop team {team_id}: {body_item}"
                    run_logger.error(error_msg)
                    iter_result["success"] = False
                    iter_result["error"] = error_msg
                    if stop_on_fail:
                        break
            
            # Add iteration result to overall results
            iteration_results.append(iter_result)
            
            # Check if we should stop the entire loop
            if not iter_result.get("success", False) and team_config.get("break_on_failure", False):
                run_logger.warning(f"Loop {team_id}: iteration {idx+1} failed, breaking loop due to break_on_failure=True.")
                break
        
        # Determine overall success
        all_succeeded = all(r.get("success", True) for r in iteration_results)
        loop_success = all_succeeded or not team_config.get("stop_on_failure", True)
        
        # Return aggregated results
        return {
            "success": loop_success,
            "team_id": team_id,
            "type": "loop",
            "iterations_count": len(iteration_results),
            "iterations": iteration_results,
            "collection_path": iterate_on,
            "collection_size": len(collection)
        }
        
    except Exception as e:
        error_msg = str(e)
        run_logger.error(f"Loop team execution failed unexpectedly: {error_msg}", exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "team_id": team_id,
            "type": "loop",
            "iterations": iteration_results
        }