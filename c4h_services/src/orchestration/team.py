"""
Team implementation for agent orchestration.
Path: c4h_services/src/orchestration/team.py

This module follows the immutable context pattern:
- The Team class treats its input context as read-only
- It does not modify the context dictionary that is passed to execute()
- Results and state transitions are returned in a result object
- The orchestrator is responsible for managing context transitions
"""

from typing import Dict, Any, List, Optional
from prefect import flow
from c4h_services.src.utils.logging import get_logger
from pathlib import Path
import json
from datetime import datetime, timezone

from c4h_services.src.intent.impl.prefect.tasks import run_agent_task
from c4h_services.src.intent.impl.prefect.models import AgentTaskConfig
from c4h_agents.lineage.event_logger import EventLogger, EventType
from c4h_agents.messages import Message, TeamHandoff, TeamResult

logger = get_logger()

class Team:
    """
    Represents a group of agents that execute in sequence.
    Acts as a Prefect flow with configurable routing.
    """
    def __init__(self, team_id: str, name: str, tasks: List[AgentTaskConfig], config: Dict[str, Any]):
        self.team_id = team_id
        self.name = name
        self.tasks = tasks
        self.config = config
        self.routing_rules = config.get("routing", {}).get("rules", [])
        self.default_next = config.get("routing", {}).get("default", None)
        
    @flow(name="team_flow")
    # Path: c4h_services/src/orchestration/team.py
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this team's agents in sequence.
        
        Args:
            context: Execution context including workflow data.
                Context follows these conventions:
                - data_context: The evolving payload/results data
                - execution_metadata: Information about workflow execution
                - config: Reference to effective configuration
            
            IMPORTANT: This method treats the context as immutable. 
            It does not modify the input context directly.
            context: Execution context including workflow data
            
        Returns:
            Dict with execution results and next team ID
        """
        logger.info("team.execution_starting", team_id=self.team_id, name=self.name)
        
        # Track results of each agent (using immutable patterns)
        # Create new result objects rather than modifying existing ones
        results = []
        team_result = {"success": True, "data": {}, "team_id": self.team_id}
        
        try:
            # Execute each agent task in sequence
            for i, task_config in enumerate(self.tasks):
                logger.info("team.task_executing", 
                        team_id=self.team_id, 
                        task_name=task_config.task_name, 
                        task_index=i)
                
                # Add team context to the task execution
                task_config_dict = {
                    "name": task_config.task_name,
                    "agent_type": task_config.agent_type,
                    "persona_key": task_config.persona_key,
                    "config": task_config.config
                    # Preserve immutability - create new dict rather than modifying
                }

                # Pass the context as-is (immutable/read-only input)
                result = run_agent_task(
                    task_config=task_config_dict, # newly created dict
                    context=context,
                    effective_config=context.get("config", {})
                )
                results.append(result)
                
                # Stop sequence on failure if configured
                if not result.get("success", False) and self.config.get("stop_on_failure", True):
                    logger.warning("team.task_failed_stopping_sequence", 
                                team_id=self.team_id,
                                task_name=task_config.task_name)
                    team_result["success"] = False
                    team_result["error"] = result.get("error")
                    break
            
            # Determine next team based on routing rules
            next_team = self._determine_next_team(results, context)
            
            # WO-BUGFIX-1: Collect only explicit output data from agents
            # After fixing ExecutionPlanExecutor and tasks.py, agents now return
            # their explicit output in the result_data field only
            team_data = {}
            for result in results:
                if result.get("success", False):
                    # Only collect from result_data which contains explicit output
                    if "result_data" in result and isinstance(result["result_data"], dict):
                        team_data.update(result["result_data"])
                    
                    # Note: We no longer look at the 'context' field since WO-BUGFIX-1
                    # ensures agents don't return accumulated context data anymore
            
            # Create final result
            team_result["data"] = team_data
            team_result["next_team"] = next_team
            
            # Prepare data handoff for next team
            if team_data and next_team:
                # Check if we should use MCP message format
                # Use MCP if: 1) messages already exist in context, or 2) MCP is explicitly enabled
                use_mcp_format = False
                existing_messages = []
                
                # Check for existing messages
                if "messages" in context:
                    existing_messages = context["messages"]
                    use_mcp_format = True
                elif "input_data" in context and isinstance(context.get("input_data"), dict):
                    if "messages" in context["input_data"]:
                        existing_messages = context["input_data"]["messages"]
                        use_mcp_format = True
                
                # Check if MCP is explicitly enabled in config
                if self.config.get("use_mcp_messages", False):
                    use_mcp_format = True
                
                if use_mcp_format:
                    # WO-BUGFIX-2: Use MCP models for standardized data handoff
                    message_content = self._format_team_output_as_message(team_data)
                    
                    team_message = Message(
                        role="assistant",
                        content=message_content,
                        metadata={
                            "source_team": self.team_id,
                            "target_team": next_team,
                            "team_data_keys": list(team_data.keys())
                        }
                    )
                    
                    # Create the handoff data with messages array
                    handoff_data = {
                        "messages": existing_messages + [team_message.model_dump()]
                    }
                    
                    # Set the input_data for next team
                    team_result["input_data"] = handoff_data
                    
                    logger.debug("team.output_data_structure.mcp",
                               team_id=self.team_id,
                               next_team=next_team,
                               data_keys=list(team_data.keys()) if team_data else [],
                               data_size=len(str(team_data)),
                               message_count=len(handoff_data["messages"]))
                else:
                    # Use legacy format for backward compatibility
                    team_result["input_data"] = team_data
                    
                    logger.debug("team.output_data_structure.legacy",
                               team_id=self.team_id,
                               next_team=next_team,
                               data_keys=list(team_data.keys()) if team_data else [],
                               data_size=len(str(team_data)))
                
                # Log transition event
                self._log_team_transition_event(team_result.get("input_data", {}), next_team, context)
            elif next_team:
                # No team data but there's a next team - pass through existing input_data
                # This handles initial teams that might not produce output
                if "input_data" in context:
                    team_result["input_data"] = context["input_data"]
                else:
                    team_result["input_data"] = {}
            
            logger.info("team.execution_completed", 
                    team_id=self.team_id, 
                    success=team_result["success"],
                    next_team=next_team)
                    
            return team_result
            
        except Exception as e:
            logger.error("team.execution_failed", team_id=self.team_id, error=str(e))
            # WO-BUGFIX-2: Return consistent structure even on error
            return {
                "success": False,
                "error": str(e),
                "team_id": self.team_id,
                "data": {},
                "next_team": None,
                "input_data": {}  # Empty input_data on failure
            }
        
    def _determine_next_team(self, results: List[Dict[str, Any]], context: Dict[str, Any]) -> Optional[str]:
        """
        Determine the next team to execute based on routing rules and results.
        
        Args:
            results: Results from agent executions
            context: Execution context
            
        Returns:
            ID of the next team or None if no next team
        """
        # First check explicit routing rules
        for rule in self.routing_rules:
            condition = rule.get("condition", "")
            if condition and self._evaluate_condition(condition, results, context):
                return rule.get("next_team")
        
        # If no rules match, use default
        return self.default_next
    
    def _evaluate_condition(self, condition: str, results: List[Dict[str, Any]], context: Dict[str, Any]) -> bool:
        """
        Evaluate a routing condition against results and context.
        
        Args:
            condition: Condition string to evaluate
            results: Results from agent executions
            context: Execution context
            
        Returns:
            True if condition evaluates to true, False otherwise
        """
        try:
            # Simple conditions based on success/failure
            if condition == "all_success":
                return all(r.get("success", False) for r in results)
            elif condition == "any_success":
                return any(r.get("success", False) for r in results)
            elif condition == "all_failure":
                return all(not r.get("success", False) for r in results)
            elif condition == "any_failure":
                return any(not r.get("success", False) for r in results)
            
            return False
        except Exception as e:
            logger.error("team.condition_evaluation_failed", 
                       team_id=self.team_id,
                       condition=condition,
                       error=str(e))
            return False
    
    def _format_team_output_as_message(self, team_data: Dict[str, Any]) -> str:
        """
        Format team output data as a message string.
        
        This method converts the team's output dictionary into a readable message
        format suitable for the Message model's content field.
        
        Args:
            team_data: The output data from the team
            
        Returns:
            Formatted message string
        """
        # Special handling for known output formats
        if "solution_design" in team_data:
            # Solution designer output is already a formatted string
            return team_data["solution_design"]
        elif "coder_result" in team_data:
            # Coder output might be structured
            coder_result = team_data["coder_result"]
            if isinstance(coder_result, str):
                return coder_result
            else:
                return json.dumps(coder_result, indent=2)
        elif "discovery_output" in team_data:
            # Discovery output might be structured
            discovery = team_data["discovery_output"]
            if isinstance(discovery, str):
                return discovery
            else:
                return json.dumps(discovery, indent=2)
        else:
            # Default: JSON serialize the entire team data
            return json.dumps(team_data, indent=2)
    
    def _log_team_transition_event(self, team_data: Dict[str, Any], next_team: str, context: Dict[str, Any]) -> None:
        """
        Log team output as a lineage event when passing data to the next team.
        This is a side effect of the data passing mechanism.
        
        Args:
            team_data: The team's output data
            next_team: The ID of the next team
            context: Execution context
        """
        try:
            # Get lineage configuration from context
            config = context.get("config", {})
            lineage_config = config.get("llm_config", {}).get("agents", {}).get("lineage", {})
            
            if not lineage_config.get("enabled", True):
                logger.debug("team.transition_event.skipped", 
                           reason="lineage_disabled",
                           team_id=self.team_id,
                           next_team=next_team)
                return
            
            # Get workflow run ID
            workflow_run_id = context.get("workflow_run_id", 
                                        context.get("system", {}).get("runid"))
            
            # Initialize event logger
            event_logger = EventLogger(
                config=lineage_config,
                run_id=workflow_run_id
            )
            
            # Create event type for team transition
            event_type = f"TEAM_TRANSITION_{self.team_id.upper()}_TO_{next_team.upper()}"
            
            # Calculate data size for tracking
            team_data_str = json.dumps(team_data) if team_data else "{}"
            data_size = len(team_data_str)
            
            # Prepare generic payload for any team transition
            payload = {
                "source_team": self.team_id,
                "target_team": next_team,
                "data_keys": list(team_data.keys()) if team_data else [],
                "data_size": data_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "workflow_run_id": workflow_run_id,
                "project_path": context.get("project", {}).get("root_path"),
                "intent": context.get("intent", {})
            }
            
            # Add summary of data being passed (first 1000 chars of each key)
            data_summary = {}
            for key, value in team_data.items():
                if isinstance(value, str):
                    data_summary[key] = value[:1000] if len(value) > 1000 else value
                elif isinstance(value, dict):
                    # For dict values, include keys
                    data_summary[key] = {"keys": list(value.keys()), "type": "dict"}
                elif isinstance(value, list):
                    data_summary[key] = {"length": len(value), "type": "list"}
                else:
                    data_summary[key] = {"type": type(value).__name__, "value": str(value)[:100]}
            
            payload["data_summary"] = data_summary
            
            # Generate event ID first so we can use it in artifact filename
            import uuid
            event_id = str(uuid.uuid4())
            
            # Get artifact threshold from config, default to 1KB
            artifact_threshold = lineage_config.get("artifact_threshold", 1000)
            
            # If data is large, save it to a separate artifact file
            if data_size > artifact_threshold:
                # Create a file path for the full data
                lineage_dir = Path(lineage_config.get("path", "workspaces/lineage"))
                if not lineage_dir.is_absolute():
                    lineage_dir = Path.cwd() / lineage_dir
                
                date_str = datetime.now().strftime('%Y%m%d')
                workflow_dir = lineage_dir / date_str / workflow_run_id
                artifacts_dir = workflow_dir / "artifacts"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                
                # Save team data to file with event ID in filename
                data_file = artifacts_dir / f"team_output_{self.team_id}_to_{next_team}_{event_id}.json"
                data_file.write_text(team_data_str)
                
                payload["data_file"] = str(data_file)
                payload["data_full_size"] = data_size
                payload["artifact_event_id"] = event_id  # Include event ID in payload
                logger.info("team.transition_data_saved_to_file",
                         team_id=self.team_id,
                         next_team=next_team,
                         file_path=str(data_file),
                         size=data_size,
                         threshold=artifact_threshold,
                         event_id=event_id)
            
            # Log the event with pre-generated ID
            actual_event_id = event_logger.log_event(
                event_type=event_type,
                payload=payload,
                step_name=f"{self.team_id}_to_{next_team}_transition",
                parent_id=context.get("parent_id"),
                execution_path=context.get("execution_path", []),
                event_id=event_id  # Pass pre-generated event ID
            )
            
            logger.info("team.transition_event_logged",
                     source_team=self.team_id,
                     target_team=next_team,
                     event_id=actual_event_id,
                     data_size=data_size,
                     data_keys=list(team_data.keys()) if team_data else [])
                     
        except Exception as e:
            # Log error but don't fail the workflow
            logger.error("team.transition_event_failed",
                       team_id=self.team_id,
                       next_team=next_team,
                       error=str(e),
                       exc_info=True)