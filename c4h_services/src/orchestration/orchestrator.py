"""
Orchestrator for managing team-based workflow execution.
Path: c4h_services/src/orchestration/orchestrator.py
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from c4h_services.src.utils.logging import get_logger
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import uuid
import yaml

from c4h_agents.config import create_config_node, deep_merge
from c4h_services.src.intent.impl.prefect.models import AgentTaskConfig
from c4h_services.src.orchestration.team import Team
from c4h_services.src.intent.impl.prefect.factories import (
    create_discovery_task,
    create_solution_task,
    create_coder_task
)

logger = get_logger()

class Orchestrator:
    """
    Manages execution of team-based workflows using Prefect.
    Handles team loading, chaining, and execution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator with configuration.
        Args:
            config: Complete configuration dictionary
        """
        # Use a unique ID for this instance for clearer logging
        self.instance_id = str(uuid.uuid4())[:4]
        # Get logger early
        global_logger = get_logger(config) # Use config for potential log settings
        self.logger = global_logger.bind(orchestrator_instance=self.instance_id)

        self.logger.info("orchestrator.__init__.start", config_keys=list(config.keys()) if config else "None")

        self.config = config if config else {} # Ensure self.config is always a dict
        self.config_node = create_config_node(self.config)
        self.teams = {}
        self.loaded_teams = set()

        # Load team configurations
        try:
             self._load_teams()
             self.logger.info("orchestrator.__init__.complete",
                    teams_loaded=len(self.teams),
                    team_ids=list(self.teams.keys()))
        except Exception as e:
             self.logger.error("orchestrator.__init__.load_teams_failed", error=str(e))
             # Attempt to load default teams as a fallback during init failure
             self.logger.warning("orchestrator.__init__.attempting_default_load_on_error")
             try:
                 self._load_default_teams()
                 self.logger.info("orchestrator.__init__.default_teams_loaded_after_error", teams_loaded=len(self.teams))
             except Exception as default_e:
                  self.logger.error("orchestrator.__init__.default_team_load_failed", error=str(default_e))

    def _load_teams(self) -> None:
        """
        Load team configurations from config.
        Creates Team instances for each team configuration.
        """
        # Access config via the instance attribute self.config_node
        teams_config = self.config_node.get_value("orchestration.teams") or {}

        # Add detailed logging about what config _load_teams is seeing
        self.logger.debug("_load_teams.config_check",
                          has_orchestration_key="orchestration" in self.config,
                          has_teams_key="teams" in self.config.get("orchestration", {}),
                          teams_config_retrieved=bool(teams_config),
                          teams_config_keys=list(teams_config.keys()) if teams_config else [])

        if not teams_config:
            self.logger.warning("orchestrator.no_teams_found") # KEEP this warning
            # Load default teams for backward compatibility
            self._load_default_teams()
            return

        # Clear existing teams before loading
        self.teams = {}
        loaded_count = 0
        for team_id, team_config in teams_config.items():
            try:
                # Get basic team info
                name = team_config.get("name", team_id)
                self.logger.debug("_load_teams.processing_team", team_id=team_id, name=name)

                # Get task configurations
                tasks = []
                for task_config in team_config.get("tasks", []):
                    agent_class = task_config.get("agent_class")
                    if not agent_class:
                        self.logger.error("orchestrator.missing_agent_class", team_id=team_id, task=task_config)
                        continue

                    # Create task config
                    # Ensure the *current* orchestrator config is used for merging task overrides
                    task_specific_override = task_config.get("config", {}) # Get overrides for this specific task
                    # --- MODIFICATION START ---
                    # Merge overrides onto a COPY of the main config, not self.config directly
                    task_base_config = self.config.copy() # Start with the current main config
                    final_task_config_for_agent = deep_merge(task_base_config, task_specific_override)
                    # --- MODIFICATION END ---

                    agent_config = AgentTaskConfig(
                        agent_class=agent_class,
                        config=final_task_config_for_agent, # Pass the isolated config for this task
                        task_name=task_config.get("name"),
                        requires_approval=task_config.get("requires_approval", False),
                        max_retries=task_config.get("max_retries", 3),
                        retry_delay_seconds=task_config.get("retry_delay_seconds", 30)
                    )
                    tasks.append(agent_config)

                # Create team
                self.teams[team_id] = Team(
                    team_id=team_id,
                    name=name,
                    tasks=tasks,
                    config=team_config # Store the team-specific config part
                )
                loaded_count += 1
                self.logger.info("orchestrator.team_loaded", team_id=team_id, name=name, tasks=len(tasks))

            except Exception as e:
                self.logger.error("orchestrator.team_load_failed", team_id=team_id, error=str(e))
        self.logger.debug("_load_teams.finished", loaded_count=loaded_count, final_team_keys=list(self.teams.keys()))

    def _load_default_teams(self) -> None:
        """
        Load default teams for backward compatibility.
        Creates default discovery, solution, and coder teams.
        """
        # Create discovery team
        discovery_task = create_discovery_task(self.config)
        self.teams["discovery"] = Team(
            team_id="discovery",
            name="Discovery Team",
            tasks=[discovery_task],
            config={
                "routing": {
                    "default": "solution"
                }
            }
        )
        
        # Create solution team
        solution_task = create_solution_task(self.config)
        self.teams["solution"] = Team(
            team_id="solution",
            name="Solution Design Team",
            tasks=[solution_task],
            config={
                "routing": {
                    "default": "coder"
                }
            }
        )
        
        # Create coder team
        coder_task = create_coder_task(self.config)
        self.teams["coder"] = Team(
            team_id="coder",
            name="Coder Team",
            tasks=[coder_task],
            config={
                "routing": {
                    "default": None  # End of flow
                }
            }
        )
        
        logger.info("orchestrator.default_teams_loaded", 
                  teams=["discovery", "solution", "coder"])
    
    def execute_workflow(
        self, 
        entry_team: str = "discovery",
        context: Dict[str, Any] = None,
        max_teams: int = 10
    ) -> Dict[str, Any]:
        """
        Execute a workflow starting from the specified team.
        
        Args:
            entry_team: ID of the team to start execution with
            context: Initial context for execution
            max_teams: Maximum number of teams to execute (prevent infinite loops)
            
        Returns:
            Final workflow result
        """
        # Use the configuration from the context if provided
        if context and "config" in context:
            # Update the orchestrator's config with the context's config
            updated_config = context["config"]
            if updated_config != self.config:
                # Config has changed, reload teams with the new config
                self.config = updated_config
                self.config_node = create_config_node(updated_config)
                # Reload teams with the new configuration
                self.teams = {}
                self._load_teams()
                logger.info("orchestrator.teams_reloaded_with_updated_config", 
                        teams_count=len(self.teams),
                        teams=list(self.teams.keys()))

        if entry_team not in self.teams:
            raise ValueError(f"Entry team {entry_team} not found")
            
        # Initialize context if needed
        if context is None:
            context = {}
            
        # Generate workflow ID if not provided
        workflow_run_id = context.get("workflow_run_id") or str(uuid.uuid4())
        if "system" not in context:
            context["system"] = {}
        context["system"]["runid"] = workflow_run_id
        context["workflow_run_id"] = workflow_run_id
        
        logger.info("orchestrator.workflow_starting", 
                entry_team=entry_team,
                workflow_run_id=workflow_run_id)
        
        # Track execution path
        execution_path = []
        team_results = {}
        
        # Initial setup
        current_team_id = entry_team
        team_count = 0
        final_result = {
            "status": "success",
            "workflow_run_id": workflow_run_id,
            "execution_id": workflow_run_id,
            "execution_path": [],
            "team_results": {},
            "data": {}
        }
        
        # Execute teams in sequence
        while current_team_id and team_count < max_teams:
            if current_team_id not in self.teams:
                logger.error("orchestrator.team_not_found", team_id=current_team_id)
                final_result["status"] = "error"
                final_result["error"] = f"Team {current_team_id} not found"
                break
                
            team = self.teams[current_team_id]
            logger.info("orchestrator.executing_team", 
                    team_id=current_team_id,
                    step=team_count + 1)
                    
            # Track this team in the execution path
            execution_path.append(current_team_id)
            
            # Execute the team
            team_result = team.execute(context)
            
            # Store team result
            team_results[current_team_id] = team_result
            
            # Update context with team result data
            if team_result.get("success", False):
                # Handle standard data
                if "data" in team_result:
                    context.update(team_result["data"])
                    final_result["data"].update(team_result["data"])
                
                # Handle special structured input_data for team-to-team communication
                if "input_data" in team_result:
                    if "input_data" not in context:
                        context["input_data"] = {}
                    context["input_data"] = team_result["input_data"]
            
            # Check for failure
            if not team_result.get("success", False):
                logger.warning("orchestrator.team_execution_failed",
                            team_id=current_team_id,
                            error=team_result.get("error"))
                final_result["status"] = "error"
                final_result["error"] = team_result.get("error")
                break
                
            # Get next team
            current_team_id = team_result.get("next_team")
            team_count += 1
            
            # Check if we've reached the end
            if not current_team_id:
                logger.info("orchestrator.workflow_completed",
                        teams_executed=team_count,
                        final_team=team.team_id)
                break
        
        # Check if we hit the team limit
        if team_count >= max_teams:
            logger.warning("orchestrator.max_teams_reached", max_teams=max_teams)
            final_result["status"] = "error"
            final_result["error"] = f"Exceeded maximum team limit of {max_teams}"
        
        # Build final result
        final_result["execution_path"] = execution_path
        final_result["team_results"] = team_results
        final_result["teams_executed"] = team_count
        final_result["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        logger.info("orchestrator.workflow_result", 
                status=final_result["status"],
                teams_executed=team_count,
                execution_path=execution_path)
                
        return final_result

    def initialize_workflow(
        self,
        project_path: Union[str, Path],
        intent_desc: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Initialize workflow configuration with consistent defaults and parameter handling.
        Args:
            project_path: Path to the project
            intent_desc: Description of the intent
            config: Base configuration (This should be the merged config from the request)

        Returns:
            Tuple of (prepared_config, context_dict)
        """
        # Use a logger specific to this instance if available, else use global
        instance_logger = getattr(self, 'logger', logger)
        instance_logger.debug("initialize_workflow.entry", config_keys=list(config.keys()))

        try:
            # Ensure config is a dictionary
            prepared_config = config.copy() if config else {}

            # Normalize project path
            if not project_path:
                project_path = prepared_config.get('project', {}).get('path')
                if not project_path:
                    instance_logger.error("initialize_workflow.missing_project_path") # Added logging
                    raise ValueError("No project path specified in arguments or config")

            # Convert Path objects to string for consistency
            if isinstance(project_path, Path):
                project_path = str(project_path)
            instance_logger.debug("initialize_workflow.project_path_normalized", path=project_path) # Added logging

            # Ensure project config exists
            if 'project' not in prepared_config:
                prepared_config['project'] = {}
            prepared_config['project']['path'] = project_path

            # Generate workflow ID with embedded timestamp if not already present
            # Use config_node for safer access
            config_node = create_config_node(prepared_config)
            workflow_id = config_node.get_value("workflow_run_id") or \
                          config_node.get_value("system.runid")
            if not workflow_id:
                time_str = datetime.now().strftime('%H%M')
                workflow_id = f"wf_{time_str}_{uuid.uuid4()}"
                instance_logger.warning("initialize_workflow.generating_new_workflow_id", new_id=workflow_id) # Added logging
            else:
                 instance_logger.debug("initialize_workflow.using_existing_workflow_id", id=workflow_id) # Added logging


            # Configure system namespace
            if 'system' not in prepared_config:
                prepared_config['system'] = {}
            prepared_config['system']['runid'] = workflow_id

            # Add workflow ID at top level for convenience
            prepared_config['workflow_run_id'] = workflow_id

            # Add timestamp information
            timestamp = datetime.now(timezone.utc).isoformat()
            if 'runtime' not in prepared_config:
                prepared_config['runtime'] = {}
            if 'workflow' not in prepared_config['runtime']:
                prepared_config['runtime']['workflow'] = {}
            prepared_config['runtime']['workflow']['start_time'] = timestamp

            # Ensure orchestration is enabled
            if 'orchestration' not in prepared_config:
                prepared_config['orchestration'] = {'enabled': True}
            else:
                prepared_config['orchestration']['enabled'] = True

            # --- DETAILED LOGGING FOR TARTXT CONFIG ---
            instance_logger.debug("initialize_workflow.checking_tartxt_config")
            # Add default configs for crucial components
            if 'llm_config' not in prepared_config: prepared_config['llm_config'] = {}
            if 'agents' not in prepared_config['llm_config']: prepared_config['llm_config']['agents'] = {}
            if 'discovery' not in prepared_config['llm_config']['agents']: prepared_config['llm_config']['agents']['discovery'] = {}

            discovery_config = prepared_config['llm_config']['agents']['discovery']
            instance_logger.debug("initialize_workflow.discovery_config_retrieved", config_keys=list(discovery_config.keys()))

            if 'tartxt_config' not in discovery_config:
                discovery_config['tartxt_config'] = {}
                instance_logger.warning("initialize_workflow.created_empty_tartxt_config") # Added logging

            tartxt_config = discovery_config['tartxt_config']
            instance_logger.debug("initialize_workflow.tartxt_config_retrieved", config_keys=list(tartxt_config.keys()), current_script_path=tartxt_config.get('script_path')) # Added logging

            # Ensure script_path is set (handle both possible key names)
            # Check if script_path is already correctly set and is a non-empty string
            current_script_path = tartxt_config.get('script_path')
            script_path_valid = isinstance(current_script_path, str) and current_script_path.strip()

            if not script_path_valid:
                instance_logger.warning("initialize_workflow.script_path_missing_or_invalid", current_value=current_script_path) # Added logging
                # Fallback 1: Check script_base_path
                script_base = tartxt_config.get('script_base_path')
                if isinstance(script_base, str) and script_base.strip():
                     tartxt_config['script_path'] = f"{script_base}/tartxt.py"
                     instance_logger.info("initialize_workflow.script_path_set_from_base", new_path=tartxt_config['script_path']) # Added logging
                else:
                    # Fallback 2: Try to locate the script in the package
                    instance_logger.info("initialize_workflow.attempting_package_lookup") # Added logging
                    script_path_found_in_package = None
                    try:
                        import c4h_agents
                        import importlib.resources as pkg_resources
                        # Use importlib.resources for better package data access
                        with pkg_resources.path(c4h_agents.skills, 'tartxt.py') as script_path_obj:
                            if script_path_obj.exists():
                                script_path_found_in_package = str(script_path_obj.resolve())
                                instance_logger.info("initialize_workflow.script_path_found_in_package", path=script_path_found_in_package) # Added logging
                    except Exception as pkg_err:
                         instance_logger.warning("initialize_workflow.package_lookup_failed", error=str(pkg_err)) # Added logging

                    if script_path_found_in_package:
                         tartxt_config['script_path'] = script_path_found_in_package
                    else:
                        # Fallback 3: Use a default relative path if nothing else worked
                        default_relative = "c4h_agents/skills/tartxt.py"
                        tartxt_config['script_path'] = default_relative
                        instance_logger.warning("initialize_workflow.using_default_relative_path", path=default_relative) # Added logging

            # Final check of the script path before exiting
            final_script_path = tartxt_config.get('script_path')
            instance_logger.debug("initialize_workflow.final_script_path_check", path=final_script_path, type=type(final_script_path).__name__) # Added logging

            # Check if the final path is None or empty string AFTER all attempts
            if not isinstance(final_script_path, str) or not final_script_path.strip():
                 instance_logger.error("initialize_workflow.final_script_path_is_invalid", path_value=final_script_path) # Added logging
                 # Raise error here to prevent the TypeError later
                 raise ValueError(f"Could not determine a valid script_path for tartxt. Final value was: {final_script_path}")

            # Ensure input_paths is set
            if 'input_paths' not in tartxt_config:
                tartxt_config['input_paths'] = ["./"]
                instance_logger.debug("initialize_workflow.set_default_input_paths") # Added logging
            # --- END OF TARTXT CONFIG LOGGING ---


            # Prepare consistent context dictionary
            context = {
                "project_path": project_path,
                "intent": intent_desc,
                "workflow_run_id": workflow_id,
                "system": {"runid": workflow_id},
                "timestamp": timestamp,
                "config": prepared_config # Pass the fully prepared config
            }

            instance_logger.info("workflow.initialized", # cite: 1460
                        workflow_id=workflow_id,
                        project_path=project_path,
                        tartxt_script_path=tartxt_config.get('script_path'), # Log final path used
                        tartxt_config_keys=list(tartxt_config.keys()))

            return prepared_config, context

        except Exception as e:
            instance_logger.error("workflow.initialization_failed", error=str(e)) # cite: 1461
            # Re-raise the exception to ensure the calling function knows about the failure
            raise