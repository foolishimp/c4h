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
        self.config = config
        self.config_node = create_config_node(config)
        # Update logger with config
        self.teams = {}
        logger = get_logger(config)
        self.loaded_teams = set()
        
        # Load team configurations
        self._load_teams()
        
        logger.info("orchestrator.initialized", 
                  teams_loaded=len(self.teams),
                  team_ids=list(self.teams.keys()))
    
    def _load_teams(self) -> None:
        """
        Load team configurations from config.
        Creates Team instances for each team configuration.
        """
        teams_config = self.config_node.get_value("orchestration.teams") or {}
        
        if not teams_config:
            logger.warning("orchestrator.no_teams_found")
            # Load default teams for backward compatibility
            self._load_default_teams()
            return
            
        for team_id, team_config in teams_config.items():
            try:
                # Get basic team info
                name = team_config.get("name", team_id)
                
                # Get task configurations
                tasks = []
                for task_config in team_config.get("tasks", []):
                    agent_class = task_config.get("agent_class")
                    if not agent_class:
                        logger.error("orchestrator.missing_agent_class", team_id=team_id, task=task_config)
                        continue
                        
                    # Create task config
                    agent_config = AgentTaskConfig(
                        agent_class=agent_class,
                        config=deep_merge(self.config, task_config.get("config", {})),
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
                    config=team_config
                )
                logger.info("orchestrator.team_loaded", team_id=team_id, name=name, tasks=len(tasks))
                
            except Exception as e:
                logger.error("orchestrator.team_load_failed", team_id=team_id, error=str(e))
    
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
            config: Base configuration
            
        Returns:
            Tuple of (prepared_config, context_dict)
        """
        try:
            # Ensure config is a dictionary
            prepared_config = config.copy() if config else {}

            # Normalize project path
            if not project_path:
                project_path = prepared_config.get('project', {}).get('path')
                if not project_path:
                    raise ValueError("No project path specified in arguments or config")

            # Convert Path objects to string for consistency
            if isinstance(project_path, Path):
                project_path = str(project_path)
                
            # Ensure project config exists
            if 'project' not in prepared_config:
                prepared_config['project'] = {}
            prepared_config['project']['path'] = project_path

            # Generate workflow ID with embedded timestamp
            time_str = datetime.now().strftime('%H%M')
            workflow_id = f"wf_{time_str}_{uuid.uuid4()}"

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

            # Add default configs for crucial components
            
            # 1. Ensure tartxt_config defaults
            if 'llm_config' not in prepared_config:
                prepared_config['llm_config'] = {}
            if 'agents' not in prepared_config['llm_config']:
                prepared_config['llm_config']['agents'] = {}
            if 'discovery' not in prepared_config['llm_config']['agents']:
                prepared_config['llm_config']['agents']['discovery'] = {}
                
            # Add default tartxt_config if not present
            discovery_config = prepared_config['llm_config']['agents']['discovery']
            if 'tartxt_config' not in discovery_config:
                discovery_config['tartxt_config'] = {}
                
            tartxt_config = discovery_config['tartxt_config']

            # Ensure script_path is set (handle both possible key names)
            if 'script_path' not in tartxt_config and 'script_base_path' not in tartxt_config:
                # Try to locate the script in the package
                import c4h_agents
                agent_path = Path(c4h_agents.__file__).parent
                script_path = agent_path / "skills" / "tartxt.py"
                if script_path.exists():
                    tartxt_config['script_path'] = str(script_path)
                else:
                    # Fallback to a relative path if the package path is not found
                    tartxt_config['script_path'] = "c4h_agents/skills/tartxt.py"
            elif 'script_base_path' in tartxt_config and 'script_path' not in tartxt_config:
                # Convert script_base_path to script_path
                script_base = tartxt_config['script_base_path']
                tartxt_config['script_path'] = f"{script_base}/tartxt.py"

            # Ensure input_paths is set
            if 'input_paths' not in tartxt_config:
                tartxt_config['input_paths'] = ["./"]
                
            # Prepare consistent context dictionary
            context = {
                "project_path": project_path,
                "intent": intent_desc,
                "workflow_run_id": workflow_id,
                "system": {"runid": workflow_id},
                "timestamp": timestamp,
                "config": prepared_config
            }
            
            logger.info("workflow.initialized", 
                    workflow_id=workflow_id,
                    project_path=project_path,
                    tartxt_config_keys=list(tartxt_config.keys()))
                    
            return prepared_config, context
            
        except Exception as e:
            logger.error("workflow.initialization_failed", error=str(e))
            raise