"""
Team implementation for agent orchestration.
Path: c4h_services/src/orchestration/team.py
"""

from typing import Dict, Any, List, Optional
from prefect import flow
from c4h_services.src.utils.logging import get_logger
from pathlib import Path

from c4h_services.src.intent.impl.prefect.tasks import run_agent_task
from c4h_services.src.intent.impl.prefect.models import AgentTaskConfig

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
            context: Execution context including workflow data
            
        Returns:
            Dict with execution results and next team ID
        """
        logger.info("team.execution_starting", team_id=self.team_id, name=self.name)
        
        # Track results of each agent
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
                task_context = {
                    **context,
                    "team_id": self.team_id,
                    "team_name": self.name,
                    "task_index": i
                }
                
                # Run the agent task
                result = run_agent_task(
                    agent_config=task_config,
                    context=task_context
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
            
            # Collect all result data
            team_data = {}
            for result in results:
                if result.get("success", False) and "result_data" in result:
                    team_data.update(result["result_data"])
            
            # Create final result
            team_result["data"] = team_data
            team_result["next_team"] = next_team
            
            # Special handling for team-to-team data passing
            if self.team_id == "discovery" and next_team == "solution":
                # Structure the data as expected by the solution designer
                team_result["input_data"] = {
                    "discovery_data": team_data,
                    "intent": context.get("intent", {}),
                    "project": context.get("project", {})
                }
            elif self.team_id == "solution" and next_team == "coder":
                # Structure data from solution to coder
                team_result["input_data"] = team_data
            
            logger.info("team.execution_completed", 
                    team_id=self.team_id, 
                    success=team_result["success"],
                    next_team=next_team)
                    
            return team_result
            
        except Exception as e:
            logger.error("team.execution_failed", team_id=self.team_id, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "team_id": self.team_id,
                "data": {},
                "next_team": None
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