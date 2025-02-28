"""
Team-based intent service implementation.
Path: c4h_services/src/intent/impl/team/service.py
"""

from typing import Dict, Any, Optional
from pathlib import Path
import structlog
from datetime import datetime
import uuid

from c4h_agents.config import load_config, load_with_app_config
from c4h_services.src.orchestration.orchestrator import Orchestrator
from c4h_services.src.intent.core.service import IntentService

logger = structlog.get_logger()

class TeamIntentService(IntentService):
    """
    Team-based implementation of intent service interface.
    Leverages orchestrator for team management and execution.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize service with configuration"""
        try:
            # Load config using existing patterns
            system_path = Path("config/system_config.yml")
            if config_path:
                self.config = load_with_app_config(system_path, config_path)
            else:
                self.config = load_config(system_path)
            
            # Create orchestrator
            self.orchestrator = Orchestrator(self.config)
            
            logger.info("team_service.initialized",
                       config_path=str(config_path) if config_path else None,
                       teams_loaded=len(self.orchestrator.teams))
            
        except Exception as e:
            logger.error("team_service.init_failed", error=str(e))
            raise

    async def process_intent(
        self,
        project_path: Path,
        intent_desc: Dict[str, Any],
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """Process refactoring intent through team-based workflow"""
        try:
            logger.info("team_service.processing",
                       project_path=str(project_path),
                       intent=intent_desc)

            # Create workflow context
            context = {
                "project_path": str(project_path),
                "intent": intent_desc,
                "workflow_run_id": str(uuid.uuid4()),
                "max_iterations": max_iterations
            }
            
            # Set project configuration
            if "project" not in context:
                context["project"] = {}
            context["project"]["path"] = str(project_path)

            # Execute workflow
            result = self.orchestrator.execute_workflow(
                entry_team="discovery",  # Start with discovery team
                context=context,
                max_teams=max_iterations * 3  # Allow multiple iterations
            )
            
            logger.info("team_service.completed",
                       workflow_id=result.get("workflow_run_id"),
                       status=result.get("status"))
                       
            # Convert to standard response format
            response = {
                "status": result.get("status", "error"),
                "workflow_data": result,
                "error": result.get("error") if result.get("status") == "error" else None
            }
            
            # Add backward compatibility fields
            if "data" in result:
                if "changes" in result["data"]:
                    response["changes"] = result["data"]["changes"]
                    
            return response

        except Exception as e:
            error_msg = str(e)
            logger.error("team_service.failed", error=error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "workflow_data": {}
            }

    async def get_status(self, intent_id: str) -> Dict[str, Any]:
        """Get status for a workflow run"""
        # Since we don't have persistence in this implementation,
        # we can only return a basic status
        try:
            return {
                "status": "unknown",
                "error": "Status lookup not implemented for team service",
                "workflow_data": {
                    "workflow_run_id": intent_id
                }
            }
            
        except Exception as e:
            logger.error("team_service.status_failed", error=str(e), intent_id=intent_id)
            return {
                "status": "error",
                "error": f"Failed to get status: {str(e)}"
            }

    async def cancel_intent(self, intent_id: str) -> bool:
        """Cancel a workflow run"""
        # Since we don't have a way to cancel running workflows in this implementation,
        # we can only return failure
        logger.error("team_service.cancel_not_implemented", intent_id=intent_id)
        return False