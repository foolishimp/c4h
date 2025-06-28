"""
Prefect implementation of intent service.
Path: c4h_services/src/intent/impl/prefect/service.py
"""

from typing import Dict, Any, Optional
from pathlib import Path
from c4h_services.src.utils.logging import get_logger
from datetime import datetime
import uuid
from prefect.client import get_client
from prefect.deployments import Deployment

# Hydra imports for configuration management
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from .flows import run_intent_workflow, run_recovery_workflow, run_rollback_workflow

logger = get_logger()

class PrefectIntentService:
    """
    Prefect-based implementation of intent service interface.
    Handles workflow orchestration while maintaining agent independence.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize service with configuration"""
        try:
            # Define the Hydra configuration path
            hydra_config_path = Path("/Users/jim/src/apps/c4h_ai_dev/conf")
            
            # Clear any existing Hydra instance
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            
            # Load configuration using Hydra
            with initialize(version_base=None, config_path=str(hydra_config_path)):
                if config_path:
                    # If a specific config path is provided, load it as an override
                    cfg = compose(config_name="config", overrides=[f"config_path={config_path}"])
                else:
                    # Load default configuration
                    cfg = compose(config_name="config")
                
                # Convert to container for use with existing code
                self.config = OmegaConf.to_container(cfg, resolve=True)
            
            # Initialize Prefect client for flow operations
            self.client = get_client()
            
            logger.info("prefect_service.initialized",
                       config_path=str(config_path) if config_path else None)
            
        except Exception as e:
            logger.error("prefect_service.init_failed", error=str(e))
            raise

    async def process_intent(
        self,
        project_path: Path,
        intent_desc: Dict[str, Any],
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """Process refactoring intent through Prefect workflow"""
        try:
            logger.info("prefect_service.processing",
                       project_path=str(project_path),
                       intent=intent_desc)

            # Execute workflow
            flow_run = await self.client.create_flow_run(
                flow=run_intent_workflow,
                parameters={
                    "project_path": project_path,
                    "intent_desc": intent_desc,
                    "config": self.config,
                    "max_iterations": max_iterations
                }
            )
            
            # Wait for completion
            flow_state = await self.client.wait_for_flow_run(
                flow_run.id,
                timeout=1800  # 30 minutes
            )

            if flow_state.is_completed():
                result = flow_state.result()
                logger.info("prefect_service.completed",
                          flow_id=flow_run.id,
                          status=result.get("status"))
                return result
            else:
                error = f"Flow failed: {flow_state.message}"
                logger.error("prefect_service.flow_failed",
                           flow_id=flow_run.id,
                           error=error)
                return {
                    "status": "error",
                    "error": error,
                    "workflow_data": {}
                }

        except Exception as e:
            error_msg = str(e)
            logger.error("prefect_service.failed", error=error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "workflow_data": {}
            }

    async def get_status(self, intent_id: str) -> Dict[str, Any]:
        """Get status from Prefect flow run"""
        try:
            # Get flow run state
            flow_run = await self.client.read_flow_run(intent_id)
            
            return {
                "status": "success",
                "workflow_data": {
                    "flow_id": flow_run.id,
                    "status": flow_run.state.name,
                    "start_time": flow_run.start_time,
                    "end_time": flow_run.end_time,
                    "duration": flow_run.total_run_time,
                    "error": flow_run.state.message if flow_run.state.is_failed() else None
                }
            }
            
        except Exception as e:
            logger.error("status.failed", error=str(e), intent_id=intent_id)
            return {
                "status": "error",
                "error": f"Failed to get status: {str(e)}"
            }

    async def cancel_intent(self, intent_id: str) -> bool:
        """Cancel Prefect flow run"""
        try:
            await self.client.set_flow_run_state(
                flow_run_id=intent_id,
                state="CANCELLED"
            )
            
            logger.info("intent.cancelled", intent_id=intent_id)
            return True
            
        except Exception as e:
            logger.error("cancel.failed", error=str(e), intent_id=intent_id)
            return False

    async def create_deployment(
        self,
        name: str,
        schedule: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a deployment for the intent workflow.
        Enables scheduling and remote execution.
        """
        try:
            deployment = await Deployment.build_from_flow(
                flow=run_intent_workflow,
                name=name,
                schedule=schedule,
                work_queue_name="intent_refactor"
            )
            
            await deployment.apply()
            
            return {
                "status": "success",
                "deployment_id": deployment.id,
                "name": deployment.name
            }
            
        except Exception as e:
            logger.error("deployment.failed", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }