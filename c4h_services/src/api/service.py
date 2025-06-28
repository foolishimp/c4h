# File: /Users/jim/src/apps/c4h_ai_dev/c4h_services/src/api/service.py

from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional, List

# Import get_logger from the shared utility
from c4h_services.src.utils.logging import get_logger
from c4h_agents.utils.logging import log_config_node # Import directly
from pathlib import Path
import uuid
import os
import logging # Import standard logging
from datetime import datetime

from c4h_services.src.api.models import (WorkflowRequest, WorkflowResponse,
                                         JobResponse, JobStatus,
                                         MultiConfigJobRequest, MergeRequest, MergeResponse)
from c4h_services.src.orchestration.orchestrator import Orchestrator
from c4h_services.src.utils.lineage_utils import load_lineage_file, prepare_context_from_lineage

# Hydra imports for configuration management
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# --- Logger Setup ---
logger = get_logger() # Initialize logger at module level is fine

# --- Global State / Storage (Okay at module level) ---
workflow_storage: Dict[str, Dict[str, Any]] = {}
job_storage: Dict[str, Dict[str, Any]] = {}
job_to_workflow_map: Dict[str, str] = {}
# Define the Hydra configuration path
hydra_config_path = Path("/Users/jim/src/apps/c4h_ai_dev/conf")

# --- Helper Functions ---
def map_workflow_to_job_changes(workflow_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Map workflow storage data to job changes format.
    Workflow data contains detailed information about changes made during execution.
    This function extracts and formats these changes for the job response.
    Args:
        workflow_data: Dictionary containing workflow execution results

    Returns:
        List of formatted change objects for job response
    """
    try:
        # Try multiple paths to find changes
        changes = None

        # Check direct changes field
        if 'changes' in workflow_data:
            changes = workflow_data['changes'] # cite: 1777
            logger.debug("jobs.changes_found_direct", count=len(changes) if changes else 0)

        # Check in data field
        elif 'data' in workflow_data and 'changes' in workflow_data['data']:
             changes = workflow_data['data']['changes'] # cite: 1778
             logger.debug("jobs.changes_found_in_data", count=len(changes) if changes else 0)

        # Check in team_results.coder.data
        elif 'team_results' in workflow_data and 'coder' in workflow_data['team_results']:
            coder_result = workflow_data['team_results']['coder'] # cite: 1778
            if 'data' in coder_result and 'changes' in coder_result['data']:
                changes = coder_result['data']['changes'] # cite: 1779
                logger.debug("jobs.changes_found_in_coder", count=len(changes) if changes else 0)

        # If no changes found
        if not changes:
            logger.warning("jobs.no_changes_found",
                         workflow_data_keys=list(workflow_data.keys())) # cite: 1780
            return []

        # Format changes for job response
        formatted_changes = []
        for change in changes:
            # Handle different change formats
            if isinstance(change, dict): # cite: 1781
                formatted_change = {}

                # Extract file path - check different field names
                if 'file' in change:
                     formatted_change['file'] = change['file'] # cite: 1782
                elif 'file_path' in change:
                    formatted_change['file'] = change['file_path'] # cite: 1782
                elif 'path' in change:
                     formatted_change['file'] = change['path'] # cite: 1783
                else:
                    # Skip changes without file information
                    continue

                # Extract change type information
                if 'change' in change:
                    formatted_change['change'] = change['change'] # cite: 1784
                elif 'type' in change:
                     formatted_change['change'] = {'type': change['type']} # cite: 1785
                elif 'success' in change:
                    # For simple success/error format
                    status = 'success' if change['success'] else 'error' # cite: 1786
                    formatted_change['change'] = {'status': status}
                    if 'error' in change and change['error']:
                        formatted_change['change']['error'] = change['error'] # cite: 1786

                formatted_changes.append(formatted_change) # cite: 1787

        logger.info("jobs.changes_mapped",
                original_count=len(changes) if changes else 0,
                formatted_count=len(formatted_changes)) # cite: 1787

        return formatted_changes
    except Exception as e:
        logger.error("jobs.mapping.changes_failed",
                  error=str(e),
                  error_type=type(e).__name__) # cite: 1788
        return []


# --- FastAPI App Creation Function ---

def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI: # Changed signature: now requires config
    """
    Create FastAPI application with team-based orchestration.
    Args:
        config: The fully loaded and merged configuration for this app instance.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="C4H Workflow Service",
        description="API for executing C4H team-based workflows",
        version="0.2.1" # Incremented version
    )

    # If no config provided, load default using Hydra
    if config is None:
        # Clear any existing Hydra instance
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        
        with initialize(version_base=None, config_path=str(hydra_config_path)):
            cfg = compose(config_name="config")
            config = OmegaConf.to_container(cfg, resolve=True)
    
    # Store the provided config in app state
    app.state.config = config
    app.state.hydra_config_path = hydra_config_path

    # Configure API logger
    api_logger = logging.getLogger("api.requests")
    # Set level based on main config if available
    log_level_str = app.state.config.get("logging", {}).get("level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    api_logger.setLevel(log_level)
    api_logger.propagate = True # Propagate to main logger setup by get_logger
    
    # Note: Orchestrator is now created per-request in create_job endpoint

    # --- run_workflow and get_workflow functions remain the same ---
    # They will use the orchestrator stored in app.state which was initialized
    # with the config passed to create_app.
    async def run_workflow(request: WorkflowRequest):
        """
        Execute a team-based workflow with the provided configuration.
        Configuration from the request is merged with the app's base configuration.
        """
        try:
            # --- Start with a clean copy of the app's config using Hydra ---
            # Clear any existing Hydra instance
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            
            with initialize(version_base=None, config_path=str(app.state.hydra_config_path)):
                # Load base configuration
                current_run_cfg = compose(config_name="config")
                
                # --- Merge system_config and app_config from the request ---
                # This allows overriding parts of the config for a specific run
                if request.system_config:
                    system_cfg = OmegaConf.create(request.system_config)
                    current_run_cfg = OmegaConf.merge(current_run_cfg, system_cfg)
                if request.app_config:
                    app_cfg = OmegaConf.create(request.app_config)
                    current_run_cfg = OmegaConf.merge(current_run_cfg, app_cfg)
                
                # Convert to container for use with existing code
                current_run_config = OmegaConf.to_container(current_run_cfg, resolve=True)

            # Check if lineage file is provided for workflow continuation
            if request.lineage_file and request.stage:
                logger.info("workflow.continuing_from_lineage",
                            lineage_file=request.lineage_file,
                            stage=request.stage,
                            keep_runid=request.keep_runid)

                try:
                    lineage_data = load_lineage_file(request.lineage_file)

                    # Add intent and project path to the config used for context prep
                    temp_config_for_context = current_run_config.copy()
                    if request.intent: temp_config_for_context['intent'] = request.intent
                    if request.project_path:
                        if 'project' not in temp_config_for_context: temp_config_for_context['project'] = {}
                        temp_config_for_context['project']['path'] = request.project_path

                    context = prepare_context_from_lineage(
                         lineage_data,
                         request.stage,
                         temp_config_for_context, # Use temp config here
                         keep_runid=request.keep_runid
                    )
                    workflow_id = context["workflow_run_id"]

                    # --- Initialize orchestrator for this specific run with potentially modified config ---
                    # This ensures the lineage continuation uses the correct merged config
                    current_orchestrator = Orchestrator(current_run_config)

                    # Pass the fully prepared config (including lineage overrides) in the context
                    context["config"] = current_run_config

                    result = current_orchestrator.execute_workflow(
                         entry_team=request.stage,
                         context=context
                    )

                    # Store result using the final workflow_id
                    workflow_storage[workflow_id] = {
                         "status": result.get("status", "error"),
                         "team_results": result.get("team_results", {}),
                         "changes": result.get("data", {}).get("changes", []),
                         "storage_path": os.path.join("workspaces", "lineage", workflow_id) if current_run_config.get("runtime", {}).get("lineage", {}).get("enabled", False) else None,
                         "source_lineage": request.lineage_file,
                         "stage": request.stage
                    }

                    return WorkflowResponse(
                        workflow_id=workflow_id,
                        status=result.get("status", "error"),
                        storage_path=workflow_storage[workflow_id].get("storage_path"),
                        error=result.get("error") if result.get("status") == "error" else None
                    )

                except Exception as e:
                     logger.error("workflow.lineage_processing_failed",
                               lineage_file=request.lineage_file,
                               stage=request.stage,
                               error=str(e), exc_info=True)
                     raise HTTPException(status_code=500, detail=f"Lineage processing failed: {str(e)}")

            # --- Standard workflow initialization ---
            # Create orchestrator for this request with the merged config
            request_orchestrator = Orchestrator(current_run_config)
            
            # initialize_workflow uses the config passed to it (current_run_config)
            prepared_config, context = request_orchestrator.initialize_workflow(
                project_path=request.project_path,
                intent_desc=request.intent,
                config=current_run_config # Pass the potentially overridden config
            )
            workflow_id = context["workflow_run_id"]

            # Ensure intent is in the prepared_config
            if 'intent' not in prepared_config:
                prepared_config['intent'] = request.intent

            logger.info("workflow.starting",
                        workflow_id=workflow_id,
                        project_path=request.project_path,
                        config_keys=list(prepared_config.keys()))

            # Pass the fully prepared config in the context
            context["config"] = prepared_config

            try:
                entry_team = prepared_config.get("orchestration", {}).get("entry_team", "discovery")

                # Execute using the request-specific orchestrator instance
                # which contains the potentially run-specific prepared_config
                result = request_orchestrator.execute_workflow(
                    entry_team=entry_team,
                    context=context
                )

                workflow_storage[workflow_id] = {
                    "status": result.get("status", "error"),
                    "team_results": result.get("team_results", {}),
                    "changes": result.get("data", {}).get("changes", []),
                    "storage_path": os.path.join("workspaces", "lineage", workflow_id) if prepared_config.get("runtime", {}).get("lineage", {}).get("enabled", False) else None
                }

                return WorkflowResponse(
                     workflow_id=workflow_id,
                     status=result.get("status", "error"),
                     storage_path=workflow_storage[workflow_id].get("storage_path"),
                     error=result.get("error") if result.get("status") == "error" else None
                )

            except Exception as e:
                logger.error("workflow.execution_failed",
                        workflow_id=workflow_id,
                        error=str(e), exc_info=True)

                workflow_storage[workflow_id] = {
                     "status": "error",
                     "error": str(e),
                     "storage_path": None
                }

                return WorkflowResponse(
                     workflow_id=workflow_id,
                     status="error",
                     error=str(e)
                )

        except Exception as e:
            logger.error("workflow.request_failed", error=str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_workflow(workflow_id: str):
        """Get workflow status and results"""
        if workflow_id in workflow_storage:
            data = workflow_storage[workflow_id]
            return WorkflowResponse(
                workflow_id=workflow_id,
                status=data.get("status", "unknown"),
                storage_path=data.get("storage_path"),
                error=data.get("error")
            )
        else:
            raise HTTPException(status_code=404, detail="Workflow not found")


    # --- API Endpoints ---
    @app.get("/health")
    async def health_check():
        # ... (health check remains the same) ...
        return {
            "status": "healthy",
            "workflows_tracked": len(workflow_storage),
            "jobs_tracked": len(job_storage),
            "teams_available": len(app.state.config.get("orchestration", {}).get("teams", {}))
        }

    @app.post("/api/v1/jobs", response_model=JobResponse)
    async def create_job(request: MultiConfigJobRequest):
        """Create a new job from multiple configuration fragments that will be merged."""
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        merged_config = None
        project_path = None
        intent = None
        workflow_request = None

        try:
            logger.info("jobs.multi_config_request_received",
                    job_id=job_id,
                    configs_count=len(request.configs))

            # Use Hydra Compose API to merge configurations
            # Clear any existing Hydra instance
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            
            with initialize(version_base=None, config_path=str(app.state.hydra_config_path)):
                # Load base configuration
                base_cfg = compose(config_name="config")
                
                # Merge each config fragment using OmegaConf
                for i, config_fragment in enumerate(request.configs):
                    logger.debug("jobs.merging_fragment", job_id=job_id, fragment_index=i, fragment_keys=list(config_fragment.keys()))
                    fragment_cfg = OmegaConf.create(config_fragment)
                    base_cfg = OmegaConf.merge(base_cfg, fragment_cfg)
                
                # Convert to container for use with existing code
                merged_config = OmegaConf.to_container(base_cfg, resolve=True)

                log_config_node(logger, merged_config, "workorder", log_prefix=f"jobs.final_merged.{job_id}")
                log_config_node(logger, merged_config, "team", log_prefix=f"jobs.final_merged.{job_id}")
                log_config_node(logger, merged_config, "runtime", log_prefix=f"jobs.final_merged.{job_id}")
                log_config_node(logger, merged_config, "llm_config.agents", log_prefix=f"jobs.final_merged.{job_id}")

                logger.debug("jobs.configs_merged",
                        job_id=job_id,
                        merged_config_keys=list(merged_config.keys()))

                # Extract project path and intent from merged config
                # Check for project path in various locations
                project_path = None
                if 'project' in merged_config and 'path' in merged_config['project']:
                    project_path = merged_config['project']['path']
                elif 'workorder' in merged_config and 'project' in merged_config['workorder'] and 'path' in merged_config['workorder']['project']:
                    project_path = merged_config['workorder']['project']['path']
                
                if not project_path:
                    logger.error("jobs.missing_project_path", job_id=job_id, config_keys=list(merged_config.keys()))
                    raise HTTPException(status_code=400, detail="Project path not found in configuration")
                
                # Extract intent
                intent = None
                if 'intent' in merged_config:
                    intent = merged_config['intent']
                elif 'workorder' in merged_config and 'intent' in merged_config['workorder']:
                    intent = merged_config['workorder']['intent']
                
                if not intent:
                    logger.error("jobs.missing_intent", job_id=job_id, config_keys=list(merged_config.keys()))
                    raise HTTPException(status_code=400, detail="Intent not found in configuration")
                
                # Extract lineage info if present
                lineage_file = None
                stage = None
                keep_runid = True
                
                if 'runtime' in merged_config and 'runtime' in merged_config['runtime']:
                    runtime_config = merged_config['runtime']['runtime']
                    if isinstance(runtime_config, dict):
                        lineage_file = runtime_config.get('lineage_file')
                        stage = runtime_config.get('stage')
                        keep_runid = runtime_config.get('keep_runid', True)
                
                # Create workflow request
                workflow_request = WorkflowRequest(
                    project_path=project_path,
                    intent=intent,
                    app_config=merged_config,
                    lineage_file=lineage_file,
                    stage=stage,
                    keep_runid=keep_runid
                )
                
                logger.info("jobs.workflow_request_created_from_merged_config",
                           job_id=job_id,
                           project_path=project_path,
                           has_intent=bool(intent))

            if not workflow_request:
                 logger.error("jobs.workflow_request_creation_failed", job_id=job_id)
                 raise HTTPException(status_code=500, detail="Internal error: Failed to create workflow request")

            logger.debug("jobs.pre_run_workflow_check",
                        job_id=job_id,
                        wf_request_type=type(workflow_request).__name__,
                        wf_req_project_path=getattr(workflow_request, 'project_path', 'N/A'),
                        wf_req_intent_type=type(getattr(workflow_request, 'intent', None)).__name__,
                        wf_req_app_config_keys=list(getattr(workflow_request, 'app_config', {}).keys()))

            try:
                # Update workflow_request's app_config to ensure it uses the final merged version
                workflow_request.app_config = merged_config
                workflow_response = await run_workflow(workflow_request)
                logger.info("jobs.workflow_executed",
                        job_id=job_id, workflow_id=workflow_response.workflow_id,
                        status=workflow_response.status)
            except Exception as e:
                logger.error("jobs.workflow_execution_failed", job_id=job_id, error=str(e), exc_info=True)
                error_detail = f"Workflow execution failed: {str(e)}"
                job_storage[job_id] = {
                    "status": "error", "error": error_detail, "created_at": datetime.now().isoformat(),
                    "workflow_id": None, "project_path": project_path, "last_updated": datetime.now().isoformat()
                }
                raise HTTPException(status_code=500, detail=error_detail)

            job_response = JobResponse(
                job_id=job_id,
                status=workflow_response.status,
                storage_path=workflow_response.storage_path,
                error=workflow_response.error
            )

            job_storage[job_id] = {
                "status": workflow_response.status,
                "storage_path": workflow_response.storage_path,
                "error": workflow_response.error,
                "created_at": datetime.now().isoformat(),
                "workflow_id": workflow_response.workflow_id,
                "project_path": project_path,
                "last_updated": datetime.now().isoformat()
            }
            job_to_workflow_map[job_id] = workflow_response.workflow_id

            logger.info("jobs.created",
                    job_id=job_id, workflow_id=workflow_response.workflow_id,
                    status=workflow_response.status)

            return job_response

        except HTTPException:
            raise
        except Exception as e:
            logger.error("jobs.creation_failed_unexpected", job_id=job_id, error=str(e), error_type=type(e).__name__, exc_info=True)
            job_storage[job_id] = {
                "status": "error", "error": f"Unexpected job creation error: {str(e)}",
                "created_at": datetime.now().isoformat(), "workflow_id": None,
                "project_path": project_path, "last_updated": datetime.now().isoformat()
            }
            raise HTTPException(status_code=500, detail=f"Job creation failed unexpectedly: {str(e)}")


    @app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
    async def get_job_status(job_id: str):
        # ... (get_job_status remains the same) ...
        try:
            if job_id not in job_storage:
                logger.error("jobs.not_found", job_id=job_id)
                raise HTTPException(status_code=404, detail="Job not found")

            workflow_id = job_to_workflow_map.get(job_id)
            if not workflow_id:
                logger.warning("jobs.workflow_mapping_missing_using_stored", job_id=job_id)
                stored_job_data = job_storage[job_id]
                return JobStatus(
                    job_id=job_id,
                    status=stored_job_data.get("status", "unknown"),
                    storage_path=stored_job_data.get("storage_path"),
                    error=stored_job_data.get("error"),
                    changes=stored_job_data.get("changes", [])
                )

            logger.info("jobs.status_request", job_id=job_id, workflow_id=workflow_id)
            workflow_data = workflow_storage.get(workflow_id, {})

            if not workflow_data:
                logger.warning("jobs.workflow_data_not_in_memory", job_id=job_id, workflow_id=workflow_id)
                workflow_data = job_storage[job_id]
                logger.debug("jobs.using_stored_job_data", job_id=job_id, last_updated=workflow_data.get("last_updated"))
            else:
                 job_storage[job_id].update({
                     "status": workflow_data.get("status"),
                     "storage_path": workflow_data.get("storage_path"),
                     "error": workflow_data.get("error"),
                     "last_checked": datetime.now().isoformat(),
                     "changes": map_workflow_to_job_changes(workflow_data)
                 })

            final_job_data = job_storage[job_id]
            job_status = JobStatus(
                job_id=job_id,
                status=final_job_data.get("status", "unknown"),
                storage_path=final_job_data.get("storage_path"),
                error=final_job_data.get("error"),
                changes=final_job_data.get("changes", [])
            )

            logger.info("jobs.status_checked",
                    job_id=job_id, workflow_id=workflow_id, status=job_status.status,
                    changes_count=len(job_status.changes) if job_status.changes else 0)

            return job_status

        except HTTPException:
            raise
        except Exception as e:
            logger.error("jobs.status_check_failed", job_id=job_id, error=str(e), error_type=type(e).__name__, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Job status check failed: {str(e)}")


    @app.post("/api/v1/configs/merge", response_model=MergeResponse)
    async def merge_configs(request: MergeRequest):
        # ... (merge_configs remains the same) ...
        try:
            # Clear any existing Hydra instance
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            
            with initialize(version_base=None, config_path=str(app.state.hydra_config_path)):
                if request.include_system_config:
                    # Load the base configuration using Hydra
                    merged_cfg = compose(config_name="config")
                    logger.debug("configs.merge.using_system_config")
                else:
                    # Start with empty config
                    merged_cfg = OmegaConf.create({})

                # Merge each config in reverse order (as per original logic)
                for i, config in enumerate(reversed(request.configs)):
                    config_cfg = OmegaConf.create(config)
                    merged_cfg = OmegaConf.merge(merged_cfg, config_cfg)
                    logger.debug(f"configs.merge.step_{i+1}", config_keys=list(config.keys()))

                # Convert to container
                merged_config = OmegaConf.to_container(merged_cfg, resolve=True)

            return MergeResponse(merged_config=merged_config)

        except Exception as e:
            logger.error("configs.merge.failed", error=str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=f"Configuration merge failed: {str(e)}")


    return app
