# File: /Users/jim/src/apps/c4h_ai_dev/c4h_services/src/api/service.py
# Correction: Removed module-level config loading and app instantiation

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional, Callable, List, Union

# Import get_logger from the shared utility
from c4h_services.src.utils.logging import get_logger
from c4h_agents.utils.logging import log_config_node # Import directly
from pathlib import Path
import uuid
import os
import json
import logging # Import standard logging
from datetime import datetime, timezone # Ensure timezone is imported

from c4h_agents.config import deep_merge, load_config, create_config_node, deepcopy
from c4h_agents.core.project import Project
from c4h_services.src.api.models import (WorkflowRequest, WorkflowResponse,
                                         JobRequest, JobResponse, JobStatus,
                                         MultiConfigJobRequest, MergeRequest, MergeResponse, JobRequestUnion)
from c4h_services.src.orchestration.orchestrator import Orchestrator
from c4h_services.src.utils.lineage_utils import load_lineage_file, prepare_context_from_lineage

# --- Logger Setup ---
logger = get_logger() # Initialize logger at module level is fine

# --- Global State / Storage (Okay at module level) ---
workflow_storage: Dict[str, Dict[str, Any]] = {}
job_storage: Dict[str, Dict[str, Any]] = {}
job_to_workflow_map: Dict[str, str] = {}
# Define the default path, but don't load it here
system_config_path_default = Path("config/system_config.yml")

# --- Helper Functions ---
# ... (map_job_to_workflow_request, extract_from_merged_config, map_workflow_to_job_changes remain the same) ...
def map_job_to_workflow_request(job_request: JobRequest) -> WorkflowRequest:
    """
    Map JobRequest to WorkflowRequest format.
    Transforms the structured job configuration to flat workflow configuration.
    """
    try:
        # Get project path from workorder
        project_path = job_request.workorder.project.path # cite: 1762

        # Get intent from workorder - convert to dictionary with exclude_none
        intent_dict = job_request.workorder.intent.dict(exclude_none=True) # cite: 1762

        # Initialize app_config with project settings
        app_config = {
            "project": job_request.workorder.project.dict(exclude_none=True), # cite: 1763
            "intent": intent_dict
        }

        # Track what's being extracted for logging
        extracted_sections = ["workorder.project", "workorder.intent"]

        # Add team configuration if provided
        if job_request.team:
            team_dict = job_request.team.dict(exclude_none=True) # cite: 1764
            for key, value in team_dict.items():
                if value:
                    app_config[key] = value # cite: 1765
                    extracted_sections.append(f"team.{key}")

        # Add runtime configuration if provided
        if job_request.runtime:
            runtime_dict = job_request.runtime.dict(exclude_none=True) # cite: 1766
            for key, value in runtime_dict.items():
                if value:
                    app_config[key] = value
                    extracted_sections.append(f"runtime.{key}") # cite: 1766

        # Extract lineage information from runtime if available
        lineage_file = None # cite: 1767
        stage = None
        keep_runid = True

        if job_request.runtime and job_request.runtime.runtime:
            runtime_config = job_request.runtime.runtime # cite: 1767
            if isinstance(runtime_config, dict):
                lineage_file = runtime_config.get("lineage_file") # cite: 1768
                stage = runtime_config.get("stage") # cite: 1768
                keep_runid = runtime_config.get("keep_runid", True) # cite: 1768

                if lineage_file:
                    extracted_sections.append("runtime.runtime.lineage_file") # cite: 1769
                if stage:
                    extracted_sections.append("runtime.runtime.stage")
                if "keep_runid" in runtime_config:
                    extracted_sections.append("runtime.runtime.keep_runid") # cite: 1769

        # Create workflow request with all parameters
        workflow_request = WorkflowRequest(
            project_path=project_path, # cite: 1770
            intent=intent_dict,
            app_config=app_config,
            lineage_file=lineage_file, # cite: 1771
            stage=stage,
            keep_runid=keep_runid
        )

        logger.debug("jobs.mapping.job_to_workflow",
                project_path=project_path,
                extracted_sections=extracted_sections,
                app_config_keys=list(app_config.keys()), # cite: 1772
                lineage_file=lineage_file,
                stage=stage)

        return workflow_request

    except Exception as e:
        logger.error("jobs.mapping.failed",
                 error=str(e),
                 error_type=type(e).__name__) # cite: 1773
        raise ValueError(f"Failed to map job request to workflow request: {str(e)}")

def extract_from_merged_config(merged_config: Dict[str, Any]) -> WorkflowRequest:
    """
    Extract necessary fields from a merged configuration and create a WorkflowRequest.
    This function handles the extraction step after all configs have been merged.

    Args:
        merged_config: The fully merged configuration dictionary

    Returns:
        WorkflowRequest object with properly mapped fields
    """
    try:
        # Create a config node for easier path-based access
        config_node = create_config_node(merged_config)

        # Extract project path - first check common paths
        project_path = config_node.get_value("project.path")
        if not project_path:
            # Fallback to workorder path
            project_path = config_node.get_value("workorder.project.path")

        if not project_path:
            logger.warning("config_extraction.missing_project_path", config_keys=list(merged_config.keys()))
            raise ValueError("Required field 'project_path' not found in configuration")

        # Extract intent
        intent_dict = config_node.get_value("intent")
        if not intent_dict:
            # Fallback to workorder intent
            intent_dict = config_node.get_value("workorder.intent")

        if not intent_dict:
            logger.warning("config_extraction.missing_intent", config_keys=list(merged_config.keys()))
            raise ValueError("Required field 'intent' not found in configuration")

        # Extract lineage information from runtime if available
        lineage_file = None
        stage = None
        keep_runid = True

        # Check common paths for lineage/stage info
        runtime_config = config_node.get_value("runtime.runtime")
        if isinstance(runtime_config, dict):
            lineage_file = runtime_config.get("lineage_file")
            stage = runtime_config.get("stage")
            keep_runid = runtime_config.get("keep_runid", True)

        # Create the workflow request
        workflow_request = WorkflowRequest(
            project_path=project_path,
            intent=intent_dict,
            app_config=merged_config, # Pass the full merged config as app_config
            lineage_file=lineage_file,
            stage=stage,
            keep_runid=keep_runid
        )

        logger.debug("config.extraction_complete",
                    project_path=project_path,
                    has_intent=bool(intent_dict),
                    app_config_keys=list(merged_config.keys()),
                    lineage_file=lineage_file,
                    stage=stage)

        return workflow_request

    except Exception as e:
        logger.error("config.extraction_failed", error=str(e), error_type=type(e).__name__)
        raise ValueError(f"Failed to extract workflow request from merged configuration: {str(e)}")

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

def create_app(config: Dict[str, Any]) -> FastAPI: # Changed signature: now requires config
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

    # Store the provided config in app state
    app.state.config = config
    # Store the default system config path for reference if needed (e.g., by merge endpoint)
    app.state.system_config_path_default = system_config_path_default

    # Create orchestrator using the provided config
    app.state.orchestrator = Orchestrator(app.state.config)
    logger.info("api.orchestrator_initialized_with_config",
               teams=len(app.state.orchestrator.teams))

    # Configure API logger
    api_logger = logging.getLogger("api.requests")
    # Set level based on main config if available
    log_level_str = app.state.config.get("logging", {}).get("level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    api_logger.setLevel(log_level)
    api_logger.propagate = True # Propagate to main logger setup by get_logger

    # --- run_workflow and get_workflow functions remain the same ---
    # They will use the orchestrator stored in app.state which was initialized
    # with the config passed to create_app.
    async def run_workflow(request: WorkflowRequest):
        """
        Execute a team-based workflow with the provided configuration.
        Configuration from the request is merged with the app's base configuration.
        """
        try:
            # --- Start with a clean copy of the app's config ---
            # This prevents mutation of the shared app.state.config
            # The orchestrator was already initialized with this config.
            current_run_config = deepcopy(app.state.config)

            # --- Merge system_config and app_config from the request ---
            # This allows overriding parts of the config for a specific run
            if request.system_config:
                 current_run_config = deep_merge(current_run_config, request.system_config)
            if request.app_config:
                 current_run_config = deep_merge(current_run_config, request.app_config)

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

                    # --- Re-initialize orchestrator for this specific run with potentially modified config ---
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
            # Use the main orchestrator instance initialized with app's config
            # initialize_workflow uses the config passed to it (current_run_config)
            prepared_config, context = app.state.orchestrator.initialize_workflow(
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

                # Execute using the main orchestrator instance, but pass the prepared context
                # which contains the potentially run-specific prepared_config
                result = app.state.orchestrator.execute_workflow(
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
            "teams_available": len(app.state.orchestrator.teams)
        }

    @app.post("/api/v1/jobs", response_model=JobResponse)
    async def create_job(request: JobRequestUnion):
        # ... (create_job logic remains the same, it calls run_workflow) ...
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        merged_config = None
        project_path = None
        intent = None
        workflow_request = None

        try:
            if isinstance(request, MultiConfigJobRequest):
                logger.info("jobs.multi_config_request_received",
                        job_id=job_id,
                        configs_count=len(request.configs))

                # Start with a deep copy of the default config from app state
                base_config = deepcopy(app.state.config) # Use app state config
                if not isinstance(base_config, dict):
                    logger.warning("App state config is not a dictionary, starting merge with empty dict.", job_id=job_id)
                    base_config = {}
                merged_config = base_config

                # Merge fragments onto the base config
                for i, config_fragment in enumerate(request.configs):
                    logger.debug("jobs.merging_fragment", job_id=job_id, fragment_index=i, fragment_keys=list(config_fragment.keys()))
                    merged_config = deep_merge(merged_config, config_fragment)

                log_config_node(logger, merged_config, "workorder", log_prefix=f"jobs.final_merged.{job_id}")
                log_config_node(logger, merged_config, "team", log_prefix=f"jobs.final_merged.{job_id}")
                log_config_node(logger, merged_config, "runtime", log_prefix=f"jobs.final_merged.{job_id}")
                log_config_node(logger, merged_config, "llm_config.agents", log_prefix=f"jobs.final_merged.{job_id}")

                logger.debug("jobs.configs_merged",
                        job_id=job_id,
                        merged_config_keys=list(merged_config.keys()))

                try:
                    workflow_request = extract_from_merged_config(merged_config)
                    logger.info("jobs.workflow_request_created_from_merged_config",
                               job_id=job_id,
                               project_path=workflow_request.project_path,
                               has_intent=bool(workflow_request.intent))
                    project_path = workflow_request.project_path
                    intent = workflow_request.intent
                except Exception as e:
                    logger.error("jobs.extract_from_merged_config_failed",
                               job_id=job_id, error=str(e), error_type=type(e).__name__, exc_info=True)
                    raise HTTPException(status_code=400, detail=f"Invalid merged configuration: {str(e)}")

            elif isinstance(request, JobRequest):
                logger.info("jobs.traditional_request_received",
                        job_id=job_id, project_path=request.workorder.project.path,
                        has_team_config=request.team is not None, has_runtime_config=request.runtime is not None)
                try:
                    workflow_request = map_job_to_workflow_request(request)
                    project_path = workflow_request.project_path
                    intent = workflow_request.intent
                    merged_config = workflow_request.app_config

                    log_config_node(logger, merged_config, "workorder", log_prefix=f"jobs.mapped.{job_id}")
                    log_config_node(logger, merged_config, "team", log_prefix=f"jobs.mapped.{job_id}")
                    log_config_node(logger, merged_config, "runtime", log_prefix=f"jobs.mapped.{job_id}")

                    logger.debug("jobs.request_mapped", job_id=job_id, workflow_request_keys=list(workflow_request.dict().keys()))
                except Exception as e:
                    logger.error("jobs.request_mapping_failed", job_id=job_id, error=str(e), exc_info=True)
                    raise HTTPException(status_code=400, detail=f"Invalid job configuration: {str(e)}")
            else:
                logger.error("jobs.unknown_request_type", job_id=job_id, request_type=type(request).__name__)
                raise HTTPException(status_code=400, detail="Unknown job request format")

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
            if request.include_system_config:
                try:
                    # Use the default path stored in app state
                    system_config = load_config(app.state.system_config_path_default)
                    merged_config = system_config.copy()
                    logger.debug("configs.merge.using_system_config", system_config_keys=list(system_config.keys()))
                except Exception as e:
                    logger.warning("configs.merge.system_config_load_failed", error=str(e))
                    # Fall back to app's current config if default load fails
                    merged_config = deepcopy(app.state.config)
            else:
                merged_config = {}

            for i, config in enumerate(reversed(request.configs)):
                 merged_config = deep_merge(merged_config, config)
                 logger.debug(f"configs.merge.step_{i+1}", config_keys=list(config.keys()))

            return MergeResponse(merged_config=merged_config)

        except Exception as e:
            logger.error("configs.merge.failed", error=str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=f"Configuration merge failed: {str(e)}")


    return app
