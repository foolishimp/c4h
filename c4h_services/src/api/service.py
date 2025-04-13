# File: /Users/jim/src/apps/c4h/c4h_services/src/api/service.py
"""
API service implementation with enhanced configuration handling and multi-config support.
Path: c4h_services/src/api/service.py
"""

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
logger = get_logger()

# --- Global State / Storage ---
workflow_storage: Dict[str, Dict[str, Any]] = {}
job_storage: Dict[str, Dict[str, Any]] = {}
job_to_workflow_map: Dict[str, str] = {}
system_config_path = Path("config/system_config.yml")

# --- Helper Functions (Moved Before Endpoints) ---

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

# --- FastAPI App Creation ---

def create_app(default_config: Dict[str, Any] = None) -> FastAPI:
    """
    Create FastAPI application with team-based orchestration.
    Args:
        default_config: Optional base configuration for the service

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="C4H Workflow Service",
        description="API for executing C4H team-based workflows",
        version="0.2.0"
    )

    # Store default config in app state
    app.state.default_config = default_config or {}
    app.state.system_config_path = system_config_path # cite: 1671

    # Create orchestrator
    app.state.orchestrator = Orchestrator(app.state.default_config) # cite: 1674
    logger.info("api.team_orchestration_initialized",
               teams=len(app.state.orchestrator.teams)) # cite: 1674

    # Align 'api.requests' logger (if necessary, see previous explanation)
    api_logger = logging.getLogger("api.requests")
    api_logger.setLevel(logging.DEBUG) # cite: 1676
    api_logger.propagate = True # cite: 1677

    async def run_workflow(request: WorkflowRequest):
        """
        Execute a team-based workflow with the provided configuration.
        Configuration from the request is merged with the default configuration.
        """
        try:
            # --- Start with a clean copy of the app's default config ---
            # This prevents mutation of the shared app.state.default_config
            config = deepcopy(app.state.default_config)

            # --- Merge system_config and app_config from the request ---
            # Note: WorkflowRequest model expects system_config and app_config separately,
            # but in the multi-config flow, they are already part of the single app_config.
            # For direct WorkflowRequest calls, we maintain this merge logic.
            if request.system_config:
                 config = deep_merge(config, request.system_config)
            if request.app_config:
                 config = deep_merge(config, request.app_config) # cite: 1679

            # Check if lineage file is provided for workflow continuation
            if request.lineage_file and request.stage:
                logger.info("workflow.continuing_from_lineage",
                            lineage_file=request.lineage_file, # cite: 1680
                            stage=request.stage,
                            keep_runid=request.keep_runid)

                try:
                    # Load lineage data
                    lineage_data = load_lineage_file(request.lineage_file) # cite: 1681

                    # Add intent to config for context preparation
                    if request.intent:
                        config['intent'] = request.intent

                    # Add project path to config if provided
                    if request.project_path:
                        if 'project' not in config:
                            config['project'] = {} # cite: 1683
                        config['project']['path'] = request.project_path

                    # Prepare context from lineage with keep_runid flag from request
                    context = prepare_context_from_lineage(
                         lineage_data, # cite: 1684
                         request.stage,
                         config,
                         keep_runid=request.keep_runid # cite: 1684
                    )

                    # Get workflow ID
                    workflow_id = context["workflow_run_id"] # cite: 1685

                    # --- Reload orchestrator with potentially modified config ---
                    # If config was updated (e.g., by prepare_context_from_lineage),
                    # ensure orchestrator uses the latest version for this execution.
                    current_orchestrator = Orchestrator(config)

                    # Execute workflow from the specified stage
                    result = current_orchestrator.execute_workflow(
                         entry_team=request.stage, # cite: 1686
                         context=context
                    )

                    # Store result
                    workflow_storage[workflow_id] = {
                         "status": result.get("status", "error"), # cite: 1687
                         "team_results": result.get("team_results", {}),
                         "changes": result.get("data", {}).get("changes", []), # cite: 1687
                         "storage_path": os.path.join("workspaces", "lineage", workflow_id) if config.get("runtime", {}).get("lineage", {}).get("enabled", False) else None, # cite: 1688 (logic adjusted for runtime)
                         "source_lineage": request.lineage_file, # cite: 1688
                         "stage": request.stage
                    }

                    return WorkflowResponse(
                        workflow_id=workflow_id, # cite: 1689
                        status=result.get("status", "error"),
                        storage_path=workflow_storage[workflow_id].get("storage_path"), # cite: 1689
                        error=result.get("error") if result.get("status") == "error" else None # cite: 1690
                    )

                except Exception as e:
                     logger.error("workflow.lineage_processing_failed",
                               lineage_file=request.lineage_file,
                               stage=request.stage, # cite: 1691
                               error=str(e))
                     raise HTTPException(status_code=500, detail=f"Lineage processing failed: {str(e)}")

            # --- Standard workflow initialization ---
            # Use the already merged config from the start of this function
            # Initialize workflow with consistent defaults using the merged config
            prepared_config, context = app.state.orchestrator.initialize_workflow( # Use the main orchestrator instance
                project_path=request.project_path,
                intent_desc=request.intent,
                config=config # Pass the config merged at the start
            )

            workflow_id = context["workflow_run_id"] # cite: 1693

            # Store intent in config (if not already there from merge)
            if 'intent' not in prepared_config:
                prepared_config['intent'] = request.intent

            # Log workflow start
            logger.info("workflow.starting",
                        workflow_id=workflow_id,
                        project_path=request.project_path, # cite: 1694
                        config_keys=list(prepared_config.keys())) # cite: 1694

            # --- Ensure Orchestrator uses the latest config for THIS execution ---
            # Pass the fully prepared config in the context
            # The orchestrator's execute_workflow method should ideally use the config from the context
            context["config"] = prepared_config

            try:
                # Get entry team from config or use default
                entry_team = prepared_config.get("orchestration", {}).get("entry_team", "discovery") # cite: 1695

                # Execute workflow using the main orchestrator instance but provide the up-to-date context/config
                result = app.state.orchestrator.execute_workflow(
                    entry_team=entry_team,
                    context=context # Pass context containing the prepared_config
                ) # cite: 1696

                # Store result
                workflow_storage[workflow_id] = {
                    "status": result.get("status", "error"),
                    "team_results": result.get("team_results", {}), # cite: 1697
                    "changes": result.get("data", {}).get("changes", []),
                    "storage_path": os.path.join("workspaces", "lineage", workflow_id) if prepared_config.get("runtime", {}).get("lineage", {}).get("enabled", False) else None # cite: 1697 (logic adjusted)
                }

                return WorkflowResponse(
                     workflow_id=workflow_id, # cite: 1698
                     status=result.get("status", "error"),
                     storage_path=workflow_storage[workflow_id].get("storage_path"), # cite: 1698
                     error=result.get("error") if result.get("status") == "error" else None
                )

            except Exception as e:
                logger.error("workflow.execution_failed",
                        workflow_id=workflow_id,
                        error=str(e)) # cite: 1699

                workflow_storage[workflow_id] = {
                     "status": "error", # cite: 1700
                     "error": str(e),
                     "storage_path": None
                }

                return WorkflowResponse(
                     workflow_id=workflow_id, # cite: 1701
                     status="error",
                     error=str(e)
                )

        except Exception as e:
            logger.error("workflow.request_failed", error=str(e)) # cite: 1701
            raise HTTPException(status_code=500, detail=str(e))

    async def get_workflow(workflow_id: str):
        """Get workflow status and results"""
        if workflow_id in workflow_storage:
            data = workflow_storage[workflow_id] # cite: 1702
            return WorkflowResponse(
                workflow_id=workflow_id,
                status=data.get("status", "unknown"),
                storage_path=data.get("storage_path"), # cite: 1703
                error=data.get("error") # cite: 1703
            )
        else:
            raise HTTPException(status_code=404, detail="Workflow not found")

    # --- API Endpoints ---

    @app.get("/health")
    async def health_check():
        """Simple health check endpoint"""
        return {
            "status": "healthy",
            "workflows_tracked": len(workflow_storage), # cite: 1704
            "teams_available": len(app.state.orchestrator.teams) # cite: 1704
        }


    @app.post("/api/v1/jobs", response_model=JobResponse)
    async def create_job(request: JobRequestUnion): # Use JobRequestUnion
        # ... (initialization code: job_id, merged_config, etc.) ...
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        merged_config = None
        project_path = None
        intent = None
        workflow_request = None

        try:
            # ... (code to handle request type: MultiConfigJobRequest or JobRequest) ...

            if isinstance(request, MultiConfigJobRequest):
                logger.info("jobs.multi_config_request_received",
                        job_id=job_id,
                        configs_count=len(request.configs))

                for i, frag in enumerate(request.configs):
                    logger.debug("jobs.config_fragment_received",
                                job_id=job_id,
                                fragment_index=i,
                                fragment_keys=list(frag.keys()))

                base_config = getattr(app.state, 'default_config', {})
                if not isinstance(base_config, dict):
                    logger.warning("Default config is not a dictionary, starting merge with empty dict.", job_id=job_id)
                    base_config = {}
                merged_config = base_config.copy()

                # --- CORRECTED MERGE ORDER ---
                for i, config_fragment in enumerate(request.configs):
                    logger.debug("jobs.merging_fragment", job_id=job_id, fragment_index=i, fragment_keys=list(config_fragment.keys()))
                    merged_config = deep_merge(merged_config, config_fragment)
                # --- END CORRECTION ---

                # --- REPLACE previous full structure log with calls to log_config_node ---
                log_config_node(logger, merged_config, "workorder", log_prefix=f"jobs.final_merged.{job_id}")
                log_config_node(logger, merged_config, "team.llm_config.agents.solution_designer", log_prefix=f"jobs.final_merged.{job_id}")
                # Optionally log the path the agent *expects* to find config at, for comparison:
                log_config_node(logger, merged_config, "llm_config.agents.solution_designer", log_prefix=f"jobs.final_merged.{job_id}")
                log_config_node(logger, merged_config, "team.llm_config.agents.discovery.tartxt_config", log_prefix=f"jobs.final_merged.{job_id}")
                # --- END REPLACEMENT ---

                logger.debug("jobs.configs_merged",
                        job_id=job_id,
                        merged_config_keys=list(merged_config.keys()))

                # ... (rest of the code for creating WorkflowRequest from MultiConfigJobRequest) ...
                try:
                    config_node = create_config_node(merged_config)
                    project_path = config_node.get_value("workorder.project.path")
                    intent = config_node.get_value("workorder.intent")
                    # ... (extract lineage/stage/keep_runid) ...
                    lineage_file = config_node.get_value("runtime.runtime.lineage_file")
                    stage = config_node.get_value("runtime.runtime.stage")
                    keep_runid_value = config_node.get_value("runtime.runtime.keep_runid")
                    keep_runid = keep_runid_value if isinstance(keep_runid_value, bool) else True

                    workflow_request = WorkflowRequest(
                        project_path=project_path,
                        intent=intent,
                        app_config=merged_config, # Pass the fully merged config
                        system_config=None,
                        lineage_file=lineage_file,
                        stage=stage,
                        keep_runid=keep_runid
                    )
                except Exception as e:
                    logger.error("jobs.workflow_request_creation_failed", job_id=job_id, error=str(e), error_type=type(e).__name__, exc_info=True)
                    raise HTTPException(status_code=400, detail=f"Invalid merged configuration ({type(e).__name__}): {str(e)}")


            elif isinstance(request, JobRequest):
                # ... (handling for traditional JobRequest - Ensure logging replacement happens here too if applicable) ...
                logger.info("jobs.traditional_request_received",
                        job_id=job_id, project_path=request.workorder.project.path,
                        has_team_config=request.team is not None, has_runtime_config=request.runtime is not None)
                try:
                    # map_job_to_workflow_request essentially performs the merge for the traditional request
                    workflow_request = map_job_to_workflow_request(request)
                    project_path = workflow_request.project_path
                    intent = workflow_request.intent
                    merged_config = workflow_request.app_config # Get the merged config resulting from mapping

                    # --- Log specific nodes for the traditional request's merged config ---
                    log_config_node(logger, merged_config, "workorder", log_prefix=f"jobs.mapped.{job_id}")
                    log_config_node(logger, merged_config, "team.llm_config.agents.solution_designer", log_prefix=f"jobs.mapped.{job_id}")
                    log_config_node(logger, merged_config, "llm_config.agents.solution_designer", log_prefix=f"jobs.mapped.{job_id}")
                    # --- End logging ---

                    logger.debug("jobs.request_mapped", job_id=job_id, workflow_request_keys=list(workflow_request.dict().keys()))
                except Exception as e:
                    logger.error("jobs.request_mapping_failed", job_id=job_id, error=str(e))
                    raise HTTPException(status_code=400, detail=f"Invalid job configuration: {str(e)}")
            else:
                logger.error("jobs.unknown_request_type", job_id=job_id, request_type=type(request).__name__)
                raise HTTPException(status_code=400, detail="Unknown job request format")

            # ... (rest of the function: pre_run_workflow_check, run_workflow call, response handling, job storage) ...
            logger.debug("jobs.pre_run_workflow_check",
                        job_id=job_id,
                        wf_request_type=type(workflow_request).__name__,
                        wf_req_project_path=getattr(workflow_request, 'project_path', 'N/A'),
                        wf_req_intent_type=type(getattr(workflow_request, 'intent', None)).__name__,
                        wf_req_app_config_keys=list(getattr(workflow_request, 'app_config', {}).keys()))

            try:
                workflow_response = await run_workflow(workflow_request)
                logger.info("jobs.workflow_executed",
                        job_id=job_id, workflow_id=workflow_response.workflow_id,
                        status=workflow_response.status)
            except Exception as e:
                logger.error("jobs.workflow_execution_failed", job_id=job_id, error=str(e))
                error_detail = f"Workflow execution failed: {str(e)}"
                raise HTTPException(status_code=500, detail=error_detail)

            job_response = JobResponse(
                job_id=job_id,
                status=workflow_response.status,
                storage_path=workflow_response.storage_path,
                error=workflow_response.error
            )

            final_project_path = project_path
            if final_project_path is None:
                logger.warning("Could not determine project_path for final job storage", job_id=job_id)

            job_storage[job_id] = {
                "status": workflow_response.status,
                "storage_path": workflow_response.storage_path,
                "error": workflow_response.error,
                "created_at": datetime.now().isoformat(),
                "workflow_id": workflow_response.workflow_id,
                "project_path": final_project_path,
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
            logger.error("jobs.creation_failed", job_id=job_id, error=str(e), error_type=type(e).__name__)
            logger.error(f"Top-level exception in create_job {job_id}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Job creation failed: {str(e)}")


    @app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
    async def get_job_status(job_id: str):
        """
        Get status of a job.
        Retrieves the workflow status and maps it to job status format.
        Args:
            job_id: Unique identifier for the job

        Returns:
            JobStatus with current status, changes, and error information
        """
        try:
            # Check if job exists
            if job_id not in job_storage:
                logger.error("jobs.not_found", job_id=job_id) # cite: 1740
                raise HTTPException(status_code=404, detail="Job not found")

            # Get workflow ID from mapping
            workflow_id = job_to_workflow_map.get(job_id) # cite: 1740
            if not workflow_id:
                logger.error("jobs.workflow_mapping_missing", job_id=job_id)
                raise HTTPException(status_code=404, detail="No workflow found for this job") # cite: 1741

            logger.info("jobs.status_request",
                    job_id=job_id,
                    workflow_id=workflow_id) # cite: 1741

            # Get workflow status
            workflow_data = workflow_storage.get(workflow_id, {}) # cite: 1742

            # If workflow data not in memory storage, try to get from API
            if not workflow_data:
                logger.debug("jobs.workflow_data_not_in_memory",
                        job_id=job_id,
                        workflow_id=workflow_id) # cite: 1742

                try:
                    # Get fresh data from workflow endpoint
                    workflow_response = await get_workflow(workflow_id) # cite: 1743
                    workflow_data = {
                         "status": workflow_response.status, # cite: 1744
                         "storage_path": workflow_response.storage_path,
                         "error": workflow_response.error
                    }
                    logger.debug("jobs.workflow_data_retrieved", # cite: 1745
                            job_id=job_id,
                            workflow_id=workflow_id,
                            status=workflow_response.status) # cite: 1745

                except Exception as e:
                    logger.error("jobs.workflow_status_fetch_failed",
                            job_id=job_id,
                            workflow_id=workflow_id,
                            error=str(e)) # cite: 1747
                    # Fall back to stored job data
                    workflow_data = job_storage[job_id] # cite: 1747
                    logger.debug("jobs.using_stored_job_data",
                             job_id=job_id, # cite: 1748
                             last_updated=workflow_data.get("last_updated")) # cite: 1748

            # Map workflow changes to job changes format
            changes = map_workflow_to_job_changes(workflow_data) # cite: 1748

            # Update job storage with latest status
            job_storage[job_id].update({
                 "status": workflow_data.get("status"), # cite: 1749
                 "storage_path": workflow_data.get("storage_path"),
                 "error": workflow_data.get("error"),
                 "last_checked": datetime.now().isoformat(),
                 "changes": changes # cite: 1749
            })

            # Create job status response
            job_status = JobStatus(
                job_id=job_id, # cite: 1750
                status=workflow_data.get("status", "unknown"), # cite: 1750
                storage_path=workflow_data.get("storage_path"), # cite: 1750
                error=workflow_data.get("error"), # cite: 1750
                changes=changes # cite: 1751
            )

            logger.info("jobs.status_checked",
                    job_id=job_id,
                    workflow_id=workflow_id,
                    status=job_status.status,
                    changes_count=len(changes) if changes else 0) # cite: 1752

            return job_status

        except HTTPException:
            # Re-raise HTTP exceptions without wrapping
            raise
        except Exception as e:
            logger.error("jobs.status_check_failed",
                     job_id=job_id, # cite: 1753
                     error=str(e),
                     error_type=type(e).__name__)
            raise HTTPException(status_code=500, detail=f"Job status check failed: {str(e)}")

    @app.post("/api/v1/configs/merge", response_model=MergeResponse)
    async def merge_configs(request: MergeRequest):
        """
        Utility endpoint to merge multiple configuration dictionaries.
        Merges configurations from right to left (rightmost has lowest priority).
        Optionally includes system_config as the base configuration.
        Args:
            request: MergeRequest containing configs to merge

        Returns:
            MergeResponse with the merged configuration
        """
        try:
            # Start with system config if requested
            if request.include_system_config: # cite: 1755
                # Try to load system config from disk
                try:
                    system_config = load_config(app.state.system_config_path) # cite: 1756
                    merged_config = system_config.copy()
                    logger.debug("configs.merge.using_system_config",
                                system_config_keys=list(system_config.keys())) # cite: 1757
                except Exception as e:
                    logger.warning("configs.merge.system_config_load_failed",
                            error=str(e)) # cite: 1757
                    # Fall back to default config
                    merged_config = app.state.default_config.copy() # cite: 1758
            else:
                # Start with empty dict
                merged_config = {}

            # Merge configurations from right to left
            for i, config in enumerate(reversed(request.configs)): # cite: 1759
                 merged_config = deep_merge(merged_config, config)
                 logger.debug(f"configs.merge.step_{i+1}", config_keys=list(config.keys())) # cite: 1759

            return MergeResponse(merged_config=merged_config) # cite: 1801

        except Exception as e:
            logger.error("configs.merge.failed", error=str(e)) # cite: 1760
            raise HTTPException(status_code=500, detail=f"Configuration merge failed: {str(e)}")


    return app

# Default app instance for direct imports
app = create_app()