"""
API service implementation with enhanced configuration handling and multi-config support.
Path: c4h_services/src/api/service.py
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional, Callable, List
from c4h_services.src.utils.logging import get_logger
from pathlib import Path
import uuid
import os
import json
import uuid
from datetime import datetime

from c4h_agents.config import deep_merge, load_config
from c4h_agents.config import create_config_node
from c4h_agents.core.project import Project
from c4h_services.src.api.models import (WorkflowRequest, WorkflowResponse, 
                                        JobRequest, JobResponse, JobStatus,
                                        MultiConfigJobRequest, MergeRequest, MergeResponse, JobRequestUnion)
from c4h_services.src.orchestration.orchestrator import Orchestrator
from c4h_services.src.utils.lineage_utils import load_lineage_file, prepare_context_from_lineage

logger = get_logger()

# In-memory storage for workflow and job results (for demonstration purposes)
# In production, this would be a database or persistent storage
workflow_storage: Dict[str, Dict[str, Any]] = {}
job_storage: Dict[str, Dict[str, Any]] = {}
job_to_workflow_map: Dict[str, str] = {}
system_config_path = Path("config/system_config.yml")

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
    app.state.system_config_path = system_config_path
    
    # Create orchestrator
    app.state.orchestrator = Orchestrator(app.state.default_config)
    logger.info("api.team_orchestration_initialized", 
               teams=len(app.state.orchestrator.teams))
    
    # Register routes
    @app.post("/api/v1/workflow", response_model=WorkflowResponse)
    async def run_workflow(request: WorkflowRequest):
        """
        Execute a team-based workflow with the provided configuration.
        Configuration from the request is merged with the default configuration.
        """
        try:
            # Merge default config with request configs
            config = deep_merge(app.state.default_config, request.system_config or {})
            if request.app_config:
                config = deep_merge(config, request.app_config)
            
            # Check if lineage file is provided for workflow continuation
            if request.lineage_file and request.stage:
                logger.info("workflow.continuing_from_lineage",
                           lineage_file=request.lineage_file,
                           stage=request.stage,
                           keep_runid=request.keep_runid)
                           
                try:
                    # Load lineage data
                    lineage_data = load_lineage_file(request.lineage_file)
                    
                    # Add intent to config for context preparation
                    if request.intent:
                        config['intent'] = request.intent
                        
                    # Add project path to config if provided
                    if request.project_path:
                        if 'project' not in config:
                            config['project'] = {}
                        config['project']['path'] = request.project_path
                    
                    # Prepare context from lineage with keep_runid flag from request
                    context = prepare_context_from_lineage(
                        lineage_data, 
                        request.stage, 
                        config,
                        keep_runid=request.keep_runid
                    )
                    
                    # Get workflow ID
                    workflow_id = context["workflow_run_id"]
                    
                    # Execute workflow from the specified stage
                    result = app.state.orchestrator.execute_workflow(
                        entry_team=request.stage,
                        context=context
                    )
                    
                    # Store result
                    workflow_storage[workflow_id] = {
                        "status": result.get("status", "error"),
                        "team_results": result.get("team_results", {}),
                        "changes": result.get("data", {}).get("changes", []),
                        "storage_path": os.path.join("workspaces", "lineage", workflow_id) if config.get("lineage", {}).get("enabled", False) else None,
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
                               error=str(e))
                    raise HTTPException(status_code=500, detail=f"Lineage processing failed: {str(e)}")
            
            # Standard workflow initialization
            # Initialize workflow with consistent defaults
            prepared_config, context = app.state.orchestrator.initialize_workflow(
                project_path=request.project_path,
                intent_desc=request.intent,
                config=config
            )
            
            workflow_id = context["workflow_run_id"]
                
            # Store intent in config
            prepared_config['intent'] = request.intent
            
            # Log workflow start
            logger.info("workflow.starting", 
                        workflow_id=workflow_id, 
                        project_path=request.project_path,
                        config_keys=list(prepared_config.keys()))
            
            # Execute workflow - pass the fully prepared config in the context
            # This is critical to ensure the orchestrator uses the current config
            context["config"] = prepared_config
            
            try:
                # Get entry team from config or use default
                entry_team = prepared_config.get("orchestration", {}).get("entry_team", "discovery")
                
                result = app.state.orchestrator.execute_workflow(
                    entry_team=entry_team,
                    context=context
                )
                
                # Store result
                workflow_storage[workflow_id] = {
                    "status": result.get("status", "error"),
                    "team_results": result.get("team_results", {}),
                    "changes": result.get("data", {}).get("changes", []),
                    "storage_path": os.path.join("workspaces", "lineage", workflow_id) if prepared_config.get("lineage", {}).get("enabled", False) else None
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
                        error=str(e))
                
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
            logger.error("workflow.request_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


    @app.get("/api/v1/workflow/{workflow_id}", response_model=WorkflowResponse)
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
            
    @app.get("/health")
    async def health_check():
        """Simple health check endpoint"""
        return {
            "status": "healthy", 
            "workflows_tracked": len(workflow_storage),
            "teams_available": len(app.state.orchestrator.teams)
        }

    """
    Enhanced Jobs API endpoints for the C4H Agent System.
    These endpoints should replace the existing ones in service.py.
    """
    
    @app.post("/api/v1/jobs", response_model=JobResponse)
    async def create_job(request: JobRequestUnion):
        """
        Create a new job with structured configuration.
        Maps the job request to workflow request format and executes the workflow.
        
        Supports both traditional JobRequest format and new MultiConfigJobRequest format.
        Job Request Structure:
        - workorder: Contains project and intent information
        - team: Contains LLM and orchestration configuration
        - runtime: Contains runtime settings and environment config
        
        Args:
            request: JobRequest or MultiConfigJobRequest containing configuration
            
        Returns:
            JobResponse with job ID and status information
        """
        try:
            # Generate a job ID with UUID and timestamp for traceability
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = f"job_{timestamp}_{str(uuid.uuid4())[:8]}"
            
            # Handle different request formats
            if isinstance(request, MultiConfigJobRequest):
                logger.info("jobs.multi_config_request_received", 
                        job_id=job_id, 
                        configs_count=len(request.configs))
                
                # Merge configurations from right to left (rightmost has lowest priority)
                merged_config = app.state.default_config.copy()
                for config in reversed(request.configs):
                    merged_config = deep_merge(merged_config, config)
                    
                logger.debug("jobs.configs_merged", 
                        job_id=job_id,
                        merged_config_keys=list(merged_config.keys()))
                
                # Create a workflow request directly from the merged config
                try:
                    # Extract project path from merged config
                    config_node = create_config_node(merged_config)
                    project_path = config_node.get_value("workorder.project.path")
                    intent = config_node.get_value("workorder.intent")
                    
                    if not project_path:
                        raise ValueError("Missing required workorder.project.path in merged configuration")
                    if not intent:
                        raise ValueError("Missing required workorder.intent in merged configuration")
                        
                    workflow_request = WorkflowRequest(
                        project_path=project_path,
                        intent=intent,
                        app_config=merged_config
                    )
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid merged configuration: {str(e)}")
            else:
                # Traditional JobRequest format
                logger.info("jobs.traditional_request_received", 
                        job_id=job_id, 
                        project_path=request.workorder.project.path,
                        has_team_config=request.team is not None,
                        has_runtime_config=request.runtime is not None)
                
                # Map job request to workflow request format
                try:
                    workflow_request = map_job_to_workflow_request(request)
                    logger.debug("jobs.request_mapped", 
                            job_id=job_id, 
                            workflow_request_keys=list(workflow_request.dict().keys()))
                except Exception as e:
                    logger.error("jobs.request_mapping_failed", 
                            job_id=job_id, 
                            error=str(e))
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid job configuration: {str(e)}"
                    )
            
            # Call the workflow endpoint with the mapped request
            try:
                workflow_response = await run_workflow(workflow_request)
                logger.info("jobs.workflow_executed", 
                        job_id=job_id, 
                        workflow_id=workflow_response.workflow_id,
                        status=workflow_response.status)
            except Exception as e:
                logger.error("jobs.workflow_execution_failed", 
                        job_id=job_id, 
                        error=str(e))
                raise HTTPException(
                    status_code=500, 
                    detail=f"Workflow execution failed: {str(e)}"
                )
            
            # Map the workflow response to a job response
            job_response = JobResponse(
                job_id=job_id,
                status=workflow_response.status,
                storage_path=workflow_response.storage_path,
                error=workflow_response.error
            )
            
            # Store job information with more context
            job_storage[job_id] = {
                "status": workflow_response.status,
                "storage_path": workflow_response.storage_path,
                "error": workflow_response.error,
                "created_at": datetime.now().isoformat(),
                "workflow_id": workflow_response.workflow_id,
                "project_path": request.workorder.project.path,
                "last_updated": datetime.now().isoformat()
            }
            
            # Store job to workflow mapping
            job_to_workflow_map[job_id] = workflow_response.workflow_id
            
            logger.info("jobs.created", 
                    job_id=job_id, 
                    workflow_id=workflow_response.workflow_id,
                    status=workflow_response.status)
            
            return job_response
            
        except HTTPException:
            # Re-raise HTTP exceptions without wrapping
            raise
        except Exception as e:
            logger.error("jobs.creation_failed", 
                    error=str(e),
                    error_type=type(e).__name__)
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
                logger.error("jobs.not_found", job_id=job_id)
                raise HTTPException(status_code=404, detail="Job not found")
                
            # Get workflow ID from mapping
            workflow_id = job_to_workflow_map.get(job_id)
            if not workflow_id:
                logger.error("jobs.workflow_mapping_missing", job_id=job_id)
                raise HTTPException(status_code=404, detail="No workflow found for this job")
            
            logger.info("jobs.status_request", 
                    job_id=job_id, 
                    workflow_id=workflow_id)
            
            # Get workflow status
            workflow_data = workflow_storage.get(workflow_id, {})
            
            # If workflow data not in memory storage, try to get from API
            if not workflow_data:
                logger.debug("jobs.workflow_data_not_in_memory", 
                        job_id=job_id, 
                        workflow_id=workflow_id)
                
                try:
                    # Get fresh data from workflow endpoint
                    workflow_response = await get_workflow(workflow_id)
                    workflow_data = {
                        "status": workflow_response.status,
                        "storage_path": workflow_response.storage_path,
                        "error": workflow_response.error
                    }
                    logger.debug("jobs.workflow_data_retrieved", 
                            job_id=job_id,
                            workflow_id=workflow_id,
                            status=workflow_response.status)
                except Exception as e:
                    logger.error("jobs.workflow_status_fetch_failed", 
                            job_id=job_id, 
                            workflow_id=workflow_id, 
                            error=str(e))
                    # Fall back to stored job data
                    workflow_data = job_storage[job_id]
                    logger.debug("jobs.using_stored_job_data",
                            job_id=job_id,
                            last_updated=workflow_data.get("last_updated"))
            
            # Map workflow changes to job changes format
            changes = map_workflow_to_job_changes(workflow_data)
            
            # Update job storage with latest status
            job_storage[job_id].update({
                "status": workflow_data.get("status"),
                "storage_path": workflow_data.get("storage_path"),
                "error": workflow_data.get("error"),
                "last_checked": datetime.now().isoformat(),
                "changes": changes
            })
            
            # Create job status response
            job_status = JobStatus(
                job_id=job_id,
                status=workflow_data.get("status", "unknown"),
                storage_path=workflow_data.get("storage_path"),
                error=workflow_data.get("error"),
                changes=changes
            )
            
            logger.info("jobs.status_checked", 
                    job_id=job_id, 
                    workflow_id=workflow_id,
                    status=job_status.status,
                    changes_count=len(changes) if changes else 0)
            
            return job_status
            
        except HTTPException:
            # Re-raise HTTP exceptions without wrapping
            raise
        except Exception as e:
            logger.error("jobs.status_check_failed", 
                    job_id=job_id, 
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
            if request.include_system_config:
                # Try to load system config from disk
                try:
                    system_config = load_config(app.state.system_config_path)
                    merged_config = system_config.copy()
                    logger.debug("configs.merge.using_system_config", 
                            system_config_keys=list(system_config.keys()))
                except Exception as e:
                    logger.warning("configs.merge.system_config_load_failed", 
                            error=str(e))
                    # Fall back to default config
                    merged_config = app.state.default_config.copy()
            else:
                # Start with empty dict
                merged_config = {}
            
            # Merge configurations from right to left
            for i, config in enumerate(reversed(request.configs)):
                merged_config = deep_merge(merged_config, config)
                logger.debug(f"configs.merge.step_{i+1}", config_keys=list(config.keys()))
            
            return MergeResponse(merged_config=merged_config)
            
        except Exception as e:
            logger.error("configs.merge.failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Configuration merge failed: {str(e)}")

    """
    Enhanced Jobs API functions for the C4H Agent System.
    These functions should be added to service.py.
    """

    def map_job_to_workflow_request(job_request: JobRequest) -> WorkflowRequest:
        """
        Map JobRequest to WorkflowRequest format.
        Transforms the structured job configuration to flat workflow configuration.
        """
        try:
            # Get project path from workorder
            project_path = job_request.workorder.project.path
            
            # Get intent from workorder - convert to dictionary with exclude_none
            intent_dict = job_request.workorder.intent.dict(exclude_none=True)
            
            # Initialize app_config with project settings
            app_config = {
                "project": job_request.workorder.project.dict(exclude_none=True),
                "intent": intent_dict
            }
            
            # Track what's being extracted for logging
            extracted_sections = ["workorder.project", "workorder.intent"]
            
            # Add team configuration if provided
            if job_request.team:
                team_dict = job_request.team.dict(exclude_none=True)
                for key, value in team_dict.items():
                    if value:
                        app_config[key] = value
                        extracted_sections.append(f"team.{key}")
            
            # Add runtime configuration if provided
            if job_request.runtime:
                runtime_dict = job_request.runtime.dict(exclude_none=True)
                for key, value in runtime_dict.items():
                    if value:
                        app_config[key] = value
                        extracted_sections.append(f"runtime.{key}")
            
            # Extract lineage information from runtime if available
            lineage_file = None
            stage = None
            keep_runid = True
            
            if job_request.runtime and job_request.runtime.runtime:
                runtime_config = job_request.runtime.runtime
                if isinstance(runtime_config, dict):
                    lineage_file = runtime_config.get("lineage_file")
                    stage = runtime_config.get("stage")
                    keep_runid = runtime_config.get("keep_runid", True)
                    
                    if lineage_file:
                        extracted_sections.append("runtime.runtime.lineage_file")
                    if stage:
                        extracted_sections.append("runtime.runtime.stage")
                    if "keep_runid" in runtime_config:
                        extracted_sections.append("runtime.runtime.keep_runid")
            
            # Create workflow request with all parameters
            workflow_request = WorkflowRequest(
                project_path=project_path,
                intent=intent_dict,
                app_config=app_config,
                lineage_file=lineage_file,
                stage=stage,
                keep_runid=keep_runid
            )
            
            logger.debug("jobs.mapping.job_to_workflow", 
                    project_path=project_path,
                    extracted_sections=extracted_sections,
                    app_config_keys=list(app_config.keys()),
                    lineage_file=lineage_file,
                    stage=stage)
            
            return workflow_request
            
        except Exception as e:
            logger.error("jobs.mapping.failed", 
                    error=str(e),
                    error_type=type(e).__name__)
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
                changes = workflow_data['changes']
                logger.debug("jobs.changes_found_direct", count=len(changes))
            
            # Check in data field
            elif 'data' in workflow_data and 'changes' in workflow_data['data']:
                changes = workflow_data['data']['changes']
                logger.debug("jobs.changes_found_in_data", count=len(changes))
            
            # Check in team_results.coder.data
            elif 'team_results' in workflow_data and 'coder' in workflow_data['team_results']:
                coder_result = workflow_data['team_results']['coder']
                if 'data' in coder_result and 'changes' in coder_result['data']:
                    changes = coder_result['data']['changes']
                    logger.debug("jobs.changes_found_in_coder", count=len(changes))
            
            # If no changes found
            if not changes:
                logger.warning("jobs.no_changes_found", 
                            workflow_data_keys=list(workflow_data.keys()))
                return []
            
            # Format changes for job response
            formatted_changes = []
            for change in changes:
                # Handle different change formats
                if isinstance(change, dict):
                    formatted_change = {}
                    
                    # Extract file path - check different field names
                    if 'file' in change:
                        formatted_change['file'] = change['file']
                    elif 'file_path' in change:
                        formatted_change['file'] = change['file_path']
                    elif 'path' in change:
                        formatted_change['file'] = change['path']
                    else:
                        # Skip changes without file information
                        continue
                    
                    # Extract change type information
                    if 'change' in change:
                        formatted_change['change'] = change['change']
                    elif 'type' in change:
                        formatted_change['change'] = {'type': change['type']}
                    elif 'success' in change:
                        # For simple success/error format
                        status = 'success' if change['success'] else 'error'
                        formatted_change['change'] = {'status': status}
                        if 'error' in change and change['error']:
                            formatted_change['change']['error'] = change['error']
                    
                    formatted_changes.append(formatted_change)
            
            logger.info("jobs.changes_mapped", 
                    original_count=len(changes), 
                    formatted_count=len(formatted_changes))
            
            return formatted_changes
        except Exception as e:
            logger.error("jobs.mapping.changes_failed", 
                    error=str(e),
                    error_type=type(e).__name__)
            return []


    return app

# Default app instance for direct imports
app = create_app()