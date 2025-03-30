"""
Mapper for transforming between Jobs API and Workflow API formats.
Path: c4h_services/src/api/jobs_mapper.py
"""

from typing import Dict, Any, Optional
from c4h_services.src.utils.logging import get_logger
from copy import deepcopy

logger = get_logger()

def job_to_workflow_request(job_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a Jobs API request to a Workflow API request.
    
    Args:
        job_request: Jobs API request format
        
    Returns:
        Dict in Workflow API request format
    """
    try:
        # Extract workorder config
        workorder = job_request.get("workorder", {})
        project = workorder.get("project", {})
        intent = workorder.get("intent", {})
        
        # Prepare app_config from team and runtime sections
        app_config = {}
        
        # Map project config
        if project:
            app_config["project"] = deepcopy(project)
        
        # Map team config (llm_config and orchestration)
        team = job_request.get("team", {})
        if team:
            if "llm_config" in team:
                app_config["llm_config"] = deepcopy(team["llm_config"])
            if "orchestration" in team:
                app_config["orchestration"] = deepcopy(team["orchestration"])
        
        # Map runtime config
        runtime = job_request.get("runtime", {})
        if runtime:
            for key, value in runtime.items():
                app_config[key] = deepcopy(value)
        
        # Construct workflow request
        workflow_request = {
            "project_path": project.get("path", ""),
            "intent": deepcopy(intent),
            "app_config": app_config
        }
        
        logger.debug("jobs_mapper.job_to_workflow", 
                    source_keys=list(job_request.keys()),
                    result_keys=list(workflow_request.keys()))
        
        return workflow_request
        
    except Exception as e:
        logger.error("jobs_mapper.job_to_workflow_failed", error=str(e))
        raise

def workflow_to_job_response(workflow_response: Dict[str, Any], job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Transform a Workflow API response to a Jobs API response.
    
    Args:
        workflow_response: Workflow API response format
        job_id: Optional override for the job ID
        
    Returns:
        Dict in Jobs API response format
    """
    try:
        # Extract workflow response fields
        workflow_id = workflow_response.get("workflow_id", "")
        status = workflow_response.get("status", "error")
        storage_path = workflow_response.get("storage_path")
        error = workflow_response.get("error")
        
        # Map to job response
        job_response = {
            "job_id": job_id or workflow_id,  # Use provided job_id if available
            "status": status,
            "storage_path": storage_path,
            "error": error
        }
        
        logger.debug("jobs_mapper.workflow_to_job_response", 
                    source_keys=list(workflow_response.keys()),
                    result_keys=list(job_response.keys()))
        
        return job_response
        
    except Exception as e:
        logger.error("jobs_mapper.workflow_to_job_response_failed", error=str(e))
        raise

def workflow_to_job_status(workflow_response: Dict[str, Any], job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Transform a Workflow API status response to a Jobs API status response.
    Includes detailed change information.
    
    Args:
        workflow_response: Workflow API status response
        job_id: Optional override for the job ID
        
    Returns:
        Dict in Jobs API status format
    """
    try:
        # Start with basic response mapping
        job_status = workflow_to_job_response(workflow_response, job_id)
        
        # Extract and transform changes if available
        changes = []
        if "changes" in workflow_response:
            for change in workflow_response["changes"]:
                job_change = {
                    "file": change.get("file", ""),
                    "success": change.get("success", True),
                    "error": change.get("error"),
                    "backup": change.get("backup")
                }
                changes.append(job_change)
        
        job_status["changes"] = changes
        
        logger.debug("jobs_mapper.workflow_to_job_status", 
                    has_changes=bool(changes),
                    change_count=len(changes))
        
        return job_status
        
    except Exception as e:
        logger.error("jobs_mapper.workflow_to_job_status_failed", error=str(e))
        raise