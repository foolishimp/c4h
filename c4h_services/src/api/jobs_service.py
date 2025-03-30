"""
Jobs API service implementation.
Path: c4h_services/src/api/jobs_service.py
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from c4h_services.src.utils.logging import get_logger
import requests
import os

from c4h_services.src.api.jobs_models import JobRequest, JobResponse, JobStatus
from c4h_services.src.api.jobs_mapper import (
    job_to_workflow_request,
    workflow_to_job_response,
    workflow_to_job_status
)
from c4h_services.src.api.jobs_storage import JobsStorage

logger = get_logger()

# Initialize router
router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

# Initialize storage
storage = JobsStorage()

# Determine workflow API URL
workflow_api_url = os.environ.get("WORKFLOW_API_URL", "http://localhost:8000")

@router.post("", response_model=JobResponse)
async def create_job(job_request: JobRequest):
    """
    Submit a new code refactoring job.
    
    Args:
        job_request: Job configuration including workorder, team, and runtime
        
    Returns:
        Job submission response with job ID and status
    """
    try:
        # Convert job request to workflow request
        workflow_request = job_to_workflow_request(job_request.dict())
        
        logger.info("jobs_service.converting_request", 
                  project_path=workflow_request.get("project_path"),
                  has_intent=bool(workflow_request.get("intent")))
        
        # Send request to workflow API
        workflow_url = f"{workflow_api_url}/api/v1/workflow"
        response = requests.post(workflow_url, json=workflow_request)
        
        # Check for errors
        if response.status_code != 200:
            logger.error("jobs_service.workflow_api_error",
                       status_code=response.status_code,
                       response=response.text)
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Workflow API error: {response.text}"
            )
            
        # Parse workflow response
        workflow_response = response.json()
        
        # Store job-to-workflow mapping
        workflow_id = workflow_response.get("workflow_id")
        job_id = storage.create_job(workflow_id)
        
        # Convert workflow response to job response
        job_response = workflow_to_job_response(workflow_response, job_id)
        
        logger.info("jobs_service.job_created",
                  job_id=job_id,
                  workflow_id=workflow_id)
                  
        return job_response
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error("jobs_service.create_job_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create job: {str(e)}"
        )

@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Get the status of a job.
    
    Args:
        job_id: ID of the job to check
        
    Returns:
        Current job status including any file changes
    """
    try:
        # Get the workflow ID for this job
        workflow_id = storage.get_workflow_id(job_id)
        
        if not workflow_id:
            logger.error("jobs_service.job_not_found", job_id=job_id)
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )
            
        # Get status from workflow API
        workflow_url = f"{workflow_api_url}/api/v1/workflow/{workflow_id}"
        response = requests.get(workflow_url)
        
        # Check for errors
        if response.status_code != 200:
            logger.error("jobs_service.workflow_status_error",
                       status_code=response.status_code,
                       response=response.text)
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Workflow API error: {response.text}"
            )
            
        # Parse workflow response
        workflow_response = response.json()
        
        # Convert workflow response to job status
        job_status = workflow_to_job_status(workflow_response, job_id)
        
        logger.info("jobs_service.job_status_retrieved",
                  job_id=job_id,
                  workflow_id=workflow_id,
                  status=job_status.get("status"))
                  
        return job_status
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error("jobs_service.get_job_status_failed", 
                   job_id=job_id,
                   error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )