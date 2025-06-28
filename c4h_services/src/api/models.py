"""
API request and response models for workflow service with enhanced configuration handling.
Path: c4h_services/src/api/models.py
"""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, Optional, List, Literal

class WorkflowRequest(BaseModel):
    """
    Request model for workflow execution.
    Contains all necessary configuration for the workflow to run.
    """
    project_path: str = Field(..., description="Path to the project to be processed")
    intent: Dict[str, Any] = Field(..., description="Intent description for the workflow")
    system_config: Optional[Dict[str, Any]] = Field(default=None, description="Base system configuration")
    app_config: Optional[Dict[str, Any]] = Field(default=None, description="Application-specific configuration overrides")
    lineage_file: Optional[str] = Field(default=None, description="Path to lineage file for workflow continuation")
    stage: Optional[Literal["discovery", "solution_designer", "coder"]] = Field(
        default=None, 
        description="Stage to continue workflow from when using lineage file"
    )
    keep_runid: Optional[bool] = Field(
        default=True,
        description="Whether to keep the original run ID from the lineage file (default: True)"
    )

class WorkflowResponse(BaseModel):
    """
    Response model for workflow operations.
    Provides workflow ID and status information.
    """
    workflow_id: str = Field(..., description="Unique identifier for the workflow")
    status: str = Field(..., description="Current status of the workflow")
    storage_path: Optional[str] = Field(default=None, description="Path to stored results if available")
    error: Optional[str] = Field(default=None, description="Error message if status is 'error'")

class WorkflowDetail(BaseModel):
    """
    Detailed workflow information model.
    Used for returning complete workflow execution details.
    """
    status: str = Field(..., description="Current status of the workflow")
    stages: Dict[str, Any] = Field(default_factory=dict, description="Results from each workflow stage")
    changes: List[Dict[str, Any]] = Field(default_factory=list, description="List of changes made by the workflow")
    events: List[Dict[str, Any]] = Field(default_factory=list, description="Execution events in chronological order")
    storage_path: Optional[str] = Field(default=None, description="Path to stored results if available")
    error: Optional[str] = Field(default=None, description="Error message if status is 'error'")
    execution_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Execution metadata and tracking info")
    source_lineage: Optional[str] = Field(default=None, description="Source lineage file if workflow was continued")
    stage: Optional[str] = Field(default=None, description="Stage that the workflow continued from if applicable")


# Jobs API Models

class JobResponse(BaseModel):
    """Response model for job operations"""
    job_id: str = Field(..., description="Unique identifier for the job")
    status: str = Field(..., description="Current status of the job")
    storage_path: Optional[str] = Field(default=None, description="Path where job results are stored")
    error: Optional[str] = Field(default=None, description="Error message if status is error")

class JobStatus(BaseModel):
    """Detailed job status information"""
    job_id: str = Field(..., description="Unique identifier for the job")
    status: str = Field(..., description="Current status of the job")
    storage_path: Optional[str] = Field(default=None, description="Path where job results are stored")
    error: Optional[str] = Field(default=None, description="Error message if status is error")
    changes: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of changes made by the job")

class MultiConfigJobRequest(BaseModel):
    """Job request model supporting multiple configuration dictionaries for server-side merging"""
    configs: List[Dict[str, Any]] = Field(..., description="List of configuration dictionaries to merge (right-to-left priority)")
    
    @model_validator(mode='after')
    def validate_configs(self) -> 'MultiConfigJobRequest':
        """Validate that configs is a non-empty list"""
        if not self.configs:
            raise ValueError("configs list cannot be empty")
        return self

class MergeRequest(BaseModel):
    """Request model for configuration merge utility endpoint"""
    configs: List[Dict[str, Any]] = Field(..., description="List of configuration dictionaries to merge (right-to-left priority)")
    include_system_config: bool = Field(default=True, description="Whether to include system_config as base")
    
    @model_validator(mode='after')
    def validate_configs(self) -> 'MergeRequest':
        """Validate that configs is a non-empty list"""
        if not self.configs:
            raise ValueError("configs list cannot be empty")
        return self

class MergeResponse(BaseModel):
    """Response model for configuration merge utility endpoint"""
    merged_config: Dict[str, Any] = Field(..., description="Resulting merged configuration")