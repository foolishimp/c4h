"""
Jobs API request and response models.
Path: c4h_services/src/api/jobs_models.py
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

# Request Models

class ProjectConfig(BaseModel):
    """Project configuration for a job."""
    path: str = Field(..., description="Path to the project directory")
    workspace_root: Optional[str] = Field(default=None, description="Directory for working files")
    source_root: Optional[str] = Field(default=None, description="Base directory for source code")
    output_root: Optional[str] = Field(default=None, description="Base directory for output files")

class IntentConfig(BaseModel):
    """Intent configuration for a job."""
    description: str = Field(..., description="Description of the refactoring intent")
    target_files: Optional[List[str]] = Field(default=None, description="Optional list of specific files to target")

class WorkorderConfig(BaseModel):
    """Workorder configuration that specifies the project and intent."""
    project: ProjectConfig
    intent: IntentConfig

class TeamConfig(BaseModel):
    """Team configuration that includes LLM and orchestration settings."""
    llm_config: Optional[Dict[str, Any]] = Field(default=None, description="LLM configuration")
    orchestration: Optional[Dict[str, Any]] = Field(default=None, description="Orchestration configuration")

class RuntimeConfig(BaseModel):
    """Runtime configuration that includes workflow, lineage, logging, and backup settings."""
    runtime: Optional[Dict[str, Any]] = Field(default=None, description="Workflow and lineage settings")
    logging: Optional[Dict[str, Any]] = Field(default=None, description="Logging settings")
    backup: Optional[Dict[str, Any]] = Field(default=None, description="Backup settings")

class JobRequest(BaseModel):
    """
    Request model for submitting a job.
    Contains workorder, team, and runtime configurations.
    """
    workorder: WorkorderConfig = Field(..., description="Workorder configuration")
    team: Optional[TeamConfig] = Field(default=None, description="Team configuration")
    runtime: Optional[RuntimeConfig] = Field(default=None, description="Runtime configuration")

# Response Models

class JobResponse(BaseModel):
    """
    Response model for job submission.
    Provides job ID and status information.
    """
    job_id: str = Field(..., description="Unique identifier for the job")
    status: str = Field(..., description="Current status of the job")
    storage_path: Optional[str] = Field(default=None, description="Path where job results are stored")
    error: Optional[str] = Field(default=None, description="Error message if status is error")

class JobChange(BaseModel):
    """Represents a change made to a file by the job."""
    file: str = Field(..., description="Path to the file that was changed")
    success: bool = Field(..., description="Whether the change was successful")
    error: Optional[str] = Field(default=None, description="Error message if the change failed")
    backup: Optional[str] = Field(default=None, description="Path to the backup of the original file")

class JobStatus(BaseModel):
    """
    Model for returning job status.
    Includes detailed information about the job's execution.
    """
    job_id: str = Field(..., description="Unique identifier for the job")
    status: str = Field(..., description="Current status of the job")
    storage_path: Optional[str] = Field(default=None, description="Path where job results are stored")
    error: Optional[str] = Field(default=None, description="Error message if status is error")
    changes: Optional[List[JobChange]] = Field(default=None, description="List of changes made by the job")