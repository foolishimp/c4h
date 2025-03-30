"""
Storage for job-to-workflow ID mapping.
Path: c4h_services/src/api/jobs_storage.py
"""

from typing import Dict, Any, Optional, List
from c4h_services.src.utils.logging import get_logger
from datetime import datetime
import uuid
import json
import os
from pathlib import Path

logger = get_logger()

class JobsStorage:
    """
    Storage for job-to-workflow ID mapping.
    Uses a simple file-based storage by default.
    """
    
    def __init__(self, storage_dir: str = "workspaces/jobs"):
        """
        Initialize the storage.
        
        Args:
            storage_dir: Directory for storing job mapping
        """
        self.storage_dir = Path(storage_dir)
        self.storage_file = self.storage_dir / "job_mapping.json"
        self._ensure_storage()
        
    def _ensure_storage(self) -> None:
        """Ensure the storage directory and file exist."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.storage_dir, exist_ok=True)
            
            # Create mapping file if it doesn't exist
            if not self.storage_file.exists():
                with open(self.storage_file, 'w') as f:
                    json.dump({}, f)
                    
            logger.debug("jobs_storage.initialized", 
                       storage_file=str(self.storage_file))
                       
        except Exception as e:
            logger.error("jobs_storage.init_failed", error=str(e))
            raise
            
    def create_job(self, workflow_id: str) -> str:
        """
        Create a new job mapping to a workflow ID.
        
        Args:
            workflow_id: The workflow ID to map to
            
        Returns:
            The new job ID
        """
        try:
            # Generate a new job ID
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Load current mapping
            mapping = self._load_mapping()
            
            # Add new mapping
            mapping[job_id] = {
                "workflow_id": workflow_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Save mapping
            self._save_mapping(mapping)
            
            logger.info("jobs_storage.job_created", 
                      job_id=job_id, 
                      workflow_id=workflow_id)
                      
            return job_id
            
        except Exception as e:
            logger.error("jobs_storage.create_job_failed", error=str(e))
            raise
            
    def get_workflow_id(self, job_id: str) -> Optional[str]:
        """
        Get the workflow ID associated with a job ID.
        
        Args:
            job_id: The job ID to look up
            
        Returns:
            The associated workflow ID, or None if not found
        """
        try:
            # Load current mapping
            mapping = self._load_mapping()
            
            # Check if job ID exists
            if job_id in mapping:
                return mapping[job_id]["workflow_id"]
                
            logger.warning("jobs_storage.job_not_found", job_id=job_id)
            return None
            
        except Exception as e:
            logger.error("jobs_storage.get_workflow_id_failed", 
                       job_id=job_id, 
                       error=str(e))
            return None
            
    def _load_mapping(self) -> Dict[str, Any]:
        """Load the job mapping from storage."""
        try:
            with open(self.storage_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error("jobs_storage.load_mapping_failed", error=str(e))
            return {}
            
    def _save_mapping(self, mapping: Dict[str, Any]) -> None:
        """Save the job mapping to storage."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(mapping, f, indent=2)
        except Exception as e:
            logger.error("jobs_storage.save_mapping_failed", error=str(e))
            raise