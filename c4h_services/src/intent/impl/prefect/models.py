"""
Path: c4h_services/src/intent/impl/prefect/models.py
Model definitions for Prefect task interfaces.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AgentTaskConfig:
    """
    Configuration for an agent task execution.
    """
    task_name: str
    agent_type: str
    config: Dict[str, Any] = None
    persona_key: Optional[str] = None
    max_retries: int = 1
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}

@dataclass
class EffectiveConfigInfo:
    """
    Information about an effective configuration snapshot.
    
    Contains details about validation, fragment metadata, and persistence
    to support configuration tracking and lineage.
    """
    snapshot_path: Path
    fragments_count: int
    run_id: str
    schema_validated: bool = False
    config_hash: Optional[str] = None
    fragment_metadata: Optional[List[Dict[str, Any]]] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        # Extract hash from filename if available
        if self.snapshot_path and self.snapshot_path.name.startswith("effective_config_"):
            parts = self.snapshot_path.stem.split("_")
            if len(parts) >= 3:
                self.config_hash = parts[2]
                
        # Generate timestamp if not provided
        if self.timestamp is None:
            from datetime import datetime, timezone
            self.timestamp = datetime.now(timezone.utc).isoformat()
            
        # Initialize empty metadata if not provided
        if self.fragment_metadata is None:
            self.fragment_metadata = []
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization and lineage tracking.
        """
        return {
            "snapshot_path": str(self.snapshot_path),
            "fragments_count": self.fragments_count,
            "run_id": self.run_id,
            "schema_validated": self.schema_validated,
            "config_hash": self.config_hash,
            "fragment_metadata": self.fragment_metadata,
            "timestamp": self.timestamp
        }