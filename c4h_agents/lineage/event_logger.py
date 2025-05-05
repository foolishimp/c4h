"""
Event logger for comprehensive workflow execution tracking.

This module provides a centralized mechanism for capturing workflow
execution details for observability, debuggability, reproducibility,
and auditability purposes.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Try importing OpenLineage, but don't fail if it's not available
try:
    from openlineage.client import OpenLineageClient
    from openlineage.client.run import RunEvent, RunState, InputDataset, OutputDataset
    from openlineage.client.facet import ParentRunFacet, DocumentationJobFacet
    OPENLINEAGE_AVAILABLE = True
except ImportError:
    OPENLINEAGE_AVAILABLE = False

# Import logging utilities
from c4h_agents.utils.logging import get_logger

logger = get_logger()


class EventType(Enum):
    """Types of events that can be logged in the system."""
    WORKFLOW_START = "WORKFLOW_START"
    WORKFLOW_END = "WORKFLOW_END"
    STEP_START = "STEP_START"
    STEP_END = "STEP_END"
    ROUTING_EVALUATION = "ROUTING_EVALUATION"
    LOOP_ITERATION_START = "LOOP_ITERATION_START"
    LOOP_ITERATION_END = "LOOP_ITERATION_END"
    CONCENTRATOR_START = "CONCENTRATOR_START"
    CONCENTRATOR_INPUT_RECEIVED = "CONCENTRATOR_INPUT_RECEIVED"
    CONCENTRATOR_END = "CONCENTRATOR_END"
    FAN_OUT_DISPATCH = "FAN_OUT_DISPATCH"
    ERROR_EVENT = "ERROR_EVENT"


class EventLogger:
    """
    Central event logger for workflow execution.
    
    This class is responsible for logging all workflow events,
    including detailed capture of LLM interactions, routing decisions,
    and error conditions.
    """
    
    def __init__(self, config: Dict[str, Any], run_id: str):
        """
        Initialize the event logger.
        
        Args:
            config: Configuration for the event logger, including paths and backends
            run_id: The workflow run ID (will be reformatted to wf_HHMMSS_UUID if needed)
        """
        self.enabled = config.get("enabled", True)
        self.sequence = 0
        self.run_id = run_id  # This may be updated during backend initialization
        
        # Extract agent metadata
        self.agent_name = "Orchestrator"
        self.agent_type = "OrchestratorFlow"
        
        # Initialize backends
        self.backends = {}
        
        # Parse config for backends
        backends_config = config.get("backends", {})
        
        # Initialize file backend if enabled
        file_config = backends_config.get("file", {})
        if file_config.get("enabled", True):
            try:
                # Setup file storage
                base_dir = Path(file_config.get("path", "workspaces/lineage"))
                # Ensure base_dir is an absolute path
                if not base_dir.is_absolute():
                    # Get the repository root directory
                    repo_root = Path(__file__).resolve().parents[2]  # Going up from c4h_agents/lineage to repository root
                    base_dir = repo_root / base_dir
                
                # Special handling for paths that might have '/apps/' followed by a subdirectory name
                base_dir_str = str(base_dir)
                if '/apps/' in base_dir_str and not '/apps/c4h_ai_dev/' in base_dir_str:
                    # This is likely a path that incorrectly points to /Users/jim/src/apps/workspaces/...
                    # instead of /Users/jim/src/apps/c4h_ai_dev/workspaces/...
                    base_dir = Path(base_dir_str.replace('/apps/', '/apps/c4h_ai_dev/'))
                
                date_str = datetime.now().strftime('%Y%m%d')
                
                # Ensure run_id has the correct format (wf_HHMMSS_UUID)
                if not run_id.startswith("wf_") or "_" not in run_id[3:]:
                    # This isn't in the standard format - reconstruct it
                    current_time = datetime.now().strftime('%H%M%S')
                    run_uuid = str(uuid.uuid4())
                    original_id = run_id  # Save for logging
                    run_id = f"wf_{current_time}_{run_uuid}"
                    # Update the instance variable to use the new run_id
                    self.run_id = run_id
                    logger.info("lineage.run_id_reformatted", 
                               original=original_id, 
                               new=run_id)
                
                # Setup directory structure according to requirements
                self.events_dir = base_dir / date_str / run_id / "events"
                self.events_dir.mkdir(parents=True, exist_ok=True)
                
                self.backends["file"] = {"enabled": True}
                logger.info("lineage.event_logger_initialized", 
                           path=str(self.events_dir),
                           run_id=run_id)
            except Exception as e:
                logger.error("lineage.file_backend_init_failed", error=str(e))
                self.backends["file"] = {"enabled": False, "error": str(e)}
        
        # Initialize OpenLineage backend if enabled
        marquez_config = None
        for key, config in backends_config.items():
            if "marquez" in key and config.get("enabled", False):
                marquez_config = config
                marquez_key = key
                break
                
        if marquez_config and OPENLINEAGE_AVAILABLE:
            try:
                # Setup OpenLineage client
                url = marquez_config.get("url", "http://localhost:5005")
                logger.info(f"lineage.marquez_backend_configuring", url=url)
                
                # Create the client with just the URL parameter
                self.client = OpenLineageClient(url=url)
                
                # Store producer info for later use in events
                self.producer_name = "c4h_agents"
                self.producer_version = "0.1.0"
                self.namespace = config.get("namespace", "c4h_agents.orchestrator")
                
                self.use_marquez = True
                self.backends["marquez"] = {"enabled": True, "url": url}
                logger.info("lineage.marquez_backend_initialized", url=url, run_id=run_id)
            except Exception as e:
                logger.error("lineage.marquez_backend_init_failed", error=str(e))
                self.backends["marquez"] = {"enabled": False, "error": str(e)}
        else:
            self.use_marquez = False
        
        # Check if any backends are enabled
        if not any(backend.get("enabled", False) for backend in self.backends.values()):
            logger.warning("lineage.all_backends_disabled")
            self.enabled = False
    
    def log_event(self, 
                 event_type: Union[EventType, str], 
                 payload: Dict[str, Any], 
                 step_name: Optional[str] = None,
                 parent_id: Optional[str] = None,
                 execution_path: Optional[List[str]] = None,
                 config_snapshot_path: Optional[str] = None,
                 config_hash: Optional[str] = None) -> str:
        """
        Log an event with the specified type and payload.
        
        Args:
            event_type: Type of event to log
            payload: Event-specific payload data
            step_name: Name of the current step being processed (if applicable)
            parent_id: ID of the parent execution (if applicable)
            execution_path: List representing the call stack to this event
            config_snapshot_path: Path to the configuration snapshot
            config_hash: Hash identifier of the configuration
            
        Returns:
            The generated event_id
        """
        if not self.enabled:
            return str(uuid.uuid4())  # Still return a UUID even if disabled
        
        # Increment sequence counter
        self.sequence += 1
        
        # Generate event_id
        event_id = str(uuid.uuid4())
        
        # Get current timestamp in UTC
        timestamp = datetime.now(timezone.utc)
        
        # Create execution path if not provided
        if execution_path is None:
            execution_path = []
        
        # Build standard event structure
        event_data = {
            "event_id": event_id,
            "event_type": event_type.value if isinstance(event_type, EventType) else event_type,
            "timestamp": timestamp.isoformat(),
            "sequence": self.sequence,
            
            "workflow": {
                "run_id": self.run_id,
                "parent_id": parent_id,
                "step_name": step_name,
                "execution_path": execution_path
            },
            
            "agent": {
                "name": self.agent_name,
                "type": self.agent_type
            },
            
            "configuration": {
                "snapshot_path": config_snapshot_path,
                "config_hash": config_hash,
                "timestamp": timestamp.isoformat()
            },
            
            "payload": payload
        }
        
        # Write to file backend
        if self.backends.get("file", {}).get("enabled", False):
            self._write_file_event(event_data, event_id, timestamp)
        
        # Emit to OpenLineage if enabled
        if self.use_marquez and self.backends.get("marquez", {}).get("enabled", False):
            self._emit_marquez_event(event_data, event_id, timestamp)
        
        return event_id
    
    def _write_file_event(self, event_data: Dict[str, Any], event_id: str, timestamp: datetime) -> None:
        """
        Write event to file system.
        
        Args:
            event_data: The event data to write
            event_id: The event ID
            timestamp: The event timestamp
        """
        try:
            # Create timestamped filename using local time (not UTC) with microsecond-based unique sequence
            local_time = datetime.now()
            time_str = local_time.strftime('%H%M%S')
            # Use microseconds to ensure unique sequence, plus agent's sequence counter for ordering
            unique_seq = f"{(local_time.microsecond // 1000):03d}{self.sequence:03d}"
            event_filename = f"{time_str}_{unique_seq}_{event_id}.json"
            
            event_file = self.events_dir / event_filename
            temp_file = self.events_dir / f"tmp_{event_filename}"
            
            # Write to temp file first (atomic operation)
            with open(temp_file, 'w') as f:
                json.dump(event_data, f, indent=2, default=str)
                
            # Rename to final filename (atomic operation)
            temp_file.rename(event_file)
            
            logger.info("lineage.event_saved",
                     path=str(event_file),
                     event_type=event_data["event_type"],
                     event_size=event_file.stat().st_size,
                     event_id=event_id)
                
        except Exception as e:
            logger.error("lineage.write_failed",
                       error=str(e),
                       events_dir=str(self.events_dir),
                       event_id=event_id)
    
    def _emit_marquez_event(self, event_data: Dict[str, Any], event_id: str, timestamp: datetime) -> None:
        """
        Emit event to Marquez/OpenLineage.
        
        Args:
            event_data: The event data to emit
            event_id: The event ID
            timestamp: The event timestamp
        """
        if not OPENLINEAGE_AVAILABLE or not hasattr(self, 'client') or not self.use_marquez:
            return
            
        try:
            logger.debug("lineage.marquez_event_preparing", 
                       event_id=event_id, 
                       event_type=event_data["event_type"])
                       
            # Create facets
            facets = {}
            
            # Add parent facet if parent_id is present
            if event_data["workflow"]["parent_id"]:
                facets["parent"] = ParentRunFacet(run={
                    "runId": event_data["workflow"]["parent_id"]
                })
            
            # Add documentation facet
            event_type = event_data["event_type"]
            step_name = event_data["workflow"]["step_name"] or "unknown"
            description = f"{event_type} - {step_name}"
            facets["documentation"] = DocumentationJobFacet(description=description)
            
            # Add configuration facets
            if event_data["configuration"]["snapshot_path"] or event_data["configuration"]["config_hash"]:
                facets["configuration"] = {
                    "snapshot_path": event_data["configuration"]["snapshot_path"],
                    "config_hash": event_data["configuration"]["config_hash"],
                    "timestamp": event_data["configuration"]["timestamp"]
                }
            
            # Create the OpenLineage event
            ol_event = RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=timestamp.isoformat(),
                producer=self.producer_name,
                run={
                    "runId": event_id,
                    "facets": facets
                },
                job={
                    "namespace": self.namespace,
                    "name": f"{self.agent_name}_{event_data['event_type']}"
                },
                inputs=[InputDataset(
                    namespace=self.namespace,
                    name=f"{event_data['event_type']}_input",
                    facets={"context": self._serialize_value(event_data["payload"])}
                )],
                outputs=[OutputDataset(
                    namespace=self.namespace,
                    name=f"{event_data['event_type']}_output",
                    facets={"payload": self._serialize_value(event_data["payload"])}
                )]
            )
            
            # Emit the event
            self.client.emit(ol_event)
            
            logger.info("lineage.marquez_event_emitted", 
                      event_id=event_id,
                      event_type=event_data["event_type"],
                      url=self.backends["marquez"].get("url", "unknown"))
                      
        except Exception as e:
            logger.error("lineage.marquez_event_failed", 
                       error=str(e),
                       event_id=event_id,
                       event_type=event_data["event_type"])
    
    def _serialize_value(self, value: Any) -> Any:
        """
        Serialize a value for inclusion in event data.
        
        Args:
            value: The value to serialize
            
        Returns:
            The serialized value
        """
        # Handle None, primitives
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        
        # Handle LLM response objects
        if hasattr(value, 'choices') and value.choices:
            try:
                # Standard LLM response format
                if hasattr(value.choices[0], 'message') and hasattr(value.choices[0].message, 'content'):
                    response_data = {
                        "content": value.choices[0].message.content,
                        "finish_reason": getattr(value.choices[0], 'finish_reason', None),
                        "model": getattr(value, 'model', None)
                    }
                    
                    # Add usage if available
                    if hasattr(value, 'usage'):
                        usage = value.usage
                        response_data["usage"] = {
                            "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                            "completion_tokens": getattr(usage, 'completion_tokens', 0),
                            "total_tokens": getattr(usage, 'total_tokens', 0)
                        }
                    return response_data
                # Handle delta format (used in streaming)
                elif hasattr(value.choices[0], 'delta') and hasattr(value.choices[0].delta, 'content'):
                    content = value.choices[0].delta.content
                    return {"content": content}
            except (AttributeError, IndexError):
                pass
        
        # Handle StreamedResponse
        if "StreamedResponse" in str(type(value)):
            try:
                if hasattr(value, 'choices') and value.choices:
                    return {"content": value.choices[0].message.content}
            except (AttributeError, IndexError):
                pass
        
        # Handle Usage objects directly 
        if type(value).__name__ == 'Usage':
            return {
                "prompt_tokens": getattr(value, 'prompt_tokens', 0),
                "completion_tokens": getattr(value, 'completion_tokens', 0),
                "total_tokens": getattr(value, 'total_tokens', 0)
            }
        
        # Handle custom objects with to_dict method
        if hasattr(value, 'to_dict'):
            return value.to_dict()
        
        # Fall back to string representation with object type indicator
        return f"{str(value)} (type: {type(value).__name__})"