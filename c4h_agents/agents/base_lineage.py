# Path: c4h_agents/agents/base_lineage.py
"""
Lineage tracking implementation leveraging existing workflow event storage.
Path: c4h_agents/agents/base_lineage.py
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import uuid
import os

# Try importing OpenLineage, but don't fail if it's not available
try:
    from openlineage.client import OpenLineageClient
    from openlineage.client.run import RunEvent, RunState, InputDataset, OutputDataset
    from openlineage.client.facet import ParentRunFacet, DocumentationJobFacet
    OPENLINEAGE_AVAILABLE = True
except ImportError:
    OPENLINEAGE_AVAILABLE = False

from c4h_agents.agents.types import LLMMessages
from c4h_agents.config import create_config_node
from c4h_agents.utils.logging import get_logger

logger = get_logger()

@dataclass
class LineageEvent:
    """Complete lineage event for LLM interaction"""
    event_id: str
    agent_name: str
    agent_type: str
    run_id: str
    parent_id: Optional[str]
    input_context: Dict[str, Any]
    messages: LLMMessages
    raw_output: Any
    metrics: Optional[Dict] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None
    step: Optional[int] = None
    execution_path: Optional[List[str]] = None
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    config_snapshot_path: Optional[str] = None
    config_hash: Optional[str] = None

class BaseLineage:
    """OpenLineage tracking implementation"""
    def __init__(self, namespace: str, agent_name: str, config: Dict[str, Any]):
        """Initialize lineage tracking"""
        self.namespace = namespace
        self.agent_name = agent_name
        self.agent_type = agent_name.split('.')[-1] if '.' in agent_name else agent_name
        self.enabled = False
        self.client = None
        self.use_marquez = False
        
        # Create a configuration node for hierarchical access
        self.config_node = create_config_node(config or {})
        
        # Debug log the configuration structure
        logger.debug(f"{agent_name}.lineage_init", 
                    has_system=self.config_node.get_value("system") is not None,
                    has_workflow_run_id=self.config_node.get_value("workflow_run_id") is not None,
                    has_runtime=self.config_node.get_value("runtime") is not None)
        
        # Get lineage config using path query
        lineage_config = self.config_node.get_value("llm_config.agents.lineage") or {}
        if not lineage_config:
            # Try runtime path as fallback
            lineage_config = self.config_node.get_value("runtime.lineage") or {}
            
        if not lineage_config:
            logger.info(f"{agent_name}.lineage_disabled", reason="no_config")
            return
            
        # Extract run ID using hierarchical path queries
        self.run_id = self._extract_run_id(self.config_node)
        logger.debug(f"{agent_name}.using_run_id", run_id=self.run_id)

        # Set global config parameters
        self.enabled = lineage_config.get("enabled", False)
        self.namespace = lineage_config.get("namespace", self.namespace)
        self.event_detail_level = lineage_config.get("event_detail_level", "full")
        self.separate_input_output = lineage_config.get("separate_input_output", False)
            
        if not self.enabled:
            logger.info(f"{agent_name}.lineage_disabled", reason="not_enabled")
            return

        # Initialize backends
        self.backends = {}
        
        # Get backend configurations
        backends_config = lineage_config.get("backends", {})
        
        # Handle backward compatibility - if no backends section, assume file backend with original config
        if not backends_config:
            file_config = {
                "enabled": True,
                "path": lineage_config.get("backend", {}).get("path", "workspaces/lineage")
            }
            backends_config = {"file": file_config}
        
        # Initialize file backend
        file_config = backends_config.get("file", {})
        if file_config.get("enabled", True):
            try:
                # Setup file storage
                base_dir = Path(file_config.get("path", "workspaces/lineage"))
                date_str = datetime.now().strftime('%Y%m%d')
                
                # Simply use the run_id directly - no need to modify it 
                # The run_id should already have the timestamp embedded from orchestrator.py
                self.lineage_dir = base_dir / date_str / self.run_id
                self.lineage_dir.mkdir(parents=True, exist_ok=True)
                
                # Create subdirectories
                (self.lineage_dir / "events").mkdir(exist_ok=True)
                (self.lineage_dir / "errors").mkdir(exist_ok=True)
                (self.lineage_dir / "inputs").mkdir(exist_ok=True)
                (self.lineage_dir / "outputs").mkdir(exist_ok=True)
                
                self.backends["file"] = {"enabled": True}
                logger.info("lineage.file_backend_initialized", path=str(self.lineage_dir), run_id=self.run_id)
            except Exception as e:
                logger.error("lineage.file_backend_init_failed", error=str(e))
                self.backends["file"] = {"enabled": False, "error": str(e)}
        
        # Initialize Marquez backend
        # Look for any marquez backend in the config
        marquez_config = None
        marquez_key = None
        for key, config in backends_config.items():
            if "marquez" in key and config.get("enabled", False):
                marquez_config = config
                marquez_key = key
                break
                
        if marquez_config:
            if OPENLINEAGE_AVAILABLE:
                try:
                    # Setup OpenLineage client
                    url = marquez_config.get("url", "http://localhost:5005")
                    logger.info(f"lineage.marquez_backend_configuring", url=url)
                    
                    # Create the client with just the URL parameter
                    self.client = OpenLineageClient(url=url)
                    
                    # We don't need to set producer on the client, it will be specified per event
                    # Store producer info for later use in events
                    self.producer_name = "c4h_agents"
                    self.producer_version = "0.1.0"
                    
                    # Log transport settings that we can't use directly
                    transport_config = marquez_config.get("transport", {})
                    if transport_config:
                        logger.debug("lineage.marquez_transport_settings", 
                                    settings=transport_config,
                                    note="Transport settings logged but not applied - OpenLineageClient doesn't support these constructor arguments")
                    
                    self.use_marquez = True
                    self.backends["marquez"] = {"enabled": True, "url": url}
                    logger.info("lineage.marquez_backend_initialized", url=url, run_id=self.run_id)
                except Exception as e:
                    logger.error("lineage.marquez_backend_init_failed", error=str(e))
                    self.backends["marquez"] = {"enabled": False, "error": str(e)}
            else:
                logger.warning("lineage.marquez_backend_unavailable", reason="openlineage_not_installed")
                self.backends["marquez"] = {"enabled": False, "error": "OpenLineage not available"}
        
        # Check if any backends are enabled
        if not any(backend.get("enabled", False) for backend in self.backends.values()):
            logger.warning("lineage.all_backends_disabled")
            self.enabled = False
        else:
            # Log active backends
            active = [name for name, config in self.backends.items() if config.get("enabled")]
            logger.info("lineage.active_backends", backends=active, count=len(active))

    def _extract_run_id(self, config_node) -> str:
        """
        Extract run ID using hierarchical path queries.
        Returns a stable run ID from the first available source.
        """
        # Query potential run ID locations in priority order
        run_id = (
            # 1. System namespace (highest priority)
            config_node.get_value("system.runid") or
            # 2. Direct context parameter
            config_node.get_value("workflow_run_id") or 
            # 3. Runtime configuration
            config_node.get_value("runtime.workflow_run_id") or
            config_node.get_value("runtime.run_id") or
            # 4. Workflow section
            config_node.get_value("runtime.workflow.id")
        )
        
        if run_id:
            return str(run_id)
            
        # Generate new UUID as fallback
        generated_id = str(uuid.uuid4())
        logger.warning("lineage.missing_run_id", agent=self.agent_name, generated_id=generated_id)
        return generated_id
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value with enhanced type handling for LLM responses"""
        # Handle None, primitives as before
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
        
        # Handle LLM response objects - check for common response structure
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
        
        # Handle StreamedResponse (from BaseLLM._get_completion_with_continuation)
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

    def _extract_lineage_metadata(self, context: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[int], Optional[List[str]]]:
        """
        Extract lineage tracking metadata from context.
        Returns tuple of (event_id, parent_id, step, execution_path)
        """
        # Create configuration node for hierarchical access
        context_node = create_config_node(context)
        
        # Generate unique event ID if not provided
        event_id = context_node.get_value("agent_execution_id") or str(uuid.uuid4())
            
        # Extract parent ID in priority order
        parent_id = (
            context_node.get_value("parent_id") or  # Explicit parent ID
            context_node.get_value("parent_run_id") or  # Alternative parent ID
            (None if context_node.get_value("workflow_run_id") == self.run_id else context_node.get_value("workflow_run_id"))  # Use workflow ID as parent if different
        )
        
        # Extract step number
        step = context_node.get_value("step") or context_node.get_value("sequence") or None
        
        # Extract or build execution path
        path = context_node.get_value("execution_path") or []
        if isinstance(path, str):
            try:
                path = json.loads(path)
            except:
                path = [path]
        
        if not path:
            path = []
        
        # Add self to path
        path = path + [f"{self.agent_type}:{event_id[:8]}"]
        
        return event_id, parent_id, step, path

    """
    Lineage tracking implementation leveraging existing workflow event storage.
    Path: c4h_agents/agents/base_lineage.py
    """

    def _write_file_event(self, event: LineageEvent) -> None:
        """Write event to file system with minimal processing"""
        if not self.enabled or not self.lineage_dir or not self.backends.get("file", {}).get("enabled", False):
            return
            
        try:
            events_dir = self.lineage_dir / "events"
            
            # Create timestamped filename but keep the original event_id in the content
            timestamp = datetime.now().strftime('%H%M%S')
            event_filename = f"{timestamp}_{event.event_id}.json"
            
            event_file = events_dir / event_filename
            temp_file = events_dir / f"tmp_{event_filename}"
            
            # Create clear, complete event structure without duplication
            event_data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "agent": {
                    "name": event.agent_name,
                    "type": event.agent_type
                },
                "workflow": {
                    "run_id": event.run_id,
                    "parent_id": event.parent_id,
                    "step": event.step,
                    "execution_path": event.execution_path
                },
                "configuration": {
                    "snapshot_path": event.config_snapshot_path,
                    "config_hash": event.config_hash,
                    "timestamp": event.timestamp.isoformat()
                },
                "llm_input": {
                    "system_message": event.messages.system if hasattr(event.messages, "system") else None,
                    "user_message": event.messages.user if hasattr(event.messages, "user") else None,
                    # Only include formatted_request if it contains different content
                    "formatted_request": (event.messages.formatted_request if hasattr(event.messages, "formatted_request") and 
                                        event.messages.formatted_request and 
                                        event.messages.formatted_request != event.messages.user 
                                        else None)
                },
                "llm_output": self._serialize_value(event.raw_output),
                "metrics": self._serialize_value(event.metrics),
                "error": event.error
            }
            
            # Write to temp file first (atomic operation)
            with open(temp_file, 'w') as f:
                json.dump(event_data, f, indent=2, default=str)
                
            # Rename to final filename (atomic operation)
            temp_file.rename(event_file)
            
            logger.info("lineage.event_saved",
                    path=str(event_file),
                    agent=event.agent_name,
                    run_id=event.run_id,
                    event_size=event_file.stat().st_size,
                    event_id=event.event_id)
                
        except Exception as e:
            logger.error("lineage.write_failed",
                        error=str(e),
                        lineage_dir=str(self.lineage_dir),
                        agent=event.agent_name,
                        event_id=event.event_id)

    def _emit_marquez_event(self, event: LineageEvent) -> None:
        """Emit event to Marquez"""
        if not OPENLINEAGE_AVAILABLE or not self.client or not self.use_marquez or not self.backends.get("marquez", {}).get("enabled", False):
            return
            
        try:
            logger.debug("lineage.marquez_event_preparing", 
                       event_id=event.event_id, 
                       agent=event.agent_name)
                       
            # Create the OpenLineage run event with producer explicitly set
            # Prepare facets dictionary with proper handling of None values
            facets = {}
            if event.parent_id:
                facets["parent"] = ParentRunFacet(run={
                    "runId": event.parent_id
                })
            facets["documentation"] = DocumentationJobFacet(description=f"Agent: {event.agent_name}")
            
            # Add configuration facets if available
            if event.config_snapshot_path or event.config_hash:
                facets["configuration"] = {
                    "snapshot_path": event.config_snapshot_path,
                    "config_hash": event.config_hash,
                    "timestamp": event.timestamp.isoformat()
                }
            
            # Create the event with producer explicitly set
            ol_event = RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=event.timestamp.isoformat(),
                producer="c4h_agents",  # Explicitly set producer in the event
                run={
                    "runId": event.event_id,
                    "facets": facets
                },
                job={
                    "namespace": self.namespace,
                    "name": event.agent_name
                },
                inputs=[InputDataset(
                    namespace=self.namespace,
                    name=f"{event.agent_name}_input",
                    facets={"context": self._serialize_value(event.input_context)}
                )],
                outputs=[OutputDataset(
                    namespace=self.namespace,
                    name=f"{event.agent_name}_output",
                    facets={
                        "metrics": event.metrics or {},
                        "output": self._serialize_value(event.raw_output)
                    }
                )]
            )
            
            logger.debug("lineage.marquez_event_sending", 
                       event_id=event.event_id,
                       agent=event.agent_name)
                       
            # Use the emit method on the client instance
            self.client.emit(ol_event)
            
            logger.info("lineage.marquez_event_emitted", 
                      event_id=event.event_id,
                      agent=event.agent_name,
                      url=self.backends["marquez"].get("url", "unknown"))
                      
        except Exception as e:
            logger.error("lineage.marquez_event_failed", 
                       error=str(e),
                       event_id=event.event_id,
                       agent=event.agent_name)
    
    def track_llm_interaction(self,
                              context: Dict[str, Any],
                              messages: LLMMessages,
                              response: Any,
                              metrics: Optional[Dict] = None) -> None:
        """Track complete LLM interaction"""
        if not self.enabled:
            logger.debug("lineage.tracking_skipped", enabled=False)
            return
            
        try:
            # Extract lineage metadata
            event_id, parent_id, step, execution_path = self._extract_lineage_metadata(context)
            
            # Extract configuration snapshot information if available in context
            config_snapshot_path = None
            config_hash = None
            
            if "config_snapshot_path" in context:
                config_snapshot_path = context["config_snapshot_path"]
            elif "config_metadata" in context and isinstance(context["config_metadata"], dict):
                if "snapshot_path" in context["config_metadata"]:
                    config_snapshot_path = context["config_metadata"]["snapshot_path"]
                if "config_hash" in context["config_metadata"]:
                    config_hash = context["config_metadata"]["config_hash"]
            
            # Also check runtime section
            if not config_snapshot_path and "runtime" in context:
                runtime = context.get("runtime", {})
                if isinstance(runtime, dict) and "config_metadata" in runtime:
                    config_metadata = runtime["config_metadata"]
                    if isinstance(config_metadata, dict):
                        if "snapshot_path" in config_metadata:
                            config_snapshot_path = config_metadata["snapshot_path"]
                        if "config_hash" in config_metadata:
                            config_hash = config_metadata["config_hash"]
            
            # Create the lineage event with snapshot information
            event = LineageEvent(
                event_id=event_id,
                agent_name=self.agent_name,
                agent_type=self.agent_type,
                run_id=self.run_id,
                parent_id=parent_id,
                input_context=context,
                messages=messages,
                raw_output=response,
                metrics=metrics,
                step=step,
                execution_path=execution_path,
                config_snapshot_path=config_snapshot_path,
                config_hash=config_hash
            )
            
            # Track to file backend
            if self.backends.get("file", {}).get("enabled", False):
                try:
                    self._write_file_event(event)
                    file_success = True
                except Exception as e:
                    logger.error("lineage.file_backend_failed", error=str(e))
                    file_success = False
            else:
                file_success = False
            
            # Track to Marquez backend
            if self.use_marquez and self.backends.get("marquez", {}).get("enabled", False):
                try:
                    self._emit_marquez_event(event)
                    marquez_success = True
                except Exception as e:
                    logger.error("lineage.marquez_backend_failed", error=str(e))
                    marquez_success = False
            else:
                marquez_success = False
            
            # Count successful backends
            backend_count = 0
            success_count = 0
            
            if "file" in self.backends and self.backends["file"].get("enabled", False):
                backend_count += 1
                if file_success:
                    success_count += 1
                    
            if "marquez" in self.backends and self.backends["marquez"].get("enabled", False):
                backend_count += 1
                if marquez_success:
                    success_count += 1
            
            # Log overall tracking status
            logger.info("lineage.event_saved",
                    agent=self.agent_name,
                    event_id=event_id,
                    parent_id=parent_id,
                    backends_succeeded=success_count,
                    backends_total=backend_count,
                    path_length=len(execution_path) if execution_path else 0)
                        
        except Exception as e:
            logger.error("lineage.track_failed", error=str(e), agent=self.agent_name)