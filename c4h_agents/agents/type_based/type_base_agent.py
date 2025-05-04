"""
Base class for all type-based agents.

This module defines the abstract base class for all type-based agents
in the new architecture. It provides common functionality and interface
definitions.
"""

import uuid
import logging
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from c4h_agents.context.execution_context import ExecutionContext
from c4h_agents.lineage.event_logger import EventLogger, EventType
from c4h_agents.agents.types import LLMMessages


class AgentResponse:
    """Structured response from agent processing."""
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        context_updates: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.metadata = metadata or {}
        self.context_updates = context_updates or {}


class BaseTypeAgent(ABC):
    """Base class for all type-based agents."""
    
    def __init__(self, persona_config: Dict[str, Any], context: Dict[str, Any], skill_registry: Any):
        """
        Initialize a type-based agent.
        
        Args:
            persona_config: Configuration for this agent's persona
            context: Initial execution context
            skill_registry: Registry for skill discovery and invocation
        """
        # Basic agent properties
        self.agent_id = str(uuid.uuid4())
        self.agent_type = self.__class__.__name__
        self.unique_name = self.agent_type  # Default to class name if not provided
        self.persona_config = persona_config
        self.context = ExecutionContext(context)
        self.skill_registry = skill_registry
        
        # Initialize logger
        self.logger = logging.getLogger(f"c4h_agents.type_based.{self.agent_type}")
        
        # Extract run ID from context for lineage tracking
        context_dict = self.context.to_dict()
        self.run_id = (
            context_dict.get("workflow_run_id") or
            (context_dict.get("system", {}) or {}).get("runid") or
            str(uuid.uuid4())
        )
        
        # Initialize event logger for lineage tracking
        self.event_logger = None
        try:
            # Use the unique name for this agent instance for clearer lineage
            if context_dict.get("agent", {}).get("key"):
                self.unique_name = context_dict["agent"]["key"]
            
            self.logger.info(f"Initializing event logger for {self.unique_name} with run_id {self.run_id}")
            
            # Update context with run_id for consistent lineage tracking
            if "system" not in context_dict:
                context_dict["system"] = {}
            context_dict["system"]["runid"] = self.run_id
            context_dict["workflow_run_id"] = self.run_id
            
            # Extract lineage config from runtime section
            lineage_config = context_dict.get("runtime", {}).get("lineage", {})
            if not lineage_config:
                # Try llm_config path as fallback
                lineage_config = context_dict.get("llm_config", {}).get("agents", {}).get("lineage", {})
                
            # Ensure the path is set correctly in the config
            if lineage_config and "backends" in lineage_config and "file" in lineage_config["backends"]:
                file_config = lineage_config["backends"]["file"]
                if "path" in file_config:
                    path = Path(file_config["path"])
                    if not path.is_absolute():
                        # Get repository root - this should always resolve to /Users/jim/src/apps/c4h_ai_dev
                        repo_root = Path(__file__).resolve().parents[3]  # Up from c4h_agents/agents/type_based to repo root
                        
                        # Handle the case where the path might be "workspaces/lineage" or similar
                        # Ensure we keep the full repo_root path in the result
                        file_config["path"] = str(repo_root / path)
                    
                    # Special handling for paths that might have '/apps/' followed by a subdirectory name
                    if '/apps/' in file_config["path"] and not '/apps/c4h_ai_dev/' in file_config["path"]:
                        # This is likely a path that incorrectly points to /Users/jim/src/apps/workspaces/...
                        # instead of /Users/jim/src/apps/c4h_ai_dev/workspaces/...
                        file_config["path"] = file_config["path"].replace('/apps/', '/apps/c4h_ai_dev/')
            
            # Initialize event logger
            self.event_logger = EventLogger(
                lineage_config,
                self.run_id
            )
            
            # Set agent info in event logger
            self.event_logger.agent_name = self.unique_name
            self.event_logger.agent_type = self.agent_type
            
            self.logger.info(f"Event logger initialized for {self.unique_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize event logger: {e}")

    def _track_llm_interaction(self, context: Dict[str, Any], messages: LLMMessages, response: Any, metrics: Optional[Dict] = None) -> None:
        """Track LLM interaction with event logger."""
        # Debug logging to check event logger status
        self.logger.info(f"Tracking LLM interaction, event logger enabled: {self.event_logger is not None and getattr(self.event_logger, 'enabled', False)}")
        
        if self.event_logger is None:
            self.logger.warning("Event tracking skipped: event_logger object is None")
            # Dump context to debug why event_logger is None
            self.logger.debug("Context dump for event_logger troubleshooting:")
            for key, value in self.context.to_dict().items():
                self.logger.debug(f"Context[{key}] = {type(value)}")
                if isinstance(value, dict) and key in ['system', 'llm_config', 'runtime']:
                    for subkey, subvalue in value.items():
                        self.logger.debug(f"  '{key}.{subkey}' = {type(subvalue)}")
                        if subkey == 'lineage' or (isinstance(subvalue, dict) and 'lineage' in subvalue):
                            self.logger.debug(f"    Found lineage config at {key}.{subkey}")
            return
            
        if not hasattr(self.event_logger, 'enabled'):
            self.logger.warning("Event tracking skipped: event_logger object has no 'enabled' attribute")
            self.logger.debug(f"Event logger object type: {type(self.event_logger)}")
            self.logger.debug(f"Event logger object dir: {dir(self.event_logger)}")
            return
            
        if not self.event_logger.enabled:
            self.logger.warning("Event tracking skipped: event_logger.enabled is False")
            # Show what backends are available
            if hasattr(self.event_logger, 'backends'):
                self.logger.debug(f"Event logger backends: {self.event_logger.backends}")
            return
        
        try:
            # Add run_id to ensure proper event tracking
            if "workflow_run_id" not in context:
                context["workflow_run_id"] = self.run_id
                self.logger.debug(f"Added workflow_run_id to context: {self.run_id}")
                
            if "system" not in context:
                context["system"] = {"runid": self.run_id}
                self.logger.debug("Added system.runid to context")
            elif "runid" not in context["system"]:
                context["system"]["runid"] = self.run_id
                self.logger.debug("Added runid to existing system context")
            
            # Extract config metadata
            config_snapshot_path = context.get("config_snapshot_path")
            config_hash = context.get("config_hash")
            
            # Debug log event logging attempt
            self.logger.info(f"Logging LLM events with event_logger for run_id: {self.run_id}")
            
            # Step Start Event
            step_name = f"llm_{self.unique_name}"
            parent_id = context.get("parent_id")
            
            # Log STEP_START event
            self.event_logger.log_event(
                EventType.STEP_START,
                {
                    "step_type": "LLMInteraction",
                    "context_keys": list(context.keys()),
                    "agent_id": self.agent_id
                },
                step_name=step_name,
                parent_id=parent_id,
                config_snapshot_path=config_snapshot_path,
                config_hash=config_hash
            )
            
            # Log STEP_END event
            self.event_logger.log_event(
                EventType.STEP_END,
                {
                    "step_type": "LLMInteraction",
                    "agent_response_summary": {
                        "success": True,
                        "content": str(response) if response else ""  # Preserve full content
                    },
                    "metrics": metrics or {},
                    "llm_input": {
                        "system": messages.system if hasattr(messages, "system") else "",
                        "user": messages.user if hasattr(messages, "user") else "",
                        "formatted_request": messages.formatted_request if hasattr(messages, "formatted_request") else ""
                    },
                    "llm_output": self.event_logger._serialize_value(response),  # Use serializer to preserve full content
                    "llm_model": {
                        "provider": context.get("provider", "unknown"),
                        "model": context.get("model", "unknown")
                    }
                },
                step_name=step_name,
                parent_id=parent_id,
                config_snapshot_path=config_snapshot_path,
                config_hash=config_hash
            )
            
            self.logger.info("Successfully logged LLM events with event_logger")
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to log LLM events: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Dump additional context for debugging
            self.logger.debug(f"Context for track_llm_interaction:")
            self.logger.debug(f"  workflow_run_id: {context.get('workflow_run_id')}")
            
            if "system" in context:
                self.logger.debug(f"  system.runid: {context['system'].get('runid')}")
                
            self.logger.debug(f"  messages: {messages}")
            self.logger.debug(f"  response type: {type(response)}")
            self.logger.debug(f"  metrics: {metrics}")
    
    @abstractmethod
    def process(self, **kwargs) -> AgentResponse:
        """
        Process input according to the agent's behavior.
        
        Args:
            **kwargs: Input parameters for processing
            
        Returns:
            AgentResponse: Structured response from processing
        """
        pass
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "persona": self.persona_config.get("persona_key", "unknown"),
            "run_id": self.run_id,
            "event_logger_enabled": self.event_logger is not None and getattr(self.event_logger, 'enabled', False)
        }