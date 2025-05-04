"""
Base class for all type-based agents.

This module defines the abstract base class for all type-based agents
in the new architecture. It provides common functionality and interface
definitions.
"""

import uuid
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from c4h_agents.context.execution_context import ExecutionContext
from c4h_agents.agents.base_lineage import BaseLineage
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
        
        # Initialize lineage tracking
        self.lineage = None
        try:
            # Use the unique name for this agent instance for clearer lineage
            if context_dict.get("agent", {}).get("key"):
                self.unique_name = context_dict["agent"]["key"]
            
            namespace = "c4h_agents.type_based"
            self.logger.info(f"Initializing lineage tracking for {self.unique_name} with run_id {self.run_id}")
            
            # Update context with run_id for consistent lineage tracking
            if "system" not in context_dict:
                context_dict["system"] = {}
            context_dict["system"]["runid"] = self.run_id
            context_dict["workflow_run_id"] = self.run_id
            
            # Initialize lineage tracker
            self.lineage = BaseLineage(
                namespace=namespace,
                agent_name=self.unique_name,
                config=context_dict
            )
            
            # Update run_id if lineage loaded an existing one
            if self.lineage and hasattr(self.lineage, 'run_id') and self.lineage.run_id != self.run_id:
                self.run_id = self.lineage.run_id
                self.logger.info(f"Lineage loaded existing run_id: {self.run_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize lineage tracking: {e}")

    def _track_llm_interaction(self, context: Dict[str, Any], messages: LLMMessages, response: Any, metrics: Optional[Dict] = None) -> None:
        """Track LLM interaction with lineage."""
        # Debug logging to check lineage status
        self.logger.info(f"Tracking LLM interaction, lineage enabled: {self.lineage is not None and getattr(self.lineage, 'enabled', False)}")
        
        if self.lineage is None:
            self.logger.warning("Lineage tracking skipped: lineage object is None")
            # Dump context to debug why lineage is None
            self.logger.debug("Context dump for lineage troubleshooting:")
            for key, value in self.context.to_dict().items():
                self.logger.debug(f"Context[{key}] = {type(value)}")
                if isinstance(value, dict) and key in ['system', 'llm_config', 'runtime']:
                    for subkey, subvalue in value.items():
                        self.logger.debug(f"  '{key}.{subkey}' = {type(subvalue)}")
                        if subkey == 'lineage' or (isinstance(subvalue, dict) and 'lineage' in subvalue):
                            self.logger.debug(f"    Found lineage config at {key}.{subkey}")
            return
            
        if not hasattr(self.lineage, 'enabled'):
            self.logger.warning("Lineage tracking skipped: lineage object has no 'enabled' attribute")
            self.logger.debug(f"Lineage object type: {type(self.lineage)}")
            self.logger.debug(f"Lineage object dir: {dir(self.lineage)}")
            return
            
        if not self.lineage.enabled:
            self.logger.warning("Lineage tracking skipped: lineage.enabled is False")
            # Show what backends are available
            if hasattr(self.lineage, 'backends'):
                self.logger.debug(f"Lineage backends: {self.lineage.backends}")
            return
        
        try:
            # Add run_id to ensure proper lineage tracking
            if "workflow_run_id" not in context:
                context["workflow_run_id"] = self.run_id
                self.logger.debug(f"Added workflow_run_id to context: {self.run_id}")
                
            if "system" not in context:
                context["system"] = {"runid": self.run_id}
                self.logger.debug("Added system.runid to context")
            elif "runid" not in context["system"]:
                context["system"]["runid"] = self.run_id
                self.logger.debug("Added runid to existing system context")
            
            # Debug log the context structure
            self.logger.info(f"Calling lineage.track_llm_interaction with run_id: {self.run_id}")
            
            # Check if lineage has file backend
            if hasattr(self.lineage, 'backends') and 'file' in self.lineage.backends:
                self.logger.debug(f"Lineage file backend config: {self.lineage.backends['file']}")
                
            # Check if lineage dir exists
            if hasattr(self.lineage, 'lineage_dir'):
                self.logger.debug(f"Lineage directory: {self.lineage.lineage_dir}")
                
            # Check message format to ensure it's compatible
            self.logger.debug(f"Messages type: {type(messages)}")
            self.logger.debug(f"Messages attributes: {dir(messages)}")
            self.logger.debug(f"System message present: {hasattr(messages, 'system')}")
            self.logger.debug(f"User message present: {hasattr(messages, 'user')}")
            
            # Track the interaction
            self.lineage.track_llm_interaction(
                context=context,
                messages=messages,
                response=response,
                metrics=metrics
            )
            
            self.logger.info("Successfully tracked LLM interaction with lineage")
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to track LLM interaction: {e}")
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
            "lineage_enabled": self.lineage is not None and getattr(self.lineage, 'enabled', False)
        }