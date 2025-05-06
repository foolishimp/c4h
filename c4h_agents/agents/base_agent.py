# File: /Users/jim/src/apps/c4h_ai_dev/c4h_agents/agents/base_agent.py
# Corrections for errors related to 'config' and 'agent_name' scope

# Necessary Imports
import json
import uuid
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import re
import structlog # Ensure structlog is imported if used directly
from c4h_agents.agents.base_config import BaseConfig
from c4h_agents.lineage.event_logger import EventLogger, EventType # New import for EventLogger
from c4h_agents.agents.lineage_context import LineageContext # Correct import
from c4h_agents.agents.base_llm import BaseLLM
from c4h_agents.agents.continuation.continuation_handler import ContinuationHandler
from c4h_agents.agents.types import AgentResponse, AgentMetrics, LLMProvider, LLMMessages, LogDetail, SkillResult
from c4h_agents.config import create_config_node # Keep create_config_node
from c4h_agents.core.project import Project
# Import get_logger and log_config_node from the correct utility path
from c4h_agents.utils.logging import get_logger, log_config_node

# Logger instance for this module (if needed, otherwise self.logger is used)
# logger = get_logger() # Removed global logger instance if self.logger is always used

class BaseAgent(BaseConfig, BaseLLM):
    """Base agent implementation"""

    # Corrected __init__ method:
    def __init__(self, full_effective_config: Dict[str, Any], unique_name: str):
        # --- Step 1: Initialize BaseConfig FIRST to set up self.config and self.config_node ---
        # BaseConfig now accepts the full config directly
        BaseConfig.__init__(self, config=full_effective_config)

        # --- Step 2: Initialize BaseLLM ---
        BaseLLM.__init__(self) # Explicitly call BaseLLM init

        # --- Step 3: Store unique_name and agent_id ---
        self.unique_name = unique_name # Store the unique name passed in
        self.agent_id = str(uuid.uuid4())

        # --- Step 4: Define agent_name using the required method ---
        # This ensures _get_agent_name() works correctly later if needed,
        # and provides a variable for logging if preferred over calling the method repeatedly.
        agent_name = self._get_agent_name() # Now returns self.unique_name

        # --- Step 5: Initialize Logger AFTER self.config is set by BaseConfig ---
        # Bind initial context including the unique agent name
        log_context = {"agent": agent_name, "agent_id": self.agent_id}
        # Add project info if project was passed and initialized in BaseConfig
        if self.project:
            log_context.update({
                "project_name": self.project.metadata.name,
                "project_version": self.project.metadata.version,
                "project_root": str(self.project.paths.root)
            })
        # Get logger using self.config which was set in BaseConfig.__init__
        # Ensure get_logger can handle the config format correctly
        self.logger = get_logger(self.config).bind(**log_context)

        # --- Step 6: Log received config using self.config and self.logger ---
        # Use self.config, not a global 'config'
        # Use agent_name variable defined above
        log_config_node(self.logger, self.config, "workorder", log_prefix=f"{agent_name}.init_received")
        log_config_node(self.logger, self.config, "team.llm_config.agents", log_prefix=f"{agent_name}.init_received")
        log_config_node(self.logger, self.config, "llm_config.agents", log_prefix=f"{agent_name}.init_received")

        # --- Step 7: Use persona configuration directly ---
        self.config_path = f"llm_config.agents.{agent_name}"
        agent_config = self.config_node.get_value(self.config_path) or {}
        
        # Get persona_key from task config - required for this approach
        self.persona_key = agent_config.get('persona_key')
        
        if not self.persona_key:
            self.logger.error(f"{agent_name}.init.missing_persona_key", 
                             agent_name=agent_name,
                             config_path=self.config_path)
            raise ValueError(f"Agent {agent_name} is missing required persona_key in configuration")
        
        # Set persona path - this is our primary configuration source
        self.persona_path = f"llm_config.personas.{self.persona_key}"
        
        # Verify persona exists
        persona_config = self.config_node.get_value(self.persona_path)
        if not persona_config:
            self.logger.error(f"{agent_name}.init.persona_not_found", 
                             persona_key=self.persona_key, 
                             persona_path=self.persona_path)
            raise ValueError(f"Persona '{self.persona_key}' not found at {self.persona_path}")
            
        self.logger.info(f"{agent_name}.init.using_persona", 
                       persona_key=self.persona_key, 
                       persona_path=self.persona_path)

        # --- Step 8: Resolve Provider, Model, Temperature primarily from persona ---
        # Use persona configuration as primary source with minimal fallbacks
        self.provider = LLMProvider(
            self.config_node.get_value(f"{self.persona_path}.provider") or \
            self.config_node.get_value("llm_config.default_provider") or \
            "anthropic"  # Ultimate fallback
        )
        
        self.model = (
            self.config_node.get_value(f"{self.persona_path}.model") or \
            self.config_node.get_value(f"llm_config.providers.{self.provider.value}.default_model") or \
            self.config_node.get_value("llm_config.default_model") or \
            "claude-3-opus-20240229"  # Ultimate fallback
        )
        
        # Temperature from persona
        temp_val = self.config_node.get_value(f"{self.persona_path}.temperature")
        self.temperature = float(temp_val) if temp_val is not None else 0.0

        self.logger.debug(f"{agent_name}.init.resolved_settings",
                          provider=self.provider.value, 
                          model=self.model, 
                          temp=self.temperature,
                          persona_key=self.persona_key)

        # --- Step 9: Continuation settings from persona ---
        # Use persona config with defaults only as fallback
        self.max_continuation_attempts = (
            self.config_node.get_value(f"{self.persona_path}.max_continuation_attempts") or 5  # Default value
        )
        
        self.continuation_token_buffer = (
            self.config_node.get_value(f"{self.persona_path}.continuation_token_buffer") or 1000  # Default value
        )

        # --- Step 10: Initialize metrics ---
        self.metrics = AgentMetrics(project=self.project.metadata.name if self.project else None)

        # --- Step 11: Set logging detail level from persona ---
        log_level_str = (
            self.config_node.get_value(f"{self.persona_path}.log_level") or
            self.config_node.get_value("logging.agent_level") or 
            self.config_node.get_value("logging.level") or 
            "basic"
        )
        self.log_level = LogDetail.from_str(log_level_str)

        # --- Step 12: Setup LiteLLM ---
        # _get_model_str uses self.provider/self.model resolved above
        self.model_str = self._get_model_str()
        # _get_provider_config uses self.config_node
        self._setup_litellm(self._get_provider_config(self.provider))

        # --- Step 13: Finalize logger binding with run_id ---
        # Get run_id from the effective config snapshot
        self.run_id = self._get_workflow_run_id() or str(uuid.uuid4())
        log_context.update({
             "provider": self.provider.serialize(),
             "model": self.model,
             "log_level": str(self.log_level),
             "run_id": self.run_id
        })
        self.logger = self.logger.bind(**log_context)

        # --- Step 14: Initialize EventLogger for Lineage ---
        self.event_logger = None
        try:
            self.logger.debug(f"{agent_name}.event_logger_init", has_runtime="runtime" in self.config, has_system="system" in self.config, has_workflow_run_id=bool(self.run_id))
            self.logger.debug(f"{agent_name}.using_run_id", run_id=self.run_id)
            
            # Extract lineage config from runtime section
            lineage_config = self.config_node.get_value("runtime.lineage") or {}
            if not lineage_config:
                # Try llm_config path as fallback
                lineage_config = self.config_node.get_value("llm_config.agents.lineage") or {}
            
            # Initialize with run_id
            self.event_logger = EventLogger(
                lineage_config,
                self.run_id
            )
            
            # Set agent type and name in event logger
            self.event_logger.agent_name = agent_name
            self.event_logger.agent_type = self.agent_type if hasattr(self, 'agent_type') else agent_name
            
            self.logger.info(f"{agent_name}.event_logger_initialized", run_id=self.run_id)
        except Exception as e:
            self.logger.error(f"{agent_name}.event_logger_init_failed", error=str(e), exc_info=True)

        # --- Step 15: Initialize Continuation Handler ---
        # Ensure it's initialized AFTER all necessary parent attributes (like logger, model_str) are set
        self._continuation_handler: Optional[ContinuationHandler] = ContinuationHandler(self)

        # --- Step 16: Final Log ---
        self.logger.info(f"{agent_name}.initialized",
                     persona_key=self.persona_key,
                     has_persona_config=bool(self.persona_path),
                     continuation_settings={
                         "max_attempts": self.max_continuation_attempts,
                         "token_buffer": self.continuation_token_buffer
                     })


    def _get_workflow_run_id(self) -> Optional[str]:
        """
        Extract workflow run ID from configuration using hierarchical path queries.
        
        IMPORTANT: This method is NOT intended to be overridden in the generic agent model.
        It provides standard configuration path resolution for workflow run IDs.
        The hierarchical search ensures compatibility with various configuration structures.
        
        Returns:
            Workflow run ID or None if not found
        """
        # Check hierarchical sources in order of priority
        # Ensure self.config_node is used
        run_id = (
            self.config_node.get_value("workflow_run_id") or
            self.config_node.get_value("system.runid") or
            self.config_node.get_value("runtime.workflow_run_id") or
            self.config_node.get_value("runtime.run_id") or
            self.config_node.get_value("runtime.workflow.id")
        )

        if run_id:
            return str(run_id)
        return None

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Main process entry point for handling requests.
        
        This is the primary method that MAY be overridden by specific agent implementations.
        It will fall back to a simple LLM interaction via _process(context) if not overridden.
        
        Note: Even when overriding this method, implementations should generally use the
        persona-based configuration approach rather than hardcoding prompts or behavior.
        
        Args:
            context: Input context containing data for processing
            
        Returns:
            Standard AgentResponse with results
        """
        return self._process(context)

    def _prepare_lineage_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context with appropriate lineage tracking IDs.
        Ensures each execution has proper parent-child relationships.
        
        IMPORTANT: This method is NOT intended to be overridden in the generic agent model.
        It provides standard lineage context handling that works for all agent types.
        
        Args:
            context: The original context dictionary
            
        Returns:
            Context with added lineage tracking identifiers
        """
        # Extract workflow run ID from context or config
        workflow_run_id = context.get("workflow_run_id", self.run_id)

        # Extract parent ID if available
        parent_id = context.get("parent_id")

        # Check if context already has agent_execution_id
        if "agent_execution_id" in context:
            # Context already has tracking IDs, preserve them
            return context

        # No tracking IDs, create them using LineageContext utility
        # If there's a parent ID, we're being called by another agent
        if parent_id:
            # We're a sub-component (skill) being called by an agent
            return LineageContext.create_skill_context(
                agent_id=parent_id, # Use agent's unique instance ID
                skill_type=self._get_agent_name(),
                workflow_run_id=workflow_run_id,
                base_context=context
            )
        else:
            # We're a top-level agent in the workflow
            return LineageContext.create_agent_context(
                workflow_run_id=workflow_run_id,
                agent_type=self._get_agent_name(),
                base_context=context
            )

    def _process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Core internal processing method for agent functionality.
        
        IMPORTANT: This method is NOT intended to be overridden in the generic agent model.
        The core process logic is standardized for all agents, following these steps:
        1. Prepare lineage context tracking
        2. Extract primary data
        3. Get prompts and messages from configuration
        4. Call LLM with appropriate logging and error handling
        5. Process and return response
        
        Instead of overriding this method, agent implementations should override the public
        'process' method to add specialized pre/post processing while still leveraging this
        core implementation when needed.
        
        Args:
            context: Context dictionary with request information
            
        Returns:
            AgentResponse with results from the LLM interaction
        """
        try:
            if self._should_log(LogDetail.DETAILED): # Use self.logger below
                self.logger.info("agent.processing", context_keys=list(context.keys()) if context else None)

            # Prepare lineage tracking context
            lineage_context = self._prepare_lineage_context(context)
            agent_execution_id = lineage_context.get("agent_execution_id")
            parent_id = lineage_context.get("parent_id")

            self.logger.debug("agent.lineage_context", # Use self.logger
                        agent_id=self.agent_id,
                        agent_execution_id=agent_execution_id,
                        parent_id=parent_id, #
                        workflow_run_id=lineage_context.get("workflow_run_id"))

            # Extract data from context
            data = self._get_data(lineage_context)

            # Prepare system and user messages with context for potential overrides
            system_message = self._get_system_message(lineage_context)
            user_message = self._format_request(data, lineage_context)

            if self._should_log(LogDetail.DEBUG): # Use self.logger below
                self.logger.debug("agent.messages",
                            system_length=len(system_message),
                            user_length=len(user_message),
                            user_message=user_message[:10] + "..." if len(user_message) > 10 else user_message)

            # Create complete messages object for LLM and lineage tracking
            messages = LLMMessages(
                system=system_message,
                user=user_message,
                formatted_request="",  # Don't store duplicate content
                raw_context=lineage_context
            )

            try:
                # Check if event logger is enabled
                event_logger_enabled = hasattr(self, 'event_logger') and self.event_logger and getattr(self.event_logger, 'enabled', False)

                # Log STEP_START event
                if event_logger_enabled:
                    try:
                        # Extract config metadata
                        config_snapshot_path = lineage_context.get("config_snapshot_path")
                        config_hash = lineage_context.get("config_hash")
                        
                        # Log step start event
                        self.event_logger.log_event(
                            EventType.STEP_START,
                            {
                                "step_type": self._get_agent_name(),
                                "context_keys": list(lineage_context.keys()),
                                "agent_id": self.agent_id
                            },
                            step_name=self._get_agent_name(),
                            parent_id=parent_id,
                            config_snapshot_path=config_snapshot_path,
                            config_hash=config_hash
                        )
                        self.logger.info("event_logger.step_start_logged",
                                         agent=self._get_agent_name(),
                                         agent_execution_id=agent_execution_id)
                    except Exception as e:
                        self.logger.error("event_logger.step_start_failed",
                                          error=str(e),
                                          error_type=type(e).__name__,
                                          agent=self._get_agent_name(),
                                          agent_execution_id=agent_execution_id)

                # Get completion with automatic continuation handling (calls BaseLLM method)
                # Pass lineage_context to support runtime configuration overrides
                content, raw_response = self._get_completion_with_continuation(
                    [
                        {"role": "system", "content": messages.system},
                        {"role": "user", "content": messages.user}
                    ],
                    context=lineage_context
                )

                # Process response - Calls THIS class's _process_response
                processed_data = self._process_response(content, raw_response)

                # Add execution metadata
                processed_data["execution_metadata"] = {
                    "agent_execution_id": agent_execution_id,
                    "parent_id": parent_id,
                    "workflow_run_id": lineage_context.get("workflow_run_id"),
                    "agent_id": self.agent_id,
                    "agent_type": self._get_agent_name(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                # Calculate metrics
                response_metrics = {"token_usage": getattr(raw_response, 'usage', {})}

                # Log STEP_END event
                if event_logger_enabled:
                    try:
                        # Extract config metadata
                        config_snapshot_path = lineage_context.get("config_snapshot_path")
                        config_hash = lineage_context.get("config_hash")
                        
                        # Create a more detailed event payload similar to the original BaseLineage
                        event_payload = {
                            "step_type": self._get_agent_name(),
                            "agent_response_summary": {
                                "success": True,
                                "content": processed_data.get("response", "")
                            },
                            "metrics": response_metrics,
                            "llm_input": {
                                "system_message": messages.system,
                                "user_message": messages.user,
                                "formatted_request": messages.formatted_request if hasattr(messages, "formatted_request") else ""
                            },
                            # Store the full raw response to preserve all data
                            "llm_output": self.event_logger._serialize_value(raw_response),
                            # Store model information
                            "llm_model": {
                                "provider": self.provider.value if hasattr(self, 'provider') else "unknown",
                                "model": self.model if hasattr(self, 'model') else "unknown"
                            },
                            # Include the full context for completeness
                            "input_context": lineage_context
                        }
                        
                        # Log the comprehensive event
                        self.event_logger.log_event(
                            EventType.STEP_END,
                            event_payload,
                            step_name=self._get_agent_name(),
                            parent_id=parent_id,
                            config_snapshot_path=config_snapshot_path,
                            config_hash=config_hash
                        )
                        self.logger.info("event_logger.step_end_logged",
                                         agent=self._get_agent_name(),
                                         agent_execution_id=agent_execution_id)
                    except Exception as e:
                        self.logger.error("event_logger.step_end_failed",
                                          error=str(e),
                                          error_type=type(e).__name__,
                                          agent=self._get_agent_name(),
                                          agent_execution_id=agent_execution_id)
                else:
                    # Log when event logging is skipped
                    self.logger.debug("event_logger.tracking_skipped",
                                     has_event_logger=hasattr(self, 'event_logger'),
                                     event_logger_enabled=getattr(self.event_logger, 'enabled', False) if hasattr(self, 'event_logger') else False,
                                     agent=self._get_agent_name())

                # Return successful response with lineage tracking metadata
                return AgentResponse(
                    success=True,
                    data=processed_data,
                    error=None,
                    messages=messages,
                    raw_output=raw_response, # Keep raw_output in AgentResponse if needed elsewhere
                    metrics=response_metrics
                )
            except Exception as e:
                # Log ERROR_EVENT
                if event_logger_enabled:
                    try:
                        # Extract config metadata
                        config_snapshot_path = lineage_context.get("config_snapshot_path")
                        config_hash = lineage_context.get("config_hash")
                        
                        # Create a comprehensive error event payload
                        error_payload = {
                            "error_message": str(e),
                            "error_type": type(e).__name__,
                            "traceback": traceback.format_exc(),
                            "location": self._get_agent_name(),
                            "llm_input": {
                                "system_message": messages.system,
                                "user_message": messages.user,
                                "formatted_request": messages.formatted_request if hasattr(messages, "formatted_request") else ""
                            },
                            "llm_model": {
                                "provider": self.provider.value if hasattr(self, 'provider') else "unknown",
                                "model": self.model if hasattr(self, 'model') else "unknown"
                            },
                            # Include the full context for completeness
                            "input_context": lineage_context
                        }
                        
                        # Log error event
                        self.event_logger.log_event(
                            EventType.ERROR_EVENT,
                            error_payload,
                            step_name=self._get_agent_name(),
                            parent_id=parent_id,
                            config_snapshot_path=config_snapshot_path,
                            config_hash=config_hash
                        )
                        self.logger.info("event_logger.error_event_logged",
                                        agent=self._get_agent_name(),
                                        agent_execution_id=agent_execution_id)
                    except Exception as event_logger_error:
                        self.logger.error("event_logger.error_event_failed",
                                        error=str(event_logger_error),
                                        original_error=str(e))

                self.logger.error("llm.completion_failed",
                        error=str(e),
                        agent_execution_id=agent_execution_id,
                        exc_info=True) # Log traceback for completion errors

                return AgentResponse(
                    success=False,
                    data={
                        "execution_metadata": {
                            "agent_execution_id": agent_execution_id,
                            "parent_id": parent_id,
                            "workflow_run_id": lineage_context.get("workflow_run_id"),
                            "agent_id": self.agent_id,
                            "agent_type": self._get_agent_name(),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "error": str(e)
                        }
                    },
                    error=f"LLM completion failed: {str(e)}",
                    messages=messages
                )
        except Exception as e:
            self.logger.error("process.failed", error=str(e), exc_info=True) # Log traceback
            return AgentResponse(success=False, data={}, error=str(e))

    def _get_prompt(self, prompt_type: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get a prompt template with support for dynamic overrides.
        
        IMPORTANT: This uses standard configuration paths based on unique_name and persona_key.
        This method is NOT intended to be overridden in the generic agent model.
        All prompt templates are sourced from the persona configuration or runtime overrides,
        NOT from agent class-specific implementations.
        
        Args:
            prompt_type: Type of prompt to retrieve (e.g., 'user')
            context: Optional runtime context that may contain overrides
            
        Returns:
            Prompt template string from context override or persona config
        """
        # Initialize variables
        prompt = None
        prompt_key = prompt_type
        
        # First check for override in context if provided
        if context:
            # Look for override in agent_config_overrides structure
            overrides = context.get('agent_config_overrides', {}).get(self.unique_name, {})
            
            # Direct prompt override for this specific prompt_type
            prompt_override_key = f"{prompt_type}_prompt"
            if prompt_override_key in overrides:
                prompt = overrides[prompt_override_key]
                self.logger.debug("prompt.from_direct_override", 
                                agent_name=self.unique_name,
                                prompt_type=prompt_type,
                                prompt_length=len(prompt) if prompt else 0)
                
            # Or check for prompt_key override
            elif 'prompt_key' in overrides:
                prompt_key = overrides['prompt_key']
                self.logger.debug("prompt.using_key_override", 
                                agent_name=self.unique_name,
                                original_key=prompt_type,
                                override_key=prompt_key)
        
        # If no direct prompt override was found, get from persona using prompt_key
        if prompt is None:
            persona_prompt_path = f"{self.persona_path}.prompts.{prompt_key}"
            prompt = self.config_node.get_value(persona_prompt_path)
            
            if prompt is None:
                # Use self.logger which is initialized
                agent_name = self._get_agent_name()
                self.logger.error("prompt.template_not_found", 
                               prompt_type=prompt_key, 
                               agent=agent_name,
                               persona_path=persona_prompt_path)
                raise ValueError(f"No prompt template '{prompt_key}' found in persona '{self.persona_key}' for agent {agent_name}")
            
        # Ensure prompt is a string before returning
        return str(prompt) if prompt is not None else ""

    def _format_request(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format request data using a template from persona configuration.
        
        Args:
            data: Data to format
            context: Optional context that may contain overrides
            
        Returns:
            Formatted request string
        """
        try:
             # Check for a format template override in context
             template = None
             template_key = "user"  # Default template key
             
             # Check for overrides in context if provided
             if context:
                 overrides = context.get('agent_config_overrides', {}).get(self.unique_name, {})
                 
                 # Direct template override
                 if 'template' in overrides:
                     template = overrides['template']
                     self.logger.debug("format_request.using_direct_template_override", 
                                     agent_name=self.unique_name,
                                     template_length=len(template) if template else 0)
                 
                 # Template key override
                 elif 'template_key' in overrides:
                     template_key = overrides['template_key']
                     self.logger.debug("format_request.using_template_key_override", 
                                     agent_name=self.unique_name,
                                     template_key=template_key)
             
             # If no direct template override, get template from persona
             if template is None:
                 # Get template using updated _get_prompt method (which also handles overrides)
                 template = self._get_prompt(template_key, context)
             
             # Format the template with data if it contains format placeholders
             if isinstance(template, str) and '{' in template:
                 try:
                     return template.format(**data)
                 except KeyError as e:
                     self.logger.warning("format_request.missing_key_in_template", error=str(e))
                     # Fall back to JSON format on template key error
                     return json.dumps(data, indent=2, default=str)
             elif template:
                 # Template exists but doesn't need formatting
                 return str(template)
             else:
                 # No template, use default JSON format
                 self.logger.debug("format_request.using_default_json")
                 return json.dumps(data, indent=2, default=str)
        except Exception as e:
             self.logger.error("format_request.failed", error=str(e))
             return str(data) # Ultimate fallback