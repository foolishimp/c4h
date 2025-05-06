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
                            agent_execution_id=agent_execution_id,
                            system=system_message[:10] + "..." if len(system_message) > 10 else system_message,
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

    def _get_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts primary data payload, ensuring it's a dictionary.
        
        IMPORTANT: This method is NOT intended to be overridden in the generic agent model.
        It provides standard data extraction that works for all agent types.
        
        Args:
            context: The context dictionary to extract data from
            
        Returns:
            Dictionary containing the primary payload data
        """
        # Use self.logger which should be initialized
        logger_to_use = self.logger
        try:
            # Prioritize 'input_data' as the primary container
            if 'input_data' in context and isinstance(context['input_data'], dict):
                 logger_to_use.debug("_get_data: using 'input_data' key")
                 return context['input_data']
            # If input_data isn't a dict or doesn't exist, use the context itself if it's a dict
            elif isinstance(context, dict):
                 logger_to_use.debug("_get_data: using context as data (input_data missing or not dict)")
                 return context
            # Fallback: create a basic dict if context isn't suitable
            logger_to_use.warning("_get_data: context is not a suitable dict, creating basic wrapper", context_type=type(context).__name__)
            return {'content': str(context)}
        except Exception as e:
            logger_to_use.error("_get_data.failed", error=str(e))
            return {}

    def _format_request(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
         """
         Format request with support for dynamic overrides.
         
         IMPORTANT: This uses standard configuration paths and template formatting.
         This method is NOT intended to be overridden in the generic agent model.
         All formatting templates are sourced from the persona configuration or runtime overrides,
         NOT from agent class-specific implementations.
         
         Args:
             data: The data to format
             context: Optional runtime context that may contain overrides
             
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

    def _get_llm_content(self, response: Any) -> Any:
        """
        Extract content from LLM response or context data with robust error handling.
        
        This method provides comprehensive content extraction from various LLM response formats.
        It robustly handles different response structures including standard LLM objects, 
        strings, nested dictionaries, and more.
        
        While agent implementations MAY override this method for highly specialized content 
        extraction, the base implementation is quite robust and should handle most cases.
        
        Args:
            response: The raw response from an LLM or other response object
            
        Returns:
            Extracted content in string format
        """
        logger_to_use = getattr(self, 'logger', get_logger())
        response_repr = repr(response)
        logger_to_use.debug("_get_llm_content.received",
                     input_type=type(response).__name__,
                     input_repr_preview=response_repr[:200] + "..." if len(response_repr) > 200 else response_repr)
        try:
            # 1. Handle standard LLM response objects
            if hasattr(response, 'choices') and response.choices:
                logger_to_use.debug("_get_llm_content.checking_choices")
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    content = response.choices[0].message.content
                    logger_to_use.debug("content.extracted_from_model", content_length=len(content) if content else 0)
                    return content
                elif hasattr(response.choices[0], 'delta') and hasattr(response.choices[0].delta, 'content'):
                    content = response.choices[0].delta.content
                    logger_to_use.debug("content.extracted_from_delta", content_length=len(content) if content else 0)
                    return content

            # 2. Handle direct string input
            if isinstance(response, str):
                logger_to_use.debug("content.extracted_direct_string", content_length=len(response))
                return response

            # 3. Handle dictionary inputs
            if isinstance(response, dict):
                logger_to_use.debug("_get_llm_content.checking_dict_keys", keys=list(response.keys()))

                # Check nested {'response': {'content': '...'}} structure FIRST
                if 'response' in response:
                    response_val = response['response']
                    logger_to_use.debug("_get_llm_content.found_response_key", response_val_type=type(response_val).__name__)
                    if isinstance(response_val, str):
                        logger_to_use.debug("content.extracted_from_dict_response_key", content_length=len(response_val))
                        return response_val
                    elif isinstance(response_val, dict) and 'content' in response_val:
                         content_val = response_val['content']
                         if isinstance(content_val, str):
                              logger_to_use.debug("content.extracted_from_nested_dict_response_content_key", content_length=len(content_val))
                              return content_val
                         else:
                              logger_to_use.warning("_get_llm_content.nested_content_not_string", nested_content_type=type(content_val).__name__)
                    # Fall through if response['response'] is dict but has no 'content'

                # Check for {'content': '...'}
                if 'content' in response and isinstance(response['content'], str):
                     logger_to_use.debug("content.extracted_from_dict_content_key", content_length=len(response['content']))
                     return response['content']

                # Check for apply_diff structure {'llm_output': {'content': '...'}}
                if 'llm_output' in response and isinstance(response['llm_output'], dict):
                    logger_to_use.debug("_get_llm_content.checking_dict_llm_output_key", llm_output_keys=list(response['llm_output'].keys()))
                    if 'content' in response['llm_output'] and isinstance(response['llm_output']['content'], str):
                         logger_to_use.debug("content.extracted_from_dict_llm_output_content_key", content_length=len(response['llm_output']['content']))
                         return response['llm_output']['content']

            # 4. Last resort fallback - convert to string
            result = str(response)
            logger_to_use.warning("content.extraction_fallback",
                        response_type=type(response).__name__,
                        content_preview=result[:100] if len(result) > 100 else result)
            return result

        except Exception as e:
            logger_to_use.error("content_extraction.failed", error=str(e), exc_info=True)
            return str(response)


    def _process_response(self, content: str, raw_response: Any) -> Dict[str, Any]:
        """
        Process LLM response into standard format.
        
        This method provides basic processing of LLM responses into a standard dictionary format.
        Advanced agent implementations may override this method to provide specialized response 
        processing, but should maintain compatibility with the expected return format.
        
        Args:
            content: The content string from the LLM
            raw_response: The raw response object from the LLM
            
        Returns:
            Dictionary containing processed response data
        """
        # Use self.logger which should be initialized in BaseAgent __init__
        logger_to_use = self.logger # Use the instance logger
        try:
            # Extract content using THIS class's _get_llm_content method
            processed_content = self._get_llm_content(content) # Calls the overridden method
            # Check if logger exists before logging
            if logger_to_use and self._should_log(LogDetail.DEBUG):
                logger_to_use.debug("agent.processing_response",
                            content_length=len(str(processed_content)) if processed_content else 0,
                            response_type=type(raw_response).__name__)

            # Create standard response structure
            response = {
                "response": processed_content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Add token usage metrics if available
            if hasattr(raw_response, 'usage'):
                usage = raw_response.usage
                usage_data = {
                    "completion_tokens": getattr(usage, 'completion_tokens', 0),
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                    "total_tokens": getattr(usage, 'total_tokens', 0)
                }
                # Check if logger exists before logging
                if logger_to_use: logger_to_use.info("llm.token_usage", **usage_data)
                response["usage"] = usage_data

            return response
        except Exception as e:
            # Check if logger exists before logging
            if logger_to_use: logger_to_use.error("response_processing.failed", error=str(e), exc_info=True) # Added traceback
            return {
                "response": str(content),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }

    def _get_required_keys(self) -> List[str]:
        """
        Returns required keys for context validation.
        
        IMPORTANT: This method is NOT intended to be overridden in the generic agent model.
        In the persona-based configuration approach, required keys should be specified in 
        the persona configuration, not through class-specific overrides.
        
        Returns:
            List of required keys for context validation
        """
        return []

    def _get_agent_name(self) -> str:
        """
        Returns the unique name assigned to this agent instance.
        
        IMPORTANT: This method is NOT intended to be overridden in the generic agent model.
        In the persona-based configuration approach, the agent name comes from the unique_name 
        parameter provided during instantiation, NOT from class-specific overrides.
        
        Returns:
            The unique name of this agent instance
        """
        if hasattr(self, 'unique_name') and self.unique_name:
             return self.unique_name
        else:
             # Fallback, though unique_name should always be set by the factory
             # Use self.logger if available, otherwise print a warning
             logger_instance = getattr(self, 'logger', None)
             if logger_instance:
                  logger_instance.warning("_get_agent_name called before unique_name was set or unique_name is empty.")
             else:
                  print("Warning: _get_agent_name called before unique_name was set or unique_name is empty.")
             # Attempt to derive from class name as a last resort, but this deviates from the new design
             class_name = self.__class__.__name__
             name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
             if name.endswith("_agent"): name = name[:-6]
             return name or "base_agent"

    def _get_system_message(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get system message with support for dynamic overrides.
        
        IMPORTANT: This uses standard configuration paths based on unique_name and persona_key.
        This method is NOT intended to be overridden in the generic agent model.
        All configuration is sourced from the persona configuration or runtime overrides,
        NOT from agent class-specific implementations.
        
        Args:
            context: Optional runtime context that may contain overrides
            
        Returns:
            System message string from context override or persona config
        """
        # Initialize system_prompt to None to track source
        system_prompt = None
        
        # First check for override in context if provided
        if context:
            # Look for override in agent_config_overrides structure
            overrides = context.get('agent_config_overrides', {}).get(self.unique_name, {})
            
            # Direct system_prompt override
            if 'system_prompt' in overrides:
                system_prompt = overrides['system_prompt']
                self.logger.debug("system_prompt.from_context_override", 
                                agent_name=self.unique_name,
                                prompt_length=len(system_prompt) if system_prompt else 0)
                
            # Or check for system_prompt_key override (key in persona.prompts)
            elif 'system_prompt_key' in overrides:
                prompt_key = overrides['system_prompt_key']
                system_prompt = self.config_node.get_value(f"{self.persona_path}.prompts.{prompt_key}")
                self.logger.debug("system_prompt.from_key_override", 
                                agent_name=self.unique_name,
                                key=prompt_key,
                                found=system_prompt is not None)
        
        # If no override was found, use default from persona
        if system_prompt is None:
            system_prompt = self.config_node.get_value(f"{self.persona_path}.prompts.system")
            
            if not system_prompt:
                self.logger.warning("system_prompt.not_found", 
                                  persona_key=self.persona_key,
                                  persona_path=f"{self.persona_path}.prompts.system")
                
        return system_prompt or ""


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


    def call_skill(self, skill_name: str, skill_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a skill with proper lineage tracking.
        
        IMPORTANT: This method is NOT intended to be overridden in the generic agent model.
        It provides standard lineage tracking for skill calls that works for all agent types.
        
        Args:
            skill_name: Name of the skill to call
            skill_context: Context to pass to the skill

        Returns:
            Result from the skill (actually returns enhanced context for skill execution)
        """
        # Use self.logger which is initialized
        logger_to_use = self.logger
        try:
            # Prepare lineage tracking context for the skill
            # Use self.run_id which should be set during BaseAgent init
            current_run_id = getattr(self, 'run_id', None)
            if not current_run_id:
                 logger_to_use.warning("call_skill invoked before run_id was set.")
                 current_run_id = "unknown_run_id" # Fallback

            lineage_skill_context = LineageContext.create_skill_context(
                agent_id=self.agent_id, # Use agent's unique instance ID
                skill_type=skill_name,
                workflow_run_id=current_run_id,
                base_context=skill_context
            )

            logger_to_use.debug("agent.calling_skill",
                    agent_id=self.agent_id,
                    skill=skill_name,
                    skill_execution_id=lineage_skill_context.get("agent_execution_id"))

            # Return enhanced context - the skill itself will handle execution
            return lineage_skill_context
        except Exception as e:
             logger_to_use.error("agent.skill_context_failed",
                    error=str(e),
                    skill=skill_name,
                    exc_info=True) # Log traceback

        # If lineage context fails, fall back to original context
        return skill_context
    
    def _invoke_skill(self, skill_identifier: str, skill_kwargs: Dict[str, Any]) -> SkillResult:
        """
        Dynamically invoke a skill by identifier.
        
        IMPORTANT: This method is NOT intended to be overridden in the generic agent model.
        It provides standard skill invocation logic for all agent types.
        
        Args:
            skill_identifier: String in format "module.Class.method" or "module.Class"
            skill_kwargs: Arguments to pass to the skill execute method
            
        Returns:
            SkillResult from the skill execution
        """
        logger_to_use = self.logger
        try:
            # Parse the skill identifier
            parts = skill_identifier.split('.')
            
            # Need at least module and class
            if len(parts) < 2:
                logger_to_use.error("skill.invalid_identifier", identifier=skill_identifier)
                return SkillResult(
                    success=False, 
                    error=f"Invalid skill identifier format: {skill_identifier}. Expected 'module.Class' or 'module.Class.method'"
                )
                
            module_path = '.'.join(parts[:-1]) if len(parts) > 2 else parts[0]
            class_name = parts[-1] if len(parts) <= 2 else parts[-2]
            method_name = 'execute' if len(parts) <= 2 else parts[-1]
            
            logger_to_use.debug("skill.resolution", 
                              module=module_path, 
                              class_name=class_name, 
                              method=method_name)
            
            # Import the module
            try:
                # Try relative to c4h_agents first
                try:
                    full_module_path = f"c4h_agents.{module_path}"
                    module = __import__(full_module_path, fromlist=[class_name])
                except (ImportError, ModuleNotFoundError):
                    # If that fails, try the direct path
                    module = __import__(module_path, fromlist=[class_name])
                    
                # Get the class
                skill_class = getattr(module, class_name)
                
            except (ImportError, ModuleNotFoundError) as e:
                logger_to_use.error("skill.module_import_failed", 
                                  module=module_path, 
                                  error=str(e))
                return SkillResult(
                    success=False,
                    error=f"Failed to import skill module '{module_path}': {str(e)}"
                )
            except AttributeError as e:
                logger_to_use.error("skill.class_not_found", 
                                  module=module_path, 
                                  class_name=class_name, 
                                  error=str(e))
                return SkillResult(
                    success=False, 
                    error=f"Skill class '{class_name}' not found in module '{module_path}': {str(e)}"
                )
                
            # Instantiate the skill with our config
            try:
                # Pass our effective config to the skill
                skill_instance = skill_class(self.config, class_name)
            except Exception as e:
                logger_to_use.error("skill.instantiation_failed", 
                                  class_name=class_name, 
                                  error=str(e))
                return SkillResult(
                    success=False, 
                    error=f"Failed to instantiate skill '{class_name}': {str(e)}"
                )
                
            # Get the method
            try:
                skill_method = getattr(skill_instance, method_name)
            except AttributeError as e:
                logger_to_use.error("skill.method_not_found", 
                                  class_name=class_name, 
                                  method_name=method_name, 
                                  error=str(e))
                return SkillResult(
                    success=False, 
                    error=f"Method '{method_name}' not found in skill '{class_name}': {str(e)}"
                )
                
            # Prepare lineage context for skill execution
            try:
                # Add lineage tracking context
                skill_context = self.call_skill(class_name, skill_kwargs)
                skill_kwargs.update(skill_context)
            except Exception as e:
                logger_to_use.warning("skill.lineage_preparation_failed", 
                                   error=str(e), 
                                   proceeding="without_lineage")
                # Proceed without lineage tracking
            
            # Call the method
            try:
                result = skill_method(**skill_kwargs)
                
                # Make sure the result is a SkillResult
                if not isinstance(result, SkillResult):
                    logger_to_use.warning("skill.invalid_result_type", 
                                       expected="SkillResult", 
                                       actual=type(result).__name__)
                    # Wrap in SkillResult if not already
                    if hasattr(result, 'success'):
                        # Has success attribute - likely compatible
                        return SkillResult(
                            success=getattr(result, 'success', False),
                            value=result,
                            error=getattr(result, 'error', None)
                        )
                    else:
                        # No success attribute - treat as raw value
                        return SkillResult(
                            success=True,
                            value=result
                        )
                        
                return result
                
            except Exception as e:
                logger_to_use.error("skill.execution_failed", 
                                  class_name=class_name, 
                                  method_name=method_name, 
                                  error=str(e),
                                  exc_info=True)
                return SkillResult(
                    success=False, 
                    error=f"Skill execution failed: {str(e)}"
                )
                
        except Exception as e:
            logger_to_use.error("skill.invocation_failed", 
                              identifier=skill_identifier, 
                              error=str(e),
                              exc_info=True)
            return SkillResult(
                success=False, 
                error=f"Skill invocation failed: {str(e)}"
            )