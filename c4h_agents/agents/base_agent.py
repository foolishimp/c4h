# Path: /Users/jim/src/apps/c4h/c4h_agents/agents/base_agent.py
"""
Base agent implementation with enhanced lineage tracking.
Path: c4h_agents/agents/base_agent.py

Updates:
- Initialize logger with configuration to ensure proper truncation
- Initialize _continuation_handler in BaseAgent init
"""

from typing import Dict, Any, Optional, List, Tuple
import structlog
import json
from pathlib import Path
from datetime import datetime, timezone # Ensure timezone is imported
import uuid
import re # Import re for _get_agent_name fallback

from c4h_agents.core.project import Project
from c4h_agents.config import create_config_node # Removed unused locate_config, get_value
from c4h_agents.utils.logging import get_logger
from .base_lineage import BaseLineage
from .lineage_context import LineageContext
from .types import LogDetail, LLMProvider, LLMMessages, AgentResponse, AgentMetrics
from .base_config import BaseConfig, log_operation
from .base_llm import BaseLLM
# Import ContinuationHandler if needed for type hint, but don't instantiate here
from .continuation.continuation_handler import ContinuationHandler


logger = get_logger() # Global logger for module-level logging

class BaseAgent(BaseConfig, BaseLLM):
    """Base agent implementation"""

    def __init__(self, config: Dict[str, Any] = None, project: Optional[Project] = None):
        # Pass full config to BaseConfig first
        BaseConfig.__init__(self, config=config, project=project)
        # Then initialize BaseLLM part AFTER BaseConfig setup config_node etc.
        BaseLLM.__init__(self) # Explicitly call BaseLLM init

        # --- FIX: Initialize _continuation_handler in BaseAgent ---
        # This ensures subclasses inheriting from BaseAgent have it initialized
        self._continuation_handler: Optional[ContinuationHandler] = None
        # --- END FIX ---


        # Create configuration node for hierarchical access (already done in BaseConfig)
        # self.config_node = create_config_node(self.config) # Redundant
        agent_name = self._get_agent_name()

        # Generate stable agent instance ID
        self.agent_id = str(uuid.uuid4())

        # Ensure system namespace exists
        if "system" not in self.config:
            self.config["system"] = {}

        # Store agent ID in system namespace
        self.config["system"]["agent_id"] = self.agent_id

        # Initialize logger with enhanced context and agent configuration
        log_context = {
            "agent": agent_name,
            "agent_id": self.agent_id,
            # provider, model, log_level added later after resolution
        }
        if self.project:
            log_context.update({
                "project_name": self.project.metadata.name,
                "project_version": self.project.metadata.version,
                "project_root": str(self.project.paths.root)
            })

        # Bind context directly to the logger obtained using the config
        # We create the logger instance here but will re-bind later with more context
        self.logger = get_logger(self.config).bind(**log_context)

        # Log the full configuration received by the agent
        try:
            # Use self.logger and pass dict directly, structlog handles serialization/truncation
            self.logger.debug(f"{agent_name}.__init__.received_config", config_data=self.config)
        except Exception as e:
            self.logger.error(f"{agent_name}.__init__.config_dump_failed", error=str(e)) # Use self.logger

        # Resolve provider, model, and temperature using hierarchical lookup
        agent_path = f"llm_config.agents.{agent_name}"
        # --- Ensure self.config_node is used for lookup ---
        provider_name = self.config_node.get_value(f"{agent_path}.provider") or self.config_node.get_value("llm_config.default_provider") or "anthropic"
        self.logger.debug(f"{agent_name}.__init__.resolved_provider", provider_name=provider_name) # Use self.logger
        self.provider = LLMProvider(provider_name)

        self.model = self.config_node.get_value(f"{agent_path}.model") or self.config_node.get_value("llm_config.default_model") or "claude-3-opus-20240229"
        self.temperature = self.config_node.get_value(f"{agent_path}.temperature") or 0
        self.logger.debug(f"{agent_name}.__init__.resolved_model", model_name=self.model) # Use self.logger

        # Continuation settings
        # --- Ensure self.config_node is used for lookup ---
        self.max_continuation_attempts = self.config_node.get_value(f"{agent_path}.max_continuation_attempts") or 5
        self.continuation_token_buffer = self.config_node.get_value(f"{agent_path}.continuation_token_buffer") or 1000

        # Initialize metrics
        self.metrics = AgentMetrics(project=self.project.metadata.name if self.project else None)

        # Set logging detail level from config
        # --- Ensure self.config_node is used for lookup ---
        log_level_str = self.config_node.get_value("logging.agent_level") or self.config_node.get_value("logging.level") or "basic"
        self.log_level = LogDetail.from_str(log_level_str)

        # Build model string and setup LiteLLM
        self.model_str = self._get_model_str() # This now correctly uses self.provider/self.model
        self._setup_litellm(self._get_provider_config(self.provider)) # _get_provider_config uses config_node

        # --- Re-bind logger with full context ---
        log_context.update({
             "provider": self.provider.serialize(),
             "model": self.model,
             "log_level": str(self.log_level)
        })
        self.logger = self.logger.bind(**log_context)
        # --- End Re-bind ---

        # Initialize lineage tracking with the full configuration
        self.lineage = None
        try:
            # Log what run_id we're using
            run_id = self._get_workflow_run_id() # Uses self.config_node
            if run_id:
                self.logger.debug(f"{agent_name}.using_workflow_run_id",
                            run_id=run_id,
                            config_keys=list(self.config.keys()),
                            source="config")

            self.lineage = BaseLineage(
                namespace="c4h_agents",
                agent_name=agent_name,
                config=self.config # Pass the current, potentially updated, config
            )

            # Store lineage run ID for consistency
            self.run_id = self.lineage.run_id

        except Exception as e:
            self.logger.error(f"{agent_name}.lineage_init_failed", error=str(e)) # Use self.logger
            # Generate run ID if lineage fails
            self.run_id = str(uuid.uuid4())

        self.logger.info(f"{agent_name}.initialized", # Use self.logger
                    run_id=self.run_id,
                    continuation_settings={
                        "max_attempts": self.max_continuation_attempts,
                        "token_buffer": self.continuation_token_buffer
                    },
                    **log_context) # Pass context directly

    def _get_workflow_run_id(self) -> Optional[str]:
        """Extract workflow run ID from configuration using hierarchical path queries"""
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
        """Main process entry point"""
        return self._process(context)

    def _prepare_lineage_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context with appropriate lineage tracking IDs.
        Ensures each execution has proper parent-child relationships.
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

            # Prepare system and user messages
            system_message = self._get_system_message()
            user_message = self._format_request(data)

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
                # Check if lineage tracking is enabled
                lineage_enabled = hasattr(self, 'lineage') and self.lineage and getattr(self.lineage, 'enabled', False)

                # Get completion with automatic continuation handling (calls BaseLLM method)
                content, raw_response = self._get_completion_with_continuation([
                    {"role": "system", "content": messages.system},
                    {"role": "user", "content": messages.user}
                ])

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

                # Track lineage if enabled
                if lineage_enabled:
                    try:
                        self.logger.debug("lineage.tracking_attempt", # Use self.logger
                                    agent=self._get_agent_name(),
                                    agent_execution_id=agent_execution_id,
                                    parent_id=parent_id, #
                                    has_context=bool(lineage_context),
                                    has_messages=bool(messages),
                                    has_metrics=hasattr(raw_response, 'usage'))

                        # Track LLM interaction with full context for event sourcing
                        if hasattr(self.lineage, 'track_llm_interaction'):
                            self.lineage.track_llm_interaction(
                                context=lineage_context,
                                messages=messages,
                                response=raw_response,
                                metrics=response_metrics
                            )
                        self.logger.info("lineage.tracking_complete", # Use self.logger
                                agent=self._get_agent_name(),
                                agent_execution_id=agent_execution_id)
                    except Exception as e: # Use self.logger below
                        self.logger.error("lineage.tracking_failed",
                                    error=str(e),
                                    error_type=type(e).__name__,
                                    agent=self._get_agent_name(),
                                    agent_execution_id=agent_execution_id)
                else:
                    # --- MODIFIED: Use self.logger ---
                    self.logger.debug("lineage.tracking_skipped",
                            has_lineage=hasattr(self, 'lineage'),
                            lineage_enabled=getattr(self.lineage, 'enabled', False) if hasattr(self, 'lineage') else False,
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
                # Handle errors with lineage tracking
                if lineage_enabled and hasattr(self.lineage, 'track_llm_interaction'):
                    try:
                        error_context = {
                            **lineage_context,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                        self.lineage.track_llm_interaction(
                            context=error_context,
                            messages=messages,
                            response={"error": str(e)},
                            metrics={"error": True}
                        )
                    except Exception as lineage_error: # Use self.logger below
                        self.logger.error("lineage.failure_tracking_failed",
                                    error=str(lineage_error),
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
        """Extracts primary data payload, ensuring it's a dictionary."""
        logger_to_use = getattr(self, 'logger', logger)
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


    def _format_request(self, context: Dict[str, Any]) -> str:
         # Default implementation, subclasses should override if specific formatting is needed
         # Safely convert context to string for basic use
         try:
              return json.dumps(context, indent=2, default=str) # Safer conversion
         except Exception:
              return str(context) # Fallback


    # --- THIS IS THE OVERRIDDEN METHOD IN BaseAgent ---
    # --- Includes fix for nested 'response' and 'llm_output' ---
    def _get_llm_content(self, response: Any) -> Any:
        """Extract content from LLM response or context data with robust error handling."""
        logger_to_use = getattr(self, 'logger', get_logger())
        response_repr = repr(response)
        logger_to_use.debug("_get_llm_content.received (BaseAgent)",
                     input_type=type(response).__name__,
                     input_repr_preview=response_repr[:200] + "..." if len(response_repr) > 200 else response_repr)
        try:
            # 1. Handle standard LLM response objects
            if hasattr(response, 'choices') and response.choices:
                logger_to_use.debug("_get_llm_content.checking_choices (BaseAgent)")
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    content = response.choices[0].message.content
                    logger_to_use.debug("content.extracted_from_model (BaseAgent)", content_length=len(content) if content else 0)
                    return content
                elif hasattr(response.choices[0], 'delta') and hasattr(response.choices[0].delta, 'content'):
                    content = response.choices[0].delta.content
                    logger_to_use.debug("content.extracted_from_delta (BaseAgent)", content_length=len(content) if content else 0)
                    return content

            # 2. Handle direct string input
            if isinstance(response, str):
                logger_to_use.debug("content.extracted_direct_string (BaseAgent)", content_length=len(response))
                return response

            # 3. Handle dictionary inputs
            if isinstance(response, dict):
                logger_to_use.debug("_get_llm_content.checking_dict_keys (BaseAgent)", keys=list(response.keys()))

                # Check nested {'response': {'content': '...'}} structure FIRST
                if 'response' in response:
                    response_val = response['response']
                    logger_to_use.debug("_get_llm_content.found_response_key (BaseAgent)", response_val_type=type(response_val).__name__)
                    if isinstance(response_val, str):
                        logger_to_use.debug("content.extracted_from_dict_response_key (string, BaseAgent)", content_length=len(response_val))
                        return response_val
                    elif isinstance(response_val, dict) and 'content' in response_val:
                         content_val = response_val['content']
                         if isinstance(content_val, str):
                              logger_to_use.debug("content.extracted_from_nested_dict_response_content_key (BaseAgent)", content_length=len(content_val))
                              return content_val
                         else:
                              logger_to_use.warning("_get_llm_content.nested_content_not_string (BaseAgent)", nested_content_type=type(content_val).__name__)
                    # Fall through if response['response'] is dict but has no 'content'

                # Check for {'content': '...'}
                if 'content' in response and isinstance(response['content'], str):
                     logger_to_use.debug("content.extracted_from_dict_content_key (BaseAgent)", content_length=len(response['content']))
                     return response['content']

                # Check for apply_diff structure {'llm_output': {'content': '...'}}
                if 'llm_output' in response and isinstance(response['llm_output'], dict):
                    logger_to_use.debug("_get_llm_content.checking_dict_llm_output_key (BaseAgent)", llm_output_keys=list(response['llm_output'].keys()))
                    if 'content' in response['llm_output'] and isinstance(response['llm_output']['content'], str):
                         logger_to_use.debug("content.extracted_from_dict_llm_output_content_key (BaseAgent)", content_length=len(response['llm_output']['content']))
                         return response['llm_output']['content']

            # 4. Last resort fallback - convert to string
            result = str(response)
            logger_to_use.warning("content.extraction_fallback (BaseAgent)",
                        response_type=type(response).__name__,
                        content_preview=result[:100] if len(result) > 100 else result)
            return result

        except Exception as e:
            logger_to_use.error("content_extraction.failed (BaseAgent)", error=str(e), exc_info=True)
            return str(response)


    def _process_response(self, content: str, raw_response: Any) -> Dict[str, Any]:
        """Process LLM response into standard format without duplicating raw output"""
        # Use self.logger which should be initialized in BaseAgent
        logger_to_use = self.logger
        try:
            # Extract content using THIS class's _get_llm_content method
            processed_content = self._get_llm_content(content) # Calls the overridden method
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
                if logger_to_use: logger_to_use.info("llm.token_usage", **usage_data)
                response["usage"] = usage_data

            return response
        except Exception as e:
            if logger_to_use: logger_to_use.error("response_processing.failed", error=str(e))
            return {
                "response": str(content),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }

    def _get_required_keys(self) -> List[str]:
        # Can be overridden by subclasses if they require specific keys in context
        return []

    def _get_agent_name(self) -> str:
        # Default implementation, should be overridden by specific agent subclasses
        class_name = self.__class__.__name__
        # Convert CamelCase to snake_case more robustly
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        if name.endswith("_agent"): # Remove suffix only if present
            name = name[:-6]
        return name or "base_agent" # Fallback


    def _get_system_message(self) -> str:
        # Use self.config_node which is initialized in BaseConfig
        agent_config_node = self.config_node.get_node(f"llm_config.agents.{self._get_agent_name()}")
        return agent_config_node.get_value("prompts.system") or ""


    def _get_prompt(self, prompt_type: str) -> str:
         # Use self.config_node which is initialized in BaseConfig
        agent_config_node = self.config_node.get_node(f"llm_config.agents.{self._get_agent_name()}")
        prompt = agent_config_node.get_value(f"prompts.{prompt_type}")
        if prompt is None: # Check explicitly for None
             # Use self.logger which is initialized
             self.logger.error("prompt.template_not_found", prompt_type=prompt_type, agent=self._get_agent_name())
             raise ValueError(f"No prompt template found for type: {prompt_type} in agent {self._get_agent_name()}")
        # Ensure prompt is a string before returning
        return str(prompt) if prompt is not None else ""


    def call_skill(self, skill_name: str, skill_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a skill with proper lineage tracking.
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