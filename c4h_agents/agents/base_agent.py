# File: /Users/jim/src/apps/c4h_ai_dev/c4h_agents/agents/base_agent.py
# Corrections for errors related to 'config' and 'agent_name' scope

# Necessary Imports (ensure these are present)
import json
import uuid
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import re
import structlog # Ensure structlog is imported if used directly
from c4h_agents.agents.base_config import BaseConfig
from c4h_agents.agents.base_lineage import BaseLineage # Correct import
from c4h_agents.agents.lineage_context import LineageContext # Correct import
from c4h_agents.agents.base_llm import BaseLLM
from c4h_agents.agents.continuation.continuation_handler import ContinuationHandler
from c4h_agents.agents.types import AgentResponse, AgentMetrics, LLMProvider, LLMMessages, LogDetail
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

        # --- Step 7: Determine Agent's Config Path ---
        # Use unique_name (agent_name) to find the config path within the snapshot
        primary_agent_path = f"team.llm_config.agents.{agent_name}"
        fallback_agent_path = f"llm_config.agents.{agent_name}"
        self.config_path = primary_agent_path # Assume primary first
        agent_config_subsection = self.config_node.get_value(primary_agent_path)

        if not isinstance(agent_config_subsection, dict):
            self.logger.debug(f"{agent_name}.init.primary_path_not_found", path=primary_agent_path)
            agent_config_subsection = self.config_node.get_value(fallback_agent_path)
            if isinstance(agent_config_subsection, dict):
                 self.config_path = fallback_agent_path # Use fallback path
                 self.logger.info(f"{agent_name}.init.using_fallback_config_path", path=self.config_path)
            else:
                 # If neither path yields a dict, log a warning. The agent might still function
                 # if it relies only on global defaults or context, but specific settings are missing.
                 self.logger.warning(f"{agent_name}.init.agent_specific_config_not_found",
                                     primary_path=primary_agent_path,
                                     fallback_path=fallback_agent_path)
                 self.config_path = fallback_agent_path # Default to fallback path even if empty

        # --- Step 8: Resolve Provider, Model, Temperature using self.config_node and self.config_path ---
        # These values should exist in the snapshot due to merging/persona injection
        self.provider = LLMProvider(self.config_node.get_value(f"{self.config_path}.provider") or \
                                    self.config_node.get_value("llm_config.default_provider") or \
                                    "anthropic") # Default if missing
        self.model = self.config_node.get_value(f"{self.config_path}.model") or \
                     self.config_node.get_value(f"llm_config.providers.{self.provider.value}.default_model") or \
                     self.config_node.get_value("llm_config.default_model") or \
                     "claude-3-opus-20240229" # Default if missing
        temp_val = self.config_node.get_value(f"{self.config_path}.temperature")
        self.temperature = float(temp_val) if temp_val is not None else 0.0

        self.logger.debug(f"{agent_name}.init.resolved_settings",
                          provider=self.provider.value, model=self.model, temp=self.temperature)

        # --- Step 9: Continuation settings ---
        # Use self.config_node and self.config_path
        self.max_continuation_attempts = self.config_node.get_value(f"{self.config_path}.max_continuation_attempts", default=5)
        self.continuation_token_buffer = self.config_node.get_value(f"{self.config_path}.continuation_token_buffer", default=1000)

        # --- Step 10: Initialize metrics ---
        self.metrics = AgentMetrics(project=self.project.metadata.name if self.project else None)

        # --- Step 11: Set logging detail level ---
        log_level_str = self.config_node.get_value("logging.agent_level") or self.config_node.get_value("logging.level") or "basic"
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

        # --- Step 14: Initialize Lineage ---
        self.lineage = None
        try:
            self.logger.debug(f"{agent_name}.lineage_init", has_runtime="runtime" in self.config, has_system="system" in self.config, has_workflow_run_id=bool(self.run_id))
            self.logger.debug(f"{agent_name}.using_run_id", run_id=self.run_id)
            # Pass self.config (the full effective snapshot)
            self.lineage = BaseLineage(
                namespace="c4h_agents",
                agent_name=agent_name, # Use variable defined above
                config=self.config
            )
            # Update run_id if lineage loaded an existing one
            if self.lineage and hasattr(self.lineage, 'run_id') and self.lineage.run_id != self.run_id:
                 self.run_id = self.lineage.run_id
                 self.logger = self.logger.bind(run_id=self.run_id)
                 self.logger.info(f"{agent_name}.lineage_loaded_existing_run", loaded_run_id=self.run_id)
        except Exception as e:
            self.logger.error(f"{agent_name}.lineage_init_failed", error=str(e), exc_info=True)

        # --- Step 15: Initialize Continuation Handler ---
        # Ensure it's initialized AFTER all necessary parent attributes (like logger, model_str) are set
        self._continuation_handler: Optional[ContinuationHandler] = ContinuationHandler(self)

        # --- Step 16: Final Log ---
        self.logger.info(f"{agent_name}.initialized",
                     continuation_settings={
                         "max_attempts": self.max_continuation_attempts,
                         "token_buffer": self.continuation_token_buffer
                     })


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

    def _format_request(self, context: Dict[str, Any]) -> str:
         # Default implementation, subclasses should override if specific formatting is needed
         # Safely convert context to string for basic use
         try:
              # Use self.logger which should be initialized
              self.logger.debug("_format_request: using default JSON dump")
              return json.dumps(context, indent=2, default=str) # Safer conversion
         except Exception as e:
              # Use self.logger which should be initialized
              self.logger.error("_format_request: failed to dump context to JSON", error=str(e))
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


    # Example: Assuming the error is in _process_response
    def _process_response(self, content: str, raw_response: Any) -> Dict[str, Any]:
        """Process LLM response into standard format without duplicating raw output"""
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
        # Can be overridden by subclasses if they require specific keys in context
        return []

    def _get_agent_name(self) -> str:
        # This method now returns the unique name assigned during instantiation
        # Ensure self.unique_name is set in __init__
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

    def _get_system_message(self) -> str:
        # Use self.config_node which is initialized in BaseConfig
        return self.config_node.get_value(f"{self.config_path}.prompts.system") or ""


    def _get_prompt(self, prompt_type: str) -> str:
         # Use self.config_node which is initialized in BaseConfig
        prompt_path = f"{self.config_path}.prompts.{prompt_type}"
        prompt = self.config_node.get_value(prompt_path)
        if prompt is None:  # Check explicitly for None
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