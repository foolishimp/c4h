# Path: /Users/jim/src/apps/c4h/c4h_agents/agents/base_llm.py
"""
LLM interaction layer providing completion and response handling.
"""

from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime, timezone # Ensure timezone is imported
import litellm
from litellm import completion
from c4h_agents.agents.types import LLMProvider, LogDetail
from c4h_agents.utils.logging import get_logger
# Import ContinuationHandler for type hint, but initialize lazily
from c4h_agents.agents.continuation.continuation_handler import ContinuationHandler

# Use get_logger() at the module level for consistency
logger = get_logger()

class BaseLLM:
    """LLM interaction layer"""

    def __init__(self):
        """Initialize LLM support"""
        # Initialize instance variables safely in __init__
        self.provider: Optional[LLMProvider] = None
        self.model: Optional[str] = None
        self.model_str: Optional[str] = None
        # self.config_node = None # <<< REMOVED THIS LINE
        self.metrics: Dict[str, Any] = {}
        self.log_level: LogDetail = LogDetail.BASIC
        # Initialize _continuation_handler here to ensure it exists
        self._continuation_handler: Optional[ContinuationHandler] = None


    def _get_completion_with_continuation(
            self,
            messages: List[Dict[str, str]],
            max_attempts: Optional[int] = None
        ) -> Tuple[str, Any]:
        """
        Get completion with automatic continuation handling.
        """
        logger_to_use = getattr(self, 'logger', logger)
        try:
            # Initialize continuation handler on first use
            # Check existence using hasattr for safety
            if not hasattr(self, '_continuation_handler') or self._continuation_handler is None:
                # Pass self (which should be the instance of BaseAgent or subclass)
                self._continuation_handler = ContinuationHandler(self)
            # Use the handler
            return self._continuation_handler.get_completion_with_continuation(messages, max_attempts)

        except AttributeError as e:
            # Log the specific error and agent type if possible
            agent_type = type(self).__name__
            logger_to_use.error(f"continuation_handler_init_failed: {str(e)}", agent_type=agent_type, exc_info=True)
            # Fall back to direct LLM call without continuation handling
            logger_to_use.warning("Falling back to direct LLM call without continuation handling")
            if not self.model_str:
                 # Ensure model_str is set before calling completion
                 try:
                      self.model_str = self._get_model_str()
                 except Exception as model_err:
                      logger_to_use.error("Failed to set model_str during fallback", error=str(model_err))
                      raise ValueError("Cannot make fallback LLM call: model not configured.") from model_err

            # --- Fallback Call ---
            try:
                 completion_params = {
                      "model": self.model_str,
                      "messages": messages
                 }
                 if hasattr(self, 'temperature'):
                      completion_params['temperature'] = self.temperature

                 response = completion(**completion_params)

                 # Basic error handling for fallback
                 if response and response.choices and hasattr(response.choices[0],'message') and hasattr(response.choices[0].message,'content'):
                      # Use the _get_llm_content from the *current instance*
                      content = self._get_llm_content(response)
                      return content, response
                 else:
                      logger_to_use.error("Fallback LLM call returned invalid response structure", response_obj=response)
                      raise ValueError("Fallback LLM call failed or returned empty response.")
            except Exception as fallback_e:
                 logger_to_use.error("Fallback LLM call itself failed", error=str(fallback_e), exc_info=True)
                 raise fallback_e # Re-raise the error from the fallback call

    def _process_response(self, content: str, raw_response: Any) -> Dict[str, Any]:
        """Basic response processing, primarily extracting content."""
        # This method might be overridden by BaseAgent.
        # This base implementation focuses only on extraction via _get_llm_content.
        logger_to_use = getattr(self, 'logger', logger)
        try:
            processed_content = self._get_llm_content(content)
            response_data = {
                "response": processed_content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_output": str(raw_response) # Keep raw for context
            }
            if hasattr(raw_response, 'usage'):
                usage = raw_response.usage
                usage_data = {
                    "completion_tokens": getattr(usage, 'completion_tokens', 0),
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                    "total_tokens": getattr(usage, 'total_tokens', 0)
                }
                logger_to_use.info("llm.token_usage", **usage_data)
                response_data["usage"] = usage_data
            return response_data
        except Exception as e:
            logger_to_use.error("base_llm._process_response.failed", error=str(e), exc_info=True)
            # Fallback on error
            return {
                "response": str(content), # Return original content string on error
                "raw_output": str(raw_response),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": f"Processing failed: {str(e)}"
            }


    def _get_model_str(self) -> str:
        """Get the appropriate model string for the provider, formatted for LiteLLM."""
        logger_to_use = getattr(self, 'logger', logger)
        # Ensure self.provider and self.model are initialized
        config_node_to_use = getattr(self, 'config_node', None) # Get config_node safely
        if not config_node_to_use: raise ValueError("config_node not available in _get_model_str (BaseLLM)")

        if not hasattr(self, 'provider') or not self.provider:
             logger_to_use.warning("_get_model_str called before provider was set.")
             agent_name = self._get_agent_name()
             provider_name = config_node_to_use.get_value(f"llm_config.agents.{agent_name}.provider") or \
                             config_node_to_use.get_value("llm_config.default_provider")
             if not provider_name:
                 raise ValueError("Provider is not set on the agent instance or in config.")
             self.provider = LLMProvider(provider_name)
             logger_to_use.debug("_get_model_str resolved provider from config", provider=self.provider.value)


        if not hasattr(self, 'model') or not self.model:
             logger_to_use.warning("_get_model_str called before model was set.")
             agent_name = self._get_agent_name()
             self.model = config_node_to_use.get_value(f"llm_config.agents.{agent_name}.model") or \
                          config_node_to_use.get_value("llm_config.default_model")
             if not self.model:
                  raise ValueError("Model is not set on the agent instance or in config.")
             logger_to_use.debug("_get_model_str resolved model from config", model=self.model)

        # Construct model string based on provider
        if self.provider:
             model_name_str = str(self.model)
             provider_prefix = self.provider.value
             return f"{provider_prefix}/{model_name_str}"
        else:
            raise ValueError("Provider is not set, cannot determine model string.")



    def _setup_litellm(self, provider_config: Dict[str, Any]) -> None:
        """
        Configure litellm with provider settings.
        Handles extended thinking configuration for Claude 3.7 Sonnet.
        """
        logger_to_use = getattr(self, 'logger', logger)
        try:
            litellm_params = provider_config.get("litellm_params", {})

            # Set retry configuration globally only if present
            if "retry" in litellm_params:
                litellm.retry = litellm_params.get("retry", True)
                max_retries = litellm_params.get("max_retries", 3)
                litellm.max_retries = int(max_retries) if str(max_retries).isdigit() else 3

                backoff = litellm_params.get("backoff", {})
                litellm.retry_wait = backoff.get("initial_delay", 1)
                litellm.max_retry_wait = backoff.get("max_delay", 30)
                litellm.retry_exponential = backoff.get("exponential", True)

            # Configure extended thinking support
            if self.provider and self.model and self.provider.value == "anthropic" and "claude-3-7-sonnet" in self.model:
                agent_name = self._get_agent_name()
                agent_path = f"llm_config.agents.{agent_name}"

                config_node_to_use = getattr(self, 'config_node', None)
                agent_thinking_config = None
                if config_node_to_use:
                     agent_thinking_config = config_node_to_use.get_value(f"{agent_path}.extended_thinking")

                if not agent_thinking_config:
                    agent_thinking_config = provider_config.get("extended_thinking", {})

                if agent_thinking_config and agent_thinking_config.get("enabled", False) is True:
                    if not hasattr(litellm, "excluded_params") or not isinstance(litellm.excluded_params, list):
                        litellm.excluded_params = []

                    if "thinking" not in litellm.excluded_params:
                         litellm.excluded_params.append("thinking")
                         logger_to_use.debug("litellm.extended_thinking_support_configured", model=self.model)


            if self._should_log(LogDetail.DEBUG):
                 retry_enabled = getattr(litellm, 'retry', 'Not Set')
                 max_retries_val = getattr(litellm, 'max_retries', 'Not Set')
                 initial_delay_val = getattr(litellm, 'retry_wait', 'Not Set')
                 max_delay_val = getattr(litellm, 'max_retry_wait', 'Not Set')

                 logger_to_use.debug("litellm.configured (Note: Global settings)",
                             provider=self.provider.serialize() if self.provider else 'Not Set',
                             retry_settings={
                                 "enabled": retry_enabled,
                                 "max_retries": max_retries_val,
                                 "initial_delay": initial_delay_val,
                                 "max_delay": max_delay_val
                             })

        except Exception as e:
            logger_to_use.error("litellm.setup_failed", error=str(e), exc_info=True)

    # --- Keep the BaseLLM version of _get_llm_content as a fallback ---
    # --- The more specific one in BaseAgent takes precedence for BaseAgent subclasses ---
    def _get_llm_content(self, response: Any) -> Any:
        """Basic content extraction from LLM response."""
        logger_to_use = getattr(self, 'logger', logger)
        logger_to_use.debug("_get_llm_content.received (BaseLLM)", input_type=type(response).__name__) # Identify which version
        try:
            # Handle different response types
            if hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    content = response.choices[0].message.content
                    logger_to_use.debug("content.extracted_from_model (BaseLLM)", content_length=len(content) if content else 0)
                    return content
                elif hasattr(response.choices[0], 'delta') and hasattr(response.choices[0].delta, 'content'):
                    content = response.choices[0].delta.content
                    logger_to_use.debug("content.extracted_from_delta (BaseLLM)", content_length=len(content) if content else 0)
                    return content

            if isinstance(response, str):
                logger_to_use.debug("content.extracted_direct_string (BaseLLM)", content_length=len(response))
                return response

            if isinstance(response, dict) and 'response' in response and isinstance(response['response'], str):
                 logger_to_use.debug("content.extracted_from_dict_response_key (BaseLLM)", content_length=len(response['response']))
                 return response['response']

            result = str(response)
            logger_to_use.warning("content.extraction_fallback (BaseLLM)",
                        response_type=type(response).__name__,
                        content_preview=result[:100] if len(result) > 100 else result)
            return result
        except Exception as e:
            logger_to_use.error("content_extraction.failed (BaseLLM)", error=str(e))
            return str(response)


    def _should_log(self, level: LogDetail) -> bool:
        """Check if current log level includes the specified detail level"""
        current_log_level = getattr(self, 'log_level', LogDetail.BASIC)
        log_levels = {
            LogDetail.MINIMAL: 0,
            LogDetail.BASIC: 1,
            LogDetail.DETAILED: 2,
            LogDetail.DEBUG: 3
        }
        target_level = LogDetail(level) if isinstance(level, str) else level
        return log_levels.get(target_level, 0) <= log_levels.get(current_log_level, 1)

    def _get_agent_name(self) -> str:
        """Placeholder: Get the agent name"""
        # Subclasses like BaseAgent should provide a more specific name
        return "base_llm"