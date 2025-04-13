# Path: /Users/jim/src/apps/c4h/c4h_agents/agents/base_llm.py
"""
LLM interaction layer providing completion and response handling.
Path: c4h_agents/agents/base_llm.py
"""

from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime, timezone # Ensure timezone is imported
import litellm
from litellm import completion
from c4h_agents.agents.types import LLMProvider, LogDetail
from c4h_agents.utils.logging import get_logger
from c4h_agents.agents.continuation.continuation_handler import ContinuationHandler  # Updated import

# Use get_logger() at the module level for consistency
logger = get_logger()

class BaseLLM:
    """LLM interaction layer"""
    # Remove instance variable declaration at class level
    # _continuation_handler = None 

    def __init__(self):
        """Initialize LLM support"""
        # Initialize instance variables safely in __init__
        self.provider: Optional[LLMProvider] = None
        self.model: Optional[str] = None
        self.model_str: Optional[str] = None # Added for clarity
        self.config_node = None # Assume set by subclass or configuration method
        self.metrics: Dict[str, Any] = {}
        self.log_level: LogDetail = LogDetail.BASIC
        self._continuation_handler: Optional[ContinuationHandler] = None # Initialize instance variable

    def _get_completion_with_continuation(
            self,
            messages: List[Dict[str, str]],
            max_attempts: Optional[int] = None
        ) -> Tuple[str, Any]:
        """
        Get completion with automatic continuation handling.
        """
        # Use self.logger if available (set by BaseAgent), otherwise use module logger
        logger_to_use = getattr(self, 'logger', logger)
        try:
            # Initialize continuation handler on first use
            if self._continuation_handler is None:
                 # Pass self (which should be the BaseAgent instance) to the handler
                self._continuation_handler = ContinuationHandler(self) 
            # Use the handler
            return self._continuation_handler.get_completion_with_continuation(messages, max_attempts)
        
        except AttributeError as e:
            logger_to_use.error(f"continuation_handler_init_failed: {str(e)}", exc_info=True) # Add traceback
            # Fall back to direct LLM call without continuation handling
            logger_to_use.warning("Falling back to direct LLM call without continuation handling")
            if not self.model_str:
                 # Ensure model_str is set before calling completion
                 self.model_str = self._get_model_str() 
            response = completion(
                model=self.model_str,
                messages=messages
                # Note: Temperature/other params from config aren't applied in this fallback
            )
            # Basic error handling for fallback
            if response and response.choices:
                 return response.choices[0].message.content, response
            else:
                 logger_to_use.error("Fallback LLM call failed or returned empty response")
                 raise ValueError("Fallback LLM call failed") # Re-raise meaningful error

    # --- Reverted _process_response to original simpler version ---
    # --- Let BaseAgent handle the complex processing ---
    def _process_response(self, content: str, raw_response: Any) -> Dict[str, Any]:
        """Basic response processing, primarily extracting content."""
        logger_to_use = getattr(self, 'logger', logger)
        try:
            # Use the helper method which should be present
            processed_content = self._get_llm_content(content) 
            response_data = {
                "response": processed_content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_output": str(raw_response) # Include raw for debugging if needed
            }
             # Add token usage metrics if available
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
        if not hasattr(self, 'provider') or not self.provider:
             # Attempt to get provider from config as a fallback if not initialized
             logger_to_use.warning("_get_model_str called before provider was set.")
             agent_name = self._get_agent_name() # Assuming this method exists and works
             # Ensure config_node is available
             config_node_to_use = getattr(self, 'config_node', None)
             if not config_node_to_use: raise ValueError("config_node not available in _get_model_str")
             
             provider_name = config_node_to_use.get_value(f"llm_config.agents.{agent_name}.provider") or \
                             config_node_to_use.get_value("llm_config.default_provider")
             if not provider_name:
                 raise ValueError("Provider is not set on the agent instance or in config.")
             self.provider = LLMProvider(provider_name)
             logger_to_use.debug("_get_model_str resolved provider from config", provider=self.provider.value)


        if not hasattr(self, 'model') or not self.model:
             logger_to_use.warning("_get_model_str called before model was set.")
             agent_name = self._get_agent_name()
             config_node_to_use = getattr(self, 'config_node', None)
             if not config_node_to_use: raise ValueError("config_node not available in _get_model_str")

             # Resolve model using hierarchy
             self.model = config_node_to_use.get_value(f"llm_config.agents.{agent_name}.model") or \
                          config_node_to_use.get_value("llm_config.default_model")
             if not self.model:
                  raise ValueError("Model is not set on the agent instance or in config.")
             logger_to_use.debug("_get_model_str resolved model from config", model=self.model)


        # Construct model string based on provider
        # --- Use standard LiteLLM provider prefixes ---
        if self.provider:
             # Ensure self.model is treated as a string
             model_name_str = str(self.model) 
             return f"{self.provider.value}/{model_name_str}"
        else:
            # This case should ideally not be reached due to checks above
             raise ValueError("Provider is not set, cannot determine model string.")



    def _setup_litellm(self, provider_config: Dict[str, Any]) -> None:
        """
        Configure litellm with provider settings.
        Handles extended thinking configuration for Claude 3.7 Sonnet.
        """
        logger_to_use = getattr(self, 'logger', logger)
        try:
            litellm_params = provider_config.get("litellm_params", {})
            
            # Set retry configuration globally
            if "retry" in litellm_params:
                litellm.success_callback = [] # Reset callbacks if setting retry
                litellm.failure_callback = []
                litellm.retry = litellm_params.get("retry", True)
                # Ensure max_retries is int or default
                max_retries = litellm_params.get("max_retries", 3)
                litellm.max_retries = int(max_retries) if isinstance(max_retries,(int,float,str)) and str(max_retries).isdigit() else 3

                # Handle backoff settings
                backoff = litellm_params.get("backoff", {})
                litellm.retry_wait = backoff.get("initial_delay", 1)
                litellm.max_retry_wait = backoff.get("max_delay", 30)
                litellm.retry_exponential = backoff.get("exponential", True)
                
            # Set rate limits if provided (Note: LiteLLM's internal limiting might be basic)
            if "rate_limit_policy" in litellm_params:
                rate_limits = litellm_params["rate_limit_policy"]
                litellm.requests_per_min = rate_limits.get("requests", 50)
                # litellm.token_limit = rate_limits.get("tokens", 4000) # Token limit might not be directly supported this way
                litellm.limit_period = rate_limits.get("period", 60)

            # Configure api base if provided
            if "api_base" in provider_config:
                litellm.api_base = provider_config["api_base"]
            
            # Configure any provider-specific configurations
            # Only configure extended thinking support for Claude 3.7 Sonnet
            # Ensure self.provider and self.model are set before this check
            if self.provider and self.model and self.provider.value == "anthropic" and "claude-3-7-sonnet" in self.model:
                # Check if extended thinking is explicitly enabled
                agent_name = self._get_agent_name()
                agent_path = f"llm_config.agents.{agent_name}"
                
                # Get extended thinking settings using config_node if available
                config_node_to_use = getattr(self, 'config_node', None)
                agent_thinking_config = None
                if config_node_to_use:
                     agent_thinking_config = config_node_to_use.get_value(f"{agent_path}.extended_thinking")

                # Fallback to provider config if not in agent config
                if not agent_thinking_config:
                    agent_thinking_config = provider_config.get("extended_thinking", {})
                
                # Only configure if explicitly enabled
                if agent_thinking_config and agent_thinking_config.get("enabled", False) is True:
                    # Ensure litellm is configured to pass through the 'thinking' parameter
                    # Ensure excluded_params is initialized as a list if it doesn't exist or isn't a list
                    if not hasattr(litellm, "excluded_params") or not isinstance(litellm.excluded_params, list):
                        litellm.excluded_params = []

                    if "thinking" not in litellm.excluded_params:
                         litellm.excluded_params.append("thinking")
                         logger_to_use.debug("litellm.extended_thinking_support_configured", model=self.model)

            
            if self._should_log(LogDetail.DEBUG):
                 # Ensure all values exist before logging
                 retry_enabled = getattr(litellm, 'retry', 'Not Set')
                 max_retries_val = getattr(litellm, 'max_retries', 'Not Set')
                 initial_delay_val = getattr(litellm, 'retry_wait', 'Not Set')
                 max_delay_val = getattr(litellm, 'max_retry_wait', 'Not Set')

                 logger_to_use.debug("litellm.configured",
                             provider=self.provider.serialize() if self.provider else 'Not Set',
                             retry_settings={
                                 "enabled": retry_enabled,
                                 "max_retries": max_retries_val,
                                 "initial_delay": initial_delay_val,
                                 "max_delay": max_delay_val
                             })

        except Exception as e:
            logger_to_use.error("litellm.setup_failed", error=str(e), exc_info=True) # Add traceback
            # Don't re-raise - litellm setup failure shouldn't be fatal

    # --- Reverted _get_llm_content to original ---
    # --- Let BaseAgent handle the complex extraction ---
    def _get_llm_content(self, response: Any) -> Any:
        """Basic content extraction from LLM response."""
        logger_to_use = getattr(self, 'logger', logger)
        try:
            # Handle different response types
            if hasattr(response, 'choices') and response.choices:
                # Standard response object
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    content = response.choices[0].message.content
                    logger_to_use.debug("content.extracted_from_model (BaseLLM)", content_length=len(content) if content else 0)
                    return content
                # Handle delta format (used in streaming)
                elif hasattr(response.choices[0], 'delta') and hasattr(response.choices[0].delta, 'content'):
                    content = response.choices[0].delta.content
                    logger_to_use.debug("content.extracted_from_delta (BaseLLM)", content_length=len(content) if content else 0)
                    return content
            
            # If we have a simple string content
            if isinstance(response, str):
                logger_to_use.debug("content.extracted_direct_string (BaseLLM)", content_length=len(response))
                return response
                
            # If response is already processed (dict with 'response' key)
            if isinstance(response, dict) and 'response' in response:
                 logger_to_use.debug("content.extracted_from_dict_response_key (BaseLLM)", content_length=len(str(response['response'])))
                 return response['response']
                
            # Last resort fallback - convert to string
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
        # Use self.log_level which should be set by BaseAgent
        current_log_level = getattr(self, 'log_level', LogDetail.BASIC)
        log_levels = {
            LogDetail.MINIMAL: 0,
            LogDetail.BASIC: 1,
            LogDetail.DETAILED: 2,
            LogDetail.DEBUG: 3
        }
        
        # Ensure level is LogDetail enum member
        target_level = LogDetail(level) if isinstance(level, str) else level
        
        return log_levels.get(target_level, 0) <= log_levels.get(current_log_level, 1)

    def _get_agent_name(self) -> str:
        """Get the agent name (placeholder method, implement as needed in subclasses)"""
        # This should ideally be overridden by BaseAgent or specific agents
        return "base_llm" # Fallback name