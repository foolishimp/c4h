"""
LLM interaction layer providing completion and response handling.
Path: c4h_agents/agents/base_llm.py
"""

from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime
import litellm
from litellm import completion
from c4h_agents.agents.types import LLMProvider, LogDetail
from c4h_agents.utils.logging import get_logger
from c4h_agents.agents.continuation.continuation_handler import ContinuationHandler  # Updated import

logger = get_logger()

class BaseLLM:
    """LLM interaction layer"""
    _continuation_handler = None

    def __init__(self):
        """Initialize LLM support"""
        self.provider = None
        self.model = None
        self.config_node = None
        self.metrics = {}
        self.log_level = LogDetail.BASIC

    def _get_completion_with_continuation(
            self,
            messages: List[Dict[str, str]],
            max_attempts: Optional[int] = None
        ) -> Tuple[str, Any]:
        """
        Get completion with automatic continuation handling.
        """
        try:
            # Initialize continuation handler on first use
            if not hasattr(self, '_continuation_handler') or self._continuation_handler is None:
                self._continuation_handler = ContinuationHandler(self)
            # Use the handler
            return self._continuation_handler.get_completion_with_continuation(messages, max_attempts)
        except AttributeError as e:
            logger.error(f"continuation_handler_init_failed: {str(e)}")
            # Fall back to direct LLM call without continuation handling
            response = completion(
                model=self.model_str,
                messages=messages
            )
            return response.choices[0].message.content, response

    def _process_response(self, content: str, raw_response: Any) -> Dict[str, Any]:
        """
        Process LLM response with comprehensive validation and logging.
        Maintains consistent response format while preserving metadata.
        
        Args:
            content: Initial content from LLM
            raw_response: Complete response object
            
        Returns:
            Standardized response dictionary with:
            - response: Processed content
            - raw_output: Original response string
            - timestamp: Processing time
            - usage: Token usage if available
            - error: Any processing errors
        """
        try:
            # Extract content using standard method
            processed_content = self._get_llm_content(content)
            
            # Debug logging for response processing
            if self._should_log(LogDetail.DEBUG):
                # Use self.config for logging if available - ensures proper truncation
                config_to_use = getattr(self, 'config', None)
                logger_to_use = getattr(self, 'logger', logger)
                
                logger_to_use.debug("agent.processing_response",
                            content_length=len(str(processed_content)) if processed_content else 0,
                            response_type=type(raw_response).__name__,
                            continuation_attempts=self.metrics.get("continuation_attempts", 0))

            # Build standard response structure
            response = {
                "response": processed_content,
                "raw_output": str(raw_response),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Log and include token usage if available
            if hasattr(raw_response, 'usage'):
                usage = raw_response.usage
                usage_data = {
                    "completion_tokens": getattr(usage, 'completion_tokens', 0),
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                    "total_tokens": getattr(usage, 'total_tokens', 0)
                }
                logger.info("llm.token_usage", **usage_data)
                response["usage"] = usage_data

            return response

        except Exception as e:
            error_msg = str(e)
            logger.error("response_processing.failed",
                        error=error_msg,
                        content_type=type(content).__name__)
            return {
                "response": str(content),
                "raw_output": str(raw_response),
                "timestamp": datetime.utcnow().isoformat(),
                "error": error_msg
            }

    # In ./c4h_agents/agents/base_llm.py

    def _get_model_str(self) -> str:
        """Get the appropriate model string for the provider, formatted for LiteLLM."""
        # Ensure self.provider and self.model are initialized
        if not hasattr(self, 'provider') or not self.provider:
             # Attempt to get provider from config as a fallback if not initialized
             # This might happen if called before full __init__ completes, though unlikely
             agent_name = self._get_agent_name() # Assuming this method exists and works
             provider_name = self.config_node.get_value(f"llm_config.agents.{agent_name}.provider") or \
                             self.config_node.get_value("llm_config.default_provider")
             if not provider_name:
                 raise ValueError("Provider is not set on the agent instance or in config.")
             self.provider = LLMProvider(provider_name)

        if not hasattr(self, 'model') or not self.model:
             # Attempt to get model from config as a fallback
             agent_name = self._get_agent_name()
             # Note: _resolve_model handles the complex hierarchy, but we might need a simpler lookup here
             # For safety, just check direct agent config or default
             self.model = self.config_node.get_value(f"llm_config.agents.{agent_name}.model") or \
                          self.config_node.get_value("llm_config.default_model")
             if not self.model:
                  raise ValueError("Model is not set on the agent instance or in config.")

        # Construct model string based on provider
        if self.provider == LLMProvider.OPENAI:
            # OpenAI models don't typically need a prefix for LiteLLM if using official API base
            # However, if using Azure OpenAI, the model name might be the deployment name.
            # Keeping openai/ prefix might be safer for potential Azure compatibility via LiteLLM proxy.
            return f"openai/{self.model}"
        elif self.provider == LLMProvider.ANTHROPIC:
            return f"anthropic/{self.model}"
        elif self.provider == LLMProvider.GEMINI:
            # --- CHANGE: Use "gemini/" prefix for standard LiteLLM compatibility ---
            return f"gemini/{self.model}"
            # --- END CHANGE ---
        # Example for XAI (adjust if LiteLLM uses a different prefix)
        elif hasattr(LLMProvider, 'XAI') and self.provider == LLMProvider.XAI:
             # Assuming LiteLLM uses 'groq/' or 'xai/' prefix - check LiteLLM docs
             # Using 'groq/' as an example, adjust if needed
             return f"groq/{self.model}" # Or potentially just self.model if no prefix needed
        else:
            # Default pattern for other potential providers
            return f"{self.provider.value}/{self.model}"



    def _setup_litellm(self, provider_config: Dict[str, Any]) -> None:
        """
        Configure litellm with provider settings.
        Handles extended thinking configuration for Claude 3.7 Sonnet.
        """
        try:
            litellm_params = provider_config.get("litellm_params", {})
            
            # Set retry configuration globally
            if "retry" in litellm_params:
                litellm.success_callback = []
                litellm.failure_callback = []
                litellm.retry = litellm_params.get("retry", True)
                litellm.max_retries = litellm_params.get("max_retries", 3)
                
                # Handle backoff settings
                backoff = litellm_params.get("backoff", {})
                litellm.retry_wait = backoff.get("initial_delay", 1)
                litellm.max_retry_wait = backoff.get("max_delay", 30)
                litellm.retry_exponential = backoff.get("exponential", True)
                
            # Set rate limits if provided
            if "rate_limit_policy" in litellm_params:
                rate_limits = litellm_params["rate_limit_policy"]
                litellm.requests_per_min = rate_limits.get("requests", 50)
                litellm.token_limit = rate_limits.get("tokens", 4000)
                litellm.limit_period = rate_limits.get("period", 60)

            # Configure api base if provided
            if "api_base" in provider_config:
                litellm.api_base = provider_config["api_base"]
            
            # Configure any provider-specific configurations
            # Only configure extended thinking support for Claude 3.7 Sonnet
            if self.provider.value == "anthropic" and "claude-3-7-sonnet" in self.model:
                # Check if extended thinking is explicitly enabled
                agent_name = self._get_agent_name()
                agent_path = f"llm_config.agents.{agent_name}"
                
                # Get extended thinking settings
                agent_thinking_config = self.config_node.get_value(f"{agent_path}.extended_thinking")
                if not agent_thinking_config:
                    agent_thinking_config = provider_config.get("extended_thinking", {})
                
                # Only configure if explicitly enabled
                if agent_thinking_config and agent_thinking_config.get("enabled", False) is True:
                    # Ensure litellm is configured to pass through the 'thinking' parameter
                    litellm.excluded_params = list(litellm.excluded_params) if hasattr(litellm, "excluded_params") else []
                    if "thinking" not in litellm.excluded_params:
                        litellm.excluded_params.append("thinking")
                        logger.debug("litellm.extended_thinking_support_configured",
                                    model=self.model)
            
            if self._should_log(LogDetail.DEBUG):
                logger.debug("litellm.configured",
                            provider=self.provider.serialize(),
                            retry_settings={
                                "enabled": litellm.retry,
                                "max_retries": litellm.max_retries,
                                "initial_delay": litellm.retry_wait,
                                "max_delay": litellm.max_retry_wait
                            })

        except Exception as e:
            logger.error("litellm.setup_failed", error=str(e))
            # Don't re-raise - litellm setup failure shouldn't be fatal

    def _get_llm_content(self, response: Any) -> Any:
            """Extract content from LLM response or context data with robust error handling."""
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

                    # Check for {'response': ...} structure
                    if 'response' in response:
                        response_val = response['response']
                        logger_to_use.debug("_get_llm_content.found_response_key", response_val_type=type(response_val).__name__) # Log type

                        # Check if response_val is the string directly
                        if isinstance(response_val, str):
                            logger_to_use.debug("content.extracted_from_dict_response_key (string)", content_length=len(response_val))
                            return response_val
                        
                        # --- Check nested dict {'response': {'content': '...'}} ---
                        elif isinstance(response_val, dict) and 'content' in response_val:
                            # --- ADDED PRINT STATEMENTS ---
                            print("DEBUG: Entered response[response][content] check") 
                            content_val = response_val['content']
                            print(f"DEBUG: Nested content type: {type(content_val).__name__}")
                            # --- END ADDED ---
                            if isinstance(content_val, str):
                                print("DEBUG: Nested content is string, returning it.") # Added Print
                                logger_to_use.debug("content.extracted_from_nested_dict_response_content_key", content_length=len(content_val))
                                return content_val
                            else:
                                print("DEBUG: Nested content is NOT string.") # Added Print
                                logger_to_use.warning("_get_llm_content.nested_content_not_string", nested_content_type=type(content_val).__name__)
                        else:
                            print("DEBUG: response[response] is dict but no 'content' key found.") # Added Print
                            logger_to_use.debug("_get_llm_content.nested_content_key_missing")


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
                # Log the exception *before* falling back
                logger_to_use.error("content_extraction.failed_with_exception", error=str(e), exc_info=True) # Log traceback
                # Return string representation on error
                return str(response)


    def _should_log(self, level: LogDetail) -> bool:
        """Check if current log level includes the specified detail level"""
        log_levels = {
            LogDetail.MINIMAL: 0,
            LogDetail.BASIC: 1,
            LogDetail.DETAILED: 2,
            LogDetail.DEBUG: 3
        }
        
        current_level = self.log_level
        target_level = LogDetail(level) if isinstance(level, str) else level
        
        return log_levels[target_level] <= log_levels[current_level]

    def _get_agent_name(self) -> str:
        """Get the agent name (placeholder method, implement as needed)"""
        # Placeholder implementation; adjust based on your actual needs
        return "default_agent"