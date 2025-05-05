"""
LLM interaction layer providing completion and response handling.
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import traceback
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

    # Add this to base_llm.py just before the _get_completion_with_continuation method

    def _call_gemini_directly(self, messages, temperature=0.7, max_output_tokens=8192):
        """
        Direct call to Gemini API bypassing LiteLLM completely.
        Uses the exact format that worked in curl test.
        """
        import os
        import json
        import requests
        import time
        from types import SimpleNamespace
        
        logger_to_use = getattr(self, 'logger', logger)
        logger_to_use.info("Using direct Gemini API call", model=self.model)
        
        # Get API key from environment
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Get provider configuration if available
        provider_config = {}
        if hasattr(self, '_get_provider_config') and hasattr(self, 'provider') and self.provider:
            provider_config = self._get_provider_config(self.provider)
        
        # Get API base from config or use default
        api_base = provider_config.get("api_base", "https://generativelanguage.googleapis.com/v1beta")
        
        # Remove trailing /models if present (will be added in URL construction)
        if api_base.endswith("/models"):
            api_base = api_base[:-7]
        
        # Construct the URL using exact model name and API key
        url = f"{api_base}/models/{self.model}:generateContent?key={api_key}"
        
        # Convert messages from ChatGPT/Claude format to Gemini format
        gemini_messages = []
        system_content = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Store system message to prepend to first user message
            if role == "system":
                system_content = content
                continue
            
            # For user messages, prepend system content if available
            if role == "user" and system_content:
                gemini_messages.append({
                    "parts": [{"text": f"{system_content}\n\n{content}"}]
                })
                system_content = None  # Only use system message once
            else:
                gemini_messages.append({
                    "parts": [{"text": content}]
                })
        
        # If system message is left over (no user message followed it)
        if system_content:
            gemini_messages.append({
                "parts": [{"text": system_content}]
            })
        
        # Create payload with proper Gemini format
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens
            }
        }
        
        # Set headers
        headers = {
            "Content-Type": "application/json"
        }
        
        # Make the request
        try:
            logger_to_use.debug("Calling Gemini API", url=url.replace(api_key, "[REDACTED]"), payload=payload)
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract content
            content = ""
            if "candidates" in result and result["candidates"]:
                for part in result["candidates"][0]["content"]["parts"]:
                    if "text" in part:
                        content += part["text"]
            
            # Get token usage
            usage = {
                "prompt_tokens": result.get("usageMetadata", {}).get("promptTokenCount", 0),
                "completion_tokens": result.get("usageMetadata", {}).get("candidatesTokenCount", 0),
                "total_tokens": result.get("usageMetadata", {}).get("totalTokenCount", 0)
            }
            
            # Create a response object with OpenAI-like structure for compatibility
            message = SimpleNamespace(
                content=content,
                role="assistant"
            )
            
            choice = SimpleNamespace(
                finish_reason="stop",
                index=0,
                message=message
            )
            
            usage_obj = SimpleNamespace(
                completion_tokens=usage["completion_tokens"],
                prompt_tokens=usage["prompt_tokens"],
                total_tokens=usage["total_tokens"]
            )
            
            response_obj = SimpleNamespace(
                id=f"gemini-{int(time.time())}",
                created=int(time.time()),
                model=self.model,
                object="chat.completion",
                choices=[choice],
                usage=usage_obj
            )
            
            logger_to_use.info("Gemini API call successful", 
                            tokens=usage["total_tokens"], 
                            content_length=len(content))
            
            return content, response_obj
            
        except Exception as e:
            logger_to_use.error("Gemini API call failed", error=str(e), exc_info=True)
            raise


    def _get_completion_with_continuation(
            self,
            messages: List[Dict[str, str]],
            max_attempts: Optional[int] = None,
            context: Optional[Dict[str, Any]] = None
        ) -> Tuple[str, Any]:
        """
        Get completion with automatic continuation handling.
        Now includes direct Gemini API integration for Gemini 2.5 models.
        """
        logger_to_use = getattr(self, 'logger', logger)
        try:
            # Check for temperature override in context
            temperature = self.temperature
            
            # Apply overrides from context if available
            if context:
                # Get the agent's unique name for context lookup
                agent_name = getattr(self, 'unique_name', '') or getattr(self, '_get_agent_name', lambda: '')()
                overrides = context.get('agent_config_overrides', {}).get(agent_name, {})
                
                # Temperature override
                if 'temperature' in overrides:
                    # Get override value (ensure it's a float)
                    temp_override = overrides['temperature']
                    if isinstance(temp_override, (int, float, str)):
                        try:
                            temperature = float(temp_override)
                            logger_to_use.debug("llm.using_temperature_override", 
                                           agent_name=agent_name,
                                           original=self.temperature,
                                           override=temperature)
                        except (ValueError, TypeError):
                            logger_to_use.warning("llm.invalid_temperature_override", 
                                              value=temp_override)
            
            # Special handling for Gemini 2.5 models - use direct API call
            if hasattr(self, 'provider') and self.provider and self.provider.value == "gemini" and "gemini-2.5" in self.model:
                logger_to_use.info("Using direct Gemini API call for Gemini 2.5", model=self.model)
                return self._call_gemini_directly(messages, temperature=temperature)
            
            # Initialize continuation handler for normal path
            if not hasattr(self, '_continuation_handler') or self._continuation_handler is None:
                self._continuation_handler = ContinuationHandler(self)
            # Use the handler for normal models and pass context
            return self._continuation_handler.get_completion_with_continuation(messages, max_attempts, context)

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

            # Fallback Call
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
            
            # Special handling for Gemini models based on error logs
            if provider_prefix == "gemini":
                # Handle Gemini model name formats
                gemini_model = model_name_str
                
                # Fix for Gemini 2.5 naming specifically
                if "2.5" in model_name_str or "2-5" in model_name_str:
                    # According to error logs, we need a different format
                    # Try the format "gemini-pro" without version numbers
                    if not model_name_str.startswith("models/"):
                        gemini_model = "gemini-pro"
                        logger_to_use.info("Using standard model name for Gemini 2.5:", 
                                        original=model_name_str, 
                                        mapped_to=gemini_model)
                
                # Return the final model string
                return f"{provider_prefix}/{gemini_model}"
            
            # Standard format for other providers
            return f"{provider_prefix}/{model_name_str}"
        else:
            raise ValueError("Provider is not set, cannot determine model string.")

    # Path: /Users/jim/src/apps/c4h/c4h_agents/agents/base_llm.py
    # Add this method or update the existing one to fix Gemini support

    def _build_completion_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Build parameters for LLM completion request."""
        try:
            # Get config_node safely from parent BaseAgent
            config_node_to_use = getattr(self.parent, 'config_node', None)
            if not config_node_to_use:
                raise ValueError("config_node not available in parent BaseAgent")
                
            # Get agent config path from parent BaseAgent
            agent_config_path = getattr(self.parent, 'config_path', None)
            if not agent_config_path:
                raise ValueError("agent_config_path not available in parent BaseAgent")
                
            # Get provider from parent (was resolved during init)
            provider = getattr(self.parent, 'provider', None)
            if not provider:
                provider_name = config_node_to_use.get_value(f"{agent_config_path}.provider") or "anthropic"
                provider = LLMProvider(provider_name)
            
            params = {"model": self.model_str, "messages": messages}
            if provider and provider.value != "openai": 
                params["temperature"] = self.temperature

            # Get provider config directly from effective config snapshot
            provider_config = config_node_to_use.get_value(f"llm_config.providers.{provider.value}") or {}

            # Get model parameters from provider config
            model_params = provider_config.get("model_params", {})
            if model_params:
                # Add model-specific parameters
                for key, value in model_params.items():
                    if key not in params:
                        params[key] = value

            # Add API base if present
            if "api_base" in provider_config:
                params["api_base"] = provider_config["api_base"]
                
                # CRITICAL FIX: For Gemini models, ensure the API base is correctly formatted
                # The debug output shows that the API base should end with /models
                if provider and provider.value == "gemini":
                    logger_to_use = getattr(self, 'logger', logger)
                    
                    # Log the API base we're using
                    logger_to_use.debug("Using Gemini API base", 
                                     api_base=params["api_base"],
                                     model=self.model)
                    
                    # Ensure the API base ends with /models
                    if not params["api_base"].endswith("/models"):
                        logger_to_use.warning("Gemini API base should end with /models - appending it")
                        if params["api_base"].endswith("/"):
                            params["api_base"] = params["api_base"] + "models"
                        else:
                            params["api_base"] = params["api_base"] + "/models"

            # Handle Claude 3.7 Sonnet extended thinking if present
            if provider and self.model and provider.value == "anthropic" and "claude-3-7-sonnet" in self.model:
                # Get extended thinking config directly from effective config snapshot
                agent_thinking_config = config_node_to_use.get_value(f"{agent_config_path}.extended_thinking")
                if not agent_thinking_config:
                    agent_thinking_config = provider_config.get("extended_thinking", {})
                if agent_thinking_config and agent_thinking_config.get("enabled", False) is True:
                    params['thinking'] = True 
                    self.logger.debug("Added 'thinking' parameter for Claude 3.7 Sonnet")

            return params
        except Exception as e:
            logger_to_use = getattr(self, 'logger', logger)
            logger_to_use.error("Completion params build failed",
                            error=str(e), stack_trace=traceback.format_exc())
            raise

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
            # Get provider and model from parent BaseAgent if available
            provider = getattr(self.parent, 'provider', None) if hasattr(self, 'parent') else None
            agent_config_path = getattr(self.parent, 'config_path', '') if hasattr(self, 'parent') else ''

            if provider and self.model and provider.value == "anthropic" and "claude-3-7-sonnet" in self.model:
                config_node_to_use = getattr(self.parent, 'config_node', None) if hasattr(self, 'parent') else None
                agent_thinking_config = config_node_to_use.get_value(f"{agent_config_path}.extended_thinking") if config_node_to_use else None

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
                             provider=provider.serialize() if provider else 'Not Set',
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