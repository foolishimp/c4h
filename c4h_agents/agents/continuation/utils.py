# Path: /Users/jim/src/apps/c4h/c4h_agents/agents/continuation/utils.py
from typing import Dict, Any, List, Optional
import logging
import traceback
import litellm # Keep litellm import
from litellm import completion
# Import the central logger utility
from c4h_agents.utils.logging import get_logger

# Removed ContentType enum import as detect_content_type is removed


def setup_logger(parent_agent) -> logging.Logger:
    """Set up logger using parent agent's logger or fallback."""
    try:
        logger_instance = getattr(parent_agent, 'logger', None)
        if logger_instance is None: # Fallback if parent doesn't have logger
            # Use the central get_logger instead of standard logging
            logger_instance = get_logger()
            logger_instance.warning("continuation.utils.setup_logger_fallback", reason="Parent agent missing logger")
        # Don't log here, let the caller log initialization if needed
        return logger_instance
    except Exception as e:
        # Fallback logger in case get_logger fails or parent_agent is problematic
        logger_instance = logging.getLogger(__name__)
        # Ensure handlers are added only once
        if not logger_instance.hasHandlers():
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             logger_instance.addHandler(handler)
             logger_instance.setLevel(logging.WARNING) # Default to WARNING for fallback
        logger_instance.error("continuation.utils.logger_setup_failed", error=str(e), stack_trace=traceback.format_exc())
        return logger_instance


def make_llm_request(params: Dict[str, Any], logger, parent_agent) -> Any: # logger parameter already exists
    """Make LLM request using LiteLLM, relying on global config."""
    try:
        # Rely on globally configured LiteLLM settings via _setup_litellm in BaseLLM
        
        safe_params = {k: v for k, v in params.items()
                      if k in ['model', 'messages', 'temperature', 'max_tokens', 'top_p', 'stream']}
        # Get provider config via parent agent to find api_base etc.
        # Ensure parent_agent has the necessary method
        provider_config = {}
        if hasattr(parent_agent, '_get_provider_config') and hasattr(parent_agent, 'provider') and parent_agent.provider:
             provider_config = parent_agent._get_provider_config(parent_agent.provider)
        else:
             # Use logger passed as argument
             logger.warning("continuation.utils.make_llm_request_missing_parent_info")


        if "api_base" in provider_config:
            safe_params["api_base"] = provider_config["api_base"]
        
        # Use logger passed as argument
        logger.debug("Making LLM request", params=safe_params) # Pass params directly to structlog
        return completion(**safe_params)
    except litellm.RateLimitError as e:
        # Use logger passed as argument
        logger.warning("Rate limit error in LLM request", error=str(e)) # Pass directly
        raise
    except Exception as e:
        # Use logger passed as argument
        logger.error("LLM request failed",
                     error=str(e), # Pass directly
                     stack_trace=traceback.format_exc())
        raise


def get_content_from_response(response: Any, logger) -> str: # logger parameter already exists
    """Extract content from LLM response."""
    try:
        if hasattr(response, 'choices') and response.choices:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                content = response.choices[0].message.content
                # Ensure content is a string before logging length
                if isinstance(content, str):
                     # Use logger passed as argument
                     logger.debug("Content extracted from response",
                                 content_preview=content[:100]) # Pass directly
                else:
                     # Use logger passed as argument
                     logger.warning("Extracted content is not string", content_type=type(content).__name__)
                     content = str(content) # Convert non-string content
                return content
        # Use logger passed as argument
        logger.warning("No content found in response structure")
        return ""
    except Exception as e:
        # Use logger passed as argument
        logger.error("Content extraction failed",
                     error=str(e), # Pass directly
                     stack_trace=traceback.format_exc())
        return ""

# Removed detect_content_type function as it was unused and depended on removed ContentType enum