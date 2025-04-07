from typing import Dict, Any, List, Optional
import logging
import traceback
import litellm
from litellm import completion
from .config import ContentType

def setup_logger(parent_agent) -> logging.Logger:
    """Set up logger using parent agent's logger or fallback."""
    try:
        logger = getattr(parent_agent, 'logger', None)
        if logger is None:
            logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)
        logger.debug("Logger initialized", extra={"logger_type": type(logger).__name__})
        return logger
    except Exception as e:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        logger.error("Logger setup failed, using fallback",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return logger

def make_llm_request(params: Dict[str, Any], logger, parent_agent) -> Any:
    """Make LLM request with rate limit handling."""
    try:
        litellm.retry = True
        litellm.max_retries = 3
        litellm.retry_wait = 2
        litellm.max_retry_wait = 60
        litellm.retry_exponential = True
        
        safe_params = {k: v for k, v in params.items()
                      if k in ['model', 'messages', 'temperature', 'max_tokens', 'top_p', 'stream']}
        provider_config = parent_agent._get_provider_config(parent_agent.provider)
        if "api_base" in provider_config:
            safe_params["api_base"] = provider_config["api_base"]
        
        logger.debug("Making LLM request", extra={"params": safe_params})
        return completion(**safe_params)
    except litellm.RateLimitError as e:
        logger.warning("Rate limit error in LLM request", extra={"error": str(e)})
        raise
    except Exception as e:
        logger.error("LLM request failed",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        raise

def get_content_from_response(response: Any, logger) -> str:
    """Extract content from LLM response."""
    try:
        if hasattr(response, 'choices') and response.choices:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                content = response.choices[0].message.content
                logger.debug("Content extracted from response",
                            extra={"content_preview": content[:100]})
                return content
        logger.warning("No content found in response")
        return ""
    except Exception as e:
        logger.error("Content extraction failed",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return ""

def detect_content_type(messages: List[Dict[str, str]], logger) -> ContentType:
    """Detect content type from messages."""
    try:
        content = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
        
        is_solution_designer = any('"changes":' in msg.get("content", "") and 
                                  '"file_path":' in msg.get("content", "") and 
                                  '"diff":' in msg.get("content", "")
                                  for msg in messages if msg.get("role") == "user")
        is_code = any("```" in msg.get("content", "") or "def " in msg.get("content", "")
                     for msg in messages if msg.get("role") == "user")
        is_json = any("json" in msg.get("content", "").lower() or 
                     msg.get("content", "").strip().startswith("{") or 
                     msg.get("content", "").strip().startswith("[")
                     for msg in messages if msg.get("role") == "user")
        is_diff = any("--- " in msg.get("content", "") and "+++ " in msg.get("content", "")
                     for msg in messages if msg.get("role") == "user")
        
        if is_solution_designer:
            detected_type = ContentType.SOLUTION_DESIGNER
        elif is_code and is_json:
            detected_type = ContentType.JSON_CODE
        elif is_code:
            detected_type = ContentType.CODE
        elif is_json:
            detected_type = ContentType.JSON
        elif is_diff:
            detected_type = ContentType.DIFF
        else:
            detected_type = ContentType.TEXT
            
        logger.debug("Content type detected", extra={"type": detected_type})
        return detected_type
    except Exception as e:
        logger.error("Content type detection failed",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return ContentType.TEXT