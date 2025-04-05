"""
Enhanced LLM response continuation handling using line number and indentation tracking with JSON formatting.
Path: c4h_agents/agents/base_llm_continuation.py
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import random
from datetime import datetime
import litellm
from litellm import completion

from c4h_agents.agents.types import LLMProvider, LogDetail
from c4h_agents.utils.logging import get_logger
from c4h_agents.agents.continuation_parsing import (
    format_with_line_numbers_and_indentation, 
    create_line_json,
    parse_json_content,
    attempt_repair_parse,
    numbered_lines_to_content,
    get_content_sample
)
from c4h_agents.agents.continuation_prompting import (
    detect_content_type,
    create_numbered_continuation_prompt
)

logger = get_logger()

class ContinuationHandler:
    """Handles LLM response continuations using line number and indentation approach with JSON formatting"""

    def __init__(self, parent_agent):
        """Initialize with parent agent reference"""
        self.parent = parent_agent
        self.model_str = parent_agent.model_str
        self.provider = parent_agent.provider
        self.temperature = parent_agent.temperature
        self.max_continuation_attempts = parent_agent.max_continuation_attempts
        
        # Rate limit handling
        self.rate_limit_retry_base_delay = 2.0
        self.rate_limit_max_retries = 5
        self.rate_limit_max_backoff = 60
        
        # Logger setup
        self.logger = getattr(parent_agent, 'logger', logger)
        
        # Simple metrics
        self.metrics = {"attempts": 0, "total_lines": 0}

    def get_completion_with_continuation(
            self, 
            messages: List[Dict[str, str]],
            max_attempts: Optional[int] = None
        ) -> Tuple[str, Any]:
        """Get completion with line-number and indentation-based continuation using JSON formatting"""
        
        attempt = 0
        max_tries = max_attempts or self.max_continuation_attempts
        accumulated_lines = []
        final_response = None
        
        # Detect content type
        content_type = detect_content_type(messages)
        
        self.logger.info("llm.continuation_starting", model=self.model_str, content_type=content_type)
        
        # Rate limit handling
        rate_limit_retries = 0
        rate_limit_backoff = self.rate_limit_retry_base_delay
        
        # Initial request
        completion_params = self._build_completion_params(messages)
        try:
            response = self._make_llm_request(completion_params)
            
            # Process initial response
            content = self._get_content_from_response(response)
            final_response = response
            
            # Log truncation detection
            finish_reason = getattr(response.choices[0], 'finish_reason', None)
            if finish_reason == 'length':
                content_lines = content.splitlines()
                last_lines = "\n".join(content_lines[-10:]) if len(content_lines) > 10 else content
                self.logger.info("llm.truncation_detected", 
                            finish_reason=finish_reason,
                            line_count=len(content_lines),
                            last_lines=last_lines)
            
            # Format initial content with line numbers and indentation
            numbered_lines = format_with_line_numbers_and_indentation(content)
            accumulated_lines = numbered_lines
            
            # Continue making requests until we're done or hit max attempts
            next_line = len(accumulated_lines) + 1
            
            while next_line > 1 and attempt < max_tries:
                attempt += 1
                self.metrics["attempts"] += 1
                
                # Create JSON array from accumulated lines for context
                context_json = create_line_json(accumulated_lines, max_context_lines=30)
                
                # Create continuation prompt with appropriate example for the content type
                continuation_prompt = create_numbered_continuation_prompt(
                    context_json, next_line, content_type)
                
                # Prepare continuation message
                cont_messages = messages.copy()
                cont_messages.append({"role": "assistant", "content": context_json})
                cont_messages.append({"role": "user", "content": continuation_prompt})
                
                self.logger.info("llm.requesting_continuation", 
                               attempt=attempt, 
                               next_line=next_line)
                
                # Make continuation request
                try:
                    cont_params = completion_params.copy()
                    cont_params["messages"] = cont_messages
                    response = self._make_llm_request(cont_params)
                    
                    # Extract content from response
                    cont_content = self._get_content_from_response(response)
                    
                    # Log continuation content sample
                    cont_lines = cont_content.splitlines()
                    first_lines = "\n".join(cont_lines[:10]) if len(cont_lines) > 10 else cont_content
                    self.logger.info("llm.continuation_received", 
                                   attempt=attempt,
                                   next_line=next_line,
                                   line_count=len(cont_lines),
                                   continuation_preview=first_lines)
                    
                    # Parse line-numbered content from JSON including indentation
                    new_lines = parse_json_content(cont_content, next_line, self.logger)
                    
                    if not new_lines:
                        self.logger.warning("llm.no_parsable_content", attempt=attempt)
                        # Try a repair attempt with more aggressive parsing
                        new_lines = attempt_repair_parse(cont_content, next_line, self.logger)
                        if not new_lines:
                            break
                    
                    # Get content before merge
                    pre_merge_content = numbered_lines_to_content(accumulated_lines)
                    pre_merge_sample = get_content_sample(pre_merge_content, next_line-5)
                    
                    # Update accumulated lines
                    accumulated_lines.extend(new_lines)
                    
                    # Get content after merge
                    post_merge_content = numbered_lines_to_content(accumulated_lines)
                    post_merge_sample = get_content_sample(post_merge_content, next_line)
                    
                    # Log the merge boundary
                    self.logger.info("llm.continuation_merged",
                                   boundary_line=next_line,
                                   new_lines_count=len(new_lines),
                                   pre_merge=pre_merge_sample,
                                   post_merge=post_merge_sample)
                    
                    finish_reason = getattr(response.choices[0], 'finish_reason', None)
                    
                    # Update final response
                    final_response = response
                    
                    if finish_reason != 'length':
                        self.logger.info("llm.continuation_segment_complete", 
                                       finish_reason=finish_reason)
                        break
                    
                    # Update next line number for next continuation
                    next_line = len(accumulated_lines) + 1
                    
                except litellm.RateLimitError as e:
                    # Handle rate limit errors with exponential backoff
                    error_msg = str(e)
                    
                    rate_limit_retries += 1
                    
                    if rate_limit_retries > self.rate_limit_max_retries:
                        self.logger.error("llm.rate_limit_max_retries_exceeded", 
                                      retry_count=rate_limit_retries,
                                      error=error_msg[:200])
                        raise
                                  
                    # Calculate backoff with jitter
                    jitter = 0.1 * rate_limit_backoff * (0.5 - random.random())
                    current_backoff = min(rate_limit_backoff + jitter, self.rate_limit_max_backoff)
                    
                    self.logger.warning("llm.rate_limit_backoff", 
                                     attempt=attempt,
                                     retry_count=rate_limit_retries,
                                     backoff_seconds=current_backoff,
                                     error=error_msg[:200])
                    
                    # Apply exponential backoff with base 2
                    time.sleep(current_backoff)
                    rate_limit_backoff = min(rate_limit_backoff * 2, self.rate_limit_max_backoff)
                    continue
                
                except Exception as e:
                    self.logger.error("llm.continuation_failed", error=str(e))
                    break
            
            # Convert accumulated lines back to raw content
            final_content = numbered_lines_to_content(accumulated_lines)
            
            # Update response content
            if final_response and hasattr(final_response, 'choices') and final_response.choices:
                # Store original length for logging
                original_length = len(final_response.choices[0].message.content) if hasattr(final_response.choices[0].message, 'content') else 0
                
                # Update with complete assembled content
                final_response.choices[0].message.content = final_content
                
                # Update finish reason to indicate proper completion
                if hasattr(final_response.choices[0], 'finish_reason'):
                    final_response.choices[0].finish_reason = 'stop'
                
                # Log the content expansion with boundary sample
                self.logger.info("llm.continuation_complete", 
                              original_length=original_length,
                              final_length=len(final_content),
                              segments_count=attempt+1,
                              final_lines_count=len(accumulated_lines),
                              final_boundary_sample=get_content_sample(final_content, len(accumulated_lines)-5))
                
            self.metrics["total_lines"] = len(accumulated_lines)
            
            return final_content, final_response
            
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            self.logger.error("llm.continuation_failed", 
                           error=error_msg, 
                           error_type=error_type)
            
            raise
    
    def _build_completion_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Build parameters for LLM completion request"""
        completion_params = {
            "model": self.model_str,
            "messages": messages,
        }

        # Only add temperature for providers that support it
        if self.provider.value != "openai":
            completion_params["temperature"] = self.temperature
            
        # Add provider-specific params
        provider_config = self.parent._get_provider_config(self.provider)
        
        # Add model-specific parameters from config
        model_params = provider_config.get("model_params", {})
        if model_params:
            logger.debug("llm.applying_model_params", 
                        provider=self.provider.serialize(),
                        params=list(model_params.keys()))
            completion_params.update(model_params)
        
        if "api_base" in provider_config:
            completion_params["api_base"] = provider_config["api_base"]
                
        return completion_params

    def _make_llm_request(self, completion_params: Dict[str, Any]) -> Any:
        """Make LLM request with rate limit handling"""
        try:
            # Get provider config
            provider_config = self.parent._get_provider_config(self.provider)
            
            # Configure litellm
            litellm.retry = True
            litellm.max_retries = 3
            litellm.retry_wait = 2
            litellm.max_retry_wait = 60
            litellm.retry_exponential = True
                
            # Filter to only supported parameters
            safe_params = {
                k: v for k, v in completion_params.items() 
                if k in ['model', 'messages', 'temperature', 'max_tokens', 'top_p', 'stream']
            }
            
            if "api_base" in provider_config:
                safe_params["api_base"] = provider_config["api_base"]
                    
            response = completion(**safe_params)
            return response
            
        except litellm.RateLimitError as e:
            logger.warning("llm.rate_limit_error", error=str(e)[:200])
            raise
            
        except Exception as e:
            logger.error("llm.request_error", error=str(e))
            raise
        
    def _get_content_from_response(self, response):
        """Extract content from LLM response"""
        if hasattr(response, 'choices') and response.choices:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                return response.choices[0].message.content
        return ""