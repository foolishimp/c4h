# File: c4h_agents/agents/continuation/continuation_handler.py

from typing import Dict, Any, List, Tuple, Optional
import time
import random
import traceback

import litellm
from .config import WINDOW_CONFIG, STITCHING_STRATEGIES, requires_json_cleaning
from .overlap_strategies import find_explicit_overlap
from .joining_strategies import join_with_explicit_overlap, clean_json_content

class ContinuationHandler:
    """Handles LLM response continuations using a simple window approach with explicit overlaps."""

    def __init__(self, parent_agent):
        self.parent = parent_agent
        self.model_str = parent_agent.model_str
        self.provider = parent_agent.provider
        self.temperature = parent_agent.temperature
        self.max_continuation_attempts = parent_agent.max_continuation_attempts
        
        # Set up logger - use parent's logger if available
        self.logger = getattr(parent_agent, 'logger', None)
        if not self.logger:
            import logging
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)
        
        self.metrics = {
            "attempts": 0, 
            "exact_matches": 0, 
            "fallback_matches": 0,
            "rate_limit_retries": 0,
            "stitching_retries": 0
        }

    def get_completion_with_continuation(
            self, messages: List[Dict[str, str]], max_attempts: Optional[int] = None
    ) -> Tuple[str, Any]:
        """Get completion with automatic continuation using window-based approach with explicit overlaps."""
        attempt = 0
        max_tries = max_attempts or self.max_continuation_attempts
        accumulated_content = ""
        final_response = None
        
        self.logger.info("Starting continuation process",
                        extra={"model": self.model_str})
        
        rate_limit_retries = 0
        rate_limit_backoff = WINDOW_CONFIG["rate_limit_retry_base_delay"]
        completion_params = self._build_completion_params(messages)
        
        try:
            # Initial request
            response = self._make_llm_request(completion_params)
            content = self._get_content_from_response(response)
            self.logger.debug("Received initial content",
                            extra={"content_preview": content[:100], "content_length": len(content)})
            
            final_response = response
            accumulated_content = content
            
            while attempt < max_tries:
                finish_reason = getattr(response.choices[0], 'finish_reason', None)
                if finish_reason != 'length':
                    self.logger.info("Continuation complete",
                                  extra={"finish_reason": finish_reason, "attempts": attempt})
                    break
                
                attempt += 1
                self.metrics["attempts"] += 1
                
                # Get context window for this continuation
                window_size = min(
                    max(len(accumulated_content) // 2, WINDOW_CONFIG["min_context_window"]), 
                    WINDOW_CONFIG["max_context_window"]
                )
                context_window = accumulated_content[-window_size:]
                
                # Get explicit overlap to request
                overlap_size = WINDOW_CONFIG["overlap_size"]
                explicit_overlap = accumulated_content[-overlap_size:] if len(accumulated_content) >= overlap_size else accumulated_content
                
                # Create continuation prompt with explicit overlap request
                continuation_prompt = self._create_prompt(context_window, explicit_overlap)
                
                # Setup continuation messages
                cont_messages = messages.copy()
                cont_messages.append({"role": "assistant", "content": accumulated_content})
                cont_messages.append({"role": "user", "content": continuation_prompt})
                
                self.logger.info("Requesting continuation",
                               extra={"attempt": attempt, "window_size": window_size, 
                                      "overlap_size": len(explicit_overlap)})
                
                # Handle stitching attempts
                stitching_success = False
                stitching_attempts = 0
                
                while stitching_attempts <= WINDOW_CONFIG["max_stitching_retries"] and not stitching_success:
                    try:
                        # Make continuation request
                        cont_params = completion_params.copy()
                        cont_params["messages"] = cont_messages
                        response = self._make_llm_request(cont_params)
                        cont_content = self._get_content_from_response(response)
                        
                        # Try to join with explicit overlap
                        joined_content, success = join_with_explicit_overlap(
                            accumulated_content, 
                            cont_content, 
                            explicit_overlap,
                            overlap_size,  # Pass the requested overlap size as a hint
                            self.logger
                        )
                        
                        if success:
                            # Successfully joined
                            self.metrics["exact_matches"] += 1
                            accumulated_content = joined_content
                            final_response = response
                            stitching_success = True
                            self.logger.debug("Successfully joined content with explicit overlap",
                                           extra={"content_length": len(accumulated_content)})
                        else:
                            # Couldn't find explicit overlap, try fallback strategies
                            stitching_attempts += 1
                            self.metrics["stitching_retries"] += 1
                            
                            if stitching_attempts <= len(STITCHING_STRATEGIES):
                                strategy = STITCHING_STRATEGIES[stitching_attempts - 1]
                                self.logger.warning(f"Stitching failed, trying {strategy['name']}",
                                                 extra={"attempt": attempt, "stitching_attempt": stitching_attempts})
                                
                                # Use progressively stronger overlap requests in fallback strategies
                                adjusted_overlap_size = overlap_size * (1 + stitching_attempts // 2)
                                cont_messages[-1]["content"] = strategy["prompt"](
                                    context_window,
                                    adjusted_overlap_size
                                )
                                continue
                            
                    except litellm.RateLimitError as e:
                        # Handle rate limits with exponential backoff
                        rate_limit_retries += 1
                        self.metrics["rate_limit_retries"] += 1
                        if rate_limit_retries > WINDOW_CONFIG["rate_limit_max_retries"]:
                            self.logger.error("Max rate limit retries exceeded",
                                           extra={"retry_count": rate_limit_retries, "error": str(e)})
                            raise
                        
                        # Calculate backoff with jitter
                        jitter = 0.1 * rate_limit_backoff * (0.5 - random.random())
                        current_backoff = min(rate_limit_backoff + jitter, WINDOW_CONFIG["rate_limit_max_backoff"])
                        self.logger.warning("Rate limit encountered, backing off",
                                         extra={"attempt": attempt, "retry_count": rate_limit_retries,
                                                "backoff_seconds": current_backoff, "error": str(e)})
                        time.sleep(current_backoff)
                        rate_limit_backoff = min(rate_limit_backoff * 2, WINDOW_CONFIG["rate_limit_max_backoff"])
                        continue
                    
                    except Exception as e:
                        # General error handling
                        self.logger.error("Continuation attempt failed",
                                       extra={"attempt": attempt, "error": str(e),
                                              "stack_trace": traceback.format_exc()})
                        stitching_attempts += 1
                        self.metrics["stitching_retries"] += 1
                        continue
                
                # If all stitching attempts failed, use a simple append with a marker
                if not stitching_success:
                    append_marker = f"\n\n--- CONTINUATION STITCHING FAILED AFTER {stitching_attempts} RETRIES ---\n\n"
                    accumulated_content += append_marker + cont_content
                    self.metrics["fallback_matches"] += 1
                    self.logger.error("All stitching retries failed, appending with marker",
                                   extra={"attempt": attempt})
                    break
            
            # Clean up content if needed
            if requires_json_cleaning(accumulated_content):
                accumulated_content = clean_json_content(accumulated_content, self.logger)
            
            # Update final response's content
            if final_response and hasattr(final_response, 'choices') and final_response.choices:
                final_response.choices[0].message.content = accumulated_content
            
            self.logger.info("Continuation process completed",
                          extra={"attempts": attempt, "metrics": self.metrics, 
                                "content_length": len(accumulated_content)})
                
            return accumulated_content, final_response
            
        except Exception as e:
            self.logger.error("Continuation process failed",
                           extra={"error": str(e), "stack_trace": traceback.format_exc(),
                                  "content_so_far": accumulated_content[:200]})
            raise

    def _create_prompt(self, context_window: str, explicit_overlap: str) -> str:
        """Create a continuation prompt that explicitly requests overlap."""
        try:
            # Create clear prompt with explicit overlap request
            prompt = f"""
I need you to continue the previous response that was interrupted due to length limits.

HERE IS THE END OF YOUR PREVIOUS RESPONSE:
------------BEGIN PREVIOUS CONTENT------------
{context_window}
------------END PREVIOUS CONTENT------------

CRITICAL CONTINUATION INSTRUCTIONS:
1. First, repeat these EXACT {len(explicit_overlap)} characters:
------------OVERLAP TO REPEAT------------
{explicit_overlap}
------------END OVERLAP------------

2. Then continue seamlessly from that point
3. Maintain identical style, formatting and organization
4. If in the middle of a code block, function, or component, respect its structure

Begin by repeating the overlap text exactly, then continue:
"""
            return prompt
        except Exception as e:
            self.logger.error("Prompt creation failed",
                           extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return f"Continue precisely from: {explicit_overlap}"

    def _build_completion_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Build parameters for LLM completion request."""
        try:
            params = {"model": self.model_str, "messages": messages}
            if self.provider.value != "openai":
                params["temperature"] = self.temperature
            provider_config = self.parent._get_provider_config(self.provider)
            params.update(provider_config.get("model_params", {}))
            if "api_base" in provider_config:
                params["api_base"] = provider_config["api_base"]
            self.logger.debug("Completion parameters built", extra={"params": params})
            return params
        except Exception as e:
            self.logger.error("Completion params build failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            raise
            
    def _make_llm_request(self, params: Dict[str, Any]) -> Any:
        """Make LLM request with rate limit handling."""
        try:
            provider_config = self.parent._get_provider_config(self.provider)
            if "api_base" in provider_config:
                params["api_base"] = provider_config["api_base"]
                
            safe_params = {k: v for k, v in params.items()
                        if k in ['model', 'messages', 'temperature', 'max_tokens', 'top_p', 'stream']}
            
            self.logger.debug("Making LLM request")
            return litellm.completion(**safe_params)
        except Exception as e:
            self.logger.error("LLM request failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            raise
            
    def _get_content_from_response(self, response: Any) -> str:
        """Extract content from LLM response."""
        try:
            if hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    content = response.choices[0].message.content
                    return content
            return ""
        except Exception as e:
            self.logger.error("Content extraction failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return ""