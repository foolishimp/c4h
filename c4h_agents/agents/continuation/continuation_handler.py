# Path: /Users/jim/src/apps/c4h/c4h_agents/agents/continuation/continuation_handler.py

from typing import Dict, Any, List, Tuple, Optional
import time
import random
import traceback
import logging # Keep standard logging for fallback

import litellm
# Import the central logger utility
from c4h_agents.utils.logging import get_logger
from .config import WINDOW_CONFIG, STITCHING_STRATEGIES, requires_json_cleaning
from .overlap_strategies import find_explicit_overlap
from .joining_strategies import join_with_explicit_overlap, clean_json_content

class ContinuationHandler:
    """Handles LLM response continuations using a simple window approach with explicit overlaps."""

    def __init__(self, parent_agent):
        self.parent = parent_agent
        # Ensure these attributes exist on parent_agent before accessing
        self.model_str = getattr(parent_agent, 'model_str', 'unknown_model')
        self.provider = getattr(parent_agent, 'provider', None)
        self.temperature = getattr(parent_agent, 'temperature', 0)
        self.max_continuation_attempts = getattr(parent_agent, 'max_continuation_attempts', 5)

        # Use parent agent's logger if available, otherwise get a default one
        # Parent logger should already be configured via get_logger(config)
        self.logger = getattr(parent_agent, 'logger', get_logger()) # Use central logger

        self.metrics = {
            "attempts": 0,
            "exact_matches": 0,
            "fallback_matches": 0,
            "rate_limit_retries": 0,
            "stitching_retries": 0
        }
        # Log initialization completion
        self.logger.debug("ContinuationHandler initialized", parent_agent_type=type(parent_agent).__name__)


    def get_completion_with_continuation(
            self, messages: List[Dict[str, str]], max_attempts: Optional[int] = None
    ) -> Tuple[str, Any]:
        """Get completion with automatic continuation using window-based approach with explicit overlaps."""
        attempt = 0
        max_tries = max_attempts or self.max_continuation_attempts
        accumulated_content = ""
        final_response = None
        
        self.logger.info("Starting continuation process", #
                        model=self.model_str) # Pass directly
        
        rate_limit_retries = 0
        rate_limit_backoff = WINDOW_CONFIG["rate_limit_retry_base_delay"]
        
        try:
            # Initial request - Use internal helper that calls parent methods if needed
            completion_params = self._build_completion_params(messages) # Build initial params
            response = self._make_llm_request(completion_params)
            content = self._get_content_from_response(response)
            self.logger.debug("Received initial content",
                            content_preview=content[:100], content_length=len(content)) # Pass directly
            
            final_response = response
            accumulated_content = content
            
            while attempt < max_tries:
                # Ensure response structure is valid before accessing choices
                if not response or not hasattr(response, 'choices') or not response.choices:
                     # --- SYNTAX FIX: Correct argument order ---
                     self.logger.error("Invalid response structure received from LLM", response_obj=response)
                     # Decide how to handle: raise error, return partial, etc.
                     # For now, let's break assuming something went wrong upstream.
                     break 

                finish_reason = getattr(response.choices[0], 'finish_reason', None)
                if finish_reason != 'length':
                    # --- SYNTAX FIX: Correct argument order ---
                    self.logger.info("Continuation complete", 
                                  finish_reason=finish_reason, attempts=attempt) # Pass directly
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
                
                # --- SYNTAX FIX: Correct argument order ---
                self.logger.info("Requesting continuation", 
                               attempt=attempt, window_size=window_size, # Pass directly
                               overlap_size=len(explicit_overlap))
                
                # Handle stitching attempts
                stitching_success = False
                stitching_attempts = 0
                
                while stitching_attempts <= WINDOW_CONFIG["max_stitching_retries"] and not stitching_success:
                    try:
                        # Make continuation request
                        cont_params = completion_params.copy() # Start with original params
                        cont_params["messages"] = cont_messages # Use updated messages
                        response = self._make_llm_request(cont_params) # Make the LLM call
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
                            final_response = response # Update final response to the latest one
                            stitching_success = True
                            self.logger.debug("Successfully joined content with explicit overlap",
                                           content_length=len(accumulated_content)) # Pass directly
                        else:
                            # Couldn't find explicit overlap, try fallback strategies
                            stitching_attempts += 1
                            self.metrics["stitching_retries"] += 1
                            
                            if stitching_attempts <= len(STITCHING_STRATEGIES):
                                strategy = STITCHING_STRATEGIES[stitching_attempts - 1]
                                # --- SYNTAX FIX: Correct argument order ---
                                self.logger.warning(f"Stitching failed, trying {strategy['name']}",
                                                 attempt=attempt, stitching_attempt=stitching_attempts) # Pass directly
                                
                                # Use progressively stronger overlap requests in fallback strategies
                                adjusted_overlap_size = overlap_size * (1 + stitching_attempts // 2)
                                # --- Ensure strategy['prompt'] is callable ---
                                if callable(strategy.get("prompt")):
                                     cont_messages[-1]["content"] = strategy["prompt"](
                                         context_window,
                                         adjusted_overlap_size
                                     )
                                else:
                                     self.logger.error("Invalid prompt strategy", strategy_name=strategy.get('name'))
                                     # Handle error - maybe break or use a default prompt
                                     break # Break stitching loop if strategy is bad
                                # --- End Ensure ---
                                continue # Retry LLM call with new prompt
                            else:
                                 # Exhausted fallback strategies for this attempt
                                 self.logger.error("Exhausted all stitching fallback strategies", attempt=attempt)
                                 break # Exit stitching loop for this attempt

                    except litellm.RateLimitError as e:
                        # Handle rate limits with exponential backoff
                        rate_limit_retries += 1
                        self.metrics["rate_limit_retries"] += 1
                        if rate_limit_retries > WINDOW_CONFIG["rate_limit_max_retries"]:
                            # --- SYNTAX FIX: Correct argument order ---
                            self.logger.error("Max rate limit retries exceeded",
                                retry_count=rate_limit_retries, error=str(e)) # Pass directly
                            raise
                        
                        # Calculate backoff with jitter
                        jitter = 0.1 * rate_limit_backoff * (0.5 - random.random())
                        current_backoff = min(rate_limit_backoff + jitter, WINDOW_CONFIG["rate_limit_max_backoff"])
                        # --- SYNTAX FIX: Correct argument order ---
                        self.logger.warning("Rate limit encountered, backing off",
                                attempt=attempt, retry_count=rate_limit_retries, # Pass directly
                                backoff_seconds=current_backoff, error=str(e))
                        time.sleep(current_backoff)
                        rate_limit_backoff = min(rate_limit_backoff * 2, WINDOW_CONFIG["rate_limit_max_backoff"])
                        continue # Retry the same stitching attempt after backoff
                    
                    except Exception as e:
                        # General error handling during a stitching attempt
                        # --- SYNTAX FIX: Correct argument order ---
                        self.logger.error("Continuation attempt failed", #
                                       attempt=attempt, error=str(e), # Pass directly
                                       stack_trace=traceback.format_exc())
                        stitching_attempts += 1
                        self.metrics["stitching_retries"] += 1
                        # Don't immediately break, allow retries via outer while loop if applicable
                        # break # Removed break here - let outer loop handle max attempts

                # If all stitching attempts failed for this continuation step
                if not stitching_success:
                    append_marker = f"\n\n--- CONTINUATION STITCHING FAILED AFTER {stitching_attempts} RETRIES ---\n\n"
                    # Use the last `cont_content` received, even though it didn't stitch
                    accumulated_content += append_marker + cont_content 
                    self.metrics["fallback_matches"] += 1
                    # --- SYNTAX FIX: Correct argument order ---
                    self.logger.error("All stitching retries failed, appending with marker", #
                                  attempt=attempt) # Pass directly
                    break # Exit the main continuation loop (attempt loop)
            
            # Clean up final accumulated content if needed
            if requires_json_cleaning(accumulated_content):
                accumulated_content = clean_json_content(accumulated_content, self.logger)
            
            # Update final response's content ONLY if final_response is valid
            if final_response and hasattr(final_response, 'choices') and final_response.choices:
                 # Ensure message structure exists before modification
                 if hasattr(final_response.choices[0], 'message'):
                      # Ensure content attribute exists before assigning
                      if hasattr(final_response.choices[0].message, 'content'):
                           final_response.choices[0].message.content = accumulated_content
                      else:
                           # Handle case where message object might be incomplete
                           self.logger.warning("Could not update final response content, message.content missing", response_obj=final_response)
                 else:
                      self.logger.warning("Could not update final response content, message structure missing", response_obj=final_response)

            # --- SYNTAX FIX: Correct argument order ---
            self.logger.info("Continuation process completed",
                           attempts=attempt, metrics=self.metrics, # Pass directly
                           content_length=len(accumulated_content))
                
            return accumulated_content, final_response
            
        except Exception as e:
             # --- SYNTAX FIX: Correct argument order ---
            self.logger.error("Continuation process failed", #
                           error=str(e), stack_trace=traceback.format_exc(), # Pass directly
                           content_so_far=accumulated_content[:200])
            raise # Re-raise the exception after logging

    def _create_prompt(self, context_window: str, explicit_overlap: str) -> str:
        """Create a continuation prompt that explicitly requests overlap."""
        try:
            # Create clear prompt with explicit overlap request
            # Ensure explicit_overlap is not empty before using len()
            overlap_len = len(explicit_overlap) if explicit_overlap else 0
            prompt = f"""
I need you to continue the previous response that was interrupted due to length limits.
HERE IS THE END OF YOUR PREVIOUS RESPONSE:
------------BEGIN PREVIOUS CONTENT------------
{context_window}
------------END PREVIOUS CONTENT------------

CRITICAL CONTINUATION INSTRUCTIONS:
1. First, repeat these EXACT {overlap_len} characters:
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
            # --- SYNTAX FIX: Correct argument order ---
            self.logger.error("Prompt creation failed",
                              error=str(e), stack_trace=traceback.format_exc()) # Pass directly
            return f"Continue precisely from: {explicit_overlap}" # Fallback prompt

    def _build_completion_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Build parameters for LLM completion request."""
        try:
            params = {"model": self.model_str, "messages": messages}
            # Ensure provider exists before checking value
            if self.provider and self.provider.value != "openai": # Example, adjust as needed for OpenAI specifics
                params["temperature"] = self.temperature

            # Ensure parent has necessary methods before calling
            provider_config = {}
            if hasattr(self.parent, '_get_provider_config') and self.provider:
                 provider_config = self.parent._get_provider_config(self.provider)
            elif hasattr(self.parent, 'config_node') and self.provider: # Fallback via config_node
                 # Ensure parent.config_node is not None before calling get_value
                 if self.parent.config_node:
                      provider_config = self.parent.config_node.get_value(f"llm_config.providers.{self.provider.value}") or {}


            params.update(provider_config.get("model_params", {}))
            if "api_base" in provider_config:
                params["api_base"] = provider_config["api_base"]

            # Add extended thinking if applicable (copied logic from BaseLLM._setup_litellm)
            if self.provider and self.model and self.provider.value == "anthropic" and "claude-3-7-sonnet" in self.model:
                 # Ensure parent has _get_agent_name method
                 agent_name = getattr(self.parent, '_get_agent_name', lambda: 'unknown_agent')()
                 agent_path = f"llm_config.agents.{agent_name}"
                 config_node_to_use = getattr(self.parent, 'config_node', None)
                 agent_thinking_config = None
                 if config_node_to_use:
                      agent_thinking_config = config_node_to_use.get_value(f"{agent_path}.extended_thinking")
                 if not agent_thinking_config:
                      agent_thinking_config = provider_config.get("extended_thinking", {})
                 if agent_thinking_config and agent_thinking_config.get("enabled", False) is True:
                      # Add 'thinking' parameter if enabled
                      params['thinking'] = True # Or appropriate value based on API
                      self.logger.debug("Added 'thinking' parameter for Claude 3.7 Sonnet")


            self.logger.debug("Completion parameters built", params=params) # Pass directly
            return params
        except Exception as e:
            # --- SYNTAX FIX: Correct argument order ---
            self.logger.error("Completion params build failed",
                            error=str(e), stack_trace=traceback.format_exc()) # Pass directly
            raise
            
    def _make_llm_request(self, params: Dict[str, Any]) -> Any:
        """Make LLM request using LiteLLM."""
        # Relies on global LiteLLM config set by BaseLLM._setup_litellm
        try:
            # Remove params not directly supported by litellm.completion if necessary
            # Or ensure they are handled by callbacks or custom logic if applicable
            # Example: 'thinking' might need custom handling or specific API kwargs
            api_kwargs = {}
            if 'thinking' in params and params['thinking'] and self.provider and self.provider.value == 'anthropic':
                 # Assuming 'thinking' needs to be passed specifically for Anthropic via completion
                 # This might require checking LiteLLM documentation for the correct way
                 # api_kwargs['thinking'] = params['thinking'] # Example, adjust as needed
                 # For now, let's just log it's present, assuming LiteLLM handles it via model_str or automatically
                 self.logger.debug("Passing 'thinking' parameter potentially via LiteLLM", model=params.get('model'))
                 # Remove it from standard params if it's not a standard LiteLLM arg
                 # del params['thinking'] # Uncomment if LiteLLM errors with unknown 'thinking' param

            safe_params = {k: v for k, v in params.items()
                        if k in ['model', 'messages', 'temperature', 'max_tokens', 'top_p', 'stream', 'api_base']} # Added api_base

            self.logger.debug("Making LLM request") # Log message remains simple
            return litellm.completion(**safe_params, **api_kwargs) # Pass api_kwargs if needed
        except Exception as e:
            # --- SYNTAX FIX: Correct argument order ---
            self.logger.error("LLM request failed",
                            error=str(e), stack_trace=traceback.format_exc()) # Pass directly
            raise
            
    def _get_content_from_response(self, response: Any) -> str:
        """Extract content from LLM response."""
        # Reusing the robust extraction logic
        try:
            if hasattr(response, 'choices') and response.choices:
                 if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                     content = response.choices[0].message.content
                     if isinstance(content, str): # Ensure it's a string
                          return content
                     else:
                          self.logger.warning("Extracted content is not string in _get_content_from_response", content_type=type(content).__name__)
                          return str(content) # Convert non-string content
            self.logger.warning("No valid content structure found in LLM response", response_obj=response)
            return ""
        except Exception as e:
            # --- SYNTAX FIX: Correct argument order ---
            self.logger.error("Content extraction failed in _get_content_from_response",
                             error=str(e), stack_trace=traceback.format_exc()) # Pass directly
            return ""