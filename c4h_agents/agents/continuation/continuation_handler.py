# Path: /Users/jim/src/apps/c4h/c4h_agents/agents/continuation/continuation_handler.py
from typing import Dict, Any, List, Tuple, Optional
import time
import random
import traceback
import logging # Keep standard logging for fallback

import litellm
# Import the central logger utility
from c4h_agents.utils.logging import get_logger
from .config import WINDOW_CONFIG, STITCHING_STRATEGIES
from .overlap_strategies import find_explicit_overlap
from .joining_strategies import join_with_explicit_overlap

class ContinuationHandler:
    """Handles LLM response continuations using a simple window approach with explicit overlaps."""

    # Path: /Users/jim/src/apps/c4h/c4h_agents/agents/continuation/continuation_handler.py
    # (Only the __init__ method is shown for brevity - replace the existing method)

    def __init__(self, parent_agent):
        self.parent = parent_agent
        # Ensure these attributes exist on parent_agent before accessing
        self.model_str = getattr(parent_agent, 'model_str', 'unknown_model')
        self.provider = getattr(parent_agent, 'provider', None)
        # --- FIX: Added missing self.model assignment ---
        self.model = getattr(parent_agent, 'model', None)
        # --- END FIX ---
        self.temperature = getattr(parent_agent, 'temperature', 0)
        self.max_continuation_attempts = getattr(parent_agent, 'max_continuation_attempts', 5)

        # Use parent agent's logger if available, otherwise get a default one
        self.logger = getattr(parent_agent, 'logger', get_logger())

        self.metrics = {
            "attempts": 0,
            "exact_matches": 0,
            "fallback_matches": 0,
            "rate_limit_retries": 0,
            "stitching_retries": 0
        }
        self.logger.debug("ContinuationHandler initialized", parent_agent_type=type(parent_agent).__name__)

    def get_completion_with_continuation(
            self, messages: List[Dict[str, str]], max_attempts: Optional[int] = None
    ) -> Tuple[str, Any]:
        """Get completion with automatic continuation using window-based approach with explicit overlaps."""
        attempt = 0
        max_tries = max_attempts or self.max_continuation_attempts
        accumulated_content = ""
        final_response = None
        
        self.logger.info("Starting continuation process", model=self.model_str)
        
        rate_limit_retries = 0
        rate_limit_backoff = WINDOW_CONFIG["rate_limit_retry_base_delay"]
        
        try:
            completion_params = self._build_completion_params(messages) 
            response = self._make_llm_request(completion_params)
            content = self._get_content_from_response(response)
            self.logger.debug("Received initial content",
                            content_preview=content[:100], content_length=len(content))
            
            final_response = response
            accumulated_content = content
            
            while attempt < max_tries:
                if not response or not hasattr(response, 'choices') or not response.choices:
                     self.logger.error("Invalid response structure received from LLM", response_obj=response)
                     break 

                finish_reason = getattr(response.choices[0], 'finish_reason', None)
                if finish_reason != 'length':
                    self.logger.info("Continuation complete", 
                                  finish_reason=finish_reason, attempts=attempt)
                    break
                
                attempt += 1
                self.metrics["attempts"] += 1
                
                window_size = min(
                    max(len(accumulated_content) // 2, WINDOW_CONFIG["min_context_window"]), 
                    WINDOW_CONFIG["max_context_window"]
                )
                context_window = accumulated_content[-window_size:]
                
                overlap_size = WINDOW_CONFIG["overlap_size"]
                explicit_overlap = accumulated_content[-overlap_size:] if len(accumulated_content) >= overlap_size else accumulated_content
                
                continuation_prompt = self._create_prompt(context_window, explicit_overlap)
                
                cont_messages = messages.copy()
                cont_messages.append({"role": "assistant", "content": accumulated_content})
                cont_messages.append({"role": "user", "content": continuation_prompt})
                
                self.logger.info("Requesting continuation", 
                               attempt=attempt, window_size=window_size, 
                               overlap_size=len(explicit_overlap))
                
                stitching_success = False
                stitching_attempts = 0
                
                while stitching_attempts <= WINDOW_CONFIG["max_stitching_retries"] and not stitching_success:
                    try:
                        cont_params = completion_params.copy() 
                        cont_params["messages"] = cont_messages 
                        response = self._make_llm_request(cont_params) 
                        cont_content = self._get_content_from_response(response)
                        
                        joined_content, success = join_with_explicit_overlap(
                            accumulated_content, 
                            cont_content, 
                            explicit_overlap,
                            overlap_size, 
                            self.logger
                        )
                        
                        if success:
                            self.metrics["exact_matches"] += 1
                            accumulated_content = joined_content
                            final_response = response 
                            stitching_success = True
                            self.logger.debug("Successfully joined content with explicit overlap",
                                           content_length=len(accumulated_content))
                        else:
                            stitching_attempts += 1
                            self.metrics["stitching_retries"] += 1
                            
                            if stitching_attempts <= len(STITCHING_STRATEGIES):
                                strategy = STITCHING_STRATEGIES[stitching_attempts - 1]
                                # Fixed syntax error here
                                self.logger.warning(f"Stitching failed, trying {strategy['name']}",
                                                 attempt=attempt, stitching_attempt=stitching_attempts) 
                                
                                adjusted_overlap_size = overlap_size * (1 + stitching_attempts // 2)
                                if callable(strategy.get("prompt")):
                                     cont_messages[-1]["content"] = strategy["prompt"](
                                         context_window,
                                         adjusted_overlap_size
                                     )
                                else:
                                     self.logger.error("Invalid prompt strategy", strategy_name=strategy.get('name'))
                                     break 
                                continue 
                            else:
                                 self.logger.error("Exhausted all stitching fallback strategies", attempt=attempt)
                                 break 

                    except litellm.RateLimitError as e:
                        rate_limit_retries += 1
                        self.metrics["rate_limit_retries"] += 1
                        if rate_limit_retries > WINDOW_CONFIG["rate_limit_max_retries"]:
                            # Fixed syntax error here
                            self.logger.error("Max rate limit retries exceeded",
                                retry_count=rate_limit_retries, error=str(e))
                            raise
                        
                        jitter = 0.1 * rate_limit_backoff * (0.5 - random.random())
                        current_backoff = min(rate_limit_backoff + jitter, WINDOW_CONFIG["rate_limit_max_backoff"])
                        # Fixed syntax error here
                        self.logger.warning("Rate limit encountered, backing off",
                                attempt=attempt, retry_count=rate_limit_retries,
                                backoff_seconds=current_backoff, error=str(e))
                        time.sleep(current_backoff)
                        rate_limit_backoff = min(rate_limit_backoff * 2, WINDOW_CONFIG["rate_limit_max_backoff"])
                        continue 
                    
                    except Exception as e:
                        # Fixed syntax error here
                        self.logger.error("Continuation attempt failed", 
                                       attempt=attempt, error=str(e), 
                                       stack_trace=traceback.format_exc())
                        stitching_attempts += 1
                        self.metrics["stitching_retries"] += 1
                
                if not stitching_success:
                    append_marker = f"\n\n--- CONTINUATION STITCHING FAILED AFTER {stitching_attempts} RETRIES ---\n\n"
                    accumulated_content += append_marker + cont_content 
                    self.metrics["fallback_matches"] += 1
                    # Fixed syntax error here
                    self.logger.error("All stitching retries failed, appending with marker", 
                                  attempt=attempt) 
                    break 
            
            # Clean up final accumulated content by removing any marker text only
            # Removed automatic brace balancing which can introduce syntax errors
            accumulated_content = self._clean_marker_text(accumulated_content)
                        # Update final response's content ONLY if final_response is valid
            if final_response and hasattr(final_response, 'choices') and final_response.choices:
                 if hasattr(final_response.choices[0], 'message'):
                      if hasattr(final_response.choices[0].message, 'content'):
                           final_response.choices[0].message.content = accumulated_content
                      else:
                           self.logger.warning("Could not update final response content, message.content missing", response_obj=final_response)
                 else:
                      self.logger.warning("Could not update final response content, message structure missing", response_obj=final_response)

            # Fixed syntax error here
            self.logger.info("Continuation process completed",
                           attempts=attempt, metrics=self.metrics, 
                           content_length=len(accumulated_content))
                
            return accumulated_content, final_response
            
        except Exception as e:
             # Fixed syntax error here
            self.logger.error("Continuation process failed", 
                           error=str(e), stack_trace=traceback.format_exc(), 
                           content_so_far=accumulated_content[:200])
            raise # Re-raise the exception after logging

    def _create_prompt(self, context_window: str, explicit_overlap: str) -> str:
        """Create a continuation prompt that explicitly requests overlap."""
        try:
            overlap_len = len(explicit_overlap) if explicit_overlap else 0
            prompt = f"""
I need you to continue code that was interrupted due to length limits.

HERE IS THE END OF THE PREVIOUS CONTENT:
```previous
{context_window}
```

CRITICAL CONTINUATION INSTRUCTIONS:
1. First, analyze the code to detect its language and structure:
   - Is it Python? Look for indentation, def/class keywords, colons followed by indented blocks
   - Is it JavaScript/TypeScript? Look for braces, function/const/let keywords, semicolons
   - Is it JSON? Look for strict object/array notation with double quotes
   - Is it Markdown? Look for heading markers, list items, code blocks
   - Is it a diff/patch? Look for +/- line prefixes, @@ markers, and file path headers

2. Before continuing, ANALYZE THE STRUCTURE FOR COMPLETENESS:
   - Check for unclosed parentheses, brackets, or braces that need closing
   - Check for incomplete function calls or expressions (e.g., missing arguments or closing parentheses)
   - Check if any string literals are unclosed
   - Check if any Python statements are incomplete (e.g., missing colons, incomplete if/else blocks)
   - For Python dicts/objects, ensure all keys have values and there are no trailing commas
   - For function calls, ensure all required arguments are provided

3. Then, repeat EXACTLY (character for character) this text:
```repeat_exactly
{explicit_overlap}
```

4. Continue the code seamlessly from that point, maintaining:
   - FOR PYTHON: 
     * Exact indentation is critical
     * Complete all function calls with proper arguments
     * Close all open parentheses, brackets, and braces
     * Maintain consistent indentation for blocks
     * Make sure all return statements return proper values
   
   - FOR JAVASCRIPT/TYPESCRIPT: 
     * Ensure all braces, parentheses, and quotes are properly balanced
     * Complete all function calls with proper arguments
     * Close all code blocks properly
   
   - FOR JSON/YAML: 
     * Maintain strict format with proper quoting
     * Ensure correct comma placement
     * Verify all objects have complete key-value pairs
   
   - FOR DIFF/PATCH:
     * Maintain proper diff syntax
     * Complete change blocks with context lines
     * Ensure all file sections are properly marked

5. VERIFY YOUR CONTINUATION:
   - Double-check that you've properly completed any statements, expressions, or blocks
   - Verify all syntax is valid for the detected language
   - Ensure no dangling tokens, incomplete expressions, or syntax errors

DO NOT include the ```repeat_exactly marker or any other markers in your response.
Output ONLY the continuation starting with the exact overlap text.

Begin your response now with the exact overlap text and continue:
"""
            return prompt
        except Exception as e:
            self.logger.error("Prompt creation failed",
                              error=str(e), stack_trace=traceback.format_exc())
            return f"Continue precisely from: {explicit_overlap}" # Fallback prompt

    def _build_completion_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Build parameters for LLM completion request."""
        try:
            params = {"model": self.model_str, "messages": messages}
            if self.provider and self.provider.value != "openai": 
                params["temperature"] = self.temperature

            provider_config = {}
            if hasattr(self.parent, '_get_provider_config') and self.provider:
                 provider_config = self.parent._get_provider_config(self.provider)
            elif hasattr(self.parent, 'config_node') and self.provider: 
                 if self.parent.config_node:
                      provider_config = self.parent.config_node.get_value(f"llm_config.providers.{self.provider.value}") or {}


            params.update(provider_config.get("model_params", {}))
            if "api_base" in provider_config:
                params["api_base"] = provider_config["api_base"]

            if self.provider and self.model and self.provider.value == "anthropic" and "claude-3-7-sonnet" in self.model:
                 agent_name = getattr(self.parent, '_get_agent_name', lambda: 'unknown_agent')()
                 agent_path = f"llm_config.agents.{agent_name}"
                 config_node_to_use = getattr(self.parent, 'config_node', None)
                 agent_thinking_config = None
                 if config_node_to_use:
                      agent_thinking_config = config_node_to_use.get_value(f"{agent_path}.extended_thinking")
                 if not agent_thinking_config:
                      agent_thinking_config = provider_config.get("extended_thinking", {})
                 if agent_thinking_config and agent_thinking_config.get("enabled", False) is True:
                      params['thinking'] = True 
                      self.logger.debug("Added 'thinking' parameter for Claude 3.7 Sonnet")


            self.logger.debug("Completion parameters built", params=params)
            return params
        except Exception as e:
            # Fixed syntax error here
            self.logger.error("Completion params build failed",
                            error=str(e), stack_trace=traceback.format_exc())
            raise
            
    def _make_llm_request(self, params: Dict[str, Any]) -> Any:
        """Make LLM request using LiteLLM."""
        try:
            api_kwargs = {}
            if 'thinking' in params and params['thinking'] and self.provider and self.provider.value == 'anthropic':
                 self.logger.debug("Passing 'thinking' parameter potentially via LiteLLM", model=params.get('model'))

            safe_params = {k: v for k, v in params.items()
                        if k in ['model', 'messages', 'temperature', 'max_tokens', 'top_p', 'stream', 'api_base']} 

            self.logger.debug("Making LLM request") 
            return litellm.completion(**safe_params, **api_kwargs) 
        except Exception as e:
            # Fixed syntax error here
            self.logger.error("LLM request failed",
                            error=str(e), stack_trace=traceback.format_exc())
            raise
            
    def _get_content_from_response(self, response: Any) -> str:
        """Extract content from LLM response."""
        try:
            if hasattr(response, 'choices') and response.choices:
                 if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                     content = response.choices[0].message.content
                     if isinstance(content, str): 
                          return content
                     else:
                          self.logger.warning("Extracted content is not string in _get_content_from_response", content_type=type(content).__name__)
                          return str(content) 
            self.logger.warning("No valid content structure found in LLM response", response_obj=response)
            return ""
        except Exception as e:
             self.logger.error("Content extraction failed in _get_content_from_response",
                             error=str(e), stack_trace=traceback.format_exc())
             return ""
             
    def _clean_marker_text(self, content: str) -> str:
        """Remove any marker text that might have been included."""
        try:
            cleaned = content
            
            # Remove marker patterns
            markers = [
                "```repeat_exactly", 
                "```previous",
                "```", 
                "------------END OVERLAP------------",
                "------------OVERLAP TO REPEAT------------", 
            ]
            
            # Check line by line to remove any lines that contain markers
            lines = cleaned.split('\n')
            cleaned_lines = []
            
            for line in lines:
                if not any(marker in line for marker in markers):
                    cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines)
        except Exception as e:
            self.logger.error("Error cleaning response artifacts", 
                           error=str(e), stack_trace=traceback.format_exc())
            return content  # Return original if cleaning fails