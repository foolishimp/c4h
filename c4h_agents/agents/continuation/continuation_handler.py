from typing import Dict, Any, List, Tuple, Optional
import time
import random
import traceback

import litellm
from .config import ContentType, CONTENT_TYPES, STITCHING_STRATEGIES, CONFIG
from .overlap_strategies import find_overlap
from .joining_strategies import join_content, clean_json_content
from .utils import setup_logger, make_llm_request, get_content_from_response, detect_content_type

class ContinuationHandler:
    """Handles LLM response continuations using a sliding window approach."""

    def __init__(self, parent_agent):
        self.parent = parent_agent
        self.model_str = parent_agent.model_str
        self.provider = parent_agent.provider
        self.temperature = parent_agent.temperature
        self.max_continuation_attempts = parent_agent.max_continuation_attempts
        
        self.logger = setup_logger(parent_agent)
        self.metrics = {
            "attempts": 0, "exact_matches": 0, "token_matches": 0, "fuzzy_matches": 0,
            "structure_matches": 0, "fallbacks": 0, "rate_limit_retries": 0,
            "append_fallbacks": 0, "stitching_retries": 0
        }

    def get_completion_with_continuation(
            self, messages: List[Dict[str, str]], max_attempts: Optional[int] = None
    ) -> Tuple[str, Any]:
        """Get completion with automatic continuation using sliding window."""
        attempt = 0
        max_tries = max_attempts or self.max_continuation_attempts
        accumulated_content = ""
        final_response = None
        
        content_type = detect_content_type(messages, self.logger)
        self.logger.info("Starting continuation process",
                        extra={"model": self.model_str, "content_type": content_type})
        
        rate_limit_retries = 0
        rate_limit_backoff = CONFIG["rate_limit_retry_base_delay"]
        completion_params = self._build_completion_params(messages)
        
        try:
            response = make_llm_request(completion_params, self.logger, self.parent)
            content = get_content_from_response(response, self.logger)
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
                
                overlap_size = CONTENT_TYPES[content_type]["overlap_size"](len(accumulated_content))
                overlap = accumulated_content[-overlap_size:]
                continuation_prompt = self._create_prompt(overlap, content_type)
                
                cont_messages = messages.copy()
                cont_messages.append({"role": "assistant", "content": accumulated_content})
                cont_messages.append({"role": "user", "content": continuation_prompt})
                
                self.logger.info("Requesting continuation",
                               extra={"attempt": attempt, "content_type": content_type,
                                      "overlap_length": len(overlap)})
                
                stitching_success = False
                stitching_attempts = 0
                
                while stitching_attempts <= CONFIG["max_stitching_retries"] and not stitching_success:
                    try:
                        cont_params = completion_params.copy()
                        cont_params["messages"] = cont_messages
                        response = make_llm_request(cont_params, self.logger, self.parent)
                        cont_content = get_content_from_response(response, self.logger)
                        
                        joined_content, join_method = join_content(
                            accumulated_content, cont_content, content_type, self.logger
                        )
                        
                        if join_method != "append_fallbacks":
                            self.metrics[join_method] += 1
                            accumulated_content = joined_content
                            final_response = response
                            stitching_success = True
                        else:
                            stitching_attempts += 1
                            self.metrics["stitching_retries"] += 1
                            
                            if stitching_attempts <= len(STITCHING_STRATEGIES):
                                strategy = STITCHING_STRATEGIES[stitching_attempts - 1]
                                self.logger.warning(f"Stitching failed, trying {strategy['name']}",
                                                  extra={"attempt": attempt, "stitching_attempt": stitching_attempts})
                                if strategy["prompt"]:
                                    cont_messages[-1]["content"] = strategy["prompt"](
                                        accumulated_content if strategy["name"] == "follow_up" else overlap,
                                        content_type
                                    )
                                continue
                            
                    except litellm.RateLimitError as e:
                        rate_limit_retries += 1
                        self.metrics["rate_limit_retries"] += 1
                        if rate_limit_retries > CONFIG["rate_limit_max_retries"]:
                            self.logger.error("Max rate limit retries exceeded",
                                            extra={"retry_count": rate_limit_retries, "error": str(e)})
                            raise
                        jitter = 0.1 * rate_limit_backoff * (0.5 - random.random())
                        current_backoff = min(rate_limit_backoff + jitter, CONFIG["rate_limit_max_backoff"])
                        self.logger.warning("Rate limit encountered, backing off",
                                          extra={"attempt": attempt, "retry_count": rate_limit_retries,
                                                 "backoff_seconds": current_backoff, "error": str(e)})
                        time.sleep(current_backoff)
                        rate_limit_backoff = min(rate_limit_backoff * 2, CONFIG["rate_limit_max_backoff"])
                        continue
                    
                    except Exception as e:
                        self.logger.error("Continuation attempt failed",
                                        extra={"attempt": attempt, "error": str(e),
                                               "stack_trace": traceback.format_exc()})
                        stitching_attempts += 1
                        self.metrics["stitching_retries"] += 1
                        continue
                
                if not stitching_success:
                    append_marker = f"\n--- CONTINUATION STITCHING FAILED AFTER RETRIES ---\n"
                    accumulated_content += append_marker + cont_content
                    self.metrics["append_fallbacks"] += 1
                    self.logger.error("All stitching retries failed, appending content",
                                    extra={"attempt": attempt, "content_type": content_type})
                    break
            
            if CONTENT_TYPES[content_type]["requires_cleaning"]:
                accumulated_content = clean_json_content(accumulated_content, self.logger)
            
            if final_response and hasattr(final_response, 'choices') and final_response.choices:
                final_response.choices[0].message.content = accumulated_content
            
            self.logger.info("Continuation process completed",
                          extra={"attempts": attempt, "content_type": content_type,
                                 "metrics": self.metrics, "content_length": len(accumulated_content)})
                
            return accumulated_content, final_response
            
        except Exception as e:
            self.logger.error("Continuation process failed",
                           extra={"error": str(e), "stack_trace": traceback.format_exc(),
                                  "content_so_far": accumulated_content[:200]})
            raise

    def _create_prompt(self, overlap: str, content_type: ContentType) -> str:
        """Create continuation prompt from config."""
        try:
            base_prompt = CONTENT_TYPES[content_type]["prompt"]
            prompt = f"{base_prompt}\n\nHere's the content to continue. Continue FROM THE EXACT END of this text:\n\n------------BEGIN CONTENT------------\n{overlap}\n------------END CONTENT------------\n\nContinue exactly from where this leaves off, maintaining the same format and structure."
            self.logger.debug("Continuation prompt created",
                            extra={"content_type": content_type, "prompt_length": len(prompt)})
            return prompt
        except Exception as e:
            self.logger.error("Prompt creation failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return f"Continue from: {overlap}"

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