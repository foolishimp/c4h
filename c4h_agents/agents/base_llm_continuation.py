from typing import Dict, Any, List, Tuple, Optional, Union
import time
import re
import json
import hashlib
import random
from datetime import datetime
from enum import Enum
import logging
import traceback

import litellm
from litellm import completion
from c4h_agents.agents.types import LLMProvider, LogDetail
from c4h_agents.utils.logging import get_logger

# Fallback to standard logger if get_logger fails
try:
    logger = get_logger()
except Exception as e:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

class ContentType(str, Enum):
    """Content types for specialized handling"""
    TEXT = "text"
    CODE = "code"
    JSON = "json"
    JSON_CODE = "json_code"
    DIFF = "diff"
    SOLUTION_DESIGNER = "solution_designer"

class ContinuationHandler:
    """
    Handles LLM response continuations using a sliding window approach.
    Uses content-aware joining and specialized overlap strategies.
    """

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
        
        # Overlap configuration
        self.min_overlap_size = 50
        self.max_overlap_size = 500
        
        # Logger setup - Use parent agent's logger if available, otherwise fallback
        self.logger = getattr(parent_agent, 'logger', logger)
        # Avoid setting level directly; assume it's configured elsewhere
        # If needed, log a debug message to confirm logger initialization
        self.logger.debug("ContinuationHandler logger initialized",
                         extra={"logger_type": type(self.logger).__name__})
        
        # Metrics tracking
        self.metrics = {
            "attempts": 0,
            "exact_matches": 0,
            "token_matches": 0,
            "fuzzy_matches": 0,
            "structure_matches": 0,
            "fallbacks": 0,
            "rate_limit_retries": 0,
            "append_fallbacks": 0
        }

    def get_completion_with_continuation(
            self,
            messages: List[Dict[str, str]],
            max_attempts: Optional[int] = None
        ) -> Tuple[str, Any]:
        """
        Get completion with automatic continuation using sliding window.
        
        Args:
            messages: List of message dictionaries with role and content
            max_attempts: Maximum number of continuation attempts
            
        Returns:
            Tuple of (accumulated_content, final_response)
        """
        attempt = 0
        max_tries = max_attempts or self.max_continuation_attempts
        accumulated_content = ""
        final_response = None
        
        content_type = self._detect_content_type(messages)
        self.logger.info("Starting continuation process",
                        extra={"model": self.model_str, "content_type": content_type})
        
        rate_limit_retries = 0
        rate_limit_backoff = self.rate_limit_retry_base_delay
        
        completion_params = self._build_completion_params(messages)
        
        try:
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
                
                overlap = self._calculate_overlap_window(accumulated_content, content_type)
                continuation_prompt = self._create_continuation_prompt(overlap, content_type)
                
                cont_messages = messages.copy()
                cont_messages.append({"role": "assistant", "content": accumulated_content})
                cont_messages.append({"role": "user", "content": continuation_prompt})
                
                self.logger.info("Requesting continuation",
                               extra={"attempt": attempt, "content_type": content_type,
                                      "overlap_length": len(overlap)})
                
                try:
                    cont_params = completion_params.copy()
                    cont_params["messages"] = cont_messages
                    response = self._make_llm_request(cont_params)
                    cont_content = self._get_content_from_response(response)
                    
                    joined_content, join_method = self._join_continuations(
                        accumulated_content, cont_content, content_type)
                    
                    self.metrics[join_method] += 1
                    accumulated_content = joined_content
                    final_response = response
                    
                except litellm.RateLimitError as e:
                    rate_limit_retries += 1
                    self.metrics["rate_limit_retries"] += 1
                    
                    if rate_limit_retries > self.rate_limit_max_retries:
                        self.logger.error("Max rate limit retries exceeded",
                                        extra={"retry_count": rate_limit_retries, "error": str(e)})
                        raise
                    
                    jitter = 0.1 * rate_limit_backoff * (0.5 - random.random())
                    current_backoff = min(rate_limit_backoff + jitter, self.rate_limit_max_backoff)
                    
                    self.logger.warning("Rate limit encountered, backing off",
                                      extra={"attempt": attempt, "retry_count": rate_limit_retries,
                                             "backoff_seconds": current_backoff, "error": str(e)})
                    
                    time.sleep(current_backoff)
                    rate_limit_backoff = min(rate_limit_backoff * 2, self.rate_limit_max_backoff)
                    continue
                    
                except Exception as e:
                    self.logger.error("Continuation attempt failed",
                                    extra={"attempt": attempt, "error": str(e),
                                           "stack_trace": traceback.format_exc()})
                    append_marker = f"\n--- CONTINUATION STITCHING FAILED ---\n"
                    accumulated_content += append_marker + cont_content
                    self.metrics["append_fallbacks"] += 1
                    break
            
            if content_type in (ContentType.JSON, ContentType.SOLUTION_DESIGNER):
                accumulated_content = self._clean_json_content(accumulated_content)
            
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

    def _detect_content_type(self, messages: List[Dict[str, str]]) -> str:
        """
        Detect content type from messages for specialized handling.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Content type string
        """
        try:
            content = ""
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    break
            
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
                
            self.logger.debug("Content type detected", extra={"type": detected_type})
            return detected_type
        except Exception as e:
            self.logger.error("Content type detection failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return ContentType.TEXT

    def _calculate_overlap_window(self, content: str, content_type: str) -> str:
        """
        Calculate appropriate overlap window based on content type.
        
        Args:
            content: Content to extract overlap from
            content_type: Type of content for specialized handling
            
        Returns:
            Overlap window
        """
        try:
            if content_type == ContentType.SOLUTION_DESIGNER:
                window_size = min(max(len(content) // 3, 200), self.max_overlap_size)
            elif content_type in (ContentType.JSON, ContentType.JSON_CODE):
                window_size = min(max(len(content) // 3, 150), self.max_overlap_size)
            elif content_type == ContentType.CODE:
                window_size = min(max(len(content) // 4, 100), self.max_overlap_size)
            else:
                window_size = min(max(len(content) // 5, 80), self.max_overlap_size)
            
            window = content[-window_size:]
            
            if content_type in (ContentType.JSON, ContentType.JSON_CODE, ContentType.SOLUTION_DESIGNER):
                adjusted_window = self._align_json_window(window)
                if adjusted_window:
                    window = adjusted_window
            elif content_type == ContentType.CODE:
                adjusted_window = self._align_code_window(window)
                if adjusted_window:
                    window = adjusted_window
            
            self.logger.debug("Overlap window calculated",
                            extra={"window_size": len(window), "content_type": content_type})
            return window
        except Exception as e:
            self.logger.error("Overlap window calculation failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return content[-self.min_overlap_size:]

    def _align_json_window(self, window: str) -> Optional[str]:
        """
        Align window with JSON structure boundaries.
        
        Args:
            window: Current window
            
        Returns:
            Adjusted window or None if no adjustment needed
        """
        try:
            open_braces = window.count('{')
            close_braces = window.count('}')
            open_brackets = window.count('[')
            close_brackets = window.count(']')
            
            if open_braces == close_braces and open_brackets == close_brackets:
                return window
                
            if open_braces > close_braces:
                brace_balance = 0
                for i in range(len(window) - 1, -1, -1):
                    if window[i] == '}':
                        brace_balance += 1
                    elif window[i] == '{':
                        brace_balance -= 1
                        if brace_balance < 0:
                            return window[i:]
                            
            if open_brackets > close_brackets:
                bracket_balance = 0
                for i in range(len(window) - 1, -1, -1):
                    if window[i] == ']':
                        bracket_balance += 1
                    elif window[i] == '[':
                        bracket_balance -= 1
                        if bracket_balance < 0:
                            return window[i:]
            
            return window
        except Exception as e:
            self.logger.error("JSON window alignment failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return window

    def _align_code_window(self, window: str) -> Optional[str]:
        """
        Align window with code block boundaries.
        
        Args:
            window: Current window
            
        Returns:
            Adjusted window or None if no adjustment needed
        """
        try:
            lines = window.split('\n')
            if len(lines) <= 1:
                return window
            
            first_line = lines[0]
            first_indent = len(first_line) - len(first_line.lstrip())
            
            if first_indent > 0:
                for i, line in enumerate(lines):
                    if line.strip() and len(line) - len(line.lstrip()) < first_indent:
                        return '\n'.join(lines[i:])
            
            return window
        except Exception as e:
            self.logger.error("Code window alignment failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return window

    def _create_continuation_prompt(self, overlap: str, content_type: str) -> str:
        """
        Create continuation prompt based on content type.
        
        Args:
            overlap: Overlap window
            content_type: Type of content
            
        Returns:
            Continuation prompt
        """
        try:
            if content_type == ContentType.SOLUTION_DESIGNER:
                prompt = """
I need you to continue the following content exactly from where it left off.

CRITICAL REQUIREMENTS:
1. You are continuing a Solution Designer response with JSON structure
2. Maintain the exact structure including proper escaping of quotation marks
3. Continue precisely where the text ends, never repeating any content
4. For diff sections, ensure proper escaping of newlines (\\n) and quotes (\")
5. Never output explanatory text or comments outside the JSON structure
6. Complete any unfinished JSON objects, arrays, or properties
"""
            elif content_type in (ContentType.JSON, ContentType.JSON_CODE):
                prompt = """
I need you to continue the following content exactly from where it left off.

CRITICAL REQUIREMENTS:
1. You are continuing a JSON structure
2. Maintain the exact structure with proper nesting
3. Continue precisely where the text ends, never repeating any content
4. Ensure proper escaping of special characters
5. Complete any unfinished JSON objects, arrays, or properties
6. Never add explanatory text or comments outside the JSON structure
"""
            elif content_type == ContentType.CODE:
                prompt = """
I need you to continue the following content exactly from where it left off.

CRITICAL REQUIREMENTS:
1. You are continuing code
2. Maintain consistent indentation and coding style
3. Continue precisely where the text ends, never repeating any content
4. Complete any unfinished functions, blocks, or statements
5. Never add explanatory text or comments outside the code
"""
            else:
                prompt = """
I need you to continue the following content exactly from where it left off.

CRITICAL REQUIREMENTS:
1. Continue precisely where the text ends, never repeating any content
2. Maintain the same style, formatting, and tone as the original
3. Do not add any explanatory text, headers, or comments
"""
            
            prompt += """
Here's the content to continue. Continue FROM THE EXACT END of this text:

------------BEGIN CONTENT------------
{}
------------END CONTENT------------

Continue exactly from where this leaves off, maintaining the same format and structure.
""".format(overlap)
            
            self.logger.debug("Continuation prompt created",
                            extra={"content_type": content_type, "prompt_length": len(prompt)})
            return prompt
        except Exception as e:
            self.logger.error("Continuation prompt creation failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return f"Continue from: {overlap}"

    def _join_continuations(
        self,
        previous: str,
        current: str,
        content_type: str
    ) -> Tuple[str, str]:
        """
        Join continuations using multiple strategies with simplified failure fallback.
        
        Args:
            previous: Previous content
            current: Current continuation
            content_type: Type of content
            
        Returns:
            Tuple of (joined_content, join_method)
        """
        self.logger.debug("Attempting to join continuations",
                        extra={"prev_length": len(previous), "curr_length": len(current),
                               "content_type": content_type})
        
        exact_match = self._find_exact_overlap(previous, current)
        if exact_match:
            joined = previous + current[len(exact_match):]
            self.logger.debug("Exact match found", extra={"overlap_length": len(exact_match)})
            return joined, "exact_matches"
        
        token_match = self._find_token_match(previous, current)
        if token_match:
            position, confidence = token_match
            if confidence >= 0.7:
                joined = previous + current[position:]
                self.logger.debug("Token match found",
                                extra={"position": position, "confidence": confidence})
                return joined, "token_matches"
        
        if content_type == ContentType.SOLUTION_DESIGNER:
            solution_join = self._join_solution_designer(previous, current)
            if solution_join:
                self.logger.debug("Solution designer join successful")
                return solution_join, "structure_matches"
        elif content_type in (ContentType.JSON, ContentType.JSON_CODE):
            json_join = self._join_json_content(previous, current)
            if json_join:
                self.logger.debug("JSON join successful")
                return json_join, "structure_matches"
        
        fuzzy_match = self._find_fuzzy_match(previous, current)
        if fuzzy_match:
            self.logger.debug("Fuzzy match found")
            return fuzzy_match, "fuzzy_matches"
        
        append_marker = f"\n--- UNABLE TO GUARANTEE STITCHING ---\n"
        joined = previous + append_marker + current
        self.logger.warning("Unable to guarantee stitching, using append fallback",
                          extra={"content_type": content_type})
        return joined, "append_fallbacks"

    def _find_exact_overlap(self, previous: str, current: str) -> Optional[str]:
        """
        Find exact overlap between previous and current content.
        
        Args:
            previous: Previous content
            current: Current continuation
            
        Returns:
            Overlap text if found, None otherwise
        """
        try:
            min_size = min(self.min_overlap_size, len(previous), len(current))
            max_size = min(self.max_overlap_size, len(previous), len(current))
            
            for size in range(max_size, min_size - 1, -10):
                overlap = previous[-size:]
                if current.startswith(overlap):
                    return overlap
            
            for size in range(min_size, max_size + 1):
                if size % 10 == 0:
                    continue
                overlap = previous[-size:]
                if current.startswith(overlap):
                    return overlap
                    
            return None
        except Exception as e:
            self.logger.error("Exact overlap detection failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return None

    def _find_token_match(self, previous: str, current: str) -> Optional[Tuple[int, float]]:
        """
        Find token-based overlap between previous and current content.
        
        Args:
            previous: Previous content
            current: Current continuation
            
        Returns:
            Tuple of (position, confidence) if match found, None otherwise
        """
        try:
            prev_tokens = self._tokenize(previous[-1000:])
            curr_tokens = self._tokenize(current[:1000])
            
            if not prev_tokens or not curr_tokens:
                self.logger.debug("No tokens available for matching")
                return None
                
            best_match_len = 0
            best_match_pos = 0
            
            for i in range(len(prev_tokens) - 4):
                prev_seq = prev_tokens[i:i+5]
                for j in range(len(curr_tokens) - 4):
                    curr_seq = curr_tokens[j:j+5]
                    if prev_seq == curr_seq:
                        match_len = 5
                        while (i + match_len < len(prev_tokens) and 
                               j + match_len < len(curr_tokens) and 
                               prev_tokens[i + match_len] == curr_tokens[j + match_len]):
                            match_len += 1
                        if match_len > best_match_len:
                            best_match_len = match_len
                            best_match_pos = j
            
            if best_match_len >= 5:
                char_pos = 0
                for k in range(best_match_pos):
                    char_pos += len(curr_tokens[k]) + 1
                confidence = min(best_match_len / 10, 1.0)
                return char_pos, confidence
                
            return None
        except Exception as e:
            self.logger.error("Token match detection failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return None

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for token matching.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        try:
            return re.findall(r'\w+|[^\w\s]', text)
        except Exception as e:
            self.logger.error("Tokenization failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return text.split()

    def _find_fuzzy_match(self, previous: str, current: str) -> Optional[str]:
        """
        Find fuzzy match using hash-based approach.
        
        Args:
            previous: Previous content
            current: Current continuation
            
        Returns:
            Joined content if match found, None otherwise
        """
        try:
            prev_norm = ''.join(previous.lower().split())
            curr_norm = ''.join(current.lower().split())
            
            for window_size in [100, 70, 50, 30]:
                if len(prev_norm) < window_size or len(curr_norm) < window_size:
                    continue
                prev_hash = hashlib.md5(prev_norm[-window_size:].encode()).hexdigest()
                for i in range(len(curr_norm) - window_size + 1):
                    curr_window = curr_norm[i:i+window_size]
                    curr_hash = hashlib.md5(curr_window.encode()).hexdigest()
                    if prev_hash == curr_hash:
                        char_pos = len(current) * i // len(curr_norm)
                        return previous + current[char_pos:]
            return None
        except Exception as e:
            self.logger.error("Fuzzy match detection failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return None

    def _join_solution_designer(self, previous: str, current: str) -> Optional[str]:
        """
        Join solution designer content with structure awareness.
        
        Args:
            previous: Previous content
            current: Current continuation
            
        Returns:
            Joined content if successful, None otherwise
        """
        try:
            open_braces = previous.count('{') - previous.count('}')
            open_brackets = previous.count('[') - previous.count(']')
            open_quotes = previous.count('"') % 2
            
            if open_braces == 0 and open_brackets == 0 and open_quotes == 0:
                if re.match(r'^\s*\{', current):
                    if re.search(r',\s*$', previous) or re.search(r'\[\s*$', previous):
                        return previous + current
                    elif re.search(r'\}\s*$', previous):
                        return previous + ',\n' + current
            
            patterns = [
                r'"file_path"\s*:\s*"[^"]+"\s*,',
                r'"type"\s*:\s*"[^"]+"\s*,',
                r'"description"\s*:\s*"[^"]+"\s*,',
                r'"diff"\s*:\s*"'
            ]
            
            for pattern in patterns:
                prev_match = re.search(f'({pattern})\\s*$', previous)
                if prev_match:
                    curr_match = re.search(f'^\\s*({pattern})', current)
                    if curr_match:
                        return previous + current[curr_match.end():]
            
            if '"diff": "' in previous and not previous.endswith('"'):
                diff_markers = ['---', '+++', '@@', '+', '-']
                for marker in diff_markers:
                    if current.startswith(marker):
                        return previous + current
            
            return None
        except Exception as e:
            self.logger.error("Solution designer join failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return None

    def _join_json_content(self, previous: str, current: str) -> Optional[str]:
        """
        Join JSON content with structure awareness.
        
        Args:
            previous: Previous content
            current: Current continuation
            
        Returns:
            Joined content if successful, None otherwise
        """
        try:
            open_braces = previous.count('{') - previous.count('}')
            open_brackets = previous.count('[') - previous.count(']')
            open_quotes = previous.count('"') % 2
            
            if open_braces > 0 or open_brackets > 0:
                prop_pattern = r'"[^"]+"\s*:\s*'
                if re.search(prop_pattern + r'$', previous):
                    return previous + current
                if previous.rstrip().endswith(','):
                    return previous + current
                if previous.rstrip().endswith('{') or previous.rstrip().endswith('['):
                    return previous + current
            
            return None
        except Exception as e:
            self.logger.error("JSON join failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return None

    def _structure_aware_join(self, previous: str, current: str, content_type: str) -> str:
        """
        Join content with structure awareness as fallback.
        
        Args:
            previous: Previous content
            current: Current continuation
            content_type: Type of content
            
        Returns:
            Joined content
        """
        try:
            previous = previous.rstrip()
            current = current.lstrip()
            
            if content_type in (ContentType.JSON, ContentType.JSON_CODE, ContentType.SOLUTION_DESIGNER):
                if previous.endswith('}') and (current.startswith('{') or current.startswith('"')):
                    return previous + ',\n' + current
                elif previous.endswith('"') and current.startswith('"'):
                    return previous + ',\n' + current
                elif previous.endswith('}') and not current.startswith('}') and not current.startswith(']'):
                    return previous + ',\n' + current
            elif content_type == ContentType.CODE:
                if not previous.endswith('\n') and not current.startswith('\n'):
                    return previous + '\n' + current
            
            return previous + '\n' + current
        except Exception as e:
            self.logger.error("Structure-aware join failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return previous + '\n' + current

    def _clean_json_content(self, content: str) -> str:
        """
        Clean up JSON content by removing artifacts and fixing structure.
        
        Args:
            content: Content to clean
            
        Returns:
            Cleaned content
        """
        try:
            if '{' in content and '}' in content:
                open_braces = content.count('{')
                close_braces = content.count('}')
                if open_braces > close_braces:
                    missing = open_braces - close_braces
                    content += '\n' + '}' * missing
                    self.logger.debug("Added missing closing braces", extra={"count": missing})
                
                open_brackets = content.count('[')
                close_brackets = content.count(']')
                if open_brackets > close_brackets:
                    missing = open_brackets - close_brackets
                    content += '\n' + ']' * missing
                    self.logger.debug("Added missing closing brackets", extra={"count": missing})
                    
            return content
        except Exception as e:
            self.logger.error("JSON cleaning failed",
                           extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return content

    def _build_completion_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Build parameters for LLM completion request.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dictionary of completion parameters
        """
        try:
            completion_params = {
                "model": self.model_str,
                "messages": messages,
            }
            if self.provider.value != "openai":
                completion_params["temperature"] = self.temperature
            
            provider_config = self.parent._get_provider_config(self.provider)
            model_params = provider_config.get("model_params", {})
            if model_params:
                completion_params.update(model_params)
            
            if "api_base" in provider_config:
                completion_params["api_base"] = provider_config["api_base"]
                
            self.logger.debug("Completion parameters built", extra={"params": completion_params})
            return completion_params
        except Exception as e:
            self.logger.error("Completion params build failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            raise

    def _make_llm_request(self, completion_params: Dict[str, Any]) -> Any:
        """
        Make LLM request with rate limit handling.
        
        Args:
            completion_params: Completion request parameters
            
        Returns:
            LLM response
        """
        try:
            litellm.retry = True
            litellm.max_retries = 3
            litellm.retry_wait = 2
            litellm.max_retry_wait = 60
            litellm.retry_exponential = True
            
            safe_params = {
                k: v for k, v in completion_params.items()
                if k in ['model', 'messages', 'temperature', 'max_tokens', 'top_p', 'stream']
            }
            
            provider_config = self.parent._get_provider_config(self.provider)
            if "api_base" in provider_config:
                safe_params["api_base"] = provider_config["api_base"]
                
            self.logger.debug("Making LLM request", extra={"params": safe_params})
            response = completion(**safe_params)
            return response
        except litellm.RateLimitError as e:
            self.logger.warning("Rate limit error in LLM request", extra={"error": str(e)})
            raise
        except Exception as e:
            self.logger.error("LLM request failed",
                           extra={"error": str(e), "stack_trace": traceback.format_exc()})
            raise

    def _get_content_from_response(self, response: Any) -> str:
        """
        Extract content from LLM response.
        
        Args:
            response: LLM response object
            
        Returns:
            Content string
        """
        try:
            if hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    content = response.choices[0].message.content
                    self.logger.debug("Content extracted from response",
                                   extra={"content_preview": content[:100]})
                    return content
            self.logger.warning("No content found in response")
            return ""
        except Exception as e:
            self.logger.error("Content extraction failed",
                            extra={"error": str(e), "stack_trace": traceback.format_exc()})
            return ""