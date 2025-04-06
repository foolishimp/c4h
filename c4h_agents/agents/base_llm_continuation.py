"""
Enhanced LLM response continuation handling using line number and indentation tracking with JSON formatting.
Path: c4h_agents/agents/base_llm_continuation.py
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import re
import json
import random
from datetime import datetime
import litellm
from litellm import completion
from c4h_agents.agents.types import LLMProvider, LogDetail
from c4h_agents.utils.logging import get_logger

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
        consecutive_failures = 0
        max_consecutive_failures = 2  # Allow up to 2 consecutive parsing failures before falling back
        
        # Detect content type
        content_type = self._detect_content_type(messages)
        
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
            
            # Format initial content with line numbers and indentation
            numbered_lines = self._format_with_line_numbers_and_indentation(content)
            accumulated_lines = numbered_lines
            
            # Continue making requests until we're done or hit max attempts
            next_line = len(accumulated_lines) + 1
            
            while next_line > 1 and attempt < max_tries:
                attempt += 1
                self.metrics["attempts"] += 1
                
                # Create JSON array from accumulated lines for context
                context_json = self._create_line_json(accumulated_lines, max_context_lines=30)
                
                # Create continuation prompt with appropriate example for the content type
                continuation_prompt = self._create_numbered_continuation_prompt(
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
                    
                    # Parse line-numbered content from JSON including indentation
                    new_lines = self._parse_json_content(cont_content, next_line)
                    
                    if not new_lines:
                        self.logger.warning("llm.no_parsable_content", attempt=attempt, content_preview=cont_content[:100])
                        # Try a repair attempt with more aggressive parsing
                        new_lines = self._attempt_repair_parse(cont_content, next_line)
                        if not new_lines:
                            consecutive_failures += 1
                            self.logger.warning("llm.repair_parse_failed", consecutive_failures=consecutive_failures)
                            
                            # If we've tried parsing multiple times without success, append raw content
                            if consecutive_failures >= max_consecutive_failures:
                                # Add a marker line to indicate parsing failure
                                marker = f"----- CONTINUATION PARSING FAILED (ATTEMPT {attempt}) -----"
                                raw_fallback = [(next_line, 0, marker)]
                                
                                # Process raw content line by line
                                raw_lines = cont_content.splitlines()
                                if raw_lines:
                                    for i, line in enumerate(raw_lines):
                                        indent = len(line) - len(line.lstrip())
                                        raw_fallback.append((next_line + i + 1, indent, line))
                                else:
                                    # No line breaks, just add the whole content
                                    raw_fallback.append((next_line + 1, 0, cont_content))
                                    
                                self.logger.warning("llm.using_raw_content_fallback", raw_lines_count=len(raw_lines) if raw_lines else 1)
                                accumulated_lines.extend(raw_fallback)
                                next_line = len(accumulated_lines) + 1
                                consecutive_failures = 0  # Reset after applying fallback
                            else:
                                # Try again with a simplified prompt next time
                                continue
                        else:
                            consecutive_failures = 0  # Reset on successful repair parse
                    else:
                        consecutive_failures = 0  # Reset on successful parse
                    
                    # Update accumulated lines
                    accumulated_lines.extend(new_lines)
                    
                    finish_reason = getattr(response.choices[0], 'finish_reason', None)
                    
                    # Update final response
                    final_response = response
                    
                    if finish_reason != 'length':
                        self.logger.info("llm.continuation_complete", 
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
                    # Add marker for error and continue with what we have
                    marker = f"----- CONTINUATION ERROR: {str(e)[:100]} -----"
                    accumulated_lines.append((next_line, 0, marker))
                    break
            
            # Convert accumulated lines back to raw content
            final_content = self._numbered_lines_to_content(accumulated_lines)
            
            # Update response content
            if final_response and hasattr(final_response, 'choices') and final_response.choices:
                final_response.choices[0].message.content = final_content
                
            self.metrics["total_lines"] = len(accumulated_lines)
            
            return final_content, final_response
            
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            self.logger.error("llm.continuation_failed", 
                        error=error_msg, 
                        error_type=error_type)
            
            raise
        
    def _format_with_line_numbers_and_indentation(self, content):
        """Format content with line numbers and indentation level tracking"""
        lines = content.splitlines()
        result = []
        
        for i, line in enumerate(lines):
            # Calculate leading whitespace (indentation)
            indent = len(line) - len(line.lstrip())
            result.append((i+1, indent, line))
        
        return result
        
    def _create_line_json(self, numbered_lines, max_context_lines=30):
        """Create JSON array with line numbers and indentation"""
        # Take last N lines for context
        context_lines = numbered_lines[-min(max_context_lines, len(numbered_lines)):]
        
        lines_data = []
        for line_num, indent, content in context_lines:
            lines_data.append({
                "line": line_num,
                "indent": indent,
                "content": content
            })
            
        return json.dumps({"lines": lines_data}, indent=2)
        
    def _create_numbered_continuation_prompt(self, context_json, next_line, content_type):
        """Create continuation prompt with numbered line and indentation instructions using JSON format"""
        # Get appropriate example based on content type
        if content_type == "code":
            example = [
                {"line": next_line, "indent": 4, "content": "def example_function():"},
                {"line": next_line+1, "indent": 8, "content": "    return \"Hello World\""},
                {"line": next_line+2, "indent": 0, "content": ""},
                {"line": next_line+3, "indent": 0, "content": "# This is a comment"}
            ]
        elif content_type == "json" or content_type == "json_code":
            example = [
                {"line": next_line, "indent": 4, "content": "\"key\": \"value\","},
                {"line": next_line+1, "indent": 4, "content": "\"nested\": {"},
                {"line": next_line+2, "indent": 8, "content": "    \"array\": ["},
                {"line": next_line+3, "indent": 12, "content": "        \"item1\","}
            ]
        elif content_type == "solution_designer":
            example = [
                {"line": next_line, "indent": 0, "content": "    {"},
                {"line": next_line+1, "indent": 2, "content": "      \"file_path\": \"path/to/file.py\","},
                {"line": next_line+2, "indent": 2, "content": "      \"type\": \"modify\","},
                {"line": next_line+3, "indent": 2, "content": "      \"description\": \"Updated function\","}
            ]
        else:
            example = [
                {"line": next_line, "indent": 0, "content": "Your continued content here"},
                {"line": next_line+1, "indent": 0, "content": "Next line of content"}
            ]

        example_json = json.dumps({"lines": example}, indent=2)

        prompt = f"""
Continue the {content_type} content from line {next_line}.

CRITICAL REQUIREMENTS:
1. Start with line {next_line} exactly
2. Use the exact same JSON format with line numbers and indentation
3. Preserve proper indentation for code/structured content
4. Do not modify or repeat any previous lines
5. Maintain exact indentation levels matching the content type
6. Do not escape newlines in content (write actual newlines, not \\n)
7. Keep all string literals intact
8. Return an array of JSON objects with line, indent, and content fields
9. For solution designer content, ensure proper formatting of diffs and JSON structure
10. **DO NOT** add any explanatory text or comments after the JSON content as it will break the parsing

Example format:
{example_json}

Previous content (for context) has been provided in the previous message.

Your continuation starting from line {next_line}:
```json
{{
  "lines": [
    // Your continuation lines here, starting with line {next_line}
  ]
}}
```
"""
        return prompt
        

    def _parse_json_content(self, content, expected_start_line):
        """Parse content with line numbers and indentation from JSON format.
        With enhanced tolerance for formatting variations and line number flexibility."""
        numbered_lines = []
        try:
            # First, try to extract JSON enclosed in a markdown code block.
            json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', content)
            if json_match:
                json_content = json_match.group(1)
            else:
                # Try to locate a JSON object directly.
                json_match = re.search(r'(\{\s*"lines"\s*:\s*\[[\s\S]+?\]\s*\})', content)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    # If nothing is found, use the raw content.
                    json_content = content
            
            # Attempt to parse the JSON.
            try:
                data = json.loads(json_content)
            except json.JSONDecodeError:
                # Try to find a JSON array and wrap it.
                array_match = re.search(r'\[\s*\{\s*"line"[\s\S]+?\}\s*\]', content)
                if array_match:
                    array_json = '{"lines": ' + array_match.group(0) + '}'
                    try:
                        data = json.loads(array_json)
                    except json.JSONDecodeError:
                        # As a last resort, try to extract line objects.
                        line_objects = self._extract_line_objects(content)
                        if line_objects:
                            data = {"lines": line_objects}
                        else:
                            data = {}
                else:
                    data = {}
            
            # Retrieve the "lines" array.
            lines = data.get("lines", [])
            if not lines and isinstance(data, list):
                lines = data
            
            # Process each line object.
            min_line_num = float('inf')
            for line_data in lines:
                try:
                    line_num = line_data.get("line")
                    indent = line_data.get("indent", 0)
                    line_text = line_data.get("content", "")
                    
                    # Keep track of lowest line number
                    if line_num and line_num < min_line_num:
                        min_line_num = line_num
                        
                    numbered_lines.append((line_num, indent, line_text))
                except (TypeError, AttributeError):
                    continue
            
            # If we found lines but they don't start at expected_start_line,
            # adjust them to maintain continuity
            if numbered_lines and min_line_num != float('inf') and min_line_num != expected_start_line:
                self.logger.warning("llm.line_number_mismatch", 
                                expected=expected_start_line, 
                                actual=min_line_num)
                
                # Adjust line numbers to match expected sequence
                offset = expected_start_line - min_line_num
                adjusted_lines = []
                for line_num, indent, text in numbered_lines:
                    adjusted_line_num = line_num + offset
                    adjusted_lines.append((adjusted_line_num, indent, text))
                numbered_lines = adjusted_lines
            
            # Sort and deduplicate based on line numbers.
            deduped_lines = []
            seen_line_nums = set()
            for ln in sorted(numbered_lines, key=lambda x: x[0] if x[0] is not None else 0):
                if ln[0] not in seen_line_nums:
                    deduped_lines.append(ln)
                    seen_line_nums.add(ln[0])
            
            # If no structured lines were parsed, fall back to appending the raw content.
            if not deduped_lines:
                self.logger.warning("llm.no_parsed_lines", fallback="appending raw content")
                # Process raw content line by line instead of as one block
                raw_lines = content.splitlines()
                if raw_lines:
                    for i, line in enumerate(raw_lines):
                        indent = len(line) - len(line.lstrip())
                        deduped_lines.append((expected_start_line + i, indent, line))
                else:
                    deduped_lines = [(expected_start_line, 0, content)]
            
            return deduped_lines
        except Exception as e:
            self.logger.error("llm.json_parse_error", error=str(e))
            # More robust fallback - process the raw content line by line
            raw_lines = content.splitlines()
            result = []
            if raw_lines:
                for i, line in enumerate(raw_lines):
                    indent = len(line) - len(line.lstrip())
                    result.append((expected_start_line + i, indent, line))
            else:
                result = [(expected_start_line, 0, content)]
            return result


    def _extract_line_objects(self, content):
        """Extract individual line objects from content using regex with enhanced pattern matching"""
        line_objects = []
        
        # Enhanced pattern matching for flexibility
        # Match both quoted and unquoted values for line and indent
        # Allow for variations in spacing and formatting
        pattern = r'\{\s*"line"\s*:\s*(\d+)[^}]*"indent"\s*:\s*(\d+)[^}]*"content"\s*:\s*"([^"]*)"'
        alt_pattern = r'\{\s*line\s*:\s*(\d+)[^}]*indent\s*:\s*(\d+)[^}]*content\s*:\s*"([^"]*)"'
        
        # Try both patterns
        for p in [pattern, alt_pattern]:
            matches = re.finditer(p, content)
            for match in matches:
                try:
                    line_num = int(match.group(1))
                    indent = int(match.group(2))
                    content = match.group(3)
                    
                    # Unescape any escaped quotes or slashes
                    content = content.replace('\\"', '"').replace('\\\\', '\\')
                    
                    line_objects.append({
                        "line": line_num,
                        "indent": indent,
                        "content": content
                    })
                except (ValueError, IndexError):
                    continue
        
        # Look for alternative formats like line=number instead of "line": number
        alt_format = r'line\s*=\s*(\d+)[^,]*indent\s*=\s*(\d+)[^,]*content\s*=\s*"([^"]*)"'
        matches = re.finditer(alt_format, content)
        for match in matches:
            try:
                line_num = int(match.group(1))
                indent = int(match.group(2))
                content = match.group(3)
                line_objects.append({
                    "line": line_num,
                    "indent": indent,
                    "content": content
                })
            except (ValueError, IndexError):
                continue
                
        return line_objects
    
    def _attempt_repair_parse(self, content, expected_start_line):
        """Attempt a more flexible repair parse with enhanced pattern recognition"""
        numbered_lines = []
        
        # First try to find JSON-like line objects with flexible patterns
        line_patterns = [
            # Standard JSON format with quotes
            r'"line"\s*:\s*(\d+)[^,}]*"indent"\s*:\s*(\d+)[^,}]*"content"\s*:\s*"([^"]*)"',
            # Alternative format without quotes for keys
            r'line\s*:\s*(\d+)[^,}]*indent\s*:\s*(\d+)[^,}]*content\s*:\s*"([^"]*)"',
            # Format with equal signs
            r'line\s*=\s*(\d+)[^,}]*indent\s*=\s*(\d+)[^,}]*content\s*=\s*"([^"]*)"',
            # Format with extra text around it
            r'line\s*(?:number|#)?\s*(?::|=)\s*(\d+)[^,}]*indent\s*(?::|=)\s*(\d+)[^,}]*content\s*(?::|=)\s*"([^"]*)"',
        ]
        
        for pattern in line_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                try:
                    line_num = int(match.group(1))
                    indent = int(match.group(2))
                    line_text = match.group(3)
                    numbered_lines.append((line_num, indent, line_text))
                except (ValueError, IndexError):
                    continue
        
        # Try looser patterns like numbered lists or prefixed lines
        if not numbered_lines:
            # Look for line numbers at the beginning of lines (e.g., "Line 5: content")
            looser_patterns = [
                r'(?:line|Line)?\s*(\d+)[^\n]*:\s*([^\n]*)',  # Line 5: content
                r'(\d+)[.:\)]\s*([^\n]*)',                    # 5. content or 5: content
                r'L(\d+):\s*([^\n]*)',                        # L5: content
            ]
            
            for pattern in looser_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    try:
                        line_num = int(match.group(1))
                        line_text = match.group(2).strip()
                        indent = len(line_text) - len(line_text.lstrip())
                        numbered_lines.append((line_num, indent, line_text))
                    except (ValueError, IndexError):
                        continue
        
        # If we found lines, sort and deduplicate them
        if numbered_lines:
            numbered_lines.sort(key=lambda x: x[0])
            self.logger.info("llm.repair_parse_successful", 
                        lines_found=len(numbered_lines),
                        pattern_match="custom_patterns")
            
            # Check if we need to adjust line numbers for continuity
            if numbered_lines and numbered_lines[0][0] != expected_start_line:
                offset = expected_start_line - numbered_lines[0][0]
                adjusted_lines = [(ln[0] + offset, ln[1], ln[2]) for ln in numbered_lines]
                numbered_lines = adjusted_lines
                self.logger.info("llm.adjusted_line_numbers", offset=offset)
            
            # Deduplicate
            deduped_lines = []
            seen_line_nums = set()
            for ln in numbered_lines:
                if ln[0] not in seen_line_nums:
                    deduped_lines.append(ln)
                    seen_line_nums.add(ln[0])
            return deduped_lines
        
        # If everything else failed, parse the content line by line
        content_lines = content.splitlines()
        if content_lines:
            self.logger.warning("llm.using_raw_lines_fallback", 
                            lines_count=len(content_lines))
            
            # Add marker to indicate we're using raw content
            result = [(expected_start_line, 0, "----- CONTINUATION PARSING FALLBACK -----")]
            
            # Add each raw line with appropriate line numbers
            for i, line in enumerate(content_lines):
                indent = len(line) - len(line.lstrip())
                result.append((expected_start_line + i + 1, indent, line))
            return result
            
        # Last resort - return raw content as a single line 
        self.logger.warning("llm.fallback_to_raw_content")
        return [(expected_start_line, 0, "----- RAW CONTINUATION CONTENT -----"), 
                (expected_start_line + 1, 0, content)]    

    def _numbered_lines_to_content(self, numbered_lines):
        """Convert numbered lines back to raw content with proper indentation"""
        # Sort by line number to ensure correct order
        sorted_lines = sorted(numbered_lines, key=lambda x: x[0])
        
        # Extract content with preserved indentation
        content_lines = [line[2] for line in sorted_lines]
        
        return "\n".join(content_lines)

    def _detect_content_type(self, messages: List[Dict[str, str]]) -> str:
        """Detect content type from messages for specialized handling"""
        content = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                break
                
        # Check for specific content types
        is_code = any("```" in msg.get("content", "") or "def " in msg.get("content", "") 
                    for msg in messages if msg.get("role") == "user")
        is_json = any("json" in msg.get("content", "").lower() or 
                    msg.get("content", "").strip().startswith("{") or 
                    msg.get("content", "").strip().startswith("[") 
                    for msg in messages if msg.get("role") == "user")
        is_diff = any("--- " in msg.get("content", "") and "+++ " in msg.get("content", "")
                    for msg in messages if msg.get("role") == "user")
                    
        # Check for solution_designer specific format
        is_solution_designer = any('"changes":' in msg.get("content", "") and 
                               '"file_path":' in msg.get("content", "") and 
                               '"diff":' in msg.get("content", "")
                               for msg in messages)
        
        content_type = "text"  # default
        if is_solution_designer:
            content_type = "solution_designer"
        elif is_code and is_json:
            content_type = "json_code"
        elif is_code:
            content_type = "code"
        elif is_json:
            content_type = "json"
        elif is_diff:
            content_type = "diff"
            
        logger.debug("llm.content_type_detected", 
                   content_type=content_type,
                   is_code=is_code, 
                   is_json=is_json, 
                   is_diff=is_diff,
                   is_solution_designer=is_solution_designer)
            
        return content_type

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