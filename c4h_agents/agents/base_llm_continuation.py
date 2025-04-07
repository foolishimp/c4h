"""
Enhanced LLM response continuation handling with robust content parsing strategies.
Path: c4h_agents/agents/base_llm_continuation.py
"""

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

logger = get_logger()

class ContentType(str, Enum):
    """Content types for specialized handling"""
    TEXT = "text"
    CODE = "code"
    JSON = "json" 
    JSON_CODE = "json_code"
    DIFF = "diff"
    SOLUTION_DESIGNER = "solution_designer"
    
class ParseResult:
    """Result of parsing a continuation response"""
    def __init__(self, success: bool, content: List[Tuple[int, int, str]] = None, error: str = None):
        self.success = success
        self.content = content or []
        self.error = error
        
    def has_content(self) -> bool:
        """Check if parsing produced any content"""
        return bool(self.content)
        
    def get_next_line(self) -> int:
        """Get the next line number after this content"""
        if not self.content:
            return 1
        return max(line[0] for line in self.content) + 1
        
class ContinuationHandler:
    """
    Robust LLM response continuation handler with content-type specialized strategies.
    Implements multiple parsing approaches and fallback mechanisms.
    """

    def __init__(self, parent_agent):
        """Initialize handler with parent agent reference"""
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
        
        # Tracking metrics
        self.metrics = {
            "attempts": 0,
            "total_lines": 0,
            "exact_matches": 0,
            "hash_matches": 0,
            "token_matches": 0,
            "fallbacks": 0,
            "parsing_errors": 0,
            "rate_limit_retries": 0
        }

    def get_completion_with_continuation(
            self, 
            messages: List[Dict[str, str]],
            max_attempts: Optional[int] = None
        ) -> Tuple[str, Any]:
        """
        Get completion with multi-strategy continuation handling.
        
        Args:
            messages: List of message dictionaries with role and content
            max_attempts: Maximum number of continuation attempts
            
        Returns:
            Tuple of (accumulated_content, final_response)
        """
        # Setup
        attempt = 0
        max_tries = max_attempts or self.max_continuation_attempts
        accumulated_lines = []
        final_response = None
        consecutive_failures = 0
        max_consecutive_failures = 2
        
        # Detect content type for specialized handling
        content_type = self._detect_content_type(messages)
        
        self.logger.info("llm.continuation_starting", 
                       model=self.model_str, 
                       content_type=content_type)
        
        # Rate limit handling
        rate_limit_retries = 0
        rate_limit_backoff = self.rate_limit_retry_base_delay
        
        # Initial request with standard parameters
        completion_params = self._build_completion_params(messages)
        
        try:
            # Make initial request
            response = self._make_llm_request(completion_params)
            
            # Process initial response
            content = self._get_content_from_response(response)
            self.logger.debug("llm.initial_content", 
                           content_preview=content[:100], 
                           content_type=type(content).__name__)
            
            final_response = response
            
            # Format initial content with line numbers and indentation
            numbered_lines = self._format_with_line_numbers_and_indentation(content)
            accumulated_lines = numbered_lines
            
            self.logger.debug("llm.initial_numbered_lines", 
                           line_count=len(accumulated_lines))
            
            # Continue making requests until we're done or hit max attempts
            next_line = len(accumulated_lines) + 1
            
            while next_line > 1 and attempt < max_tries:
                attempt += 1
                self.metrics["attempts"] += 1
                
                # Select appropriate continuation strategy based on content type
                if content_type == ContentType.SOLUTION_DESIGNER:
                    continuation_prompt = self._create_solution_designer_continuation(
                        accumulated_lines, next_line)
                elif content_type in (ContentType.JSON, ContentType.JSON_CODE):
                    continuation_prompt = self._create_json_continuation(
                        accumulated_lines, next_line)
                else:
                    # Default line-number based continuation
                    continuation_prompt = self._create_numbered_continuation_prompt(
                        self._create_line_json(accumulated_lines), next_line, content_type)
                
                # Prepare continuation message
                cont_messages = messages.copy()
                cont_messages.append({"role": "assistant", "content": 
                                    self._numbered_lines_to_content(accumulated_lines)})
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
                    
                    # Try multiple parsing strategies
                    parse_result = self._parse_continuation(
                        cont_content, next_line, content_type)
                    
                    if parse_result.success and parse_result.has_content():
                        # Successful parsing
                        new_lines = parse_result.content
                        consecutive_failures = 0
                    else:
                        # Try repair parsing as fallback
                        self.logger.warning("llm.parsing_failed", 
                                         attempt=attempt, 
                                         error=parse_result.error)
                        
                        # Try specialized content type repair
                        if content_type == ContentType.SOLUTION_DESIGNER:
                            new_lines = self._repair_solution_designer_parse(
                                cont_content, next_line)
                        else:
                            new_lines = self._attempt_repair_parse(
                                cont_content, next_line)
                        
                        if not new_lines:
                            consecutive_failures += 1
                            self.logger.warning("llm.repair_parse_failed", 
                                             consecutive_failures=consecutive_failures)
                            
                            # Multiple failures, use raw content fallback
                            if consecutive_failures >= max_consecutive_failures:
                                self.metrics["fallbacks"] += 1
                                # Add marker and fall back to raw content
                                new_lines = self._create_raw_fallback(
                                    cont_content, next_line, attempt)
                                consecutive_failures = 0  # Reset after fallback
                            else:
                                # Try again with a different prompt
                                continue
                        else:
                            consecutive_failures = 0  # Reset on successful repair
                    
                    # Successful continuation, update accumulated lines
                    accumulated_lines.extend(new_lines)
                    
                    # Check if response is complete
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
                    self.metrics["rate_limit_retries"] += 1
                    
                    if rate_limit_retries > self.rate_limit_max_retries:
                        self.logger.error("llm.rate_limit_max_retries_exceeded", 
                                       retry_count=rate_limit_retries,
                                       error=error_msg[:200])
                        raise
                                
                    # Calculate backoff with jitter
                    jitter = 0.1 * rate_limit_backoff * (0.5 - random.random())
                    current_backoff = min(rate_limit_backoff + jitter, 
                                       self.rate_limit_max_backoff)
                    
                    self.logger.warning("llm.rate_limit_backoff", 
                                     attempt=attempt,
                                     retry_count=rate_limit_retries,
                                     backoff_seconds=current_backoff,
                                     error=error_msg[:200])
                    
                    # Apply exponential backoff with base 2
                    time.sleep(current_backoff)
                    rate_limit_backoff = min(rate_limit_backoff * 2, 
                                          self.rate_limit_max_backoff)
                    continue
                
                except Exception as e:
                    self.logger.error("llm.continuation_failed", error=str(e))
                    # Add marker for error but continue with what we have
                    marker = f"----- CONTINUATION ERROR: {str(e)[:100]} -----"
                    accumulated_lines.append((next_line, 0, marker))
                    self.metrics["parsing_errors"] += 1
                    break
            
            # Convert accumulated lines back to raw content
            final_content = self._numbered_lines_to_content(accumulated_lines)
            
            # Clean up content if needed for specific formats
            if content_type in (ContentType.JSON, ContentType.JSON_CODE, ContentType.SOLUTION_DESIGNER):
                final_content = self._clean_json_content(final_content, content_type)
            
            # Update metrics
            self.metrics["total_lines"] = len(accumulated_lines)
            
            # Update response content
            if final_response and hasattr(final_response, 'choices') and final_response.choices:
                final_response.choices[0].message.content = final_content
            
            self.logger.info("llm.continuation_complete", 
                          attempts=attempt,
                          content_type=content_type,
                          metrics=self.metrics)
                
            return final_content, final_response
            
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            self.logger.error("llm.continuation_failed", 
                           error=error_msg, 
                           error_type=error_type)
            
            raise

    def _parse_continuation(self, content: str, expected_start_line: int, 
                           content_type: str) -> ParseResult:
        """
        Parse continuation content using multiple strategies.
        
        Args:
            content: Response content to parse
            expected_start_line: Expected starting line number
            content_type: Type of content for specialized handling
            
        Returns:
            ParseResult with success status and parsed content
        """
        try:
            # 1. Try to parse as JSON format first
            json_lines = self._parse_json_content(content, expected_start_line)
            if json_lines:
                return ParseResult(True, json_lines)
                
            # 2. Try content-type specific parsing
            if content_type == ContentType.SOLUTION_DESIGNER:
                sd_lines = self._parse_solution_designer_content(content, expected_start_line)
                if sd_lines:
                    return ParseResult(True, sd_lines)
            
            # 3. Try pattern-based extraction
            pattern_lines = self._extract_line_patterns(content, expected_start_line)
            if pattern_lines:
                return ParseResult(True, pattern_lines)
                
            # 4. All strategies failed
            return ParseResult(False, error="No parsing strategy succeeded")
            
        except Exception as e:
            self.logger.error("llm.parse_continuation_failed", error=str(e))
            return ParseResult(False, error=f"Parsing error: {str(e)}")

    def _parse_solution_designer_content(self, content: str, expected_start_line: int) -> List[Tuple[int, int, str]]:
        """
        Special parsing for solution_designer content with diff awareness.
        
        Args:
            content: Content to parse
            expected_start_line: Expected starting line number
            
        Returns:
            List of (line_number, indent, content) tuples
        """
        try:
            # Try to find diff sections and format them properly
            if "--- " in content and "+++ " in content:
                # Extract raw diff blocks
                diff_pattern = r'(-{3}.*?)\n(\+{3}.*?)(?=\n-{3}|\Z)'
                diff_matches = re.finditer(diff_pattern, content, re.DOTALL)
                
                numbered_lines = []
                current_line = expected_start_line
                
                for match in diff_matches:
                    diff_block = match.group(0)
                    lines = diff_block.splitlines()
                    
                    for line in lines:
                        # Preserve indentation
                        indent = len(line) - len(line.lstrip())
                        numbered_lines.append((current_line, indent, line))
                        current_line += 1
                
                if numbered_lines:
                    return numbered_lines
                    
            # If no diff blocks found, try other strategies
            # Look for file_path fields which typically indicate solution_designer objects
            file_path_pattern = r'"file_path"\s*:\s*"([^"]+)"'
            if re.search(file_path_pattern, content):
                # Try to extract as regular text with minimal parsing
                lines = content.splitlines()
                numbered_lines = []
                
                for i, line in enumerate(lines):
                    indent = len(line) - len(line.lstrip())
                    numbered_lines.append((expected_start_line + i, indent, line))
                
                return numbered_lines
                
            return None
            
        except Exception as e:
            self.logger.error("llm.solution_designer_parse_error", error=str(e))
            return None

    def _extract_line_patterns(self, content: str, expected_start_line: int) -> List[Tuple[int, int, str]]:
        """
        Extract line patterns with regular expressions.
        
        Args:
            content: Content to parse
            expected_start_line: Expected starting line number
            
        Returns:
            List of (line_number, indent, content) tuples
        """
        numbered_lines = []
        
        # Look for patterns like "Line X: content" or "X: content"
        patterns = [
            r'(?:line|Line)?\s*(\d+)\s*:\s*(.*?)$',  # Line 5: content
            r'(\d+)[.:\)]\s*(.*?)$',                 # 5. content or 5: content
            r'L(\d+):\s*(.*?)$'                      # L5: content
        ]
        
        for pattern in patterns:
            line_matches = []
            for match in re.finditer(pattern, content, re.MULTILINE):
                try:
                    line_num = int(match.group(1))
                    line_text = match.group(2).strip()
                    indent = len(line_text) - len(line_text.lstrip())
                    line_matches.append((line_num, indent, line_text))
                except (ValueError, IndexError):
                    continue
            
            if line_matches:
                # Check if we need to adjust line numbers
                line_matches.sort(key=lambda x: x[0])
                if line_matches[0][0] != expected_start_line:
                    offset = expected_start_line - line_matches[0][0]
                    line_matches = [(ln + offset, indent, text) 
                                  for ln, indent, text in line_matches]
                
                return line_matches
        
        return numbered_lines

    def _create_raw_fallback(self, content: str, expected_start_line: int, 
                             attempt: int) -> List[Tuple[int, int, str]]:
        """
        Create fallback content when all parsing fails.
        
        Args:
            content: Raw content to use as fallback
            expected_start_line: Expected starting line number
            attempt: Current continuation attempt number
            
        Returns:
            List of (line_number, indent, content) tuples
        """
        # Add marker line to indicate parsing failure
        marker = f"----- CONTINUATION PARSING FAILED (ATTEMPT {attempt}) -----"
        raw_fallback = [(expected_start_line, 0, marker)]
        
        # Process raw content line by line
        raw_lines = content.splitlines()
        if raw_lines:
            for i, line in enumerate(raw_lines):
                indent = len(line) - len(line.lstrip())
                raw_fallback.append((expected_start_line + i + 1, indent, line))
        else:
            # No line breaks, just add the whole content
            raw_fallback.append((expected_start_line + 1, 0, content))
            
        self.logger.warning("llm.using_raw_content_fallback", 
                         raw_lines_count=len(raw_lines) if raw_lines else 1)
        
        return raw_fallback

    def _create_solution_designer_continuation(self, accumulated_lines: List[Tuple[int, int, str]], 
                                            next_line: int) -> str:
        """Create specialized continuation prompt for solution_designer content."""
        # Get context lines
        context_json = self._create_line_json(accumulated_lines, max_context_lines=30)
        
        # Create example with solution_designer format
        example = [
            {"line": next_line, "indent": 0, "content": "    {"},
            {"line": next_line+1, "indent": 2, "content": "      \"file_path\": \"path/to/file.py\","},
            {"line": next_line+2, "indent": 2, "content": "      \"type\": \"modify\","},
            {"line": next_line+3, "indent": 2, "content": "      \"description\": \"Updated function\","}
        ]
        example_json = json.dumps({"lines": example}, indent=2)
        
        # Special prompt for solution_designer - FIX HERE - DOUBLE THE CURLY BRACES
        prompt = f"""
    Continue the solution_designer content from line {next_line}.

    CRITICAL REQUIREMENTS FOR SOLUTION DESIGNER FORMAT:
    1. Start with line {next_line} exactly
    2. Use the exact JSON format with line numbers and indentation
    3. NEVER output raw diff content directly (+/- prefixed lines)
    4. ALL content must be in the JSON lines array format
    5. Each line of diff content must be inside a {{"line": X, "indent": Y, "content": "..."}} structure
    6. Preserve proper escaping of special characters in the content field
    7. Remember you are continuing inside a larger JSON structure 
    8. Maintain proper escaping of quotes (\") and newlines (\\n) in diff content

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
"""
        return prompt

    def _create_json_continuation(self, accumulated_lines: List[Tuple[int, int, str]], 
                                next_line: int) -> str:
        """
        Create specialized continuation prompt for JSON content.
        
        Args:
            accumulated_lines: Previously accumulated content lines
            next_line: Next line number to continue from
            
        Returns:
            Continuation prompt optimized for JSON format
        """
        # Get context lines
        context_json = self._create_line_json(accumulated_lines, max_context_lines=30)
        
        # Create example with JSON format
        example = [
            {"line": next_line, "indent": 4, "content": "\"key\": \"value\","},
            {"line": next_line+1, "indent": 4, "content": "\"nested\": {"},
            {"line": next_line+2, "indent": 8, "content": "    \"array\": ["},
            {"line": next_line+3, "indent": 12, "content": "        \"item1\","}
        ]
        example_json = json.dumps({"lines": example}, indent=2)
        
        # Special prompt for JSON
        prompt = f"""
Continue the JSON content from line {next_line}.

CRITICAL REQUIREMENTS FOR JSON FORMAT:
1. Start with line {next_line} exactly
2. Use the exact JSON format with line numbers and indentation
3. Preserve proper JSON structure and nesting
4. Each line should maintain appropriate indentation
5. Remember to properly escape nested quotes and special characters
6. Be aware of arrays, objects, and strings that need to be completed
7. ALL content must be in the lines array JSON format, never raw JSON objects

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

    def _format_with_line_numbers_and_indentation(self, content: str) -> List[Tuple[int, int, str]]:
        """
        Format content with line numbers and indentation tracking.
        
        Args:
            content: Raw content to format
            
        Returns:
            List of (line_number, indent, content) tuples
        """
        lines = content.splitlines()
        result = []
        
        for i, line in enumerate(lines):
            # Calculate leading whitespace (indentation)
            indent = len(line) - len(line.lstrip())
            result.append((i+1, indent, line))
        
        return result
        
    def _create_line_json(self, numbered_lines: List[Tuple[int, int, str]], 
                         max_context_lines: int = 30) -> str:
        """
        Create JSON array with line numbers and indentation.
        
        Args:
            numbered_lines: List of (line_number, indent, content) tuples
            max_context_lines: Maximum number of context lines to include
            
        Returns:
            JSON string representing the context lines
        """
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
        
    def _create_numbered_continuation_prompt(self, context_json: str, 
                                           next_line: int, 
                                           content_type: str) -> str:
        """
        Create continuation prompt with numbered line and indentation instructions.
        
        Args:
            context_json: JSON string with context lines
            next_line: Next line number
            content_type: Type of content being continued
            
        Returns:
            Continuation prompt
        """
        # Get appropriate example based on content type
        if content_type == ContentType.CODE:
            example = [
                {"line": next_line, "indent": 4, "content": "def example_function():"},
                {"line": next_line+1, "indent": 8, "content": "    return \"Hello World\""},
                {"line": next_line+2, "indent": 0, "content": ""},
                {"line": next_line+3, "indent": 0, "content": "# This is a comment"}
            ]
        elif content_type in (ContentType.JSON, ContentType.JSON_CODE):
            example = [
                {"line": next_line, "indent": 4, "content": "\"key\": \"value\","},
                {"line": next_line+1, "indent": 4, "content": "\"nested\": {"},
                {"line": next_line+2, "indent": 8, "content": "    \"array\": ["},
                {"line": next_line+3, "indent": 12, "content": "        \"item1\","}
            ]
        elif content_type == ContentType.SOLUTION_DESIGNER:
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

    def _parse_json_content(self, content: str, expected_start_line: int) -> List[Tuple[int, int, str]]:
        """
        Parse content with line numbers and indentation from JSON format.
        
        Args:
            content: Content to parse
            expected_start_line: Expected starting line number
            
        Returns:
            List of (line_number, indent, content) tuples or None if parsing fails
        """
        numbered_lines = []
        try:
            # Try to extract content from different formats
            json_content = None
            
            # 1. Look for markdown code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', content)
            if json_match:
                json_content = json_match.group(1)
                
            # 2. Look for direct JSON object
            elif not json_content:
                json_match = re.search(r'(\{\s*"lines"\s*:\s*\[[\s\S]+?\]\s*\})', content)
                if json_match:
                    json_content = json_match.group(1)
                    
            # 3. Use raw content as fallback
            if not json_content:
                json_content = content
            
            # Try to parse the JSON content
            try:
                data = json.loads(json_content)
            except json.JSONDecodeError:
                # Look for JSON array and wrap it
                array_match = re.search(r'\[\s*\{\s*"line"[\s\S]+?\}\s*\]', content)
                if array_match:
                    try:
                        array_json = '{"lines": ' + array_match.group(0) + '}'
                        data = json.loads(array_json)
                    except json.JSONDecodeError:
                        # Extract line objects from text
                        line_objects = self._extract_line_objects(content)
                        if line_objects:
                            data = {"lines": line_objects}
                        else:
                            return None
                else:
                    return None
            
            # Extract lines from the data
            lines = data.get("lines", [])
            if not lines and isinstance(data, list):
                lines = data
            
            # Process line objects
            min_line_num = float('inf')
            for line_data in lines:
                try:
                    line_num = line_data.get("line")
                    if line_num is None:
                        continue
                        
                    indent = line_data.get("indent", 0)
                    line_text = line_data.get("content", "")
                    
                    # Track lowest line number
                    if line_num < min_line_num:
                        min_line_num = line_num
                        
                    numbered_lines.append((line_num, indent, line_text))
                except (TypeError, AttributeError):
                    continue
            
            # Check if we need to adjust line numbers
            if numbered_lines and min_line_num != float('inf') and min_line_num != expected_start_line:
                # Adjust line numbers
                offset = expected_start_line - min_line_num
                adjusted_lines = []
                for line_num, indent, text in numbered_lines:
                    adjusted_line_num = line_num + offset
                    adjusted_lines.append((adjusted_line_num, indent, text))
                numbered_lines = adjusted_lines
            
            # Sort and deduplicate
            if numbered_lines:
                # Sort by line number
                numbered_lines.sort(key=lambda x: x[0])
                
                # Deduplicate
                deduped_lines = []
                seen_line_nums = set()
                for ln in numbered_lines:
                    if ln[0] not in seen_line_nums:
                        deduped_lines.append(ln)
                        seen_line_nums.add(ln[0])
                
                return deduped_lines
            
            return None
            
        except Exception as e:
            self.logger.error("llm.json_parse_error", error=str(e))
            return None

    def _extract_line_objects(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract individual line objects using regex.
        
        Args:
            content: Content to extract from
            
        Returns:
            List of line objects or empty list if extraction fails
        """
        line_objects = []
        
        # Try multiple patterns
        patterns = [
            # Standard JSON format
            r'\{\s*"line"\s*:\s*(\d+)[^}]*"indent"\s*:\s*(\d+)[^}]*"content"\s*:\s*"([^"]*)"',
            # Alternative format without quotes for keys
            r'\{\s*line\s*:\s*(\d+)[^}]*indent\s*:\s*(\d+)[^}]*content\s*:\s*"([^"]*)"',
            # Format with equal signs
            r'line\s*=\s*(\d+)[^,}]*indent\s*=\s*(\d+)[^,}]*content\s*=\s*"([^"]*)"'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                try:
                    line_num = int(match.group(1))
                    indent = int(match.group(2))
                    content_text = match.group(3)
                    
                    # Unescape content
                    content_text = content_text.replace('\\"', '"').replace('\\\\', '\\')
                    
                    line_objects.append({
                        "line": line_num,
                        "indent": indent,
                        "content": content_text
                    })
                except (ValueError, IndexError):
                    continue
                    
        return line_objects

    def _repair_solution_designer_parse(self, content: str, expected_start_line: int) -> List[Tuple[int, int, str]]:
        """
        Special repair for solution_designer content.
        
        Args:
            content: Content to repair
            expected_start_line: Expected starting line number
            
        Returns:
            List of (line_number, indent, content) tuples or None if repair fails
        """
        try:
            # Check if this is solution_designer content by looking for specific patterns
            is_solution = re.search(r'"file_path"\s*:|"diff"\s*:|"type"\s*:\s*"(?:create|modify|delete)"', content)
            if not is_solution:
                return None
                
            # Try to handle the solution_designer format specially
            # 1. Look for diff content
            if '+++' in content and '---' in content:
                # This likely contains raw diff content
                lines = content.splitlines()
                numbered_lines = []
                
                for i, line in enumerate(lines):
                    # Calculate indentation
                    indent = len(line) - len(line.lstrip())
                    
                    # Special handling for diff lines
                    if line.startswith('+') or line.startswith('-') or line.startswith('@'):
                        # Increase indent for diff lines for better readability
                        indent += 2
                    
                    numbered_lines.append((expected_start_line + i, indent, line))
                
                return numbered_lines
            
            # 2. Try to extract any objects
            file_path_match = re.search(r'"file_path"\s*:\s*"([^"]+)"', content)
            if file_path_match:
                # This looks like a solution_designer object
                # Just format as regular text with basic structure
                lines = content.splitlines()
                numbered_lines = []
                
                for i, line in enumerate(lines):
                    indent = len(line) - len(line.lstrip())
                    numbered_lines.append((expected_start_line + i, indent, line))
                
                return numbered_lines
            
            # Couldn't repair with special handling
            return None
            
        except Exception as e:
            self.logger.error("llm.solution_designer_repair_failed", error=str(e))
            return None

    def _attempt_repair_parse(self, content: str, expected_start_line: int) -> List[Tuple[int, int, str]]:
        """
        Attempt flexible repair parsing for general content.
        
        Args:
            content: Content to repair parse
            expected_start_line: Expected starting line number
            
        Returns:
            List of (line_number, indent, content) tuples or None if repair fails
        """
        numbered_lines = []
        
        # Try flexible pattern matches
        patterns = [
            # Line number at start of line
            r'^\s*(\d+)[\s:.-]+(.*)$',
            # Common patterns like "Line X:"
            r'(?:line|Line)?\s*(\d+)[^\n]*:\s*([^\n]*)',
            # Numbered list items
            r'(\d+)[.:\)]\s*([^\n]*)',
            # Line prefixes
            r'L(\d+):\s*([^\n]*)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                try:
                    line_num = int(match.group(1))
                    line_text = match.group(2).strip()
                    indent = len(line_text) - len(line_text.lstrip())
                    numbered_lines.append((line_num, indent, line_text))
                except (ValueError, IndexError):
                    continue
        
        # If we found lines, sort and adjust line numbers if needed
        if numbered_lines:
            numbered_lines.sort(key=lambda x: x[0])
            
            # Check if we need to adjust line numbers
            if numbered_lines[0][0] != expected_start_line:
                offset = expected_start_line - numbered_lines[0][0]
                adjusted_lines = [(ln[0] + offset, ln[1], ln[2]) for ln in numbered_lines]
                numbered_lines = adjusted_lines
            
            # Deduplicate
            deduped_lines = []
            seen_line_nums = set()
            for ln in numbered_lines:
                if ln[0] not in seen_line_nums:
                    deduped_lines.append(ln)
                    seen_line_nums.add(ln[0])
            
            return deduped_lines
        
        # If all pattern matching failed, fall back to raw content processing
        raw_lines = content.splitlines()
        if raw_lines:
            # Process raw content line by line
            result = []
            for i, line in enumerate(raw_lines):
                indent = len(line) - len(line.lstrip())
                result.append((expected_start_line + i, indent, line))
            return result
        
        # No lines found at all
        return None

    def _numbered_lines_to_content(self, numbered_lines: List[Tuple[int, int, str]]) -> str:
        """
        Convert numbered lines back to raw content.
        
        Args:
            numbered_lines: List of (line_number, indent, content) tuples
            
        Returns:
            Raw content string
        """
        # Sort by line number to ensure correct order
        sorted_lines = sorted(numbered_lines, key=lambda x: x[0])
        
        # Extract content with preserved indentation
        content_lines = [line[2] for line in sorted_lines]
        
        return "\n".join(content_lines)

    def _clean_json_content(self, content: str, content_type: str) -> str:
        """
        Clean JSON content by removing artifacts and fixing structure.
        
        Args:
            content: Content to clean
            content_type: Type of content
            
        Returns:
            Cleaned content
        """
        # Remove any trailing {"lines": []} pattern
        lines_pattern = r'\s*\{\s*"lines"\s*:\s*\[\s*\]\s*\}\s*$'
        if re.search(lines_pattern, content):
            cleaned_content = re.sub(lines_pattern, '', content)
            self.logger.info("llm.removed_trailing_lines_array", 
                          original_length=len(content),
                          cleaned_length=len(cleaned_content))
            content = cleaned_content
        
        # For JSON content, ensure we have a complete object/array
        if content_type in (ContentType.JSON, ContentType.JSON_CODE, ContentType.SOLUTION_DESIGNER):
            if '{"lines":' in content:
                # Try to extract any JSON array if embedded in lines array
                if content.startswith('[') and ']' in content:
                    # Find end of array
                    array_end = content.find(']') + 1
                    if array_end > 0:
                        try:
                            # Validate JSON array
                            potential_json = content[:array_end]
                            json.loads(potential_json)
                            if len(potential_json) < len(content):
                                content = potential_json
                        except json.JSONDecodeError:
                            pass
                
                # Try to extract any JSON object
                elif content.startswith('{') and '}' in content:
                    # Find end of object
                    obj_end = content.find('}') + 1
                    if obj_end > 0:
                        try:
                            # Validate JSON object
                            potential_json = content[:obj_end]
                            json.loads(potential_json)
                            if len(potential_json) < len(content):
                                content = potential_json
                        except json.JSONDecodeError:
                            pass
        
        return content

    def _detect_content_type(self, messages: List[Dict[str, str]]) -> str:
        """
        Detect content type from messages for specialized handling.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Content type string
        """
        # Extract content from messages
        content = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                break
        
        # Check for solution_designer format
        is_solution_designer = any('"changes":' in msg.get("content", "") and 
                                '"file_path":' in msg.get("content", "") and 
                                '"diff":' in msg.get("content", "")
                                for msg in messages)
                                
        # Check for specific content types
        is_code = any("```" in msg.get("content", "") or "def " in msg.get("content", "")
                    for msg in messages if msg.get("role") == "user")
        is_json = any("json" in msg.get("content", "").lower() or 
                    msg.get("content", "").strip().startswith("{") or 
                    msg.get("content", "").strip().startswith("[")
                    for msg in messages if msg.get("role") == "user")
        is_diff = any("--- " in msg.get("content", "") and "+++ " in msg.get("content", "")
                    for msg in messages if msg.get("role") == "user")
        
        # Determine content type
        if is_solution_designer:
            return ContentType.SOLUTION_DESIGNER
        elif is_code and is_json:
            return ContentType.JSON_CODE
        elif is_code:
            return ContentType.CODE
        elif is_json:
            return ContentType.JSON
        elif is_diff:
            return ContentType.DIFF
        else:
            return ContentType.TEXT

    def _build_completion_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Build parameters for LLM completion request.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dictionary of completion parameters
        """
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
            completion_params.update(model_params)
        
        if "api_base" in provider_config:
            completion_params["api_base"] = provider_config["api_base"]
                
        return completion_params

    def _make_llm_request(self, completion_params: Dict[str, Any]) -> Any:
        """
        Make LLM request with rate limit handling.
        
        Args:
            completion_params: Completion request parameters
            
        Returns:
            LLM response
        """
        try:
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
            
            # Add API base from provider config
            provider_config = self.parent._get_provider_config(self.provider)
            if "api_base" in provider_config:
                safe_params["api_base"] = provider_config["api_base"]
                    
            response = completion(**safe_params)
            return response
            
        except litellm.RateLimitError as e:
            self.logger.warning("llm.rate_limit_error", error=str(e)[:200])
            raise
            
        except Exception as e:
            self.logger.error("llm.request_error", error=str(e))
            raise
        
    def _get_content_from_response(self, response: Any) -> str:
        """
        Extract content from LLM response.
        
        Args:
            response: LLM response object
            
        Returns:
            Content string
        """
        if hasattr(response, 'choices') and response.choices:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                return response.choices[0].message.content
        return ""