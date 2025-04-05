"""
LLM response parsing utilities for continuation handling.
Path: c4h_agents/agents/continuation_parsing.py
"""

import json
import re
from typing import List, Tuple, Any
from c4h_agents.utils.logging import get_logger

logger = get_logger()

def get_content_sample(content: str, line_number: int, context_lines: int = 5) -> str:
    """Extract sample lines around a specific line number"""
    lines = content.splitlines()
    start = max(0, line_number - context_lines - 1)
    end = min(len(lines), line_number + context_lines)
    
    result = []
    for i in range(start, end):
        line_marker = ">>> " if i == line_number - 1 else "    "
        result.append(f"{line_marker}{i+1}: {lines[i]}")
    
    return "\n".join(result)

def format_with_line_numbers_and_indentation(content: str) -> List[Tuple[int, int, str]]:
    """Format content with line numbers and indentation level tracking"""
    lines = content.splitlines()
    result = []
    
    for i, line in enumerate(lines):
        # Calculate leading whitespace (indentation)
        indent = len(line) - len(line.lstrip())
        result.append((i+1, indent, line))
    
    return result

def create_line_json(numbered_lines: List[Tuple[int, int, str]], max_context_lines: int = 30) -> str:
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

def parse_json_content(content: str, expected_start_line: int, logger_instance = None) -> List[Tuple[int, int, str]]:
    """Parse content with line numbers and indentation from JSON format"""
    log = logger_instance or logger
    numbered_lines = []
    
    try:
        # Extract JSON from response content
        json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', content)
        if json_match:
            json_content = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'(\{\s*"lines"\s*:\s*\[[\s\S]+?\]\s*\})', content)
            if json_match:
                json_content = json_match.group(1)
            else:
                # Fall back to using the entire content
                json_content = content
        
        # Parse the JSON
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError:
            # Try again with a more aggressive approach to find JSON
            array_match = re.search(r'\[\s*\{\s*"line"[\s\S]+?\}\s*\]', content)
            if array_match:
                # Add wrapping to make it valid JSON
                array_json = '{"lines": ' + array_match.group(0) + '}'
                try:
                    data = json.loads(array_json)
                except json.JSONDecodeError:
                    # Individual line objects
                    line_objects = extract_line_objects(content) 
                    if line_objects:
                        data = {"lines": line_objects}
                    else:
                        return []
            else:
                return []
        
        # Get the lines array
        lines = data.get("lines", [])
        if not lines and isinstance(data, list):
            # Handle case where the array is the top-level element
            lines = data
        
        # Process each line
        for line_data in lines:
            try:
                line_num = line_data.get("line")
                indent = line_data.get("indent", 0)
                content = line_data.get("content", "")
                
                # Only add if it's the expected line number or after
                if line_num >= expected_start_line:
                    numbered_lines.append((line_num, indent, content))
            except (TypeError, AttributeError):
                # Skip invalid line data
                continue
        
        # Sort by line number
        numbered_lines.sort(key=lambda x: x[0])
        return numbered_lines
    
    except Exception as e:
        log.error("llm.json_parse_error", error=str(e))
        return []

def extract_line_objects(content: str) -> List[dict]:
    """Extract individual line objects from content using regex"""
    line_objects = []
    # Match pattern for individual line objects
    pattern = r'\{\s*"line"\s*:\s*(\d+)\s*,\s*"indent"\s*:\s*(\d+)\s*,\s*"content"\s*:\s*"([^"]*)"\s*\}'
    matches = re.finditer(pattern, content)
    
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
            
    return line_objects

def attempt_repair_parse(content: str, expected_start_line: int, logger_instance = None) -> List[Tuple[int, int, str]]:
    """More aggressive parsing attempt for broken JSON"""
    log = logger_instance or logger
    # Try to manually extract line number, indent, and content
    numbered_lines = []
    
    # Look for patterns like "line": 42, "indent": 4, "content": "some content"
    line_pattern = r'"line"\s*:\s*(\d+)[^\d].*?"indent"\s*:\s*(\d+)[^}]*"content"\s*:\s*"([^"]*)"'
    matches = re.finditer(line_pattern, content)
    
    for match in matches:
        try:
            line_num = int(match.group(1))
            indent = int(match.group(2))
            line_content = match.group(3)
            
            # Only add if it's the expected line number or after
            if line_num >= expected_start_line:
                numbered_lines.append((line_num, indent, line_content))
        except (ValueError, IndexError):
            continue
    
    # If we found any lines, sort them and return
    if numbered_lines:
        numbered_lines.sort(key=lambda x: x[0])
        log.info("llm.repair_parse_successful", lines_found=len(numbered_lines))
        return numbered_lines
    
    # Last resort: try to extract any numbered lines from the text
    line_pattern = r'(?:line|Line)?\s*(\d+)[^\n]*:\s*([^\n]*)'
    matches = re.finditer(line_pattern, content)
    
    for match in matches:
        try:
            line_num = int(match.group(1))
            line_content = match.group(2).strip()
            
            # Use a default indent of 0
            if line_num >= expected_start_line:
                indent = len(line_content) - len(line_content.lstrip())
                numbered_lines.append((line_num, indent, line_content))
        except (ValueError, IndexError):
            continue
            
    # Sort any found lines
    if numbered_lines:
        numbered_lines.sort(key=lambda x: x[0])
        log.info("llm.fallback_parse_successful", lines_found=len(numbered_lines))
        
    return numbered_lines

def numbered_lines_to_content(numbered_lines: List[Tuple[int, int, str]]) -> str:
    """Convert numbered lines back to raw content with proper indentation"""
    # Sort by line number to ensure correct order
    sorted_lines = sorted(numbered_lines, key=lambda x: x[0])
    
    # Extract content with preserved indentation
    content_lines = [line[2] for line in sorted_lines]
    
    return "\n".join(content_lines)