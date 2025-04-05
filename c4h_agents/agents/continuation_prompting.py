"""
Prompt creation utilities for LLM continuation handling.
Path: c4h_agents/agents/continuation_prompting.py
"""

import json
from typing import List, Dict, Any
from c4h_agents.utils.logging import get_logger

logger = get_logger()

def detect_content_type(messages: List[Dict[str, str]]) -> str:
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

def create_numbered_continuation_prompt(context_json: str, next_line: int, content_type: str) -> str:
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
