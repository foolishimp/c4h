from enum import Enum
from typing import Dict, Callable, Optional, List, Union

class ContentType(str, Enum):
    """Content types for specialized handling"""
    TEXT = "text"
    CODE = "code"
    JSON = "json"
    JSON_CODE = "json_code"
    DIFF = "diff"
    SOLUTION_DESIGNER = "solution_designer"

# Define the JSON prompt separately to avoid circular reference
JSON_PROMPT = """
CRITICAL REQUIREMENTS:
1. You are continuing a JSON structure
2. Maintain the exact structure with proper nesting
3. Continue precisely where the text ends, never repeating any content
4. Ensure proper escaping of special characters
5. Complete any unfinished JSON objects, arrays, or properties
6. Never add explanatory text or comments outside the JSON structure
"""

# Content type configurations
CONTENT_TYPES: Dict[ContentType, Dict[str, Union[str, Callable[[int], int], bool]]] = {
    ContentType.SOLUTION_DESIGNER: {
        "prompt": """
CRITICAL REQUIREMENTS:
1. You are continuing a Solution Designer response with JSON structure
2. Maintain the exact structure including proper escaping of quotation marks
3. Continue precisely where the text ends, never repeating any content
4. For diff sections, ensure proper escaping of newlines (\\n) and quotes (\")
5. Never output explanatory text or comments outside the JSON structure
6. Complete any unfinished JSON objects, arrays, or properties
""",
        "overlap_size": lambda length: min(max(length // 3, 200), 500),
        "requires_cleaning": True
    },
    ContentType.JSON: {
        "prompt": JSON_PROMPT,
        "overlap_size": lambda length: min(max(length // 3, 150), 500),
        "requires_cleaning": True
    },
    ContentType.JSON_CODE: {
        "prompt": JSON_PROMPT,  # Reuse the extracted JSON prompt
        "overlap_size": lambda length: min(max(length // 3, 150), 500),
        "requires_cleaning": True
    },
    ContentType.CODE: {
        "prompt": """
CRITICAL REQUIREMENTS:
1. You are continuing code
2. Maintain consistent indentation and coding style
3. Continue precisely where the text ends, never repeating any content
4. Complete any unfinished functions, blocks, or statements
5. Never add explanatory text or comments outside the code
""",
        "overlap_size": lambda length: min(max(length // 4, 100), 500),
        "requires_cleaning": False
    },
    ContentType.DIFF: {
        "prompt": """
CRITICAL REQUIREMENTS:
1. You are continuing a diff
2. Preserve the exact format with +/- line prefixes
3. Continue precisely where the text ends, never repeating any content
""",
        "overlap_size": lambda length: min(max(length // 5, 80), 500),
        "requires_cleaning": False
    },
    ContentType.TEXT: {
        "prompt": """
CRITICAL REQUIREMENTS:
1. Continue precisely where the text ends, never repeating any content
2. Maintain the same style, formatting, and tone as the original
3. Do not add any explanatory text, headers, or comments
""",
        "overlap_size": lambda length: min(max(length // 5, 80), 500),
        "requires_cleaning": False
    }
}

# Stitching retry strategies
STITCHING_STRATEGIES: List[Dict[str, Union[str, Callable[[str, str], str], None]]] = [
    {"name": "resubmission", "prompt": None},  # No prompt change, just retry
    {"name": "follow_up", "prompt": lambda content, content_type: f"""
The previous continuation attempt failed to align properly. Please continue exactly from the end of this content:

--- PREVIOUS CONTENT ---
{content[-500:]}
--- END PREVIOUS CONTENT ---

CRITICAL REQUIREMENTS:
1. Start precisely at the end of the provided content
2. Do not repeat any previous content
3. Maintain the same format and structure as the previous content
""" + (CONTENT_TYPES[content_type]["prompt"] if content_type in CONTENT_TYPES else "")},
    {"name": "overlap_request", "prompt": lambda overlap, content_type: f"""
The previous continuation did not align correctly. Please provide the continuation starting with this exact overlap:

--- REQUIRED OVERLAP ---
{overlap}
--- END REQUIRED OVERLAP ---

CRITICAL REQUIREMENTS:
1. Begin your response with the exact overlap provided above
2. Continue seamlessly from where the overlap ends
3. Do not add any additional text or comments before the overlap
""" + (CONTENT_TYPES[content_type]["prompt"] if content_type in CONTENT_TYPES else "")}
]

# General configuration
CONFIG = {
    "rate_limit_retry_base_delay": 2.0,
    "rate_limit_max_retries": 5,
    "rate_limit_max_backoff": 60,
    "min_overlap_size": 50,
    "max_overlap_size": 500,
    "max_stitching_retries": 2
}