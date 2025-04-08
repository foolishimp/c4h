# File: c4h_agents/agents/continuation/config.py

from typing import Dict, Callable, List, Union

# Simple configuration for window-based continuation strategy
WINDOW_CONFIG = {
    # Size of explicit overlap to request in characters
    "overlap_size": 150,
    
    # Window context size
    "min_context_window": 1000,
    "max_context_window": 2000,
    
    # Retry settings
    "max_stitching_retries": 3,
    
    # Rate limit handling
    "rate_limit_retry_base_delay": 2.0,
    "rate_limit_max_retries": 5,
    "rate_limit_max_backoff": 60,
}

# Fallback stitching strategies - simplified
STITCHING_STRATEGIES: List[Dict[str, Union[str, Callable[[str, int], str], None]]] = [
    {"name": "retry_with_longer_overlap", "prompt": lambda context_window, overlap_size: f"""
I need you to continue the previous response that was interrupted due to length limits.

HERE IS THE END OF YOUR PREVIOUS RESPONSE:
------------BEGIN PREVIOUS CONTENT------------
{context_window}
------------END PREVIOUS CONTENT------------

CRITICAL CONTINUATION INSTRUCTIONS:
1. First, repeat these EXACT {overlap_size} characters:
------------OVERLAP TO REPEAT------------
{context_window[-overlap_size:]}
------------END OVERLAP------------

2. Then continue seamlessly from that point
3. Maintain identical style, formatting and organization

Begin by repeating the overlap text exactly, then continue:
"""},
    {"name": "explicit_content_match", "prompt": lambda context_window, overlap_size: f"""
The previous continuation did not align correctly. Please try again with a different approach.

HERE IS THE END OF YOUR PREVIOUS RESPONSE:
{context_window}

CRITICAL INSTRUCTIONS:
1. First, repeat these EXACT {overlap_size*2} characters:
```
{context_window[-(overlap_size*2):]}
```

2. Then continue directly from that point
3. Do not add any explanations or commentary

Start your response with the exact text between the triple backticks above:
"""}
]

# Simplified cleaning function
def requires_json_cleaning(content: str) -> bool:
    """Check if content appears to be JSON and may need cleaning."""
    return ('{' in content and '}' in content) or ('[' in content and ']' in content)