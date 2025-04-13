# Path: /Users/jim/src/apps/c4h/c4h_agents/skills/shared/markdown_utils.py
"""
Shared utilities for handling markdown code blocks.
Path: src/skills/shared/markdown_utils.py
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
# Import the central logger utility
from c4h_agents.utils.logging import get_logger

logger = get_logger() # Use centralized logger

@dataclass
class CodeBlock:
    """Represents a parsed code block"""
    content: str
    language: Optional[str] = None
    raw: str = ""

def extract_code_block(content: str) -> CodeBlock:
    """
    Extract code from markdown code blocks.
    Handles both fenced and inline code blocks.
    
    Args:
        content: String potentially containing markdown code blocks
        
    Returns:
        CodeBlock with extracted content and metadata
    """
    # Ensure logger is initialized if called early
    logger_to_use = logger if logger else get_logger()
    try:
        if not isinstance(content, str):
             logger_to_use.warning("extract_code_block received non-string input", input_type=type(content).__name__)
             # Handle non-string input gracefully - return as is or raise error?
             # Returning original content within CodeBlock for now
             return CodeBlock(content=str(content), raw=str(content))

        content_stripped = content.strip()
        raw = content # Store original input
        language = None
        extracted_content = content_stripped # Default to stripped original

        # Handle fenced code blocks (```...```)
        if content_stripped.startswith('```') and content_stripped.endswith('```'):
            lines = content_stripped.split('\n')
            # Extract language if specified on the first line (e.g., ```python)
            first_line = lines[0][3:].strip()
            if first_line:
                language = first_line
                # Content starts from the second line
                content_lines = lines[1:-1] # Exclude first and last lines (fences)
            else:
                # No language specified, content starts from second line
                content_lines = lines[1:-1] # Exclude fences

            extracted_content = '\n'.join(content_lines)
            logger_to_use.debug("markdown.extracted_fenced_code", language=language)

        # Handle inline code blocks (`...`) - check if the entire string is wrapped
        elif content_stripped.startswith('`') and content_stripped.endswith('`'):
             # Avoid stripping if it's just using backticks within the text
             # Only treat as inline block if it's ONLY the backtick-wrapped content
             if content_stripped.count('`') == 2:
                  extracted_content = content_stripped[1:-1]
                  logger_to_use.debug("markdown.extracted_inline_code")
             # else: treat as normal text containing backticks
        
        # Log the outcome
        logger_to_use.debug("markdown.extracted_code",
                    original_length=len(raw), # Pass parameters directly
                    cleaned_length=len(extracted_content), #
                    language=language,
                    content_preview=extracted_content[:100] if extracted_content else None)
                    
        return CodeBlock(
            content=extracted_content, # Return the processed content
            language=language,
            raw=raw # Keep original raw input
        )
        
    except Exception as e:
        logger_to_use.error("markdown.extraction_failed", error=str(e)) # Pass error directly
        # Fallback: return original content wrapped in CodeBlock
        return CodeBlock(content=content, raw=content)


def is_code_block(content: str) -> bool:
    """
    Check if content appears to be primarily a markdown code block.
    Args:
        content: String to check
        
    Returns:
        bool indicating if content is likely a code block
    """
    if not isinstance(content, str):
         return False
         
    content_stripped = content.strip()
    # Check for fenced block structure
    is_fenced = content_stripped.startswith('```') and content_stripped.endswith('```')
    # Check for simple inline structure (only backticks at start/end)
    is_inline = content_stripped.startswith('`') and content_stripped.endswith('`') and content_stripped.count('`') == 2
    
    return is_fenced or is_inline