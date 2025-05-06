# Path: /Users/jim/src/apps/c4h/c4h_agents/agents/continuation/joining_strategies.py

from typing import Tuple
import re
import traceback
import json
# Import the central logger utility
from c4h_agents.utils.logging import get_logger
from .overlap_strategies import find_explicit_overlap

# Use central logger at module level if needed, though functions receive it
logger = get_logger()

def join_with_explicit_overlap(previous: str, current: str, explicit_overlap: str, requested_size: int, logger) -> Tuple[str, bool]:
    """
    Join previous and current content using explicit overlap.
    
    Args:
        previous: Accumulated content so far
        current: New continuation content
        explicit_overlap: The overlap text we asked the LLM to repeat
        requested_size: The size of overlap we requested (hint)
        logger: Logger instance (passed from caller)
        
    Returns:
        (joined_content, success) tuple
    """
    try:
        # Clean any potential marker text that might have been included
        current_cleaned = current
        markers = ["```repeat_exactly", "```previous", "```"]
        for marker in markers:
            current_cleaned = current_cleaned.replace(marker, "")
        
        # Find the overlap position
        success, position = find_explicit_overlap(previous, current_cleaned, explicit_overlap, requested_size, logger)
        
        if success and position > 0:
            # Join the content at the identified position
            joined_content = previous + current_cleaned[position:]
            logger.debug("Successfully joined with explicit overlap", 
                        overlap_size=position, joined_length=len(joined_content))
            return joined_content, True
            
        # If explicit overlap matching failed, try some simple structural heuristics
        # This can handle case where the LLM didn't repeat properly but continued sensibly
        
        if _is_valid_append_point(previous, current_cleaned):
            logger.info("Used structural heuristics for joining",
                       method="valid_append_point")
            return previous + current_cleaned, True
            
        # Try a more aggressive approach to find the overlap if all else fails
        if explicit_overlap in current_cleaned:
            position = current_cleaned.find(explicit_overlap) + len(explicit_overlap)
            logger.info("Found overlap using direct search as fallback")
            return previous + current_cleaned[position:], True
            
        # If no matches at all, we return failure
        logger.warning("Could not join content with any method")
        return previous, False
        
    except Exception as e:
        logger.error("Join with explicit overlap failed",
                    error=str(e),
                    stack_trace=traceback.format_exc())
        return previous, False

def _is_valid_append_point(previous: str, current: str) -> bool:
    """Check if it's reasonable to directly append content.""" # Simplified Heuristics
    try:
        # Basic structural checks... 
        # Ensure strings are not empty before accessing indices
        if not previous or not current:
             return False
             
        # Check for complete sentences at the end of previous
        if previous.rstrip().endswith(('.', '!', '?', ':', ';')):
            return True
            
        # Check for natural break points in code or structured text
        if (previous.rstrip().endswith(('{', '[', '(', ',', ';')) or 
            previous.rstrip().endswith(('function', 'class', 'if', 'else', 'for', 'while'))):
            return True
        
        # Check for incomplete words (don't append if we'd be joining mid-word)
        if previous[-1].isalnum() and current[0].isalnum():
            return False
            
        # Check for block delimiters in markdown or structured text
        if (previous.rstrip().endswith(('```', '===', '---')) or
            current.lstrip().startswith(('```', '===', '---'))):
            return True
        
        # For JSON-like content, check structural integrity (simplistic)
        open_braces = previous.count('{') - previous.count('}')
        open_brackets = previous.count('[') - previous.count(']')
        
        if (open_braces > 0 or open_brackets > 0) and (current.lstrip()[0] in ['"', '{', '[', ']', '}', ',']):
            return True

    except IndexError:
         # Handle potential index errors if strings are very short
         return False
    except Exception as e:
         # Use global logger as this is a static-like helper
         logger.error("_is_valid_append_point.failed", error=str(e))
         return False

    # Default to false - better to be cautious
    return False
        
def clean_json_content(content: str, logger) -> str:
    """Clean up JSON-like content by fixing structure."""
    try:
        # Fix unbalanced braces
        if '{' in content or '}' in content: # Check if braces are present at all
            open_braces = content.count('{')
            close_braces = content.count('}')
            if open_braces > close_braces:
                missing = open_braces - close_braces
                content += '\n' + '}' * missing
                logger.debug("Added missing closing braces", count=missing) # Pass directly
            
            # Fix unbalanced brackets
            open_brackets = content.count('[')
            close_brackets = content.count(']')
            if open_brackets > close_brackets:
                missing = open_brackets - close_brackets
                content += '\n' + ']' * missing #
                logger.debug("Added missing closing brackets", count=missing) # Pass directly
                
            # Check for special markers (Ensure they are balanced)
            change_begins = content.count('===CHANGE_BEGIN===')
            change_ends = content.count('===CHANGE_END===')
            
            if change_begins > change_ends:
                missing = change_begins - change_ends
                content += '\n===CHANGE_END===' * missing
                logger.debug("Added missing CHANGE_END markers", count=missing) # Pass directly
        
        # Attempt light validation - try to parse and fix if it looks like JSON
        # Only attempt parse if it starts/ends like JSON
        content_stripped = content.strip()
        if (content_stripped.startswith('{') and content_stripped.endswith('}')) or \
           (content_stripped.startswith('[') and content_stripped.endswith(']')):
            try:
                json.loads(content)  # Just to validate
            except json.JSONDecodeError:
                # Not valid JSON - might be close though, just log it
                logger.warning("Content appears to be JSON but is not valid")
        
        return content
    except Exception as e:
        logger.error("Content cleaning failed",
                      error=str(e), # Pass directly
                      stack_trace=traceback.format_exc())
        return content