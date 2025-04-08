# File: c4h_agents/agents/continuation/joining_strategies.py

from typing import Tuple
import re
import traceback
import json
from .overlap_strategies import find_explicit_overlap

def join_with_explicit_overlap(previous: str, current: str, explicit_overlap: str, requested_size: int, logger) -> Tuple[str, bool]:
    """
    Join previous and current content using explicit overlap.
    
    Args:
        previous: Accumulated content so far
        current: New continuation content
        explicit_overlap: The overlap text we asked the LLM to repeat
        requested_size: The size of overlap we requested (hint)
        logger: Logger instance
        
    Returns:
        (joined_content, success) tuple
    """
    try:
        # Find the overlap position
        success, position = find_explicit_overlap(previous, current, explicit_overlap, requested_size, logger)
        
        if success and position > 0:
            # Join the content at the identified position
            joined_content = previous + current[position:]
            logger.debug("Successfully joined with explicit overlap", 
                         extra={"overlap_size": position, "joined_length": len(joined_content)})
            return joined_content, True
            
        # If explicit overlap matching failed, try some simple structural heuristics
        # This can handle case where the LLM didn't repeat properly but continued sensibly
        
        # 1. Check if we can just append (good for well-structured content)
        if _is_valid_append_point(previous, current):
            logger.info("Used structural heuristics for joining", 
                      extra={"method": "valid_append_point"})
            return previous + current, True
            
        # 2. If no matches at all, we return failure
        logger.warning("Could not join content with any method")
        return previous, False
        
    except Exception as e:
        logger.error("Join with explicit overlap failed",
                    extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return previous, False

def _is_valid_append_point(previous: str, current: str) -> bool:
    """
    Check if it's reasonable to directly append content based on structural cues.
    Uses simple heuristics to detect if the content appears to be at a valid break point.
    """
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
    
    # For JSON-like content, check structural integrity
    open_braces = previous.count('{') - previous.count('}')
    open_brackets = previous.count('[') - previous.count(']')
    
    if (open_braces > 0 or open_brackets > 0) and (current.lstrip()[0] in ['"', '{', '[', ']', '}', ',']):
        return True
    
    # Default to false - better to be cautious
    return False
        
def clean_json_content(content: str, logger) -> str:
    """Clean up JSON-like content by fixing structure."""
    try:
        # Fix unbalanced braces
        if '{' in content and '}' in content:
            open_braces = content.count('{')
            close_braces = content.count('}')
            if open_braces > close_braces:
                missing = open_braces - close_braces
                content += '\n' + '}' * missing
                logger.debug("Added missing closing braces", extra={"count": missing})
            
            # Fix unbalanced brackets
            open_brackets = content.count('[')
            close_brackets = content.count(']')
            if open_brackets > close_brackets:
                missing = open_brackets - close_brackets
                content += '\n' + ']' * missing
                logger.debug("Added missing closing brackets", extra={"count": missing})
                
            # Check for special markers
            change_begins = content.count('===CHANGE_BEGIN===')
            change_ends = content.count('===CHANGE_END===')
            
            if change_begins > change_ends:
                missing = change_begins - change_ends
                content += '\n===CHANGE_END===' * missing
                logger.debug("Added missing CHANGE_END markers", extra={"count": missing})
            
        # Attempt light validation - try to parse and fix
        try:
            # See if the content looks like valid JSON
            if (content.strip().startswith('{') and content.strip().endswith('}')) or \
               (content.strip().startswith('[') and content.strip().endswith(']')):
                json.loads(content)  # Just to validate
        except json.JSONDecodeError:
            # Not valid JSON - might be close though, just log it
            logger.warning("Content appears to be JSON but is not valid")
        
        return content
    except Exception as e:
        logger.error("Content cleaning failed",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return content