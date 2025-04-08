# File: c4h_agents/agents/continuation/overlap_strategies.py

from typing import Optional, Tuple
import traceback
import re

def find_explicit_overlap(previous: str, current: str, explicit_overlap: str, requested_size: int, logger) -> Tuple[bool, int]:
    """
    Find an explicit overlap in the continuation response.
    
    Args:
        previous: The accumulated content so far
        current: The new continuation content
        explicit_overlap: The overlap text we asked the LLM to repeat
        requested_size: The size of overlap we requested (as a hint)
        logger: Logger instance
        
    Returns:
        (success, position) where success is a boolean indicating if overlap was found,
        and position is the index where the continuation should start
    """
    try:
        logger.debug("Searching for explicit overlap", 
                    extra={
                        "explicit_overlap_len": len(explicit_overlap),
                        "requested_size": requested_size,
                        "current_content_len": len(current)
                    })
        
        # Quick sanity check
        if not explicit_overlap or not current:
            return False, 0
            
        # Method 1: Check if the response starts with the exact overlap
        if current.startswith(explicit_overlap):
            logger.debug("Found exact overlap at beginning of content")
            return True, len(explicit_overlap)
        
        # Method 2: Search for overlap within first part of the response
        search_limit = min(len(current), len(explicit_overlap) * 3)
        search_area = current[:search_limit]
        
        # Try to find the entire overlap
        if explicit_overlap in search_area:
            position = search_area.find(explicit_overlap) + len(explicit_overlap)
            logger.debug("Found complete overlap in search area", 
                        extra={"position": position})
            return True, position
            
        # Method 3: Try matching the last N characters of explicit_overlap with beginning of current
        # This helps when the LLM slightly rephrases but gets the end of the overlap correct
        min_match_size = min(50, len(explicit_overlap) // 2)  # Try at least 50 chars or half the overlap
        
        for match_size in range(len(explicit_overlap) - 5, min_match_size, -5):
            match_text = explicit_overlap[-match_size:]
            if current.startswith(match_text):
                logger.debug("Found partial overlap at beginning", 
                            extra={"match_size": match_size})
                return True, match_size
        
        # Method 4: Look for reasonable matches with some flexibility
        # For example, whitespace differences, minor formatting changes
        def normalize(text):
            # Remove excess whitespace
            normalized = re.sub(r'\s+', ' ', text)
            # Normalize quotes
            normalized = normalized.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
            # Remove backticks (from code blocks)
            normalized = normalized.replace('`', '')
            return normalized.strip()
            
        norm_overlap = normalize(explicit_overlap)
        norm_current = normalize(current[:search_limit])
        
        if norm_overlap and norm_current and norm_current.startswith(norm_overlap):
            # Try to estimate how many characters this represents in the original
            approx_pos = min(len(explicit_overlap) + 20, len(current))
            logger.debug("Found normalized match", 
                        extra={"approximated_position": approx_pos})
            return True, approx_pos
            
        # Last resort: Use the requested size as a hint if we can't find a match
        # This assumes the LLM tried to follow instructions but didn't repeat exactly
        if requested_size > 0 and len(current) > requested_size:
            logger.warning("Using requested size as fallback", 
                         extra={"requested_size": requested_size})
            return True, requested_size
            
        # If we got here, we couldn't find any reasonable overlap
        logger.warning("Could not find explicit overlap", 
                    extra={
                        "overlap_preview": explicit_overlap[:50] + "..." if len(explicit_overlap) > 50 else explicit_overlap,
                        "current_preview": current[:50] + "..." if len(current) > 50 else current
                    })
        return False, 0
        
    except Exception as e:
        logger.error("Error finding explicit overlap",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return False, 0