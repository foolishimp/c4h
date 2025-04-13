# Path: /Users/jim/src/apps/c4h/c4h_agents/agents/continuation/overlap_strategies.py

from typing import Optional, Tuple
import traceback
import re
# Import the central logger utility
from c4h_agents.utils.logging import get_logger

# Use central logger at module level if needed, though function receives it
logger = get_logger()

def find_explicit_overlap(previous: str, current: str, explicit_overlap: str, requested_size: int, logger) -> Tuple[bool, int]:
    """
    Find an explicit overlap in the continuation response.
    
    Args:
        previous: The accumulated content so far
        current: The new continuation content
        explicit_overlap: The overlap text we asked the LLM to repeat
        requested_size: The size of overlap we requested (as a hint)
        logger: Logger instance (passed from caller)
        
    Returns:
        (success, position) where success is a boolean indicating if overlap was found,
        and position is the index where the continuation should start
    """
    try:
        # Ensure logger is valid
        logger_to_use = logger if logger else get_logger()

        logger_to_use.debug("Searching for explicit overlap", #
                    explicit_overlap_len=len(explicit_overlap), # Pass directly
                    requested_size=requested_size,
                    current_content_len=len(current)
                    )
        
        # Quick sanity check
        if not explicit_overlap or not current:
             logger_to_use.warning("find_explicit_overlap received empty string", 
                           has_overlap=bool(explicit_overlap), has_current=bool(current))
             return False, 0
            
        # Method 1: Check if the response starts with the exact overlap
        if current.startswith(explicit_overlap):
            logger_to_use.debug("Found exact overlap at beginning of content")
            return True, len(explicit_overlap)
        
        # Method 2: Search for overlap within first part of the response
        search_limit = min(len(current), len(explicit_overlap) * 3) # Search reasonable area
        search_area = current[:search_limit]
        
        # Try to find the entire overlap
        overlap_pos_in_search = search_area.find(explicit_overlap)
        if overlap_pos_in_search != -1: # Check if found
            position = overlap_pos_in_search + len(explicit_overlap)
            logger_to_use.debug("Found complete overlap in search area", 
                         position=position) # Pass directly
            return True, position
            
        # Method 3: Try matching the last N characters of explicit_overlap with beginning of current
        # This helps when the LLM slightly rephrases but gets the end of the overlap correct
        min_match_size = min(50, len(explicit_overlap) // 2) # Try at least 50 chars or half the overlap
        
        # Iterate from larger potential overlaps down to minimum
        for match_size in range(len(explicit_overlap) - 5, min_match_size -1 , -5): # Ensure range includes min_match_size
             if match_size <= 0: continue # Skip if match_size becomes non-positive
             match_text = explicit_overlap[-match_size:]
             if current.startswith(match_text):
                 logger_to_use.debug("Found partial overlap at beginning", 
                              match_size=match_size) # Pass directly
                 return True, match_size
        
        # Method 4: Look for reasonable matches with some flexibility
        # For example, whitespace differences, minor formatting changes
        def normalize(text):
            # Remove excess whitespace and normalize line endings
            normalized = re.sub(r'\s+', ' ', text.replace('\r\n', '\n').replace('\r', '\n'))
            # Normalize quotes (simple version)
            normalized = normalized.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
            # Remove backticks (from code blocks) - might be too aggressive, consider context
            # normalized = normalized.replace('`', '') 
            return normalized.strip()
            
        norm_overlap = normalize(explicit_overlap)
        # Normalize a slightly larger search area for robustness
        norm_current_search_area = normalize(current[:search_limit + 50]) 
        
        if norm_overlap and norm_current_search_area and norm_current_search_area.startswith(norm_overlap):
            # Estimate original position more carefully based on normalized match length
            # This is still an approximation
            # Find where the normalized overlap *ends* in the normalized current text
            norm_end_pos = len(norm_overlap)
            # Count non-whitespace chars in normalized overlap to estimate original chars
            orig_char_estimate = len(re.sub(r'\s', '', norm_overlap)) 
            # Use a position slightly larger than the estimate as the split point
            approx_pos = min(orig_char_estimate + int(orig_char_estimate * 0.1), len(current)) # Add 10% buffer
            
            logger_to_use.debug("Found normalized match", 
                         approximated_position=approx_pos, 
                         norm_overlap_len=len(norm_overlap),
                         orig_char_estimate=orig_char_estimate) # Pass directly
            return True, approx_pos
            
        # Last resort: Use the requested size as a hint if we can't find a match
        # This assumes the LLM tried to follow instructions but didn't repeat exactly
        if requested_size > 0 and len(current) > requested_size:
            logger_to_use.warning("Using requested size as fallback heuristic for overlap", 
                         requested_size=requested_size) # Pass directly
            # Be cautious with this fallback, maybe return False if too uncertain?
            # For now, return True as per previous logic, but it's risky.
            return True, requested_size 
            
        # If we got here, we couldn't find any reasonable overlap
        logger_to_use.warning("Could not find explicit overlap", #
                    # Pass directly
                        overlap_preview=explicit_overlap[:50] + "..." if len(explicit_overlap) > 50 else explicit_overlap,
                        current_preview=current[:50] + "..." if len(current) > 50 else current
                    )
        return False, 0
        
    except Exception as e:
        # Use logger_to_use here as well
        logger_to_use.error("Error finding explicit overlap",
                     error=str(e), stack_trace=traceback.format_exc()) # Pass directly
        return False, 0