from typing import Tuple, Optional
import re
import traceback
from .config import ContentType, CONTENT_TYPES
from .overlap_strategies import find_overlap
from .utils import setup_logger

def join_content(previous: str, current: str, content_type: ContentType, logger) -> Tuple[str, str]:
    """Join continuations using config-driven strategies."""
    overlap, method = find_overlap(previous, current, logger)
    if overlap and method != "none":
        return previous + current[len(overlap):], method
    
    if content_type == ContentType.SOLUTION_DESIGNER:
        result = _join_solution_designer(previous, current, logger)
        if result:
            return result, "structure_matches"
    elif content_type in (ContentType.JSON, ContentType.JSON_CODE):
        result = _join_json_content(previous, current, logger)
        if result:
            return result, "structure_matches"
    
    append_marker = f"\n--- UNABLE TO GUARANTEE STITCHING ---\n"
    joined = previous + append_marker + current
    logger.warning("Unable to guarantee stitching, using append fallback",
                   extra={"content_type": content_type})
    return joined, "append_fallbacks"

def _join_solution_designer(previous: str, current: str, logger) -> Optional[str]:
    """Join solution designer content with structure awareness."""
    try:
        open_braces = previous.count('{') - previous.count('}')
        open_brackets = previous.count('[') - previous.count(']')
        open_quotes = previous.count('"') % 2
        
        if open_braces == 0 and open_brackets == 0 and open_quotes == 0:
            if re.match(r'^\s*\{', current):
                if re.search(r',\s*$', previous) or re.search(r'\[\s*$', previous):
                    return previous + current
                elif re.search(r'\}\s*$', previous):
                    return previous + ',\n' + current
        
        patterns = [
            r'"file_path"\s*:\s*"[^"]+"\s*,',
            r'"type"\s*:\s*"[^"]+"\s*,',
            r'"description"\s*:\s*"[^"]+"\s*,',
            r'"diff"\s*:\s*"'
        ]
        
        for pattern in patterns:
            prev_match = re.search(f'({pattern})\\s*$', previous)
            if prev_match:
                curr_match = re.search(f'^\\s*({pattern})', current)
                if curr_match:
                    return previous + current[curr_match.end():]
        
        if '"diff": "' in previous and not previous.endswith('"'):
            diff_markers = ['---', '+++', '@@', '+', '-']
            for marker in diff_markers:
                if current.startswith(marker):
                    return previous + current
        return None
    except Exception as e:
        logger.error("Solution designer join failed",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return None

def _join_json_content(previous: str, current: str, logger) -> Optional[str]:
    """Join JSON content with structure awareness."""
    try:
        open_braces = previous.count('{') - previous.count('}')
        open_brackets = previous.count('[') - previous.count(']')
        open_quotes = previous.count('"') % 2
        
        if open_braces > 0 or open_brackets > 0:
            prop_pattern = r'"[^"]+"\s*:\s*'
            if re.search(prop_pattern + r'$', previous):
                return previous + current
            if previous.rstrip().endswith(','):
                return previous + current
            if previous.rstrip().endswith('{') or previous.rstrip().endswith('['):
                return previous + current
        return None
    except Exception as e:
        logger.error("JSON join failed",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return None

def clean_json_content(content: str, logger) -> str:
    """Clean up JSON content by fixing structure."""
    try:
        if '{' in content and '}' in content:
            open_braces = content.count('{')
            close_braces = content.count('}')
            if open_braces > close_braces:
                missing = open_braces - close_braces
                content += '\n' + '}' * missing
                logger.debug("Added missing closing braces", extra={"count": missing})
            
            open_brackets = content.count('[')
            close_brackets = content.count(']')
            if open_brackets > close_brackets:
                missing = open_brackets - close_brackets
                content += '\n' + ']' * missing
                logger.debug("Added missing closing brackets", extra={"count": missing})
        return content
    except Exception as e:
        logger.error("JSON cleaning failed",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return content