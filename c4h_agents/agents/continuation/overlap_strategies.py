from typing import Optional, Tuple, List
import traceback
from pathlib import Path
import re
import hashlib
from .config import CONFIG
from .utils import setup_logger

def find_overlap(previous: str, current: str, logger) -> Tuple[Optional[str], str]:
    """Find the best overlap between previous and current content."""
    # Exact match
    overlap = _find_exact_overlap(previous, current, logger)
    if overlap:
        return overlap, "exact_matches"
    
    # Token match
    token_result = _find_token_match(previous, current, logger)
    if token_result:
        position, confidence = token_result
        if confidence >= 0.7:
            return current[:position], "token_matches"
    
    # Fuzzy match
    fuzzy_result = _find_fuzzy_match(previous, current, logger)
    if fuzzy_result:
        return fuzzy_result, "fuzzy_matches"
    
    return None, "none"

def _find_exact_overlap(previous: str, current: str, logger) -> Optional[str]:
    """Find exact overlap between previous and current content."""
    try:
        min_size = min(CONFIG["min_overlap_size"], len(previous), len(current))
        max_size = min(CONFIG["max_overlap_size"], len(previous), len(current))
        
        for size in range(max_size, min_size - 1, -10):
            overlap = previous[-size:]
            if current.startswith(overlap):
                return overlap
        
        for size in range(min_size, max_size + 1):
            if size % 10 == 0:
                continue
            overlap = previous[-size:]
            if current.startswith(overlap):
                return overlap
        return None
    except Exception as e:
        logger.error("Exact overlap detection failed",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return None

def _find_token_match(previous: str, current: str, logger) -> Optional[Tuple[int, float]]:
    """Find token-based overlap between previous and current content."""
    try:
        prev_tokens = _tokenize(previous[-1000:], logger)
        curr_tokens = _tokenize(current[:1000], logger)
        
        if not prev_tokens or not curr_tokens:
            logger.debug("No tokens available for matching")
            return None
            
        best_match_len, best_match_pos = 0, 0
        
        for i in range(len(prev_tokens) - 4):
            prev_seq = prev_tokens[i:i+5]
            for j in range(len(curr_tokens) - 4):
                curr_seq = curr_tokens[j:j+5]
                if prev_seq == curr_seq:
                    match_len = 5
                    while (i + match_len < len(prev_tokens) and 
                           j + match_len < len(curr_tokens) and 
                           prev_tokens[i + match_len] == curr_tokens[j + match_len]):
                        match_len += 1
                    if match_len > best_match_len:
                        best_match_len = match_len
                        best_match_pos = j
        
        if best_match_len >= 5:
            char_pos = sum(len(t) + 1 for t in curr_tokens[:best_match_pos])
            confidence = min(best_match_len / 10, 1.0)
            return char_pos, confidence
        return None
    except Exception as e:
        logger.error("Token match detection failed",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return None

def _find_fuzzy_match(previous: str, current: str, logger) -> Optional[str]:
    """Find fuzzy match using hash-based approach."""
    try:
        prev_norm = ''.join(previous.lower().split())
        curr_norm = ''.join(current.lower().split())
        
        for window_size in [100, 70, 50, 30]:
            if len(prev_norm) < window_size or len(curr_norm) < window_size:
                continue
            prev_hash = hashlib.md5(prev_norm[-window_size:].encode()).hexdigest()
            for i in range(len(curr_norm) - window_size + 1):
                curr_window = curr_norm[i:i+window_size]
                curr_hash = hashlib.md5(curr_window.encode()).hexdigest()
                if prev_hash == curr_hash:
                    char_pos = len(current) * i // len(curr_norm)
                    return current[char_pos:]
        return None
    except Exception as e:
        logger.error("Fuzzy match detection failed",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return None

def _tokenize(text: str, logger) -> List[str]:
    """Simple tokenization for token matching."""
    try:
        return re.findall(r'\w+|[^\w\s]', text)
    except Exception as e:
        logger.error("Tokenization failed",
                     extra={"error": str(e), "stack_trace": traceback.format_exc()})
        return text.split()