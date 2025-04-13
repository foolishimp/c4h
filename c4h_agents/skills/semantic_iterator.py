"""
Semantic iterator with improved preprocessing for more reliable extraction.
Path: c4h_agents/skills/semantic_iterator.py
"""

from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
import json
import re
from config import locate_config
from c4h_agents.agents.base_agent import BaseAgent, AgentResponse 
from skills.shared.types import ExtractConfig
from skills._semantic_fast import FastExtractor, FastItemIterator
from skills._semantic_slow import SlowExtractor, SlowItemIterator
from enum import Enum
from c4h_agents.utils.logging import get_logger

logger = get_logger()

class ExtractionComplete(StopIteration):
    """Custom exception to signal clean completion of extraction"""
    pass

class ExtractionError(Exception):
    """Custom exception for extraction errors"""
    pass

class ExtractionMode(str, Enum):
    """Available extraction modes"""
    FAST = "fast"      # Direct extraction from structured data
    SLOW = "slow"      # Sequential item-by-item extraction

@dataclass
class ExtractorState:
    """Internal state for extraction process"""
    mode: str
    position: int = 0
    content: Any = None
    config: Optional[ExtractConfig] = None
    iterator: Optional[Union[FastItemIterator, SlowItemIterator]] = None
    expected_items: int = 0  # Added to track expected item count

class SemanticIterator(BaseAgent):
    """
    Agent responsible for semantic extraction using configurable modes.
    Follows standard BaseAgent pattern while maintaining iterator protocol.
    
    Enhanced with robust preprocessing to handle JSON-wrapped content
    and escape sequence normalization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize iterator with configuration."""
        super().__init__(config=config)
        
        # Get iterator-specific config
        iterator_config = locate_config(self.config or {}, self._get_agent_name())
        extractor_config = iterator_config.get('extractor_config', {})
        
        # Initialize extraction state
        self._state = ExtractorState(
            mode=extractor_config.get('mode', 'fast'),
            position=0
        )
        
        # Configure extractors
        self._allow_fallback = extractor_config.get('allow_fallback', True)
        self._fast_extractor = FastExtractor(config=config)
        self._slow_extractor = SlowExtractor(config=config)
        
        logger.info("semantic_iterator.initialized",
                   mode=self._state.mode,
                   allow_fallback=self._allow_fallback)

    def _get_agent_name(self) -> str:
        """Get agent name for config lookup"""
        return "semantic_iterator"

    def _preprocess_content(self, content: Any) -> str:
        """
        Preprocess content before extraction by handling JSON wrapping and normalizing escape sequences.
        
        This is a critical new function to improve extraction reliability.
        """
        try:
            # Start with logging the input type
            logger.debug("preprocessing.start", 
                       content_type=type(content).__name__,
                       is_dict=isinstance(content, dict),
                       is_str=isinstance(content, str))
            
            # Handle JSON object with response field
            if isinstance(content, dict):
                # Check if this is a response wrapper
                if "response" in content:
                    logger.info("preprocessing.unwrapping_json_response")
                    content = content["response"]
                elif "llm_output" in content and isinstance(content["llm_output"], dict):
                    if "content" in content["llm_output"]:
                        logger.info("preprocessing.unwrapping_llm_output_content")
                        content = content["llm_output"]["content"]
                # Convert any remaining dict to JSON string
                elif not isinstance(content, str):
                    logger.debug("preprocessing.converting_dict_to_json")
                    content = json.dumps(content, indent=2)
            
            # Convert to string if needed
            if not isinstance(content, str):
                logger.debug("preprocessing.converting_to_string")
                content = str(content)
            
            # Handle triple-escaped sequences that might appear in nested JSON
            # This ensures consistent handling of escaped characters
            if '\\\\\\\\' in content or '\\\\\"' in content:
                logger.info("preprocessing.normalizing_triple_escapes")
                content = content.replace('\\\\\\\\', '\\\\')
                content = content.replace('\\\\\"', '\\"')
            
            # Try to detect any JSON string wrapping the actual content
            if content.strip().startswith('"') and content.strip().endswith('"'):
                try:
                    # Check if this might be a JSON string that needs unescaping
                    decoded = json.loads(f"{{{content}}}")
                    if isinstance(decoded, dict) and len(decoded) == 1:
                        inner_value = next(iter(decoded.values()))
                        if isinstance(inner_value, str) and "===CHANGE_BEGIN===" in inner_value:
                            logger.info("preprocessing.unwrapped_json_string")
                            content = inner_value
                except (json.JSONDecodeError, ValueError):
                    # Not valid JSON, keep original
                    pass
            
            # Detect standard change blocks
            change_count = content.count("===CHANGE_BEGIN===")
            if change_count > 0:
                logger.info("preprocessing.detected_change_blocks", count=change_count)
            
            logger.debug("preprocessing.complete", 
                        result_type=type(content).__name__, 
                        length=len(content) if hasattr(content, "__len__") else 0,
                        change_count=change_count)
            
            return content
        except Exception as e:
            logger.error("preprocessing.failed", error=str(e))
            # Return original content on error
            if isinstance(content, str):
                return content
            return str(content)

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Process extraction request following standard agent interface"""
        try:
            # Get input data from context
            content = context.get('input_data', context)
            
            # Apply preprocessing before extraction
            preprocessed_content = self._preprocess_content(content)
            
            # Get extraction parameters
            instruction = context.get('instruction', '')
            format_hint = context.get('format', 'json')

            extract_config = ExtractConfig(
                instruction=instruction,
                format=format_hint
            )
            
            self._state = ExtractorState(
                mode=self._state.mode,
                content=preprocessed_content,
                config=extract_config
            )
            
            results = []
            try:
                for item in self:
                    if item == "NO_MORE_ITEMS":
                        break
                    results.append(item)
            except StopIteration:
                pass
            
            if not results:
                return AgentResponse(
                    success=False,
                    data={},
                    error="No items could be extracted"
                )
            
            return AgentResponse(
                success=True,
                data={
                    "results": results,
                    "count": len(results),
                    "format": format_hint
                }
            )

        except Exception as e:
            logger.error("semantic_iterator.process_failed", error=str(e))
            return AgentResponse(
                success=False,
                data={},
                error=str(e)
            )

    def __iter__(self) -> Iterator[Any]:
        """Initialize iteration in configured mode"""
        logger.debug("iterator.starting", mode=self._state.mode)
        
        if not self._state.content or not self._state.config:
            raise ValueError("Iterator not configured. Call process() first.")
        
        if self._state.mode == ExtractionMode.FAST:
            # Try fast extraction first
            self._state.iterator = self._fast_extractor.create_iterator(
                self._state.content,
                self._state.config
            )
            
            # Check for completeness of fast extraction
            fast_items_count = len(self._state.iterator) if self._state.iterator else 0
            
            # Estimate expected items by counting patterns in content
            if isinstance(self._state.content, str):
                expected_items = self._estimate_item_count(self._state.content)
                self._state.expected_items = expected_items
                
                # Determine if fast extraction was successful but incomplete
                if fast_items_count > 0 and fast_items_count < expected_items and self._allow_fallback:
                    logger.info("extraction.incomplete_fast_results", 
                              found=fast_items_count, 
                              expected=expected_items) # Logged the counts

                    # Fallback to slow extraction for more reliable results
                    logger.info("extraction.fallback_to_slow", reason="incomplete_results")
                    self._state.mode = ExtractionMode.SLOW
                    # --- FIX: Explicitly initialize SlowItemIterator ---
                    self._state.iterator = self._slow_extractor.create_iterator(
                        self._state.content,
                        self._state.config
                    )
                    # --- END FIX ---
            
            # Check if fast extraction completely failed or if fallback is explicitly chosen
            # Removed check for has_items as it doesn't exist on SlowItemIterator
            if not self._state.iterator.has_items() and self._allow_fallback:
                logger.info("extraction.fallback_to_slow", reason="no_fast_results")
                self._state.mode = ExtractionMode.SLOW
                self._state.iterator = self._slow_extractor.create_iterator(
                    self._state.content,
                    self._state.config
                )
        else:
            # Start with slow extraction
            self._state.iterator = self._slow_extractor.create_iterator(
                self._state.content,
                self._state.config
            )
            
        return self

    def __next__(self) -> Any:
        """Get next item using the appropriate iterator"""
        try:
            if not self._state.iterator:
                logger.warning("iterator.next_called_without_iterator")
                raise StopIteration
                
            # Directly call next on the current iterator (could be Fast or Slow)
            next_item = next(self._state.iterator)
            # Increment position tracking after successfully getting an item
            self._state.position += 1
            logger.debug("iterator.yielded_item",
                        mode=self._state.mode,
                        position=self._state.position) # Log position *after* yielding
            return next_item
            
        except StopIteration:
            logger.debug("iterator.complete", 
                        mode=self._state.mode,
                        position=self._state.position)
            raise
        except Exception as e:
            logger.error("iterator.error",
                        error=str(e),
                        mode=self._state.mode,
                        position=self._state.position)
            raise StopIteration

    def _estimate_item_count(self, content: str) -> int:
        """Estimate number of items in content based on key patterns"""
        # Count ===CHANGE_BEGIN=== occurrences as most reliable estimate
        change_begin_count = content.count("===CHANGE_BEGIN===")
        change_end_count = content.count("===CHANGE_END===")
        
        # Also count file_path occurrences as alternate estimate
        file_path_count = content.count('"file_path"')
        
        # Additional pattern checks for verification
        change_type_count = content.count('"type"')
        diff_count = content.count('"diff"')
        
        # Use most reliable count, prioritizing change markers
        if change_begin_count > 0 and change_begin_count == change_end_count:
            estimate = change_begin_count
        else:
            # Take the most conservative estimate to avoid missing items
            estimate = min(file_path_count, max(1, change_type_count))
        
        logger.debug("iterator.estimated_items", 
                   estimate=estimate,
                   change_blocks=change_begin_count,
                   file_paths=file_path_count, 
                   change_types=change_type_count,
                   diffs=diff_count)
        
        return estimate
        
    def configure(self, content: Any, config: ExtractConfig) -> 'SemanticIterator':
        """
        Legacy configuration method for prefect compatibility.
        Sets up iterator state using configuration.
        """
        logger.warning("semantic_iterator.using_deprecated_configure")
        
        # Apply preprocessing before setting up state
        preprocessed_content = self._preprocess_content(content)
        
        self._state = ExtractorState(
            mode=self._state.mode,
            content=preprocessed_content,
            config=config,
            position=0
        )
        
        logger.debug("iterator.configured",
                    mode=self._state.mode,
                    content_type=type(preprocessed_content).__name__)
                    
        return self
