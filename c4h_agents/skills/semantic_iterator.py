"""
Semantic iterator with standardized BaseAgent implementation.
Path: c4h_agents/skills/semantic_iterator.py
"""

from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
import json
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

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Process extraction request following standard agent interface"""
        try:
            content = context.get('input_data', context)
            if isinstance(content, dict):
                content = json.dumps(content, indent=2)
            instruction = context.get('instruction', '')
            format_hint = context.get('format', 'json')

            extract_config = ExtractConfig(
                instruction=instruction,
                format=format_hint
            )
            
            self._state = ExtractorState(
                mode=self._state.mode,
                content=content,
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
                              expected=expected_items)
                    
                    # Fallback to slow extraction for more reliable results
                    logger.info("extraction.fallback_to_slow", reason="incomplete_results")
                    self._state.mode = ExtractionMode.SLOW
                    self._state.iterator = self._slow_extractor.create_iterator(
                        self._state.content,
                        self._state.config
                    )
            
            # Check if fast extraction completely failed
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
                raise StopIteration
                
            return next(self._state.iterator)
            
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
        # Count file_path occurrences as base estimate
        file_path_count = content.count('"file_path"')
        
        # Additional pattern checks for verification
        change_type_count = content.count('"type"')
        diff_count = content.count('"diff"')
        
        # Take min to avoid overcounting
        estimate = min(file_path_count, max(1, change_type_count))
        
        logger.debug("iterator.estimated_items", 
                   estimate=estimate,
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
        
        self._state = ExtractorState(
            mode=self._state.mode,
            content=content,
            config=config,
            position=0
        )
        
        logger.debug("iterator.configured",
                    mode=self._state.mode,
                    content_type=type(content).__name__)
                    
        return self