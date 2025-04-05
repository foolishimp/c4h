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

    def _serialize_content(self, content: Any) -> Any:
        """Safely serialize content for processing"""
        # Handle ModelResponse and similar objects
        if hasattr(content, "choices") and hasattr(content.choices[0], "message"):
            try:
                # Extract the actual content from ModelResponse
                return content.choices[0].message.content
            except (AttributeError, IndexError):
                logger.warning("semantic_iterator.content_extraction_failed", 
                             error="Failed to extract content from response")
                return str(content)
        
        # Handle raw response objects
        if hasattr(content, "model_dump_json"):
            try:
                return json.loads(content.model_dump_json())
            except Exception as e:
                logger.warning("semantic_iterator.model_dump_failed", error=str(e))
                return str(content)
                
        # If it's a dict, we need to check if it contains any non-serializable objects
        if isinstance(content, dict):
            try:
                # Test if the content is JSON serializable
                sanitized = {}
                for k, v in content.items():
                    if k == "raw_output" or k == "raw_response":
                        if hasattr(v, "model_dump"):
                            sanitized[k] = v.model_dump()
                        else:
                            # For other non-serializable objects, convert to string
                            sanitized[k] = str(v)
                    else:
                        sanitized[k] = v
                return sanitized
            except (TypeError, ValueError) as e:
                # If dict contains non-serializable objects, convert to string
                logger.warning("semantic_iterator.dict_serialization_failed", error=str(e))
                return str(content)
                
        # Default case, just return the content
        return content

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Process extraction request following standard agent interface"""
        try:
            # Handle potential ModelResponse objects or other non-serializable content
            if "input_data" in context and hasattr(context["input_data"], "get"):
                input_data = context["input_data"]
                # Special handling for response and raw_output
                for key in ["response", "raw_output", "raw_content"]:
                    if key in input_data:
                        input_data[key] = self._serialize_content(input_data[key])
                content = input_data
            else:
                content = self._serialize_content(context.get('input_data', context))
                
            # Convert to string if it's a dict for proper processing
            if isinstance(content, dict):
                try:
                    content = json.dumps(content, default=str)
                except (TypeError, ValueError) as e:
                    logger.warning("semantic_iterator.json_serialization_failed", error=str(e))
                    content = str(content)
                    
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
            except Exception as e:
                logger.error("semantic_iterator.extraction_failed", error=str(e))
                return AgentResponse(
                    success=False,
                    data={},
                    error=f"Extraction failed: {str(e)}"
                )
            
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
            
            # Check if fast extraction succeeded
            if not self._state.iterator.has_items() and self._allow_fallback:
                logger.info("extraction.fallback_to_slow")
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

    def configure(self, content: Any, config: ExtractConfig) -> 'SemanticIterator':
        """
        Legacy configuration method for prefect compatibility.
        Sets up iterator state using configuration.
        """
        logger.warning("semantic_iterator.using_deprecated_configure")
        
        # Safely serialize the content
        content = self._serialize_content(content)
        
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