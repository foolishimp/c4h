# Path: /Users/jim/src/apps/c4h/c4h_agents/skills/semantic_iterator.py
"""
Semantic iterator with improved preprocessing for more reliable extraction.
Path: c4h_agents/skills/semantic_iterator.py
"""

from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
import json
import re
# Ensure locate_config is imported if needed, or rely on BaseAgent's config handling
from c4h_agents.config import locate_config 
from c4h_agents.agents.base_agent import BaseAgent, AgentResponse 
from c4h_agents.skills.shared.types import ExtractConfig
from c4h_agents.skills._semantic_fast import FastExtractor, FastItemIterator
from c4h_agents.skills._semantic_slow import SlowExtractor, SlowItemIterator
from enum import Enum
from c4h_agents.utils.logging import get_logger

# Use central logger
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
        # Pass positional parameters to BaseAgent
        super().__init__(config, "semantic_iterator")
        
        # The unique_name is now stored by BaseAgent
        
        # Get iterator-specific config using inherited method
        iterator_config = self._get_agent_config() # Gets llm_config.agents.semantic_iterator
        extractor_config = iterator_config.get('extractor_config', {})
        
        # Initialize extraction state using config values
        self._state = ExtractorState(
            mode=extractor_config.get('mode', 'fast'),
            position=0 # Reset position on init
        )
        
        # Configure extractors, passing the full agent config
        self._allow_fallback = extractor_config.get('allow_fallback', True)
        self._fast_extractor = FastExtractor(config=self.config) # Pass full config
        self._slow_extractor = SlowExtractor(config=self.config) # Pass full config
        
        # Use self.logger inherited from BaseAgent
        self.logger.info("semantic_iterator.initialized",
                   mode=self._state.mode,
                   allow_fallback=self._allow_fallback)

    def _get_agent_name(self) -> str:
        """Get agent name for config lookup"""
        return "semantic_iterator"

    def _preprocess_content(self, content: Any) -> str:
        """
        Preprocess content before extraction by handling JSON wrapping and normalizing escape sequences.
        """
        # Use self.logger inherited from BaseAgent
        logger_to_use = self.logger 
        try:
            logger_to_use.debug("preprocessing.start", 
                       content_type=type(content).__name__,
                       is_dict=isinstance(content, dict),
                       is_str=isinstance(content, str))
            
            # Handle if content is already a plain string (most common case after Coder fix)
            if isinstance(content, str):
                 # Check for JSON string wrapping
                 if content.strip().startswith('"') and content.strip().endswith('"'):
                      try:
                           maybe_decoded = json.loads(content)
                           if isinstance(maybe_decoded, str): # It was a JSON encoded string
                                content = maybe_decoded
                                logger_to_use.info("preprocessing.unwrapped_json_string")
                      except (json.JSONDecodeError, ValueError): pass # Keep original if not valid JSON string
                 # Normalize escapes if needed
                 if '\\\\' in content or '\\"' in content: # Simplified check for common issues
                      # Be careful with normalization, might corrupt valid content
                      # content = content.replace('\\\\', '\\').replace('\\"', '"') # Example - use cautiously
                      logger_to_use.debug("preprocessing.detected_escapes_in_string")
                 # Fall through to return the (potentially modified) string

            # Handle dictionary input (less likely now but keep for robustness)
            elif isinstance(content, dict):
                # Check standard wrapper keys first
                if "response" in content:
                    logger_to_use.info("preprocessing.unwrapping_dict_response_key")
                    content = content["response"] # Recurse/reprocess unwrapped content
                    return self._preprocess_content(content) # Re-run preprocessing on unwrapped part
                elif "content" in content:
                     logger_to_use.info("preprocessing.unwrapping_dict_content_key")
                     content = content["content"]
                     return self._preprocess_content(content)
                elif "llm_output" in content and isinstance(content["llm_output"], dict):
                     if "content" in content["llm_output"]:
                          logger_to_use.info("preprocessing.unwrapping_llm_output_content")
                          content = content["llm_output"]["content"]
                          return self._preprocess_content(content) # Re-run on extracted string
                else:
                     # If it's a dict but not a known wrapper, convert to JSON string
                     logger_to_use.debug("preprocessing.converting_unrecognized_dict_to_json")
                     content = json.dumps(content, indent=2)
            
            # Convert any other type to string as fallback
            if not isinstance(content, str):
                logger_to_use.debug("preprocessing.converting_final_to_string", original_type=type(content).__name__)
                content = str(content)
            
            # Detect standard change blocks in the final string
            change_count = content.count("===CHANGE_BEGIN===")
            if change_count > 0:
                logger_to_use.info("preprocessing.detected_change_blocks", count=change_count)
            
            logger_to_use.debug("preprocessing.complete", 
                        result_type=type(content).__name__, 
                        length=len(content) if hasattr(content, "__len__") else 0,
                        change_count=change_count)
            
            return content # Should be a string by now
        except Exception as e:
            logger_to_use.error("preprocessing.failed", error=str(e), exc_info=True)
            # Return original content converted to string on error
            return str(content)

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Process extraction request following standard agent interface"""
        # Use self.logger inherited from BaseAgent
        logger_to_use = self.logger
        try:
            # Get input data - expects string under 'input_data' after Coder fix
            content_input = context.get('input_data', context) # Prioritize 'input_data'
            logger_to_use.debug("iterator.process.received_input", input_type=type(content_input).__name__)

            # Apply preprocessing before extraction
            preprocessed_content = self._preprocess_content(content_input)
            
            # Get extraction parameters (these usually come from Agent config, not context)
            # Use inherited methods to get config values
            agent_config = self._get_agent_config() # Get config for 'semantic_iterator'
            instruction = agent_config.get('instruction', '') # Example if instruction was in config
            format_hint = agent_config.get('format', 'json')   # Example if format was in config

            # If instruction/format are expected in context instead:
            # instruction = context.get('instruction', agent_config.get('instruction', '')) 
            # format_hint = context.get('format', agent_config.get('format', 'json'))

            # --- Create ExtractConfig ---
            # Ensure ExtractConfig is created correctly
            # We might not actually need ExtractConfig if instruction/format are not used by iterator itself
            # If extractors use prompts from their *own* config, ExtractConfig might be redundant here.
            # Let's assume for now the config is primarily for mode selection.
            extract_config_obj = ExtractConfig(instruction=instruction, format=format_hint) # Create the object
            
            # Reset state for this process call
            self._state = ExtractorState(
                mode=self._state.mode, # Keep mode from init unless overridden
                content=preprocessed_content,
                config=extract_config_obj, # Store the config object
                position=0 # Reset position
            )
            
            results = []
            try:
                # Initialize the iterator by calling __iter__
                iterator_instance = self.__iter__()
                # Loop through the iterator instance
                for item in iterator_instance:
                    # NO_MORE_ITEMS check is usually handled by StopIteration
                    # if item == "NO_MORE_ITEMS": 
                    #     break 
                    results.append(item)
            except StopIteration:
                # This is the normal way iteration ends
                logger_to_use.debug("iterator.process.iteration_complete")
                pass 
            except Exception as iter_err:
                 # Catch other errors during iteration
                 logger_to_use.error("iterator.process.iteration_error", error=str(iter_err), exc_info=True)
                 return AgentResponse(
                     success=False,
                     data={},
                     error=f"Error during item extraction: {str(iter_err)}"
                 )

            # Check if any results were actually extracted
            if not results:
                 # Log why extraction might have failed (e.g., deterministic parse failed and LLM fallback also yielded nothing)
                 logger_to_use.warning("iterator.process.no_items_extracted", final_iterator_mode=self._state.mode)
                 return AgentResponse(
                     success=False, # Consider if this should be True if input was valid but empty
                     data={},
                     error="No items could be extracted from the input content."
                 )
            
            # Successfully extracted items
            logger_to_use.info("iterator.process.extraction_successful", count=len(results), format=format_hint)
            return AgentResponse(
                success=True,
                data={
                    "results": results,
                    "count": len(results),
                    "format": format_hint
                }
            )

        except Exception as e:
            logger_to_use.error("semantic_iterator.process_failed", error=str(e), exc_info=True) # Log traceback
            return AgentResponse(
                success=False,
                data={},
                error=str(e)
            )

    def __iter__(self) -> Iterator[Any]:
        """Initialize iteration in configured mode"""
        # Use self.logger inherited from BaseAgent
        logger_to_use = self.logger
        logger_to_use.debug("iterator.starting", mode=self._state.mode)
        
        if self._state.content is None or self._state.config is None:
             logger_to_use.error("iterator.not_configured")
             raise ValueError("Iterator not configured. Call process() first.")
        
        # Reset position for new iteration
        self._state.position = 0 
        
        if self._state.mode == ExtractionMode.FAST:
            # Try fast extraction first
            # Pass the config object to the extractor
            self._state.iterator = self._fast_extractor.create_iterator(
                self._state.content,
                self._state.config # Pass ExtractConfig object
            )
            
            # Check for completeness of fast extraction
            # Ensure iterator is not None before checking length
            fast_items_count = len(self._state.iterator) if self._state.iterator else 0
            
            # Estimate expected items by counting patterns in content
            if isinstance(self._state.content, str):
                expected_items = self._estimate_item_count(self._state.content)
                self._state.expected_items = expected_items
                
                # Determine if fast extraction was successful but incomplete
                # Ensure iterator exists before checking has_items
                fast_had_items = self._state.iterator and self._state.iterator.has_items()

                # Condition 1: Fast extractor ran, found *some* items, but fewer than expected -> Fallback
                if fast_had_items and fast_items_count < expected_items and self._allow_fallback:
                    logger_to_use.info("extraction.incomplete_fast_results", 
                              found=fast_items_count, 
                              expected=expected_items) 
                    
                    logger_to_use.info("extraction.fallback_to_slow", reason="incomplete_results")
                    self._state.mode = ExtractionMode.SLOW
                    # --- FIX: Explicitly initialize SlowItemIterator ---
                    self._state.iterator = self._slow_extractor.create_iterator(
                        self._state.content,
                        self._state.config # Pass ExtractConfig object
                    )
                    # --- END FIX ---
                
                # Condition 2: Fast extractor ran but found *no* items -> Fallback
                # Check has_items on the FastItemIterator specifically
                elif isinstance(self._state.iterator, FastItemIterator) and not self._state.iterator.has_items() and self._allow_fallback:
                     logger_to_use.info("extraction.fallback_to_slow", reason="no_fast_results")
                     self._state.mode = ExtractionMode.SLOW
                     self._state.iterator = self._slow_extractor.create_iterator(
                          self._state.content,
                          self._state.config # Pass ExtractConfig object
                     )
            # else: content wasn't string, can't estimate, proceed with fast results or lack thereof

        # If mode is explicitly SLOW or fallback occurred above
        if self._state.mode == ExtractionMode.SLOW:
             # Ensure slow iterator is initialized if not already done by fallback
             if not isinstance(self._state.iterator, SlowItemIterator):
                  logger_to_use.debug("iterator.initializing_slow_iterator", reason="Initial mode or failed fast init")
                  self._state.iterator = self._slow_extractor.create_iterator(
                       self._state.content,
                       self._state.config # Pass ExtractConfig object
                  )
            
        # Check if iterator initialization failed
        if self._state.iterator is None:
             logger_to_use.error("iterator.initialization_failed", mode=self._state.mode)
             # Set state to prevent __next__ from running with None iterator
             self._state.iterator = iter([]) # Empty iterator

        return self # Return self as the iterator


    def __next__(self) -> Any:
        """Get next item using the appropriate iterator"""
        # Use self.logger inherited from BaseAgent
        logger_to_use = self.logger
        try:
            if not self._state.iterator:
                logger_to_use.warning("iterator.next_called_without_iterator")
                raise StopIteration
                
            # Directly call next on the current iterator (could be Fast or Slow)
            next_item = next(self._state.iterator)
            
            # Increment position tracking *after* successfully getting an item
            self._state.position += 1
            logger_to_use.debug("iterator.yielded_item",
                        mode=self._state.mode,
                        position=self._state.position) # Log position *after* yielding
            return next_item
            
        except StopIteration:
            # This is the expected end of iteration
            logger_to_use.debug("iterator.complete", 
                        mode=self._state.mode,
                        position=self._state.position)
            raise # Re-raise StopIteration to signal end
        except Exception as e:
            # Log other unexpected errors during iteration
            logger_to_use.error("iterator.error",
                        error=str(e),
                        mode=self._state.mode,
                        position=self._state.position,
                        exc_info=True) # Include traceback
            raise StopIteration # Raise StopIteration to gracefully end iteration on error

    def _estimate_item_count(self, content: str) -> int:
        """Estimate number of items in content based on key patterns"""
        # Use self.logger inherited from BaseAgent
        logger_to_use = self.logger
        # Count ===CHANGE_BEGIN=== occurrences as most reliable estimate
        change_begin_count = content.count("===CHANGE_BEGIN===")
        change_end_count = content.count("===CHANGE_END===")
        
        # Also count file_path occurrences as alternate estimate (less reliable)
        # Using a simpler regex for file_path counting to avoid complexity
        file_path_count = len(re.findall(r"FILE:", content)) 
        
        # Use most reliable count, prioritizing change markers if balanced
        estimate = 0
        if change_begin_count > 0 and change_begin_count == change_end_count:
            estimate = change_begin_count
        elif change_begin_count > 0: # Use begin count if ends don't match
             estimate = change_begin_count
             logger_to_use.warning("iterator.marker_mismatch", begins=change_begin_count, ends=change_end_count)
        elif file_path_count > 0: # Fallback to FILE: count
             estimate = file_path_count
             logger_to_use.debug("iterator.estimating_with_file_key", estimate=estimate)
        # else: estimate remains 0
        
        logger_to_use.debug("iterator.estimated_items", 
                   estimate=estimate,
                   change_blocks=change_begin_count)
                   # Removed less reliable counts like change_type/diff
        
        return estimate
        
    # The deprecated 'configure' method has been removed.
    # Use 'process' with appropriate context instead.