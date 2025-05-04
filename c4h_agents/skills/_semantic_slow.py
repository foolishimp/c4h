"""
Slow extraction mode with lazy LLM calls.
Path: c4h_agents/skills/_semantic_slow.py
"""

from typing import Dict, Any, Optional 
from c4h_agents.agents.base_agent import BaseAgent, AgentResponse 
from c4h_agents.skills.shared.types import ExtractConfig
import json
from c4h_agents.config import locate_config
from c4h_agents.utils.logging import get_logger

logger = get_logger()
class ExtractionError(Exception):
    """Custom exception for extraction errors"""
    pass

class ExtractionComplete(StopIteration):
    """Custom exception to signal clean completion of extraction"""
    pass

"""
Slow extraction mode with lazy LLM calls.
Path: c4h_agents/skills/_semantic_slow.py
"""

class SlowItemIterator:
    """Iterator for slow extraction results with lazy LLM calls"""
    
    def __init__(self, extractor: 'SlowExtractor', content: Any, config: ExtractConfig):
        """Initialize iterator with extraction parameters"""
        self._extractor = extractor
        self._content = content
        self._config = config
        self._position = 0
        self._exhausted = False
        self._max_attempts = 10 # Safety limit
        self._returned_items = set()  # Track returned items
        self._current_attempt = 0  # Track retries for current position

    def __iter__(self):
        return self

    def _get_content_key(self, content: Dict[str, Any]) -> str:
        """Get unique key for content based on full definition"""
        try:
            if isinstance(content, dict):
                # Sort the content for consistent string representation
                items = []
                for k in sorted(content.keys()):
                    items.append(f"{k}:{content[k]}")
                return "|".join(items)
            return str(content)
        except Exception as e:
            logger.error("iterator.key_generation_failed", error=str(e))
            return str(content)

    def __next__(self) -> Any:
        """Get next item using lazy extraction"""
        logger.debug("slow_iterator.next_called", 
                    position=self._position,
                    attempt=self._current_attempt,
                    max_attempts=self._max_attempts)

        if self._exhausted:
            logger.debug("slow_iterator.already_exhausted")
            raise StopIteration

        if self._position >= self._max_attempts:
            logger.warning("slow_iterator.max_attempts_reached",
                        max_attempts=self._max_attempts)
            self._exhausted = True
            raise StopIteration

        try:
            # Run extraction using the extractor instance
            agent_response = self._extractor.process({
                'content': self._content,
                'config': self._config,
                'position': self._position
            })

            if not agent_response.success:
                logger.error("slow_iterator.extraction_failed", 
                        error=agent_response.error,
                        position=self._position)
                self._exhausted = True
                raise StopIteration

            # Get response content from standard location
            content = agent_response.data.get('response')

            logger.debug("slow_iterator.response",
                        position=self._position,
                        content_type=type(content).__name__,
                        content_preview=str(content)[:100] if content else None)

            # Handle various empty/none cases
            if content is None or (isinstance(content, str) and not content.strip()):
                self._current_attempt += 1
                if self._current_attempt >= 3:  # Max retries per position
                    logger.error("slow_iterator.max_retries_for_position",
                            position=self._position)
                    self._exhausted = True
                    raise StopIteration
                logger.warning("slow_iterator.empty_response_retry",
                            position=self._position,
                            attempt=self._current_attempt)
                return next(self)  # Retry this position

            # Fix: Check for NO_MORE_ITEMS at the start of the response string
            if isinstance(content, str) and content.strip().upper().startswith("NO_MORE_ITEMS"):
                logger.info("slow_iterator.no_more_items", 
                        position=self._position)
                self._exhausted = True
                raise StopIteration

            # Try JSON parsing if string
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    content = parsed
                    logger.debug("slow_iterator.parsed_json",
                            position=self._position)
                except json.JSONDecodeError:
                    pass  # Keep as string if not JSON

            # Generate content key using full content
            content_key = self._get_content_key(content)
            
            # Check for duplicates using full content
            if content_key in self._returned_items:
                logger.warning("slow_iterator.duplicate_item",
                            position=self._position,
                            content_preview=str(content)[:100])
                self._current_attempt += 1
                if self._current_attempt >= 3:
                    self._exhausted = True
                    raise StopIteration
                return next(self)  # Try next position instead of stopping

            # Success - update state
            self._returned_items.add(content_key)
            self._position += 1
            self._current_attempt = 0  # Reset attempt counter for next position

            logger.info("slow_iterator.item_extracted",
                    position=self._position - 1,
                    item_type=type(content).__name__)

            return content

        except Exception as e:
            logger.error("slow_iterator.error", 
                        error=str(e),
                        position=self._position)
            self._exhausted = True
            raise StopIteration


class SlowExtractor(BaseAgent):
    """Implements slow extraction mode using iterative LLM queries"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with parent agent configuration"""
        # Pass positional parameters to BaseAgent
        super().__init__(config, "semantic_slow_extractor")
        
        # The unique_name is now stored by BaseAgent
        
        # Get our config section
        slow_cfg = locate_config(self.config or {}, self._get_agent_name())
        
        # Validate template at initialization
        template = self._get_prompt('extract')
        if '{ordinal}' not in template:
            logger.error("slow_extractor.invalid_template", 
                        reason="Missing {ordinal} placeholder")
            raise ValueError("Slow extractor requires {ordinal} in prompt template")

        logger.info("slow_extractor.initialized",
                   settings=slow_cfg)

    def _get_agent_name(self) -> str:
        return "semantic_slow_extractor"

    def _format_request(self, context: Dict[str, Any]) -> str:
        """Format extraction request for slow mode using config template"""
        if not context.get('config'):
            logger.error("slow_extractor.missing_config")
            raise ValueError("Extract config required")

        position = context.get('position', 0)
        ordinal = self._get_ordinal(position + 1)
        
        # Format instruction with ordinal
        instruction = context['config'].instruction.format(ordinal=ordinal)

        # Format complete request using template
        request = self._get_prompt('extract').format(
            ordinal=ordinal,
            content=context.get('content', ''),
            instruction=instruction,
            format=context['config'].format
        )

        logger.debug("slow_extractor.request",
                   position=position,
                   ordinal=ordinal,
                   request_length=len(request))

        return request

    @staticmethod
    def _get_ordinal(n: int) -> str:
        """Generate ordinal string for a number"""
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10 if n % 100 not in [11, 12, 13] else 0, 'th')
        return f"{n}{suffix}"

    def create_iterator(self, content: Any, config: ExtractConfig) -> SlowItemIterator:
        """Create iterator for slow extraction"""
        logger.debug("slow_extractor.creating_iterator",
                    content_type=type(content).__name__)
        return SlowItemIterator(self, content, config)