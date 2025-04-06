"""
Fast extraction mode implementation using standardized LLM response handling.
Path: c4h_agents/skills/_semantic_fast.py
"""

from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
import json
import re
from c4h_agents.agents.base_agent import BaseAgent, AgentResponse 
from skills.shared.types import ExtractConfig
from config import locate_config
from c4h_agents.utils.logging import get_logger

logger = get_logger()

class FastItemIterator:
    """Iterator for fast extraction results with indexing support"""
    def __init__(self, items: List[Any]):
        self._items = items if items else []
        self._position = 0
        logger.debug("fast_iterator.initialized", items_count=len(self._items))

    def __iter__(self):
        return self

    def __next__(self):
        if self._position >= len(self._items):
            raise StopIteration
        item = self._items[self._position]
        self._position += 1
        return item

    def __len__(self):
        """Support length checking"""
        return len(self._items)

    def __getitem__(self, idx):
        """Support array-style access"""
        return self._items[idx]

    def has_items(self) -> bool:
        """Check if iterator has any items"""
        return bool(self._items)

class FastExtractor(BaseAgent):
    """Implements fast extraction mode using direct LLM parsing"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with parent agent configuration"""
        super().__init__(config=config)
        
        # Get our config section
        fast_cfg = locate_config(self.config or {}, self._get_agent_name())
        
        logger.info("fast_extractor.initialized",
                   settings=fast_cfg)

    def _get_agent_name(self) -> str:
        return "semantic_fast_extractor"

    def _format_request(self, context: Dict[str, Any]) -> str:
        """Format extraction request for fast mode with improved content extraction"""
        # --- Essential Setup ---
        if not context.get('config'):
            logger.error("fast_extractor.missing_config")
            raise ValueError("Extract config required")

        # Get the prompt template
        extract_template = self._get_prompt('extract')

        # --- Extract Core Content ---
        raw_input_content = context.get('content', '')
        
        # Convert to string if needed
        if isinstance(raw_input_content, dict):
            # Use the full content - let the LLM handle the extraction
            # This is more reliable than regex-based extraction
            content_str_for_prompt = json.dumps(raw_input_content, indent=2)
            logger.debug("fast_extractor.using_json_content", 
                       content_length=len(content_str_for_prompt))
        else:
            content_str_for_prompt = str(raw_input_content)
            logger.debug("fast_extractor.using_raw_input")

        # --- Format Final Prompt ---
        try:
            final_prompt = extract_template.format(
                content=content_str_for_prompt,
                instruction=context['config'].instruction,
                format=context['config'].format
            )
            
            # Add escape sequence handling instructions
            escape_handling_instructions = """
CRITICAL ESCAPE SEQUENCE HANDLING:
- Ensure all backslashes in diff content are properly double-escaped (\\\\)
- Ensure all quotes in diff content are properly escaped (\")
- When processing JSON with nested escape sequences, maintain exact character representation
"""
            final_prompt = final_prompt + escape_handling_instructions
            
            logger.debug("fast_extractor.formatted_prompt",
                        prompt_length=len(final_prompt),
                        content_length=len(content_str_for_prompt))
            return final_prompt
        except KeyError as e:
            logger.error("fast_extractor.format_key_error", error=str(e))
            raise ValueError(f"Prompt template formatting failed. Missing key: {e}")
        except Exception as e:
            logger.error("fast_extractor.format_failed", error=str(e))
            raise
        
    def create_iterator(self, content: Any, config: ExtractConfig) -> FastItemIterator:
        """Create iterator for fast extraction - synchronous interface"""
        try:
            logger.debug("fast_extractor.creating_iterator",
                        content_type=type(content).__name__)
                            
            # Use synchronous process instead of async
            result = self.process({
                'content': content,
                'config': config
            })

            if not result.success:
                logger.warning("fast_extraction.failed", error=result.error)
                return FastItemIterator([])

            # Get response content using standardized helper
            extracted_content = self._get_llm_content(result.data.get('response'))
            if extracted_content is None:
                logger.error("fast_extraction.no_content")
                return FastItemIterator([])
                
            # Attempt to parse the content
            items = []
            expected_count = 0
            try:
                # Use a simplified parsing approach
                # 1. Try to parse as complete JSON
                if isinstance(extracted_content, str):
                    try:
                        parsed_content = json.loads(extracted_content)
                        logger.info("fast_extraction.standard_parse_successful")
                        
                        # Handle both array and object formats
                        if isinstance(parsed_content, list):
                            items = parsed_content
                            expected_count = len(items)
                        elif isinstance(parsed_content, dict):
                            # Check for changes array in dict
                            if "changes" in parsed_content and isinstance(parsed_content["changes"], list):
                                items = parsed_content["changes"]
                                expected_count = len(items)
                            else:
                                items = [parsed_content]
                                expected_count = 1
                    except json.JSONDecodeError:
                        # Fall back to a simpler object extraction
                        logger.warning("fast_extraction.json_parse_failed", 
                                    error="Failed to parse complete JSON, trying extract_objects")
                        # Extract objects with the simpler method
                        simple_items = self._extract_simple_objects(extracted_content)
                        if simple_items:
                            items = simple_items
                            expected_count = len(self._count_potential_objects(extracted_content))
                            logger.info("fast_extraction.simple_extraction_successful", 
                                      objects_found=len(items),
                                      objects_expected=expected_count)
                else:
                    # Non-string content, just pass through
                    items = extracted_content if isinstance(extracted_content, list) else [extracted_content]
                    expected_count = len(items)
                    
                # Validate parsed objects
                validated_items = []
                for item in items:
                    if isinstance(item, dict) and "file_path" in item:
                        validated_items.append(item)
                    else:
                        logger.warning("fast_extraction.invalid_item_skipped", 
                                      item=str(item)[:100])
                
                # Log completion status
                if validated_items:
                    logger.info("fast_extraction.complete", 
                               items_found=len(validated_items),
                               items_expected=expected_count)
                    
                    # Check if we should fall back to slow extraction
                    if len(validated_items) < expected_count and self._allow_fallback:
                        logger.warning("fast_extraction.partial_results", 
                                     extracted=len(validated_items), 
                                     expected=expected_count)
                        # Don't fall back here - return what we have
                        # The semantic_iterator will handle fallback if needed
                else:
                    logger.warning("fast_extraction.no_valid_items")
                    
                return FastItemIterator(validated_items)

            except Exception as e:
                logger.error("fast_extraction.parse_error", error=str(e))
                return FastItemIterator([])

        except Exception as e:
            logger.error("fast_extraction.failed", error=str(e))
            return FastItemIterator([])

    def _normalize_diff_escapes(self, diff_content: str) -> str:
        """Normalize escape sequences in diff content for reliable JSON parsing"""
        # First level: normalize backslashes
        escaped = diff_content.replace('\\', '\\\\')
        # Second level: escape quotes
        escaped = escaped.replace('"', '\\"')
        return escaped
    
    def _count_potential_objects(self, content: str) -> List[int]:
        """Count potential objects based on standard patterns"""
        file_path_count = len(re.findall(r'"file_path"', content))
        type_count = len(re.findall(r'"type"\s*:', content)) 
        diff_count = len(re.findall(r'"diff"\s*:', content))
        
        # Return positions of potential objects
        return list(range(max(file_path_count, type_count, diff_count)))
        
    def _extract_simple_objects(self, content: str) -> List[Dict]:
        """Extract objects using simple patterns, falling back to core fields extraction"""
        objects = []
        
        # Try to find complete objects first
        object_pattern = r'\{\s*"file_path".*?"\s*\}'
        matches = re.finditer(object_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                obj_text = match.group(0)
                # Try to parse as JSON
                obj = json.loads(obj_text)
                if "file_path" in obj:
                    objects.append(obj)
                    logger.debug("fast_extraction.object_extracted")
            except json.JSONDecodeError:
                logger.debug("fast_extraction.object_parse_failed")
        
        # If we found objects, return them
        if objects:
            return objects
            
        # Fallback: extract core fields individually
        file_paths = re.findall(r'"file_path"\s*:\s*"([^"]+)"', content)
        types = re.findall(r'"type"\s*:\s*"([^"]+)"', content)
        descriptions = re.findall(r'"description"\s*:\s*"([^"]+)"', content)
        
        # Extract diff sections with careful handling
        diff_sections = []
        diff_pattern = r'"diff"\s*:\s*"(.*?)(?="}\s*$|",\s*")'
        matches = re.finditer(diff_pattern, content, re.DOTALL)
        for match in matches:
            diff_text = match.group(1)
            # Un-escape the diff content
            diff_text = diff_text.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            diff_sections.append(diff_text)
        
        # Assemble objects from extracted fields
        count = min(len(file_paths), len(types))
        for i in range(count):
            obj = {
                "file_path": file_paths[i],
                "type": types[i],
                "description": descriptions[i] if i < len(descriptions) else "No description",
                "diff": diff_sections[i] if i < len(diff_sections) else ""
            }
            objects.append(obj)
            
        return objects

    def _find_matching_bracket(self, text: str, start_pos: int, 
                              open_char: str = '{', close_char: str = '}') -> int:
        """Find the matching closing bracket position for a given opening bracket"""
        stack = []
        for i in range(start_pos, len(text)):
            if text[i] == open_char:
                stack.append(i)
            elif text[i] == close_char:
                if stack:
                    stack.pop()
                    if not stack:
                        return i  # This is the matching closing bracket
        return -1  # No matching bracket found