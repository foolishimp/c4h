"""
Fast extraction mode implementation using standardized LLM response handling.
Path: c4h_agents/skills/_semantic_fast.py
"""

from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
import json
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
        """Format extraction request for fast mode using config template"""
        if not context.get('config'):
            logger.error("fast_extractor.missing_config")
            raise ValueError("Extract config required")

        extract_template = self._get_prompt('extract')
        return extract_template.format(
            content=context.get('content', ''),
            instruction=context['config'].instruction,
            format=context['config'].format
        )

    def _safely_serialize_content(self, content: Any) -> Any:
        """Safely convert ModelResponse and other objects to serializable form"""
        # If it's a ModelResponse object, extract the actual content
        if hasattr(content, "choices") and hasattr(content.choices[0], "message"):
            try:
                return content.choices[0].message.content
            except (AttributeError, IndexError):
                logger.warning("fast_extractor.content_extraction_failed")
                return str(content)
        
        # If it's a dictionary with raw_output or raw_response
        if isinstance(content, dict):
            for key in ["raw_output", "raw_response"]:
                if key in content and not isinstance(content[key], (str, int, float, bool, list, dict, type(None))):
                    # Convert non-serializable objects to strings
                    content[key] = str(content[key])
            return content
        
        # Default case - return as is
        return content

    def create_iterator(self, content: Any, config: ExtractConfig) -> FastItemIterator:
        """Create iterator for fast extraction - synchronous interface"""
        try:
            # Safely handle potential ModelResponse objects
            content = self._safely_serialize_content(content)
            
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
                
            try:
                # Parse JSON with more robust error handling
                if isinstance(extracted_content, str):
                    # Find the specific offending character for debugging
                    try:
                        json.loads(extracted_content)
                    except json.JSONDecodeError as e:
                        problem_char = ord(extracted_content[e.pos]) if e.pos < len(extracted_content) else -1
                        logger.warning("fast_extraction.specific_char_issue", 
                                    position=e.pos, 
                                    char_code=problem_char,
                                    line=e.lineno, 
                                    column=e.colno)
                    
                    # More aggressive sanitization to handle ALL control and non-ASCII characters
                    # Only keep printable ASCII (32-126) plus basic whitespace
                    sanitized_content = ""
                    for i, ch in enumerate(extracted_content):
                        # Keep only safe characters: printable ASCII or basic whitespace
                        if (32 <= ord(ch) <= 126) or ch in '\n\r\t':
                            sanitized_content += ch
                        else:
                            # Replace with space and log the specific character that was removed
                            sanitized_content += ' '
                            if i >= 14720 and i <= 14740:  # Log only near the problematic area
                                logger.debug("fast_extraction.removed_char", 
                                        position=i, 
                                        char_code=ord(ch))
                    
                    try:
                        items = json.loads(sanitized_content)
                        logger.info("fast_extraction.aggressive_sanitization_successful")
                    except json.JSONDecodeError as e:
                        # Try extracting partial valid JSON
                        try:
                            # For this specific case, try to directly cut the problem area
                            # Assuming the start is valid JSON
                            problem_area = max(0, e.pos - 100)
                            before_problem = sanitized_content[:problem_area]
                            after_problem = sanitized_content[e.pos + 100:]
                            
                            # Look for valid structural elements
                            if before_problem.count('[') > before_problem.count(']'):
                                # We're in an array, try to find a valid ]
                                if ']' in after_problem:
                                    end_pos = after_problem.find(']') + len(before_problem) + 100
                                    patched_content = sanitized_content[:end_pos+1]
                                    items = json.loads(patched_content)
                                    logger.info("fast_extraction.array_patched_successfully", 
                                            original_len=len(sanitized_content),
                                            patched_len=len(patched_content))
                                else:
                                    return FastItemIterator([])
                            else:
                                # Try to extract valid objects
                                objects = self._extract_valid_objects(sanitized_content)
                                if objects:
                                    items = objects
                                else:
                                    return FastItemIterator([])
                        except Exception as recovery_err:
                            logger.error("fast_extraction.recovery_failed", error=str(recovery_err))
                            # Fall back to partial JSON extraction as last resort
                            try:
                                # Look for valid JSON objects using regex
                                import re
                                # Find objects between { and }
                                object_pattern = re.compile(r'\{[^{}]*(\{[^{}]*\}[^{}]*)*\}')
                                objects = [json.loads(m.group(0)) for m in object_pattern.finditer(sanitized_content)]
                                
                                # Find arrays between [ and ]
                                array_pattern = re.compile(r'\[[^\[\]]*(\[[^\[\]]*\][^\[\]]*)*\]')
                                arrays = [json.loads(m.group(0)) for m in array_pattern.finditer(sanitized_content)]
                                
                                if objects:
                                    items = objects
                                    logger.info("fast_extraction.regex_extracted_objects", 
                                            count=len(objects))
                                elif arrays:
                                    array = arrays[0]
                                    if isinstance(array, list):
                                        items = array
                                        logger.info("fast_extraction.regex_extracted_array", 
                                                count=len(array))
                                    else:
                                        items = [array]
                                else:
                                    return FastItemIterator([])
                            except Exception:
                                logger.error("fast_extraction.all_recovery_methods_failed")
                                return FastItemIterator([])
                else:
                    items = extracted_content
                    
                # Normalize to list
                if isinstance(items, dict):
                    items = [items]
                elif not isinstance(items, list):
                    items = []
                    
                logger.info("fast_extraction.complete", items_found=len(items))
                return FastItemIterator(items)

            except json.JSONDecodeError as e:
                logger.error("fast_extraction.parse_error", error=str(e))
                return FastItemIterator([])

        except Exception as e:
            logger.error("fast_extraction.failed", error=str(e))
            return FastItemIterator([])
            
    def _extract_valid_objects(self, content: str) -> List[Dict]:
        """Extract valid JSON objects even from malformed JSON"""
        objects = []
        start_idx = 0
        
        while start_idx < len(content):
            # Find opening braces
            obj_start = content.find('{', start_idx)
            arr_start = content.find('[', start_idx)
            
            # Determine which comes first
            if obj_start < 0 and arr_start < 0:
                break  # No more JSON structures
            
            if (obj_start >= 0 and arr_start >= 0 and obj_start < arr_start) or arr_start < 0:
                # Object starts first
                start_pos = obj_start
                for end_pos in range(start_pos + 1, len(content)):
                    # Try parsing this substring
                    try:
                        obj = json.loads(content[start_pos:end_pos+1])
                        objects.append(obj)
                        start_idx = end_pos + 1
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    start_idx = len(content)  # No valid object found
                    
            else:
                # Array starts first
                start_pos = arr_start
                for end_pos in range(start_pos + 1, len(content)):
                    # Try parsing this substring
                    try:
                        arr = json.loads(content[start_pos:end_pos+1])
                        if isinstance(arr, list):
                            objects.extend(arr)
                        else:
                            objects.append(arr)
                        start_idx = end_pos + 1
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    start_idx = len(content)  # No valid array found
        
        return objects
            
    def _extract_json_objects(self, text: str) -> List[Dict]:
        """Extract valid JSON objects from potentially malformed text"""
        objects = []
        # Look for objects that start with { and end with }
        object_start = text.find('{')
        while object_start >= 0:
            # Find the corresponding closing brace
            object_end = self._find_matching_bracket(text, object_start)
            if object_end > object_start:
                # Try to parse this segment as JSON
                try:
                    obj_text = text[object_start:object_end+1]
                    obj = json.loads(obj_text)
                    objects.append(obj)
                    logger.debug("fast_extraction.object_extracted", 
                               start=object_start,
                               end=object_end,
                               length=len(obj_text))
                except json.JSONDecodeError:
                    # Not valid JSON, skip this segment
                    pass
                
                # Move to the next potential object
                object_start = text.find('{', object_end + 1)
            else:
                # No valid closing bracket found
                break
                
        # Look for arrays that start with [ and end with ]
        array_start = text.find('[')
        if array_start >= 0:
            array_end = self._find_matching_bracket(text, array_start, open_char='[', close_char=']')
            if array_end > array_start:
                try:
                    array_text = text[array_start:array_end+1]
                    array = json.loads(array_text)
                    if isinstance(array, list) and array:
                        # If we found a valid array, return its elements
                        objects.extend(array)
                        logger.debug("fast_extraction.array_extracted",
                                   start=array_start,
                                   end=array_end,
                                   items=len(array))
                except json.JSONDecodeError:
                    # Not valid JSON, ignore
                    pass
                    
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