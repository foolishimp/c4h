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
        content_str_for_prompt = ""

        # Extract essential content with more aggressive pattern matching
        if isinstance(raw_input_content, dict):
            # Try to extract JSON array directly using regex pattern matching
            content_to_search = json.dumps(raw_input_content, indent=2)
            
            # Look for a JSON array pattern first
            array_match = re.search(r'\[\s*\{\s*"file_path"[^\]]*\}\s*\]', content_to_search, re.DOTALL)
            if array_match:
                content_str_for_prompt = array_match.group(0)
                logger.debug("fast_extractor.extracted_json_array", content_length=len(content_str_for_prompt))
            # Or look for a "changes" array
            elif '"changes"' in content_to_search:
                changes_match = re.search(r'"changes"\s*:\s*(\[\s*\{.*?\}\s*\])', content_to_search, re.DOTALL)
                if changes_match:
                    content_str_for_prompt = changes_match.group(1)
                    logger.debug("fast_extractor.extracted_changes_array", content_length=len(content_str_for_prompt))
            # Fall back to response field if available
            elif "response" in raw_input_content:
                response_content = raw_input_content.get("response", "")
                if isinstance(response_content, str):
                    content_str_for_prompt = response_content
                    logger.debug("fast_extractor.using_response_field", content_length=len(content_str_for_prompt))
            else:
                # Last resort: use the full content
                content_str_for_prompt = json.dumps(raw_input_content, indent=2)
                logger.debug("fast_extractor.using_full_content", content_length=len(content_str_for_prompt))
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
                
            try:
                # Check if content contains diff sections that need special handling
                has_diffs = isinstance(extracted_content, str) and "diff" in extracted_content and \
                           ("---" in extracted_content or "+++" in extracted_content)
                
                if has_diffs:
                    logger.info("fast_extraction.diff_content_detected")
                    try:
                        # Try special diff-aware parsing
                        items = self._parse_with_diff_handling(extracted_content)
                        if items:
                            logger.info("fast_extraction.diff_handling_successful", items_count=len(items))
                            # Normalize to list
                            if isinstance(items, dict):
                                items = [items]
                            elif not isinstance(items, list):
                                items = []
                                
                            logger.info("fast_extraction.complete", items_found=len(items))
                            return FastItemIterator(items)
                    except Exception as e:
                        logger.warning("fast_extraction.diff_handling_failed", error=str(e))
                        # Continue to regular parsing methods
                
                # Standard JSON parsing with more robust error handling
                if isinstance(extracted_content, str):
                    # Find specific problematic characters for debugging
                    try:
                        json.loads(extracted_content)
                    except json.JSONDecodeError as e:
                        problem_char = ord(extracted_content[e.pos]) if e.pos < len(extracted_content) else -1
                        logger.warning("fast_extraction.specific_char_issue", 
                                    position=e.pos, 
                                    char_code=problem_char,
                                    line=e.lineno, 
                                    column=e.colno)
                    
                    # Try to parse complete structure first
                    try:
                        items = json.loads(extracted_content)
                        logger.info("fast_extraction.standard_parse_successful")
                    except json.JSONDecodeError:
                        # Try to extract objects using regex approach
                        items = self._extract_json_objects(extracted_content)
                        if items:
                            logger.info("fast_extraction.object_extraction_successful", objects_found=len(items))
                        else:
                            # If that fails, fall back to extracting partial valid objects
                            items = self._extract_valid_objects(extracted_content)
                            if items:
                                logger.info("fast_extraction.partial_extraction_successful", objects_found=len(items))
                            else:
                                logger.error("fast_extraction.all_parsing_methods_failed")
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

    def _parse_with_diff_handling(self, content: str) -> List[Dict]:
        """Parse JSON content with special handling for diff sections"""
        # Try to parse as array first, strip potential surrounding text
        content = content.strip()
        
        # Try to find JSON array pattern
        array_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
        if array_match:
            content = array_match.group(0)
        
        # Extract each object separately to handle problematic diffs
        objects = []
        
        # Find the start position of each object in the array
        object_starts = [m.start() for m in re.finditer(r'\{\s*"file_path"|\{\s*"type"|\{\s*"description"', content)]
        
        if not object_starts:
            # Try different object pattern if none found
            object_starts = [m.start() for m in re.finditer(r'\s*\{', content)]
            if not object_starts:
                logger.warning("fast_extraction.no_object_boundaries_found")
                return self._extract_json_objects(content)
        
        # Add end of content as final boundary
        if content.rstrip().endswith(']'):
            # Remove the closing bracket temporarily
            content = content.rstrip()[:-1].rstrip()
            
        # Process each object
        for i in range(len(object_starts)):
            start = object_starts[i]
            end = object_starts[i+1] if i+1 < len(object_starts) else len(content)
            
            obj_text = content[start:end].strip()
            # Fix unclosed objects
            if not obj_text.endswith('}'):
                obj_text += '}'
            # Remove trailing comma if present
            if obj_text.endswith(',}'):
                obj_text = obj_text[:-2] + '}'
                
            # Handle diff section specially
            obj_text = self._escape_diff_section(obj_text)
            
            try:
                obj = json.loads(obj_text)
                objects.append(obj)
                logger.debug("fast_extraction.object_parsed", 
                          object_index=i, 
                          object_keys=list(obj.keys()) if isinstance(obj, dict) else None)
            except json.JSONDecodeError as e:
                logger.warning("fast_extraction.object_parse_error", 
                             index=i, 
                             error=str(e), 
                             object_start=start, 
                             object_length=len(obj_text))
                # Try aggressive repair of this object
                try:
                    # Extract known fields directly
                    file_path_match = re.search(r'"file_path"\s*:\s*"([^"]+)"', obj_text)
                    type_match = re.search(r'"type"\s*:\s*"([^"]+)"', obj_text)
                    desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', obj_text)
                    
                    # Extract diff section with raw handling
                    diff_match = re.search(r'"diff"\s*:\s*"(.+?)(?="}\s*$|",\s*")', obj_text, re.DOTALL)
                    
                    if file_path_match and type_match and diff_match:
                        repaired_obj = {
                            "file_path": file_path_match.group(1),
                            "type": type_match.group(1),
                            "description": desc_match.group(1) if desc_match else "No description",
                            "diff": diff_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                        }
                        objects.append(repaired_obj)
                        logger.info("fast_extraction.object_manually_repaired", index=i)
                except Exception as repair_error:
                    logger.error("fast_extraction.repair_failed", error=str(repair_error))
                
        return objects

    def _escape_diff_section(self, text: str) -> str:
        """Escape special characters in diff section properly"""
        if '"diff":' not in text:
            return text
            
        # Find diff section using regex with relaxed pattern
        diff_pattern = r'"diff"\s*:\s*"(.*?)(?="}\s*$|",\s*")'
        
        def escape_diff(match):
            diff_content = match.group(1)
            # Double escape backslashes first
            escaped = diff_content.replace('\\', '\\\\')
            # Then escape newlines and quotes
            escaped = escaped.replace('\n', '\\n').replace('"', '\\"')
            return f'"diff": "{escaped}'
        
        # Replace with properly escaped content
        try:
            fixed_text = re.sub(diff_pattern, escape_diff, text, flags=re.DOTALL)
            return fixed_text
        except Exception as e:
            logger.error("fast_extraction.diff_escape_failed", error=str(e))
            return text
            
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
        """Extract valid JSON objects from potentially malformed text with continuation metadata"""
        objects = []
        
        # Try to extract complete array first
        try:
            array_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
            if array_match:
                array_text = array_match.group(0)
                array = json.loads(array_text)
                if isinstance(array, list) and array:
                    logger.debug("fast_extraction.array_extracted", 
                            items=len(array))
                    return array
        except json.JSONDecodeError:
            pass
        
        # Try to extract changes object (solution designer format)
        try:
            changes_match = re.search(r'{\s*"changes"\s*:\s*\[\s*\{.*?\}\s*\]\s*}', text, re.DOTALL)
            if changes_match:
                changes_obj = json.loads(changes_match.group(0))
                if "changes" in changes_obj and isinstance(changes_obj["changes"], list):
                    logger.debug("fast_extraction.changes_extracted", 
                            items=len(changes_obj["changes"]))
                    return changes_obj["changes"]
        except json.JSONDecodeError:
            pass
        
        # Fallback to object-by-object extraction
        object_start = text.find('{')
        while object_start >= 0:
            object_end = self._find_matching_bracket(text, object_start)
            if object_end > object_start:
                try:
                    obj_text = text[object_start:object_end+1]
                    if '"diff"' in obj_text:
                        obj_text = self._escape_diff_section(obj_text)
                    obj = json.loads(obj_text)
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                object_start = text.find('{', object_end + 1)
            else:
                break
        
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