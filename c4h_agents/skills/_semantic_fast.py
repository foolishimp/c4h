# Path: c4h_agents/skills/_semantic_fast.py
"""
Fast extraction mode implementation using standardized LLM response handling.
Refactored to prioritize deterministic parsing for known formats.
"""

from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
import json
import re # Added for deterministic parsing
from c4h_agents.agents.base_agent import BaseAgent, AgentResponse
from skills.shared.types import ExtractConfig # Assuming this path is correct relative to execution
# from config import locate_config # locate_config might not be needed directly if config comes via BaseAgent
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
    """
    Implements fast extraction mode.
    Prioritizes deterministic parsing for '===CHANGE_BEGIN===' format,
    falling back to LLM-based extraction if deterministic parsing fails.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with parent agent configuration"""
        super().__init__(config=config)
        # locate_config is handled by BaseAgent's _get_agent_config now
        fast_cfg = self._get_agent_config()
        logger.info("fast_extractor.initialized", settings=fast_cfg)

    def _get_agent_name(self) -> str:
        """Get agent name for config lookup"""
        # Consistent with system_config.yml
        return "semantic_fast_extractor"

    def _format_request(self, context: Dict[str, Any]) -> str:
        """Format extraction request for LLM fallback mode"""
        # --- Essential Setup ---
        if not context.get('config'):
            logger.error("fast_extractor.missing_config")
            raise ValueError("Extract config required")

        # Get the prompt template (used for LLM fallback)
        # Ensure prompt is fetched correctly using BaseAgent's method
        extract_template = self._get_prompt('extract')

        # --- Extract Core Content ---
        raw_input_content = context.get('content', '')

        # Convert to string if needed
        if isinstance(raw_input_content, dict):
            content_str_for_prompt = json.dumps(raw_input_content, indent=2)
            logger.debug("fast_extractor_llm.using_json_content",
                       content_length=len(content_str_for_prompt))
        else:
            content_str_for_prompt = str(raw_input_content)
            logger.debug("fast_extractor_llm.using_raw_input")

        # --- Format Final Prompt ---
        try:
            # Using the prompt defined in system_config.yml
            final_prompt = extract_template.format(
                content=content_str_for_prompt,
                instruction=context['config'].instruction,
                format=context['config'].format
            )

            # Add escape sequence handling instructions (important for LLM)
            escape_handling_instructions = """
CRITICAL ESCAPE SEQUENCE HANDLING:
- Ensure all backslashes in diff content are properly double-escaped (\\\\)
- Ensure all quotes in diff content are properly escaped (\")
- When processing JSON with nested escape sequences, maintain exact character representation
"""
            final_prompt = final_prompt + escape_handling_instructions

            logger.debug("fast_extractor_llm.formatted_prompt",
                        prompt_length=len(final_prompt),
                        content_length=len(content_str_for_prompt))
            return final_prompt
        except KeyError as e:
            logger.error("fast_extractor_llm.format_key_error", error=str(e))
            raise ValueError(f"Prompt template formatting failed. Missing key: {e}")
        except Exception as e:
            logger.error("fast_extractor_llm.format_failed", error=str(e))
            raise

    def _try_deterministic_parse(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """
        Attempt to parse the input content using deterministic logic
        specifically for the '===CHANGE_BEGIN===' format.

        Args:
            content: The input string potentially containing change blocks.

        Returns:
            A list of parsed change dictionaries if successful and format matches,
            otherwise None.
        """
        if not isinstance(content, str) or "===CHANGE_BEGIN===" not in content:
            logger.debug("deterministic_parse.skipping", reason="Input not string or marker not found")
            return None

        logger.debug("deterministic_parse.attempting", content_length=len(content))
        parsed_items = []
        # Regex to capture blocks between markers, handling potential variations
        pattern = re.compile(
            r"===CHANGE_BEGIN===\s*"
            r"FILE:\s*(?P<file_path>.*?)\s*"
            r"TYPE:\s*(?P<type>.*?)\s*"
            r"DESCRIPTION:\s*(?P<description>.*?)\s*"
            r"DIFF:\s*(?P<diff>.*?)\s*"
            r"===CHANGE_END===",
            re.DOTALL | re.MULTILINE
        )

        matches_found = 0
        for match in pattern.finditer(content):
            matches_found += 1
            data = match.groupdict()
            # Basic validation
            if data.get('file_path') and data.get('type') and data.get('diff') is not None:
                # Strip leading/trailing whitespace from extracted fields
                item = {
                    "file_path": data['file_path'].strip(),
                    "type": data['type'].strip(),
                    "description": data['description'].strip(),
                    # Remove potential leading newline from diff that regex might capture
                    "diff": data['diff'].strip('\n')
                }
                # Ensure file_path is not empty after stripping
                if item["file_path"]:
                    parsed_items.append(item)
                    logger.debug("deterministic_parse.item_extracted", file_path=item["file_path"])
                else:
                    logger.warning("deterministic_parse.item_skipped", reason="Empty file_path after stripping", matched_data=data)
            else:
                logger.warning("deterministic_parse.item_skipped", reason="Missing required fields", matched_data=data)

        # Only return results if we actually found matches using this pattern
        if matches_found > 0:
             # Estimate expected count for logging comparison
            estimated_count = content.count("===CHANGE_BEGIN===")
            if len(parsed_items) == estimated_count:
                logger.info("deterministic_parse.success", items_found=len(parsed_items))
            else:
                 logger.warning("deterministic_parse.partial_success",
                                items_found=len(parsed_items),
                                expected_markers=estimated_count)
            return parsed_items
        else:
            logger.info("deterministic_parse.failed", reason="No matching blocks found")
            return None

    def create_iterator(self, content: Any, config: ExtractConfig) -> FastItemIterator:
        """
        Create iterator for fast extraction.
        Tries deterministic parsing first, falls back to LLM if needed.
        """
        logger.debug("fast_extractor.creating_iterator", content_type=type(content).__name__)

        # --- Attempt Deterministic Parsing First ---
        deterministic_items = None
        if isinstance(content, str): # Deterministic parsing only works on strings
            try:
                deterministic_items = self._try_deterministic_parse(content)
            except Exception as e:
                logger.error("deterministic_parse.unexpected_error", error=str(e))
                deterministic_items = None # Ensure fallback if deterministic parser errors

        if deterministic_items is not None:
            # Deterministic parsing succeeded (even if it found zero items, it means the pattern was attempted)
            logger.info("fast_extractor.using_deterministic_results", items_count=len(deterministic_items))
            return FastItemIterator(deterministic_items)

        # --- Fallback to LLM-Based Extraction ---
        logger.info("fast_extractor.falling_back_to_llm")
        try:
            # Use synchronous process method inherited from BaseAgent
            result = self.process({
                'content': content,
                'config': config
            })

            if not result.success:
                logger.warning("fast_extractor_llm.failed", error=result.error)
                return FastItemIterator([]) # Return empty iterator on LLM failure

            # Get response content using standardized helper
            extracted_content = self._get_llm_content(result.data.get('response'))
            if extracted_content is None:
                logger.error("fast_extractor_llm.no_content")
                return FastItemIterator([])

            # Attempt to parse the LLM output (expecting JSON)
            items = []
            try:
                if isinstance(extracted_content, str):
                    try:
                        parsed_content = json.loads(extracted_content)
                        logger.info("fast_extractor_llm.standard_parse_successful")

                        # Handle both array and object formats
                        if isinstance(parsed_content, list):
                            items = parsed_content
                        elif isinstance(parsed_content, dict):
                            # Check for changes array in dict
                            if "changes" in parsed_content and isinstance(parsed_content["changes"], list):
                                items = parsed_content["changes"]
                            else:
                                items = [parsed_content] # Wrap single object in list
                    except json.JSONDecodeError:
                        logger.warning("fast_extractor_llm.json_parse_failed",
                                    error="Failed to parse complete JSON from LLM, trying simple object extraction")
                        # Fall back to simpler regex extraction on the LLM output string
                        items = self._extract_simple_objects(extracted_content)
                        logger.info("fast_extractor_llm.simple_extraction_used", objects_found=len(items))
                else:
                    # Non-string content from LLM (less likely but handle)
                    items = extracted_content if isinstance(extracted_content, list) else [extracted_content]

                # Validate parsed objects from LLM
                validated_items = []
                for item in items:
                    if isinstance(item, dict) and item.get("file_path"): # Check non-empty file_path
                        validated_items.append(item)
                    else:
                        logger.warning("fast_extractor_llm.invalid_item_skipped", item=str(item)[:100])

                logger.info("fast_extractor_llm.complete", items_found=len(validated_items))
                return FastItemIterator(validated_items)

            except Exception as e:
                logger.error("fast_extractor_llm.parse_error", error=str(e))
                return FastItemIterator([]) # Return empty on parsing error

        except Exception as e:
            logger.error("fast_extractor_llm_fallback.process_failed", error=str(e))
            return FastItemIterator([]) # Return empty on general LLM processing error


    # --- Helper methods (mostly used by LLM fallback path) ---

    def _normalize_diff_escapes(self, diff_content: str) -> str:
        """Normalize escape sequences in diff content for reliable JSON parsing"""
        # This logic might be less critical if deterministic parsing succeeds,
        # but useful if LLM output needs cleaning.
        try:
            # Minimal normalization: Ensure standard JSON escapes are attempted
            escaped = diff_content.replace('\\', '\\\\')
            escaped = escaped.replace('"', '\\"')
            escaped = escaped.replace('\n', '\\n')
            escaped = escaped.replace('\r', '\\r')
            escaped = escaped.replace('\t', '\\t')
            return escaped
        except Exception:
            return diff_content # Return original on error

    def _count_potential_objects(self, content: str) -> List[int]:
        """Count potential objects based on standard patterns"""
        # Used for logging/comparison in the LLM fallback path
        file_path_count = len(re.findall(r'"file_path"', content))
        # Count distinct blocks as a better estimate for LLM fallback
        block_count = len(re.findall(r'\{\s*"file_path"', content))
        return list(range(max(file_path_count, block_count)))

    def _extract_simple_objects(self, content: str) -> List[Dict]:
        """Extract objects using simple patterns, falling back to core fields extraction.
           Used as a fallback for LLM JSON parsing failure."""
        objects = []
        try:
            # Try to find complete JSON objects first within the string
            object_pattern = r'\{\s*"file_path":.*?"diff":.*?\}(?=\s*,|\s*\])' # Non-greedy match until potential end
            matches = re.finditer(object_pattern, content, re.DOTALL)
            found_json_objects = False
            for match in matches:
                try:
                    obj_text = match.group(0)
                    obj = json.loads(obj_text)
                    if obj.get("file_path"): # Basic validation
                        objects.append(obj)
                        found_json_objects = True
                except json.JSONDecodeError:
                    logger.debug("fast_extractor_llm.simple_json_object_parse_failed", text=obj_text[:100])
                    continue # Try next match

            if found_json_objects:
                 logger.info("fast_extractor_llm.simple_json_extraction_successful", count=len(objects))
                 return objects

            # If JSON object parsing failed, fall back to field extraction (less reliable)
            logger.warning("fast_extractor_llm.falling_back_to_field_regex")
            file_paths = re.findall(r'"file_path"\s*:\s*"([^"]+)"', content)
            types = re.findall(r'"type"\s*:\s*"([^"]+)"', content)
            descriptions = re.findall(r'"description"\s*:\s*"([^"]*)"', content) # Allow empty description

            # Extract diff sections - more complex regex needed
            diff_sections = []
            # Try to capture content between "diff": "..."
            diff_pattern = r'"diff"\s*:\s*"(.*?)(?<!\\)"' # Match until unescaped quote
            matches = re.finditer(diff_pattern, content, re.DOTALL)
            for match in matches:
                diff_text = match.group(1)
                # Basic un-escaping for diff content extracted via regex
                try:
                    diff_text = diff_text.encode('utf-8').decode('unicode_escape')
                except Exception:
                    logger.warning("fast_extractor_llm.diff_unescape_failed", preview=diff_text[:50])
                    # Use raw if unescaping fails
                diff_sections.append(diff_text)

            # Assemble objects from extracted fields
            count = len(file_paths) # Base count on file_paths as it's mandatory
            objects = []
            for i in range(count):
                obj = {
                    "file_path": file_paths[i],
                    "type": types[i] if i < len(types) else "unknown",
                    "description": descriptions[i] if i < len(descriptions) else "No description",
                    "diff": diff_sections[i] if i < len(diff_sections) else ""
                }
                objects.append(obj)

            if objects:
                 logger.info("fast_extractor_llm.field_regex_extraction_successful", count=len(objects))
            else:
                 logger.warning("fast_extractor_llm.field_regex_extraction_failed")

            return objects
        except Exception as e:
            logger.error("fast_extractor_llm.simple_extraction_error", error=str(e))
            return []


    # No changes needed to _find_matching_bracket or other BaseAgent methods
    # _process_response, _get_llm_content etc are inherited from BaseAgent
    # and used primarily by the LLM fallback path within self.process()