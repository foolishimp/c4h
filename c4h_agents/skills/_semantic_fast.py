# Path: c4h_agents/skills/_semantic_fast.py
"""
Fast extraction mode implementation using standardized LLM response handling.
Refactored to prioritize deterministic parsing for known formats.
"""

from typing import List, Dict, Any, Optional, Iterator, Union
import json
import re
from c4h_agents.agents.base_agent import BaseAgent, AgentResponse
from skills.shared.types import ExtractConfig
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
            logger.debug("fast_iterator.exhausted")
            raise StopIteration
        item = self._items[self._position]
        self._position += 1
        logger.debug("fast_iterator.yielded_item", position=self._position - 1, item_preview=str(item)[:100])
        return item

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def has_items(self) -> bool:
        return bool(self._items)

class FastExtractor(BaseAgent):
    """
    Implements fast extraction mode.
    Prioritizes deterministic parsing for '===CHANGE_BEGIN===' format,
    falling back to LLM-based extraction if deterministic parsing fails.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config=config)
        fast_cfg = self._get_agent_config()
        logger.info("fast_extractor.initialized", settings=fast_cfg)

    def _get_agent_name(self) -> str:
        return "semantic_fast_extractor"

    def _format_request(self, context: Dict[str, Any]) -> str:
        """Format extraction request for LLM fallback mode"""
        if not context.get('config'):
            logger.error("fast_extractor.missing_config")
            raise ValueError("Extract config required")

        extract_template = self._get_prompt('extract')
        raw_input_content = context.get('content', '')

        if isinstance(raw_input_content, dict):
            content_str_for_prompt = json.dumps(raw_input_content, indent=2)
            logger.debug("fast_extractor_llm.using_json_content", content_length=len(content_str_for_prompt))
        else:
            content_str_for_prompt = str(raw_input_content)
            logger.debug("fast_extractor_llm.using_raw_input", content_length=len(content_str_for_prompt))

        try:
            final_prompt = extract_template.format(
                content=content_str_for_prompt,
                instruction=context['config'].instruction,
                format=context['config'].format
            )
            escape_handling_instructions = """
CRITICAL ESCAPE SEQUENCE HANDLING:
- Ensure all backslashes in diff content are properly double-escaped (\\\\)
- Ensure all quotes in diff content are properly escaped (\")
- When processing JSON with nested escape sequences, maintain exact character representation
"""
            final_prompt += escape_handling_instructions
            logger.debug("fast_extractor_llm.formatted_prompt", prompt_length=len(final_prompt))
            return final_prompt
        except KeyError as e:
            logger.error("fast_extractor_llm.format_key_error", error=str(e), missing_key=str(e))
            raise ValueError(f"Prompt template formatting failed. Missing key: {e}")
        except Exception as e:
            logger.error("fast_extractor_llm.format_failed", error=str(e))
            raise

    def _try_deterministic_parse(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Attempt deterministic parsing for '===CHANGE_BEGIN===' format"""
        if not isinstance(content, str):
            logger.debug("deterministic_parse.skipping", reason="Input is not a string", content_type=type(content).__name__)
            return None

        if "===CHANGE_BEGIN===" not in content:
            logger.debug("deterministic_parse.skipping", reason="No change markers found")
            return None

        logger.debug("deterministic_parse.attempting", content_length=len(content))
        parsed_items = []
        # Improved regex: More robust whitespace handling, non-greedy matching for fields,
        # and explicit handling of content until ===CHANGE_END===
        pattern = re.compile(
            r"===CHANGE_BEGIN===\s*"
            r"FILE:\s*(?P<file_path>[^\n]*?)\s*" # Match until newline
            r"TYPE:\s*(?P<type>[^\n]*?)\s*"       # Match until newline
            r"DESCRIPTION:\s*(?P<description>[^\n]*?)\s*" # Match until newline
            r"DIFF:\s*(?P<diff>.*?)\s*"
            r"===CHANGE_END===",
            re.DOTALL | re.MULTILINE
        )

        matches = list(pattern.finditer(content))
        logger.debug("deterministic_parse.matches_found", count=len(matches))

        for match in matches:
            data = match.groupdict()
            logger.debug("deterministic_parse.match_data", file_path=data.get('file_path'), type=data.get('type'))

            if not data.get('file_path'):
                logger.warning("deterministic_parse.missing_field", field="file_path", match_data=data)
                continue
            if not data.get('type'):
                logger.warning("deterministic_parse.missing_field", field="type", match_data=data)
                continue
            if data.get('diff') is None:
                logger.warning("deterministic_parse.missing_field", field="diff", match_data=data)
                continue

            item = {
                "file_path": data['file_path'].strip(),
                "type": data['type'].strip(),
                "description": data['description'].strip(),
                "diff": data['diff'].strip('\n')
            }
            parsed_items.append(item)
            logger.debug("deterministic_parse.item_extracted", file_path=item["file_path"])

        if parsed_items:
            logger.info("deterministic_parse.success", items_found=len(parsed_items))
        else:
            logger.warning("deterministic_parse.failed", reason="No valid items extracted")
        return parsed_items if parsed_items else None

    def create_iterator(self, content: Any, config: ExtractConfig) -> FastItemIterator:
        """Create iterator, trying deterministic parsing then LLM fallback"""
        logger.debug("fast_extractor.creating_iterator", content_type=type(content).__name__)

        deterministic_items = None
        if isinstance(content, str):
            try:
                deterministic_items = self._try_deterministic_parse(content)
            except Exception as e:
                logger.error("deterministic_parse.unexpected_error", error=str(e))

        if deterministic_items is not None:
            logger.info("fast_extractor.using_deterministic_results", items_count=len(deterministic_items))
            return FastItemIterator(deterministic_items)

        logger.info("fast_extractor.falling_back_to_llm")
        try:
            result = self.process({
                'content': content,
                'config': config
            })
            if not result.success:
                logger.warning("fast_extractor_llm.failed", error=result.error)
                return FastItemIterator([])

            extracted_content = self._get_llm_content(result.data.get('response'))
            if extracted_content is None:
                logger.error("fast_extractor_llm.no_content")
                return FastItemIterator([])

            logger.debug("fast_extractor_llm.response", response_preview=str(extracted_content)[:200])

            items = []
            try:
                if isinstance(extracted_content, str):
                    try:
                        parsed_content = json.loads(extracted_content)
                        logger.info("fast_extractor_llm.standard_parse_successful")
                        if isinstance(parsed_content, list):
                            items = parsed_content
                        elif isinstance(parsed_content, dict):
                            if "changes" in parsed_content and isinstance(parsed_content["changes"], list):
                                items = parsed_content["changes"]
                            else:
                                items = [parsed_content]
                    except json.JSONDecodeError:
                        logger.warning("fast_extractor_llm.json_parse_failed", error="Failed to parse JSON, trying simple object extraction")
                        items = self._extract_simple_objects(extracted_content)
                else:
                    items = extracted_content if isinstance(extracted_content, list) else [extracted_content]

                validated_items = []
                for item in items:
                    if isinstance(item, dict) and item.get("file_path"):
                        validated_items.append(item)
                    else:
                        logger.warning("fast_extractor_llm.invalid_item_skipped", item_preview=str(item)[:100])

                logger.info("fast_extractor_llm.complete", items_found=len(validated_items))
                return FastItemIterator(validated_items)
            except Exception as e:
                logger.error("fast_extractor_llm.parse_error", error=str(e))
                return FastItemIterator([])
        except Exception as e:
            logger.error("fast_extractor_llm_fallback.process_failed", error=str(e))
            return FastItemIterator([])

    def _extract_simple_objects(self, content: str) -> List[Dict]:
        """Fallback extraction for LLM response parsing failures"""
        objects = []
        try:
            object_pattern = r'\{\s*"file_path":.*?"diff":.*?\}(?=\s*,|\s*\])'
            matches = re.finditer(object_pattern, content, re.DOTALL)
            for match in matches:
                try:
                    obj_text = match.group(0)
                    obj = json.loads(obj_text)
                    if obj.get("file_path"):
                        objects.append(obj)
                except json.JSONDecodeError:
                    logger.debug("fast_extractor_llm.simple_json_object_parse_failed", text=obj_text[:100])

            if objects:
                logger.info("fast_extractor_llm.simple_json_extraction_successful", count=len(objects))
                return objects

            logger.warning("fast_extractor_llm.falling_back_to_field_regex")
            file_paths = re.findall(r'"file_path"\s*:\s*"([^"]+)"', content)
            types = re.findall(r'"type"\s*:\s*"([^"]+)"', content)
            descriptions = re.findall(r'"description"\s*:\s*"([^"]*)"', content)
            diff_pattern = r'"diff"\s*:\s*"(.*?)(?<!\\)"'
            diff_sections = [match.group(1).encode('utf-8').decode('unicode_escape') for match in re.finditer(diff_pattern, content, re.DOTALL)]

            count = len(file_paths)
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
