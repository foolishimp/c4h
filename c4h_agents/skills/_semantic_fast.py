# Path: c4h_agents/skills/_semantic_fast.py
"""
Fast extraction mode implementation using standardized LLM response handling.
Refactored to prioritize deterministic parsing for known formats.
"""

from typing import List, Dict, Any, Optional, Iterator, Union
import json
import re
from c4h_agents.agents.base_agent import BaseAgent, AgentResponse
from c4h_agents.skills.shared.types import ExtractConfig
from c4h_agents.utils.logging import get_logger

# Create a global logger for the non-class code
logger = get_logger()

class FastItemIterator:
    """Iterator for fast extraction results with indexing support"""
    def __init__(self, items: List[Any], parent_logger=None):
        self._items = items if items else []
        self._position = 0
        # Use logger from parent if available, otherwise use module-level logger
        self.logger = parent_logger or logger
        self.logger.debug("fast_iterator.initialized", items_count=len(self._items))

    def __iter__(self):
        return self

    def __next__(self):
        if self._position >= len(self._items):
            self.logger.debug("fast_iterator.exhausted")
            raise StopIteration
        item = self._items[self._position]
        self._position += 1
        self.logger.debug("fast_iterator.yielded_item", position=self._position - 1, item_preview=str(item)[:100])
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
        # Create a customized config if needed to prevent errors
        if config is None:
            config = {}
        
        # Clone our configuration to avoid modifying the original
        patched_config = dict(config)
        
        # Ensure the fast extractor has a persona key in the config 
        if 'llm_config' not in patched_config:
            patched_config['llm_config'] = {}
        if 'agents' not in patched_config['llm_config']:
            patched_config['llm_config']['agents'] = {}
        if 'semantic_fast_extractor' not in patched_config['llm_config']['agents']:
            patched_config['llm_config']['agents']['semantic_fast_extractor'] = {}
        if 'persona_key' not in patched_config['llm_config']['agents']['semantic_fast_extractor']:
            # Use a default persona that should exist in config
            parent_persona = config.get('llm_config', {}).get('agents', {}).get('semantic_iterator', {}).get('persona_key')
            patched_config['llm_config']['agents']['semantic_fast_extractor']['persona_key'] = parent_persona or "discovery_v1"
            
        # Pass positional parameters to BaseAgent
        super().__init__(patched_config, "semantic_fast_extractor")
        
        # The unique_name is now stored by BaseAgent
        fast_cfg = self._get_agent_config()
        self.logger.info("fast_extractor.initialized", settings=fast_cfg)

    def _get_agent_name(self) -> str:
        return "semantic_fast_extractor"

    def _format_request(self, context: Dict[str, Any]) -> str:
        """Format extraction request for LLM fallback mode"""
        if not context.get('config'):
            self.logger.error("fast_extractor.missing_config")
            raise ValueError("Extract config required")

        extract_template = self._get_prompt('extract')
        raw_input_content = context.get('content', '')

        if isinstance(raw_input_content, dict):
            content_str_for_prompt = json.dumps(raw_input_content, indent=2)
            self.logger.debug("fast_extractor_llm.using_json_content", content_length=len(content_str_for_prompt))
        else:
            content_str_for_prompt = str(raw_input_content)
            self.logger.debug("fast_extractor_llm.using_raw_input", content_length=len(content_str_for_prompt))

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
            self.logger.debug("fast_extractor_llm.formatted_prompt", prompt_length=len(final_prompt))
            return final_prompt
        except KeyError as e:
            self.logger.error("fast_extractor_llm.format_key_error", error=str(e), missing_key=str(e))
            raise ValueError(f"Prompt template formatting failed. Missing key: {e}")
        except Exception as e:
            self.logger.error("fast_extractor_llm.format_failed", error=str(e))
            raise

    def _try_deterministic_parse(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Attempt deterministic parsing for '===CHANGE_BEGIN===' format"""
        if not isinstance(content, str):
            self.logger.debug("deterministic_parse.skipping", reason="Input is not a string", content_type=type(content).__name__)
            return None

        if "===CHANGE_BEGIN===" not in content:
            self.logger.debug("deterministic_parse.skipping", reason="No change markers found")
            return None

        # --- ADDED: Log the exact content being parsed ---
        self.logger.debug("deterministic_parse.attempting", content_length=len(content), content_to_parse=content)
        # --- END ADDED ---

        parsed_items = []
        # Refined regex again: More explicit boundaries, especially for DIFF content
        pattern = re.compile(
            r"===CHANGE_BEGIN===\s*"
            r"FILE:(?P<file_path>.*?)\n"                # Capture non-greedily until newline
            r"TYPE:(?P<type>.*?)\n"                  # Capture non-greedily until newline
            r"DESCRIPTION:(?P<description>.*?)\n"      # Capture non-greedily until newline
            r"DIFF:\s*\n(?P<diff>.*?)\n?"               # Capture DIFF content after newline, non-greedily
            r"===CHANGE_END===",
            # Using DOTALL allows '.' to match newlines within the DIFF section
            re.DOTALL | re.MULTILINE
        )

        matches = list(pattern.finditer(content))
        self.logger.debug("deterministic_parse.matches_found", count=len(matches))

        for match in matches:
            data = match.groupdict()
            self.logger.debug("deterministic_parse.match_data", file_path=data.get('file_path'), type=data.get('type'))

            if not data.get('file_path'):
                self.logger.warning("deterministic_parse.missing_field", field="file_path", match_data=data)
                continue
            if not data.get('type'):
                self.logger.warning("deterministic_parse.missing_field", field="type", match_data=data)
                continue
            if data.get('diff') is None:
                # Allow empty diffs, but log a warning if it was truly missing vs empty
                if "DIFF:" not in match.group(0).split("DESCRIPTION:", 1)[1]:
                     self.logger.warning("deterministic_parse.missing_field", field="diff", match_data=data)
                # If DIFF: is present but content is empty, that's okay (e.g., delete file)
                data['diff'] = '' # Ensure key exists even if empty

            item = {
                "file_path": data['file_path'].strip(),
                "type": data['type'].strip(),
                "description": data['description'].strip(),
                # Strip only leading/trailing whitespace/newlines from the diff itself
                "diff": data['diff'].strip()
            }
            parsed_items.append(item)
            self.logger.debug("deterministic_parse.item_extracted", file_path=item["file_path"])

        if parsed_items:
            self.logger.info("deterministic_parse.success", items_found=len(parsed_items))
        else:
            self.logger.warning("deterministic_parse.failed", reason="No valid items extracted")
        return parsed_items if parsed_items else None

    def create_iterator(self, content: Any, config: ExtractConfig) -> FastItemIterator:
        """Create iterator, trying deterministic parsing then LLM fallback"""
        self.logger.debug("fast_extractor.creating_iterator", content_type=type(content).__name__)

        deterministic_items = None
        if isinstance(content, str):
            try:
                deterministic_items = self._try_deterministic_parse(content)
            except Exception as e:
                self.logger.error("deterministic_parse.unexpected_error", error=str(e))

        if deterministic_items is not None:
            self.logger.info("fast_extractor.using_deterministic_results", items_count=len(deterministic_items))
            return FastItemIterator(deterministic_items, self.logger)

        self.logger.info("fast_extractor.falling_back_to_llm")
        try:
            result = self.process({
                'content': content,
                'config': config
            })
            if not result.success:
                self.logger.warning("fast_extractor_llm.failed", error=result.error)
                return FastItemIterator([], self.logger)

            extracted_content = self._get_llm_content(result.data.get('response'))
            if extracted_content is None:
                self.logger.error("fast_extractor_llm.no_content")
                return FastItemIterator([], self.logger)

            self.logger.debug("fast_extractor_llm.response", response_preview=str(extracted_content)[:200])

            items = []
            try:
                if isinstance(extracted_content, str):
                    try:
                        parsed_content = json.loads(extracted_content)
                        self.logger.info("fast_extractor_llm.standard_parse_successful")
                        if isinstance(parsed_content, list):
                            items = parsed_content
                        elif isinstance(parsed_content, dict):
                            if "changes" in parsed_content and isinstance(parsed_content["changes"], list):
                                items = parsed_content["changes"]
                            else:
                                items = [parsed_content]
                    except json.JSONDecodeError:
                        self.logger.warning("fast_extractor_llm.json_parse_failed", error="Failed to parse JSON, trying simple object extraction")
                        items = self._extract_simple_objects(extracted_content)
                else:
                    items = extracted_content if isinstance(extracted_content, list) else [extracted_content]

                validated_items = []
                for item in items:
                    if isinstance(item, dict) and item.get("file_path"):
                        validated_items.append(item)
                    else:
                        self.logger.warning("fast_extractor_llm.invalid_item_skipped", item_preview=str(item)[:100])

                self.logger.info("fast_extractor_llm.complete", items_found=len(validated_items))
                return FastItemIterator(validated_items, self.logger)
            except Exception as e:
                self.logger.error("fast_extractor_llm.parse_error", error=str(e))
                return FastItemIterator([], self.logger)
        except Exception as e:
            self.logger.error("fast_extractor_llm_fallback.process_failed", error=str(e))
            return FastItemIterator([], self.logger)

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
                    self.logger.debug("fast_extractor_llm.simple_json_object_parse_failed", text=obj_text[:100])

            if objects:
                self.logger.info("fast_extractor_llm.simple_json_extraction_successful", count=len(objects))
                return objects

            self.logger.warning("fast_extractor_llm.falling_back_to_field_regex")
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
                self.logger.info("fast_extractor_llm.field_regex_extraction_successful", count=len(objects))
            else:
                self.logger.warning("fast_extractor_llm.field_regex_extraction_failed")
            return objects
        except Exception as e:
            self.logger.error("fast_extractor_llm.simple_extraction_error", error=str(e))
            return []