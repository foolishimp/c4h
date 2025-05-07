"""
Primary coder agent implementation using semantic extraction.
Handles iterating through change blocks provided by SolutionDesigner
and applying them using AssetManager.

Path: c4h_agents/agents/coder.py
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import re

from c4h_agents.agents.base_agent import BaseAgent, AgentResponse
from c4h_agents.agents.types import SkillResult
from c4h_agents.skills.semantic_merge import SemanticMerge
from c4h_agents.skills.semantic_iterator import SemanticIterator
from c4h_agents.skills.asset_manager import AssetManager
from c4h_agents.utils.logging import get_logger

# Use central logger
logger = get_logger()

@dataclass
class CoderMetrics:
    """Detailed metrics for code processing operations"""
    total_changes_processed: int = 0
    successful_changes: int = 0
    failed_changes: int = 0
    start_time: str = ""
    end_time: str = ""
    processing_time: float = 0.0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to plain dictionary for serialization"""
        return asdict(self)

class Coder(BaseAgent):
    """Handles code modifications using semantic extraction and asset management"""

    def __init__(self, full_effective_config: Dict[str, Any], unique_name: str = "coder"):
        """Initialize coder with configuration"""
        super().__init__(full_effective_config, unique_name)

        # Use self.logger inherited from BaseAgent after super().__init__()
        # Bind logger immediately for consistent context
        self.logger = self.logger.bind(agent_name=self._get_agent_name(), agent_type="Coder")

        # Get agent specific config
        coder_config = self._get_agent_config()

        # Determine backup settings from the full config
        backup_config = self.config_node.get_value("backup") or {}
        if not backup_config:
            runtime_config = self.config_node.get_value("runtime") or {}
            backup_config = runtime_config.get("backup", {})

        backup_enabled_val = backup_config.get("enabled", True)
        # Robust boolean conversion
        backup_enabled = str(backup_enabled_val).lower() == 'true' if isinstance(backup_enabled_val, str) else bool(backup_enabled_val)
        backup_path_str = backup_config.get("path", "workspaces/backups")

        # Resolve backup path
        backup_path = Path(backup_path_str)
        if not backup_path.is_absolute():
            project_path = self.config_node.get_value("project.path")
            if project_path:
                backup_path = Path(project_path) / backup_path
            else:
                backup_path = Path.cwd() / backup_path

        # Setup skills through _invoke_skill mechanism instead of direct instantiation
        # This ensures consistent skill management

        # Initialize metrics
        self.operation_metrics = CoderMetrics(start_time=datetime.now(timezone.utc).isoformat())
        self.logger.info("initialized", backup_path=str(backup_path.resolve()), backup_enabled=backup_enabled)

    def _get_agent_name(self) -> str:
        """Provide agent name for config/logging."""
        return self.unique_name

    def _parse_raw_change_block(self, raw_block: str) -> Optional[Dict[str, Any]]:
        """
        Parses a raw text block (including markers) into a structured dictionary.
        Needed when SemanticIterator falls back and yields raw text.
        """
        logger_to_use = self.logger
        logger_to_use.debug("parsing_raw_change_block", block_length=len(raw_block))
        if not isinstance(raw_block, str) or not raw_block.strip().startswith("===CHANGE_BEGIN==="):
            logger_to_use.warning("invalid_raw_block_format", block_preview=raw_block[:100])
            return None

        # Basic parsing using regex (similar to FastExtractor's logic)
        pattern = re.compile(
            r"FILE:(?P<file_path>.*?)\n"
            r"TYPE:(?P<type>.*?)\n"
            r"DESCRIPTION:(?P<description>.*?)\n"
            r"DIFF:\s*\n(?P<diff>.*?)\n?$", # Match DIFF content until the end (before ===CHANGE_END===)
            re.DOTALL | re.MULTILINE
        )

        # Remove markers before matching
        content_inside_markers = raw_block.strip()
        if "===CHANGE_BEGIN===" in content_inside_markers and "===CHANGE_END===" in content_inside_markers:
            content_inside_markers = content_inside_markers[content_inside_markers.find("===CHANGE_BEGIN===") + len("===CHANGE_BEGIN==="):content_inside_markers.rfind("===CHANGE_END===")].strip()
        else:
            # Just try to work with what we have
            content_inside_markers = content_inside_markers.replace("===CHANGE_BEGIN===", "").replace("===CHANGE_END===", "").strip()

        match = pattern.search(content_inside_markers)
        if not match:
            logger_to_use.error("failed_to_parse_raw_block", block_content=content_inside_markers[:200])
            return None

        data = match.groupdict()
        parsed_item = {
            "file_path": data.get('file_path', '').strip(),
            "type": data.get('type', '').strip(),
            "description": data.get('description', '').strip(),
            "diff": data.get('diff', '').strip() # Strip whitespace from diff edges
        }

        # Handle content extraction from diff (modern format compatible)
        if "```" in parsed_item["diff"]:
            # This is the new format, extract content between code blocks
            diff_content = parsed_item["diff"]
            content_start = diff_content.find("```") + 3
            content_start = diff_content.find("\n", content_start) + 1
            content_end = diff_content.rfind("```")
            if content_end > content_start:
                parsed_item["content"] = diff_content[content_start:content_end].strip()
            else:
                # Just use the whole diff as content if we can't parse it properly
                parsed_item["content"] = diff_content.strip()
        else:
            # For backward compatibility, use diff as content
            parsed_item["content"] = parsed_item["diff"]

        # Ensure we have file_path and action type
        if not parsed_item["file_path"] or not parsed_item["type"]:
            logger_to_use.error("missing_essential_fields_in_parsed_block", parsed_data=parsed_item)
            return None

        # Default params for asset_manager
        if parsed_item["type"].strip().lower() == "create" or parsed_item["type"].strip().lower() == "modify":
            parsed_item["action"] = "write"
        elif parsed_item["type"].strip().lower() == "delete":
            parsed_item["action"] = "delete"
        else:
            parsed_item["action"] = "write"  # Default

        # Map fields to asset_manager expected format
        parsed_item["path"] = parsed_item["file_path"]
        
        logger_to_use.debug("raw_block_parsed_successfully", file_path=parsed_item["file_path"])
        return parsed_item

    def _extract_data_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the primary data payload from the context."""
        # Check direct input_data first
        if 'input_data' in context and context['input_data']:
            return context['input_data']
        
        # Check response in context
        if 'response' in context:
            return context['response']
            
        # If no specific input found, return the full context
        return context

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Process code changes using semantic extraction and asset management."""
        logger_to_use = self.logger
        logger_to_use.info("process_start", context_keys=list(context.keys()))

        # Reset metrics at the start of processing
        self.operation_metrics = CoderMetrics(start_time=datetime.now(timezone.utc).isoformat())
        final_error = None
        change_results = []

        try:
            # Get input data (typically the SolutionDesigner's output)
            data = self._extract_data_from_context(context)

            # Extract the actual content string (potentially nested)
            content_to_iterate = None
            if isinstance(data, dict) and 'response' in data:
                content_to_iterate = data['response']
            elif isinstance(data, str):
                content_to_iterate = data
            else:
                content_to_iterate = str(data)

            # Ensure content is a string before passing to iterator
            if not isinstance(content_to_iterate, str):
                logger_to_use.warning("extracted_content_not_string", content_type=type(content_to_iterate).__name__)
                content_to_iterate = str(content_to_iterate)

            # --- Call SemanticIterator to get results ---
            iterator_result = self._invoke_skill('semantic_iterator', {
                'input_data': content_to_iterate,
                'format': 'json'
            })

            iterator_success = False
            iterator_error = None
            extracted_items = []
            
            if isinstance(iterator_result, SkillResult):
                iterator_success = iterator_result.success
                iterator_error = iterator_result.error
                if iterator_success and iterator_result.value:
                    extracted_items = iterator_result.value
            else:
                # Handle the case where we get an AgentResponse or other response type
                iterator_success = getattr(iterator_result, 'success', False)
                iterator_error = getattr(iterator_result, 'error', "Unknown iterator error")
                if iterator_success:
                    # Try to get results from various possible response structures
                    if hasattr(iterator_result, 'data') and isinstance(iterator_result.data, dict):
                        extracted_items = iterator_result.data.get('results', [])
                    elif hasattr(iterator_result, 'value'):
                        extracted_items = iterator_result.value

            if not iterator_success:
                logger_to_use.error("iterator_failed", error=iterator_error)
                final_error = f"Iterator failed: {iterator_error or 'Unknown error'}"
            else:
                item_count = len(extracted_items)
                logger_to_use.info("iterator_completed", items_found=item_count)

                if item_count > 0:
                    for index, item in enumerate(extracted_items):
                        self.operation_metrics.total_changes_processed += 1
                        logger_to_use.debug("processing_item", 
                                          index=index, 
                                          item_type=type(item).__name__)

                        change_action_dict = None
                        # Check if the item needs parsing (i.e., it's a raw string block)
                        if isinstance(item, str) and "===CHANGE_BEGIN===" in item:
                            logger_to_use.info("parsing_raw_block", index=index)
                            change_action_dict = self._parse_raw_change_block(item)
                        elif isinstance(item, dict):
                            # It's already a dictionary, ensure it has the right keys
                            change_action_dict = item.copy()
                            
                            # Map fields if needed
                            if 'file_path' in change_action_dict and 'path' not in change_action_dict:
                                change_action_dict['path'] = change_action_dict['file_path']
                            
                            # Add action if not present
                            if 'action' not in change_action_dict:
                                if 'type' in change_action_dict:
                                    item_type = change_action_dict['type'].lower()
                                    if item_type in ('create', 'modify'):
                                        change_action_dict['action'] = 'write'
                                    elif item_type == 'delete':
                                        change_action_dict['action'] = 'delete'
                                    else:
                                        change_action_dict['action'] = 'write'  # Default
                        else:
                            logger_to_use.error("invalid_item_type", item_type=type(item).__name__, index=index)
                            self.operation_metrics.failed_changes += 1
                            self.operation_metrics.error_count += 1
                            change_results.append({
                                'success': False,
                                'path': 'unknown',
                                'error': f"Invalid item type: {type(item).__name__}"
                            })
                            continue

                        if not change_action_dict:
                            logger_to_use.error("failed_to_parse_item", index=index)
                            self.operation_metrics.failed_changes += 1
                            self.operation_metrics.error_count += 1
                            change_results.append({
                                'success': False,
                                'path': 'unknown',
                                'error': "Failed to parse item"
                            })
                            continue

                        # Ensure we have required fields
                        if 'path' not in change_action_dict or not change_action_dict['path']:
                            if 'file_path' in change_action_dict:
                                change_action_dict['path'] = change_action_dict['file_path']
                            else:
                                logger_to_use.error("missing_file_path", index=index)
                                self.operation_metrics.failed_changes += 1
                                self.operation_metrics.error_count += 1
                                change_results.append({
                                    'success': False,
                                    'path': 'unknown',
                                    'error': "Missing file path"
                                })
                                continue

                        # Ensure we have content for write operations
                        if change_action_dict.get('action') == 'write' and 'content' not in change_action_dict:
                            if 'diff' in change_action_dict:
                                # Try to extract content from diff
                                diff = change_action_dict['diff']
                                if "```" in diff:
                                    content_start = diff.find("```") + 3
                                    content_start = diff.find("\n", content_start) + 1
                                    content_end = diff.rfind("```")
                                    if content_end > content_start:
                                        change_action_dict['content'] = diff[content_start:content_end].strip()
                                    else:
                                        change_action_dict['content'] = diff.strip()
                                else:
                                    change_action_dict['content'] = diff

                        # Call asset_manager
                        try:
                            asset_result = self._invoke_skill('asset_manager', change_action_dict)
                            
                            result_success = False
                            result_error = None
                            
                            if isinstance(asset_result, SkillResult):
                                result_success = asset_result.success
                                result_error = asset_result.error
                                result_value = asset_result.value
                            else:
                                # Handle the case where result is not a SkillResult
                                result_success = getattr(asset_result, 'success', False)
                                result_error = getattr(asset_result, 'error', "Unknown error")
                                result_value = getattr(asset_result, 'value', None) or getattr(asset_result, 'data', {})
                            
                            # Build a result dictionary
                            result_dict = {
                                'success': result_success,
                                'path': change_action_dict.get('path', 'unknown'),
                                'error': result_error,
                                'backup': result_value.get('backup_path') if isinstance(result_value, dict) else None
                            }
                            
                            change_results.append(result_dict)
                            
                            if result_success:
                                self.operation_metrics.successful_changes += 1
                                logger_to_use.info("change_applied_successfully", 
                                                 file=change_action_dict.get('path', 'unknown'))
                            else:
                                self.operation_metrics.failed_changes += 1
                                self.operation_metrics.error_count += 1
                                logger_to_use.error("change_application_failed", 
                                                  file=change_action_dict.get('path', 'unknown'), 
                                                  error=result_error)
                        except Exception as e:
                            self.operation_metrics.failed_changes += 1
                            self.operation_metrics.error_count += 1
                            logger_to_use.error("asset_manager_exception", 
                                             file=change_action_dict.get('path', 'unknown'), 
                                             error=str(e))
                            change_results.append({
                                'success': False,
                                'path': change_action_dict.get('path', 'unknown'),
                                'error': str(e)
                            })
                else:
                    logger_to_use.info("no_changes_to_apply")

            # Determine overall success
            success = iterator_success and self.operation_metrics.error_count == 0
            
            # Update final_error if process failed but no specific error was caught
            if not success and not final_error:
                if self.operation_metrics.error_count > 0:
                    final_error = f"{self.operation_metrics.error_count} change processing errors occurred."
                else:
                    final_error = "Coder task failed due to unknown error."
                    
            # Finalize metrics
            self.operation_metrics.end_time = datetime.now(timezone.utc).isoformat()
            try:
                start_dt = datetime.fromisoformat(self.operation_metrics.start_time)
                end_dt = datetime.fromisoformat(self.operation_metrics.end_time)
                self.operation_metrics.processing_time = round((end_dt - start_dt).total_seconds(), 3)
            except ValueError:
                self.operation_metrics.processing_time = 0.0
                logger_to_use.warning("failed_to_calculate_processing_time")

            metrics_dict = self.operation_metrics.to_dict()
            logger_to_use.info("process_finished", 
                             success=success, 
                             metrics=metrics_dict, 
                             final_error=final_error)

            # Return structured results
            return AgentResponse(
                success=success,
                data={
                    "changes": change_results,
                    "metrics": metrics_dict
                },
                error=final_error
            )

        except Exception as e:
            logger_to_use.error("process_uncaught_exception", error=str(e))
            self.operation_metrics.error_count += 1
            self.operation_metrics.end_time = datetime.now(timezone.utc).isoformat()
            
            try:
                start_dt = datetime.fromisoformat(self.operation_metrics.start_time)
                end_dt = datetime.fromisoformat(self.operation_metrics.end_time)
                self.operation_metrics.processing_time = round((end_dt - start_dt).total_seconds(), 3)
            except (ValueError, TypeError):
                self.operation_metrics.processing_time = 0.0
                
            metrics_dict = self.operation_metrics.to_dict()
            
            return AgentResponse(
                success=False,
                data={"metrics": metrics_dict},
                error=str(e)
            )