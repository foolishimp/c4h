# File: /Users/jim/src/apps/c4h_ai_dev/c4h_agents/agents/coder.py
"""
Primary coder agent implementation using semantic extraction.
Handles iterating through change blocks provided by SolutionDesigner
and applying them using AssetManager.
"""
from typing import Dict, Any, Optional, List # Added Optional, List
from dataclasses import dataclass, field, asdict # Use asdict directly
from datetime import datetime, timezone
from pathlib import Path
import json # For parsing string if needed
import re # For parsing raw string block

from c4h_agents.agents.base_agent import BaseAgent, AgentResponse
from c4h_agents.skills.semantic_merge import SemanticMerge
from c4h_agents.skills.semantic_iterator import SemanticIterator
from c4h_agents.skills.asset_manager import AssetManager, AssetResult # Import AssetResult
from c4h_agents.utils.logging import get_logger

# Use central logger
logger = get_logger()

@dataclass
class CoderMetrics:
    """Detailed metrics for code processing operations"""
    total_changes_processed: int = 0 # Renamed for clarity
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
    """Handles code modifications using semantic processing"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize coder with configuration"""
        super().__init__(config=config)

        # Use self.logger inherited from BaseAgent after super().__init__()
        # Bind logger immediately for consistent context
        self.logger = self.logger.bind(agent_name=self._get_agent_name(), agent_type="Coder")

        # Get agent specific config (though Coder might not have much unique config now)
        coder_config = self._get_agent_config() # Uses persona if generic, or llm_config.agents.coder if specific

        # Determine backup settings from the full config
        backup_config = self.config_node.get_value("backup") or {} # Check top-level first
        if not backup_config:
             runtime_config = self.config_node.get_value("runtime") or {} # Fallback to runtime
             backup_config = runtime_config.get("backup", {})

        backup_enabled_val = backup_config.get("enabled", True) # Default to True if not specified
        # Robust boolean conversion
        backup_enabled = str(backup_enabled_val).lower() == 'true' if isinstance(backup_enabled_val, str) else bool(backup_enabled_val)
        backup_path_str = backup_config.get("path", "workspaces/backups")

        # Resolve backup path relative to project if project path exists
        backup_path = Path(backup_path_str)
        if not backup_path.is_absolute() and self.project:
             backup_path = self.project.paths.root / backup_path

        # Create semantic tools, passing the agent's full config
        # These skills will inherit config resolution from BaseAgent
        self.iterator = SemanticIterator(config=self.config)
        self.merger = SemanticMerge(config=self.config)
        self.asset_manager = AssetManager(
            backup_enabled=backup_enabled,
            backup_dir=backup_path.resolve(), # Ensure resolved path
            merger=self.merger,
            config=self.config,
            # logger=self.logger # Pass logger if AssetManager accepts it
        )

        # Initialize metrics
        self.operation_metrics = CoderMetrics(start_time=datetime.now(timezone.utc).isoformat())
        self.logger.info("initialized", backup_path=str(backup_path.resolve()), backup_enabled=backup_enabled)


    def _get_agent_name(self) -> str:
         """Provide agent name for config/logging."""
         # If this becomes generic, it should return self.unique_name
         # For now, keep as specific Coder
         return "coder"

    def _parse_raw_change_block(self, raw_block: str) -> Optional[Dict[str, Any]]:
        """
        Parses a raw text block (including markers) into a structured dictionary.
        Needed when SemanticIterator falls back and yields raw text.
        """
        logger_to_use = self.logger # Use instance logger
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
        content_inside_markers = raw_block.strip()[len("===CHANGE_BEGIN==="):-len("===CHANGE_END===")].strip()

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

        # Validate essential fields
        if not parsed_item["file_path"] or not parsed_item["type"]:
            logger_to_use.error("missing_essential_fields_in_parsed_block", parsed_data=parsed_item)
            return None

        logger_to_use.debug("raw_block_parsed_successfully", file_path=parsed_item["file_path"])
        return parsed_item


    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Process code changes using semantic extraction and asset management."""
        logger_to_use = self.logger # Use initialized instance logger
        logger_to_use.info("process_start", context_keys=list(context.keys()))
        logger_to_use.debug("input_context_received", data=context)

        # Reset metrics at the start of processing
        self.operation_metrics = CoderMetrics(start_time=datetime.now(timezone.utc).isoformat())
        final_error = None
        change_results = [] # Store AssetResult objects

        try:
            # Get input data (typically the SolutionDesigner's output)
            data = self._get_data(context)

            # Extract the actual content string (potentially nested)
            content_to_iterate = self._get_llm_content(data)

            # Ensure content is a string before passing to iterator
            if not isinstance(content_to_iterate, str):
                logger_to_use.warning("extracted_content_not_string", content_type=type(content_to_iterate).__name__)
                content_to_iterate = str(content_to_iterate)

            # --- Call SemanticIterator.process to get results ---
            # This handles the fast/slow fallback internally
            iterator_response: AgentResponse = self.iterator.process({'input_data': content_to_iterate})

            if not iterator_response.success:
                logger_to_use.error("iterator_failed", error=iterator_response.error)
                final_error = f"Iterator failed: {iterator_response.error}"
                # Fall through to return error state at the end
            else:
                # --- CORRECTED ITERATION LOGIC ---
                extracted_items = iterator_response.data.get("results", [])
                item_count = iterator_response.data.get("count", 0)
                logger_to_use.info("iterator_completed", items_yielded=item_count)

                if item_count > 0 and extracted_items:
                    for index, item in enumerate(extracted_items):
                        self.operation_metrics.total_changes_processed += 1
                        logger_to_use.debug("processing_yielded_item",
                                            index=index,
                                            item_type=type(item).__name__,
                                            item_preview=repr(item)[:150] + "...")

                        change_action_dict = None
                        # Check if the item needs parsing (i.e., it's a raw string block)
                        if isinstance(item, str) and item.strip().startswith("===CHANGE_BEGIN==="):
                             logger_to_use.info("parsing_raw_block_from_iterator", index=index)
                             change_action_dict = self._parse_raw_change_block(item)
                             if not change_action_dict:
                                 logger_to_use.error("failed_to_parse_raw_block_item", index=index)
                                 self.operation_metrics.failed_changes += 1
                                 self.operation_metrics.error_count += 1
                                 change_results.append(AssetResult(success=False, path=Path("unknown"), error="Failed to parse raw block from iterator"))
                                 continue # Skip to next item
                        elif isinstance(item, dict):
                             # Assume it's already the correct dictionary format
                             change_action_dict = item
                        else:
                             logger_to_use.error("iterator_yielded_invalid_type", item_type=type(item).__name__, index=index)
                             self.operation_metrics.failed_changes += 1
                             self.operation_metrics.error_count += 1
                             change_results.append(AssetResult(success=False, path=Path("unknown"), error=f"Iterator yielded unexpected type: {type(item).__name__}"))
                             continue # Skip to next item

                        # Ensure we have a valid dictionary before calling AssetManager
                        if change_action_dict and isinstance(change_action_dict, dict):
                             # Call AssetManager with the structured dictionary
                             asset_result: AssetResult = self.asset_manager.process_action(change_action_dict)
                             change_results.append(asset_result) # Store the AssetResult

                             if asset_result.success:
                                 self.operation_metrics.successful_changes += 1
                                 logger_to_use.info("change_applied_successfully", file=str(asset_result.path))
                             else:
                                 self.operation_metrics.failed_changes += 1
                                 self.operation_metrics.error_count += 1
                                 logger_to_use.error("change_application_failed", file=str(asset_result.path), error=asset_result.error)
                        # else: error handled above

                elif item_count == 0:
                     logger_to_use.info("process_complete_nop", reason="Iterator returned zero change items.")
                # --- END CORRECTED ITERATION LOGIC ---

            # Determine overall success
            # Success only if iterator didn't fail AND no errors occurred during change processing.
            success = (iterator_response and iterator_response.success and self.operation_metrics.error_count == 0)

            # Update final_error if process failed but no specific error was caught earlier
            if not success and not final_error:
                 iter_error = iterator_response.error if iterator_response else "Iterator processing failed"
                 processing_errors = f"{self.operation_metrics.error_count} change processing errors occurred." if self.operation_metrics.error_count > 0 else ""
                 final_error = iter_error or processing_errors or "Coder task failed due to processing errors."
            elif success and not change_results: # Successful NOP case
                 final_error = None # Ensure error is None

            # Finalize metrics
            self.operation_metrics.end_time = datetime.now(timezone.utc).isoformat()
            try:
                 if self.operation_metrics.start_time and isinstance(self.operation_metrics.start_time, str):
                      start_dt = datetime.fromisoformat(self.operation_metrics.start_time)
                      end_dt = datetime.fromisoformat(self.operation_metrics.end_time)
                      self.operation_metrics.processing_time = round((end_dt - start_dt).total_seconds(), 3)
                 else:
                      self.operation_metrics.processing_time = 0.0
            except ValueError:
                 self.operation_metrics.processing_time = 0.0
                 logger_to_use.warning("failed_to_calculate_processing_time",
                                        start=self.operation_metrics.start_time,
                                        end=self.operation_metrics.end_time)

            metrics_dict = self.operation_metrics.to_dict()
            logger_to_use.info("process_finished", success=success, metrics=metrics_dict, final_error=final_error)

            # Return structured results
            return AgentResponse(
                success=success,
                data={
                    # Use the structured AssetResult list
                    "changes": [
                        {
                            "file": str(r.path) if r.path else "unknown",
                            "success": r.success,
                            "error": r.error,
                            "backup": str(r.backup_path) if r.backup_path else None
                        }
                        for r in change_results # Iterate over AssetResult objects
                    ],
                    "metrics": metrics_dict
                },
                error=final_error
            )

        except Exception as e:
            logger_to_use.error("process_uncaught_exception", error=str(e), exc_info=True)
            # Update metrics on failure
            if hasattr(self, 'operation_metrics'):
                 self.operation_metrics.error_count += 1
                 self.operation_metrics.end_time = datetime.now(timezone.utc).isoformat()
                 if self.operation_metrics.start_time and isinstance(self.operation_metrics.start_time, str):
                      try:
                           start_dt = datetime.fromisoformat(self.operation_metrics.start_time)
                           end_dt = datetime.fromisoformat(self.operation_metrics.end_time)
                           self.operation_metrics.processing_time = round((end_dt - start_dt).total_seconds(), 3)
                      except ValueError:
                           self.operation_metrics.processing_time = 0.0

                 metrics_dict = {}
                 try:
                     metrics_dict = self.operation_metrics.to_dict()
                 except Exception: # Fallback if to_dict fails
                     metrics_dict = vars(self.operation_metrics)

                 return AgentResponse(success=False, data={"metrics": metrics_dict}, error=str(e))
            else:
                 # Fallback if metrics not initialized
                 return AgentResponse(success=False, data={}, error=str(e))