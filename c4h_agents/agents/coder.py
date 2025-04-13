# Path: /Users/jim/src/apps/c4h/c4h_agents/agents/coder.py
"""
Primary coder agent implementation using semantic extraction.
"""
from typing import Dict, Any
from dataclasses import dataclass, field # Import field, asdict
import dataclasses # Import dataclasses for asdict

from datetime import datetime, timezone
from pathlib import Path

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
    total_changes: int = 0
    successful_changes: int = 0
    failed_changes: int = 0
    start_time: str = ""
    end_time: str = ""
    processing_time: float = 0.0
    error_count: int = 0

    # --- FIX: Add to_dict method using dataclasses.asdict ---
    def to_dict(self) -> Dict[str, Any]:
         """Convert metrics to plain dictionary for serialization"""
         return dataclasses.asdict(self)
    # --- END FIX ---

class Coder(BaseAgent):
    """Handles code modifications using semantic processing"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize coder with configuration"""
        super().__init__(config=config)

        # Use self.logger inherited from BaseAgent after super().__init__()
        logger_to_use = self.logger

        coder_config = self._get_agent_config()

        runtime_config = self.config_node.get_value("runtime") or {}
        backup_config = runtime_config.get("backup", {})
        backup_enabled_val = backup_config.get("enabled", False)
        backup_enabled = str(backup_enabled_val).lower() == 'true' if isinstance(backup_enabled_val, str) else bool(backup_enabled_val)
        backup_path = Path(backup_config.get("path", "workspaces/backups"))

        # Create semantic tools, passing the agent's config
        self.iterator = SemanticIterator(config=self.config)
        self.merger = SemanticMerge(config=self.config)
        # Pass logger to AssetManager if it accepts it
        self.asset_manager = AssetManager(
            backup_enabled=backup_enabled,
            backup_dir=backup_path,
            merger=self.merger,
            config=self.config,
            # logger=logger_to_use # Uncomment if AssetManager accepts logger
        )

        # Initialize metrics
        self.operation_metrics = CoderMetrics(start_time=datetime.now(timezone.utc).isoformat())
        logger_to_use.info("coder.initialized", backup_path=str(backup_path), backup_enabled=backup_enabled)


    def _get_agent_name(self) -> str:
         """Provide agent name for config/logging."""
         return "coder"

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Process code changes using semantic extraction"""
        logger_to_use = self.logger # Use initialized instance logger
        logger_to_use.info("coder.process_start", context_keys=list(context.keys()))
        logger_to_use.debug("coder.input_data", data=context)

        # Reset metrics at the start of processing
        self.operation_metrics = CoderMetrics(start_time=datetime.now(timezone.utc).isoformat())
        final_error = None

        try:
            # Get input data using the inherited method
            data = self._get_data(context)

            # Extract the actual content string using the corrected _get_llm_content from BaseAgent
            content = self._get_llm_content(data)

            # Ensure content is a string before passing to iterator
            if not isinstance(content, str):
                logger_to_use.warning("coder.extracted_content_not_string", content_type=type(content).__name__)
                content = str(content)

            # Get changes from iterator
            # Pass only the extracted content string under 'input_data'
            iterator_result = self.iterator.process({'input_data': content})

            if not iterator_result.success:
                logger_to_use.error("coder.iterator_failed", error=iterator_result.error)
                final_error = f"Iterator failed: {iterator_result.error}"
                # Fall through to return error state at the end

            # Process each change yielded by the iterator (if any)
            results = []
            processed_count = 0
            # Check success and if results exist before iterating
            # Also check if iterator_result itself is not None
            if iterator_result and iterator_result.success and iterator_result.data and iterator_result.data.get("results"):
                 try:
                      # Use the iterator instance directly from self.iterator
                      for change in self.iterator: # Iterate through items found
                           processed_count += 1
                           logger_to_use.debug("coder.processing_change",
                                        type=type(change).__name__,
                                        change_preview=repr(change)[:150] + "..." if len(repr(change)) > 150 else repr(change) )

                           if not isinstance(change, dict):
                                logger_to_use.error("coder.iterator_yielded_non_dict", item_type=type(change).__name__)
                                self.operation_metrics.failed_changes += 1
                                self.operation_metrics.error_count += 1
                                results.append(AssetResult(success=False, path=Path("unknown"), error="Iterator yielded non-dictionary item"))
                                continue

                           result = self.asset_manager.process_action(change)

                           if result.success:
                                self.operation_metrics.successful_changes += 1
                           else:
                                self.operation_metrics.failed_changes += 1
                                self.operation_metrics.error_count += 1

                           self.operation_metrics.total_changes += 1
                           results.append(result)

                 except Exception as iter_err:
                      logger_to_use.error("coder.iteration_error", error=str(iter_err), exc_info=True)
                      self.operation_metrics.error_count += 1
                      final_error = f"Error during change processing: {str(iter_err)}"
            elif iterator_result and iterator_result.success: # Handle case where iterator succeeded but found 0 items
                 logger_to_use.info("coder.process_complete_nop", reason="Iterator returned zero change items.")


            # Determine overall success
            # Success if iterator succeeded AND no errors occurred during change processing. Zero changes is success.
            success = iterator_result and iterator_result.success and self.operation_metrics.error_count == 0

            # Update final_error if process failed but no specific error was caught
            if not success and not final_error:
                 # If iterator_result exists and has an error, use that, otherwise use a generic message
                 iter_error = iterator_result.error if iterator_result else "Iterator processing failed"
                 final_error = iter_error or "Coder task failed due to processing errors."
            elif success and not results: # Successful NOP case
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
                 logger_to_use.warning("Failed to calculate processing_time due to invalid timestamp format",
                                        start=self.operation_metrics.start_time,
                                        end=self.operation_metrics.end_time)

            # Use the new to_dict() method for metrics
            metrics_dict = self.operation_metrics.to_dict()

            return AgentResponse(
                success=success,
                data={
                    "changes": [
                        {
                            "file": str(r.path) if r.path else "unknown",
                            "success": r.success,
                            "error": r.error,
                            "backup": str(r.backup_path) if r.backup_path else None
                        }
                        for r in results
                    ],
                    "metrics": metrics_dict
                },
                error=final_error
            )

        except Exception as e:
            logger_to_use.error("coder.process_failed", error=str(e), exc_info=True)
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
                 # Use the to_dict method safely
                 if hasattr(self.operation_metrics, 'to_dict') and callable(self.operation_metrics.to_dict):
                      metrics_dict = self.operation_metrics.to_dict()
                 else: # Fallback if method somehow missing
                     metrics_dict = vars(self.operation_metrics)

                 return AgentResponse(success=False, data={"metrics": metrics_dict}, error=str(e))
            else:
                 # Fallback if metrics not initialized
                 return AgentResponse(success=False, data={}, error=str(e))