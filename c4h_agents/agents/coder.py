# Path: /Users/jim/src/apps/c4h/c4h_agents/agents/coder.py
"""
Primary coder agent implementation using semantic extraction.
Path: c4h_agents/agents/coder.py
"""
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from c4h_agents.agents.base_agent import BaseAgent, AgentResponse 
from c4h_agents.skills.semantic_merge import SemanticMerge
from c4h_agents.skills.semantic_iterator import SemanticIterator
from c4h_agents.skills.asset_manager import AssetManager, AssetResult
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

class Coder(BaseAgent):
    """Handles code modifications using semantic processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize coder with configuration"""
        super().__init__(config=config)
        
        # Get coder-specific config using hierarchical lookup via BaseConfig/BaseAgent
        # self._get_agent_config() is inherited from BaseConfig
        coder_config = self._get_agent_config() 
        
        # Get runtime backup settings using hierarchical lookup via BaseConfig/BaseAgent
        # self.config_node is initialized in BaseConfig
        runtime_config = self.config_node.get_value("runtime") or {}
        backup_config = runtime_config.get("backup", {})
        # --- Ensure boolean conversion for enabled flag ---
        backup_enabled_val = backup_config.get("enabled", False) 
        backup_enabled = str(backup_enabled_val).lower() == 'true' if isinstance(backup_enabled_val, str) else bool(backup_enabled_val)
        # --- End Ensure ---
        backup_path = Path(backup_config.get("path", "workspaces/backups"))
        
        # Create semantic tools, passing the agent's config
        self.iterator = SemanticIterator(config=self.config) # Pass full config
        self.merger = SemanticMerge(config=self.config)     # Pass full config
        self.asset_manager = AssetManager(
            backup_enabled=backup_enabled,
            backup_dir=backup_path,
            merger=self.merger,
            config=self.config # Pass full config
        )
        
        # Initialize metrics
        self.operation_metrics = CoderMetrics(start_time=datetime.now(timezone.utc).isoformat()) # Set start time
        # Use self.logger inherited from BaseAgent
        self.logger.info("coder.initialized", backup_path=str(backup_path), backup_enabled=backup_enabled)


    def _get_agent_name(self) -> str:
         """Provide agent name for config/logging."""
         # Overrides the generic name from BaseAgent if needed, otherwise inherited version is fine
         return "coder" # Explicitly name this agent

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Process code changes using semantic extraction"""
        # Use self.logger inherited from BaseAgent
        self.logger.info("coder.process_start", context_keys=list(context.keys()))
        self.logger.debug("coder.input_data", data=context)

        try:
            # Get input data from context using the inherited method
            # _get_data is simple, just returns context if dict, else {'content': str(context)}
            data = self._get_data(context) 
            
            # Extract the actual content string using the corrected _get_llm_content
            # This method is inherited from BaseAgent and has the necessary logic
            # It expects the dictionary structure passed in 'data' variable
            content = self._get_llm_content(data) 
            
            # Ensure content is a string before passing to iterator
            if not isinstance(content, str):
                self.logger.warning("coder.extracted_content_not_string", content_type=type(content).__name__)
                # Attempt to stringify if not a string, as a fallback.
                content = str(content)

            # Get changes from iterator 
            # Pass only the extracted content string to the iterator under the key it expects
            iterator_result = self.iterator.process({'input_data': content}) 

            if not iterator_result.success:
                self.logger.error("coder.iterator_failed", error=iterator_result.error)
                # Ensure metrics reflect the error state
                self.operation_metrics.error_count += 1
                self.operation_metrics.end_time = datetime.now(timezone.utc).isoformat()
                if self.operation_metrics.start_time: # Calculate duration if possible
                     try:
                          start_dt = datetime.fromisoformat(self.operation_metrics.start_time)
                          end_dt = datetime.fromisoformat(self.operation_metrics.end_time)
                          self.operation_metrics.processing_time = (end_dt - start_dt).total_seconds()
                     except ValueError: pass # Ignore format errors
                
                return AgentResponse(
                    success=False, 
                    data={"metrics": self.operation_metrics.to_dict()}, # Include metrics in error response
                    error=f"Iterator failed: {iterator_result.error}"
                )
            
            # Reset metrics for this run before processing changes
            self.operation_metrics = CoderMetrics(start_time=datetime.now(timezone.utc).isoformat())

            # Process each change yielded by the iterator
            results = []
            processed_count = 0
            try:
                 # Use the iterator instance directly from self.iterator
                 for change in self.iterator: # This should now work correctly if iterator initialized properly
                      processed_count += 1
                      self.logger.debug("coder.processing_change", 
                                   type=type(change).__name__, # Log type
                                   change_preview=repr(change)[:150] + "..." if len(repr(change)) > 150 else repr(change) ) # Log preview

                      # Ensure 'change' is a dictionary before passing to asset manager
                      if not isinstance(change, dict):
                           self.logger.error("coder.iterator_yielded_non_dict", item_type=type(change).__name__)
                           self.operation_metrics.failed_changes += 1
                           self.operation_metrics.error_count += 1
                           results.append(AssetResult(success=False, path=Path("unknown"), error="Iterator yielded non-dictionary item"))
                           continue # Skip non-dict items

                      result = self.asset_manager.process_action(change)
                      
                      if result.success:
                           self.operation_metrics.successful_changes += 1
                      else:
                           self.operation_metrics.failed_changes += 1
                           self.operation_metrics.error_count += 1
                      
                      self.operation_metrics.total_changes += 1
                      results.append(result)

            except Exception as iter_err:
                 # Catch errors during iteration itself
                 self.logger.error("coder.iteration_error", error=str(iter_err), exc_info=True)
                 self.operation_metrics.error_count += 1
                 # Return partial results if any processing happened before error
                 # Decide if the overall process is a success based on partial results? Maybe not.
                 success = False # Mark as failed if iteration breaks
                 final_error = f"Error during change processing: {str(iter_err)}"


            # Finalize metrics
            self.operation_metrics.end_time = datetime.now(timezone.utc).isoformat()
            try:
                 start_dt = datetime.fromisoformat(self.operation_metrics.start_time)
                 end_dt = datetime.fromisoformat(self.operation_metrics.end_time)
                 self.operation_metrics.processing_time = (end_dt - start_dt).total_seconds()
            except ValueError: pass # Ignore format errors


            # Determine overall success - requires at least one successful change if changes were attempted
            success = bool(results) and self.operation_metrics.successful_changes > 0 
            final_error = None
            if self.operation_metrics.error_count > 0:
                 success = False # Mark as failed if any errors occurred
                 final_error = f"{self.operation_metrics.error_count} errors occurred during processing."
            elif not results: # No items were iterated over successfully
                 success = False
                 final_error = iterator_result.error or "No changes processed by iterator."


            return AgentResponse(
                success=success,
                data={
                    "changes": [
                        {
                            "file": str(r.path) if r.path else "unknown", # Handle potential None path
                            "success": r.success,
                            "error": r.error,
                            "backup": str(r.backup_path) if r.backup_path else None
                        }
                        for r in results # Use the collected results
                    ],
                    "metrics": self.operation_metrics.to_dict() # Use to_dict() method
                },
                error=final_error # Report accumulated error if any
            )

        except Exception as e:
            self.logger.error("coder.process_failed", error=str(e), exc_info=True) # Log traceback
            # Update metrics on failure
            if hasattr(self, 'operation_metrics'): # Check if metrics initialized
                 self.operation_metrics.error_count += 1
                 self.operation_metrics.end_time = datetime.now(timezone.utc).isoformat()
                 if self.operation_metrics.start_time:
                      try:
                           start_dt = datetime.fromisoformat(self.operation_metrics.start_time)
                           end_dt = datetime.fromisoformat(self.operation_metrics.end_time)
                           self.operation_metrics.processing_time = (end_dt - start_dt).total_seconds()
                      except ValueError: pass
                 return AgentResponse(success=False, data={"metrics": self.operation_metrics.to_dict()}, error=str(e))
            else:
                 # Fallback if metrics not initialized
                 return AgentResponse(success=False, data={}, error=str(e))