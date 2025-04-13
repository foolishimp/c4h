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
from c4h_agents.skills.asset_manager import AssetManager
from c4h_agents.utils.logging import get_logger

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
        
        # Get coder-specific config 
        coder_config = self._get_agent_config()
        
        # Get runtime backup settings
        runtime_config = self.config_node.get_value("runtime") or {}
        backup_config = runtime_config.get("backup", {})
        backup_enabled = backup_config.get("enabled", False)  # Default to false
        backup_path = Path(backup_config.get("path", "workspaces/backups"))
        
        # Create semantic tools
        self.iterator = SemanticIterator(config=config)
        self.merger = SemanticMerge(config=config)
        self.asset_manager = AssetManager(
            backup_enabled=backup_enabled,
            backup_dir=backup_path,
            merger=self.merger,
            config=config
        )
        
        # Initialize metrics
        self.operation_metrics = CoderMetrics()
        logger.info("coder.initialized", backup_path=str(backup_path))


    def process(self, context: Dict[str, Any]) -> AgentResponse:
            """Process code changes using semantic extraction"""
            logger.info("coder.process_start", context_keys=list(context.keys()))
            logger.debug("coder.input_data", data=context)

            try:
                # Get content from input
                data = self._get_data(context)
                content = self._get_llm_content(data.get('input_data', {}))
                if not isinstance(content, str):
                    logger.warning("coder.extracted_content_not_string", content_type=type(content).__name__)
                    # Attempt to stringify if not a string, as a fallback. Ideally it should be string here.
                    content = str(content)

                # Get changes from iterator 
                # --- FIX: Pass only the extracted content string to the iterator ---
                iterator_result = self.iterator.process({'input_data': content}) # Pass content string as 'input_data'
                # --- END FIX ---

                if not iterator_result.success:
                    logger.error("coder.iterator_failed", error=iterator_result.error)
                    return AgentResponse(
                        success=False, 
                        data={},
                        error=f"Iterator failed: {iterator_result.error}"
                    )
                
                # Process each change
                results = []
                for change in self.iterator:
                    logger.debug("coder.processing_change", 
                                type=type(change).__name__,  # See what type we're dealing with
                                change=repr(change) )              # Original logging
                    result = self.asset_manager.process_action(change)
                    
                    if result.success:
                        self.operation_metrics.successful_changes += 1
                    else:
                        self.operation_metrics.failed_changes += 1
                        self.operation_metrics.error_count += 1
                    
                    self.operation_metrics.total_changes += 1
                    results.append(result)

                success = bool(results) and any(r.success for r in results)
                self.operation_metrics.end_time = datetime.now(timezone.utc).isoformat()
                
                return AgentResponse(
                    success=success,
                    data={
                        "changes": [
                            {
                                "file": str(r.path),
                                "success": r.success,
                                "error": r.error,
                                "backup": str(r.backup_path) if r.backup_path else None
                            }
                            for r in results
                        ],
                        "metrics": vars(self.operation_metrics)
                    },
                    error=None if success else "No changes were successful"
                )

            except Exception as e:
                logger.error("coder.process_failed", error=str(e))
                self.operation_metrics.error_count += 1
                return AgentResponse(success=False, data={}, error=str(e))