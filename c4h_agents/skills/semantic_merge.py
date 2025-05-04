"""
Semantic merge implementation for code modifications.
Path: c4h_agents/skills/semantic_merge.py
"""

from typing import Dict, Any, Optional
from pathlib import Path
from c4h_agents.agents.base_agent import BaseAgent, AgentResponse 
from c4h_agents.config import locate_config
from c4h_agents.utils.logging import get_logger

logger = get_logger()

class SemanticMerge(BaseAgent):
    """Handles merging of code modifications."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize merger with configuration."""
        # Pass positional parameters to BaseAgent
        super().__init__(config, "semantic_merge")
        
        # The unique_name is now stored by BaseAgent
        
        # Get merge-specific configs
        merge_config = locate_config(self.config or {}, self._get_agent_name())
        merge_config = merge_config.get('merge_config', {})
        self.preserve_formatting = merge_config.get('preserve_formatting', True)
        self.allow_partial = merge_config.get('allow_partial', False)
        
        logger.info("semantic_merge.initialized")

    def _get_agent_name(self) -> str:
        """Get agent name for config lookup"""
        return "semantic_merge"

    def _get_original_content(self, file_path: str) -> Optional[str]:
        """Read original file content if it exists"""
        try:
            # If self.project is defined, use its path resolution
            if self.project:
                path = self.project.resolve_path(file_path)
            else:
                # Otherwise use direct path
                path = Path(file_path)
                
            # Check if path exists and is a file
            if path.is_file():
                try:
                    content = path.read_text()
                    logger.debug("merge.read_original_success", 
                                file_path=str(path), 
                                content_length=len(content))
                    return content
                except Exception as e:
                    logger.error("merge.read_file_error", 
                                file_path=str(path), 
                                error=str(e))
                    return None
            
            logger.debug("merge.file_not_found", file_path=str(path))
            return None
            
        except Exception as e:
            logger.error("merge.read_original_failed", error=str(e))
            return None

    def _format_request(self, context: Dict[str, Any]) -> str:
        """Format merge request using merge prompt template"""
        try:
            # Get merge prompt template
            merge_template = self._get_prompt('merge')
            
            # Extract key merge components
            original = context.get('original', '')
            diff = context.get('diff', '')
            is_create = context.get('type') == 'create'
            
            # For create operations with no original content, use a special marker
            if is_create and not original and diff:
                # Add a special note for creation operations
                merge_template = merge_template.replace(
                    "If either is missing, return the error: \"Missing required [original|diff] content\".",
                    "For new file creation (when original is empty), focus on generating the complete new file content."
                )
                
                logger.debug("merge.create_mode_prompt_update", 
                          file_path=context.get('file_path'))
            
            # Log merge components at debug level
            logger.debug("merge.request_prepared",
                       has_original=bool(original),
                       diff_lines=len(diff.split('\n')) if diff else 0)
            
            return merge_template.format(
                original=original or "# New file - no original content",
                diff=diff
            )
            
        except Exception as e:
            logger.error("merge.format_failed", error=str(e))
            raise

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Process merge by passing entire context to LLM"""
        logger.info("merge.starting")

        try:
            # Validate required inputs
            if not isinstance(context, dict):
                return AgentResponse(
                    success=False,
                    error="Invalid context format",
                    data={}
                )

            file_path = context.get('file_path')
            diff = context.get('diff')
            content = context.get('content')
            operation_type = context.get('type', 'modify')
            
            if not file_path:
                return AgentResponse(
                    success=False,
                    error="Missing required file_path",
                    data=context
                )
            
            # Handle content-only create operations directly (no diff needed)
            if operation_type == 'create' and content and not diff:
                logger.info("merge.direct_content", 
                           file_path=file_path, 
                           content_length=len(content))
                return AgentResponse(
                    success=True,
                    data={
                        "response": content,
                        "raw_output": content
                    }
                )
                
            # Require diff for non-create operations (regular merges)
            if not diff and operation_type != 'create':
                return AgentResponse(
                    success=False,
                    error=f"Missing required diff for {file_path}",
                    data=context
                )

            # Try to get original from context
            original = context.get('original')
            
            # If not in context but file exists, read it directly
            if not original:
                # Try to read file content directly from disk
                file_content = self._get_original_content(file_path)
                if file_content:
                    original = file_content
                    logger.info("merge.file_content_read", 
                               file_path=file_path, 
                               content_length=len(file_content))
            
            # For non-existent files on create operations, use empty string
            if not original and operation_type == 'create':
                original = ""
                logger.info("merge.create_operation", 
                           file_path=file_path, 
                           using_empty_original=True)
            # For other operations, require content unless allow_partial is True
            elif not original and not self.allow_partial:
                return AgentResponse(
                    success=False,
                    error=f"Original file not found: {file_path}",
                    data=context
                )
            
            if not original:
                logger.warning("merge.missing_original", 
                             file=file_path,
                             proceeding=True)

            # Prepare merge context with complete information
            merge_context = {
                **context,
                'original': original or '',  # Empty string for new files
                'file_path': str(file_path),
                'type': operation_type  # Ensure operation type is passed
            }

            # Let LLM handle merge with complete context
            result = super().process(merge_context)
            
            if not result.success:
                logger.warning("merge.failed", error=result.error)
                return result
                
            # Check for the error message in the response
            response_content = result.data.get("response", "")
            if response_content.strip() == "Missing required original content" and operation_type == 'create':
                # For create operations, this means we need to handle it specially
                # Instead of using the error as content, try to extract content from the diff
                logger.warning("merge.detected_error_response", 
                             file_path=file_path,
                             error=response_content)
                             
                if diff:
                    # Try to extract content from the diff for the new file
                    extracted_content = self._extract_content_from_diff(diff)
                    if extracted_content:
                        logger.info("merge.extracted_diff_content", 
                                  file_path=file_path,
                                  content_length=len(extracted_content))
                        return AgentResponse(
                            success=True,
                            data={
                                "response": extracted_content,
                                "raw_output": extracted_content
                            }
                        )
                    else:
                        return AgentResponse(
                            success=False,
                            error="Failed to extract content from diff for new file",
                            data=context
                        )
                        
            return AgentResponse(
                success=True,
                data={
                    "response": result.data.get("response", ""),
                    "raw_output": result.data.get("raw_output")
                }
            )

        except Exception as e:
            logger.error("merge.failed", error=str(e))
            return AgentResponse(success=False, data={}, error=str(e))
            
    def _extract_content_from_diff(self, diff: str) -> Optional[str]:
        """
        Extract content from a unified diff for a new file.
        This handles the case where we need to create a new file and
        have a diff but no original content.
        """
        try:
            # Process the diff to extract added lines (for new files)
            lines = diff.splitlines()
            added_lines = []
            in_hunk = False
            
            for line in lines:
                if line.startswith("@@"):
                    in_hunk = True
                    continue
                    
                if in_hunk and line.startswith("+"):
                    # Remove the leading + for added lines
                    added_lines.append(line[1:])
                    
            if added_lines:
                return "\n".join(added_lines)
                
            return None
            
        except Exception as e:
            logger.error("merge.extract_diff_failed", error=str(e))
            return None