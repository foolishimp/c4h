"""
Semantic formatting using configured prompts.
Path: c4h_agents/skills/semantic_formatter.py
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from c4h_agents.agents.base_agent import BaseAgent, AgentResponse 
from c4h_agents.utils.logging import get_logger

logger = get_logger()
@dataclass
class FormatResult:
    """Result of semantic formatting"""
    success: bool
    value: str
    raw_response: str
    error: Optional[str] = None

class SemanticFormatter(BaseAgent):
    """Handles text formatting using LLM"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize formatter with config."""
        # Pass positional parameters to BaseAgent
        super().__init__(config, "semantic_formatter")
        
        # The unique_name is now stored by BaseAgent

    def _get_agent_name(self) -> str:
        return "semantic_formatter"

    def _format_request(self, context: Dict[str, Any]) -> str:
        """Format request using config template"""
        format_template = self._get_prompt('format')
        
        return format_template.format(
            content=context.get('content', ''),
            instruction=context.get('instruction', '')
        )

    def format(self,
               content: Any,
               instruction: str,
               **context: Any) -> FormatResult:
        """Format content using configured prompts"""
        try:
            request = {
                'content': content,
                'instruction': instruction,
                **context
            }
            
            response = self.process(request)
            
            if not response.success:
                return FormatResult(
                    success=False,
                    value="",
                    raw_response=str(response.data),
                    error=response.error
                )

            return FormatResult(
                success=True,
                value=response.data.get("response", ""),
                raw_response=response.data.get("raw_content", "")
            )
            
        except Exception as e:
            logger.error("formatting.failed", error=str(e))
            return FormatResult(
                success=False,
                value="",
                raw_response="",
                error=str(e)
            )