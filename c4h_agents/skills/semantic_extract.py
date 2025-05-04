"""
Semantic extraction using configured prompts.
Path: src/skills/semantic_extract.py
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from c4h_agents.agents.base_agent import BaseAgent, AgentResponse 
from skills.shared.markdown_utils import extract_code_block, is_code_block
from c4h_agents.utils.logging import get_logger

logger = get_logger()

@dataclass
class ExtractResult:
    """Result of semantic extraction"""
    success: bool
    value: Any
    raw_response: str
    error: Optional[str] = None

class SemanticExtract(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize extractor with config."""
        # Pass positional parameters to BaseAgent
        super().__init__(config, "semantic_extract")
        
        # The unique_name is now stored by BaseAgent

    def _get_agent_name(self) -> str:
        return "semantic_extract"

    def _format_request(self, context: Dict[str, Any]) -> str:
        """Format extraction request using config template"""
        extract_template = self._get_prompt('extract')
        
        return extract_template.format(
            content=context.get('content', ''),
            instruction=context.get('instruction', ''),
            format=context.get('format_hint', 'default')
        )

    def extract(self,
                content: Any,
                instruction: str,
                format_hint: str = "default",
                **context: Any) -> ExtractResult:
        """Extract information using configured prompts"""
        try:
            request = {
                'content': content,
                'instruction': instruction,
                'format_hint': format_hint,
                **context
            }
            
            response = self.process(request)
            
            if not response.success:
                return ExtractResult(
                    success=False,
                    value=None,
                    raw_response=str(response.data),
                    error=response.error
                )

            extracted_content = response.data.get("response")
            if extracted_content and is_code_block(extracted_content):
                logger.debug("extract.processing_code_block")
                code_block = extract_code_block(extracted_content)
                processed_content = code_block.content
            else:
                processed_content = extracted_content

            return ExtractResult(
                success=True,
                value=processed_content,
                raw_response=response.data.get("raw_content", "")
            )
            
        except Exception as e:
            logger.error("extraction.failed", error=str(e))
            return ExtractResult(
                success=False,
                value=None,
                raw_response="",
                error=str(e)
            )

    def _process_llm_response(self, content: str, raw_response: Any) -> Dict[str, Any]:
        """Process raw LLM response - parent class will call this"""
        # We handle code block processing in extract() instead
        return super()._process_llm_response(content, raw_response)