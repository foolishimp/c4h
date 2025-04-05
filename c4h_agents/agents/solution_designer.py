"""
Solution designer implementation focused on synchronous operation.
Path: src/agents/solution_designer.py
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json
from c4h_agents.agents.base_agent import BaseAgent, LogDetail, AgentResponse 
from config import locate_config
from c4h_agents.utils.logging import get_logger

logger = get_logger()

class SolutionDesigner(BaseAgent):
    """Designs specific code modifications based on intent and discovery analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize designer with configuration."""
        super().__init__(config=config)
        logger.info("solution_designer.initialized")

    def _get_agent_name(self) -> str:
        """Get agent name for config lookup"""
        return "solution_designer"


    def _format_request(self, context: Dict[str, Any]) -> str:
        """Format solution design request"""
        try:
            # Get discovery data
            discovery_data = context.get('discovery_data', {})
            raw_output = discovery_data.get('raw_output', '')
            
            # Get intent from context or config
            intent = context.get('intent', {})
            if isinstance(intent, dict):
                intent_desc = intent.get('description', '')
            else:
                intent_desc = str(intent)

            # Log request components
            if self._should_log(LogDetail.DEBUG):
                logger.debug("solution_designer.format_request",
                            has_discovery=bool(raw_output),
                            discovery_length=len(raw_output),
                            intent_length=len(intent_desc),
                            iteration=context.get('iteration', 0))
                    
                logger.debug("solution_designer.request_preview",
                            intent_preview=intent_desc[:100] + "..." if len(intent_desc) > 100 else intent_desc,
                            discovery_preview=raw_output[:100] + "..." if len(raw_output) > 100 else raw_output)

            # Get solution template
            solution_template = self._get_prompt('solution')
            if self._should_log(LogDetail.DEBUG):
                logger.debug("solution_designer.template_loaded",
                            template_length=len(solution_template))

            # Format request
            formatted_request = solution_template.format(
                source_code=raw_output,
                intent=intent_desc
            )

            if self._should_log(LogDetail.DEBUG):
                logger.debug("solution_designer.request_formatted",
                            request_length=len(formatted_request))

            return formatted_request

        except Exception as e:
            logger.error("solution_designer.format_failed", 
                        error=str(e),
                        error_type=type(e).__name__)
            raise

    def _extract_context_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract consistent data whether from nested or flat context"""
        try:
            if 'input_data' in context:
                input_data = context['input_data']
                discovery_data = input_data.get('discovery_data', {})
                raw_output = discovery_data.get('raw_output', '')
                intent = input_data.get('intent', {})
                intent_desc = intent.get('description', '') if isinstance(intent, dict) else str(intent)
            else:
                discovery_data = context.get('discovery_data', {})
                raw_output = discovery_data.get('raw_output', '')
                intent = context.get('intent', {})
                intent_desc = intent.get('description', '') if isinstance(intent, dict) else str(intent)

            return {
                'source_code': raw_output,
                'intent': intent_desc,
                'iteration': context.get('iteration', 0)
            }

        except Exception as e:
            logger.error("solution_designer.context_extraction_failed", error=str(e))
            return {}

    def _validate_input(self, context: Dict[str, Any]) -> bool:
        """Validate required input data is present"""
        discovery_data = None
        raw_output = None
            
        if 'input_data' in context:
            discovery_data = context['input_data'].get('discovery_data', {})
        elif 'discovery_data' in context:
            discovery_data = context['discovery_data']

        if hasattr(discovery_data, 'raw_output'):
            raw_output = discovery_data.raw_output
        elif isinstance(discovery_data, dict):
            raw_output = discovery_data.get('raw_output')

        logger.info("solution_designer.validate_input", 
                has_discovery=bool(raw_output),
                has_context=bool(context))

        return bool(raw_output)

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Process solution design request"""
        try:
            logger.info("agent.processing", context_keys=list(context.keys()))
            
            # Format the request directly
            formatted_request = self._format_request(context)
            
            # Get system message
            system_message = self._get_system_message()
            
            # Create messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": formatted_request}
            ]
            
            # Get completion
            content, raw_response = self._get_completion_with_continuation(messages)
            
            # Process using our custom response processor 
            processed_data = self._process_response(content, raw_response)
            
            return AgentResponse(
                success=True,
                data=processed_data
            )

        except Exception as e:
            logger.error("solution_designer.process_failed", error=str(e))
            return AgentResponse(success=False, data={}, error=str(e))

    def _process_response(self, content: str, raw_response: Any) -> Dict[str, Any]:
        """Process LLM response into standard format"""
        try:
            # Extract content using standard helper
            content = self._get_llm_content(content)
            if not content:
                raise ValueError("No content in LLM response")
                
            # Log content length to help debug continuation issues
            self.logger.info("solution_designer.response_received", 
                        content_length=len(content) if content else 0)

            return {
                "changes": content,
                "raw_output": raw_response,
                "raw_content": content,
                "response": content,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            # Continue processing even if there's an error
            self.logger.error("solution_designer.response_processing_failed", 
                            error=str(e), 
                            content_type=type(content).__name__,
                            content_length=len(str(content)) if content else 0)
            
            # Return what we have, even if processing failed
            return {
                "error": str(e),
                "raw_output": raw_response,
                "raw_content": str(content),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _get_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from context with discovery results"""
        try:
            input_data = {}
            if isinstance(context, dict):
                if 'input_data' in context:
                    input_data = context['input_data']
                else:
                    input_data = context

                # Always log what we found
                logger.debug("solution_designer.data_extraction",
                            has_discovery='discovery_data' in input_data,
                            has_intent='intent' in input_data,
                            context_keys=list(context.keys()))
                    
                return input_data
                
            return {'content': str(context)}
                
        except Exception as e:
            logger.error("solution_designer.get_data_failed", error=str(e))
            return {}