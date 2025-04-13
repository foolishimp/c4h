"""
Base agent implementation with enhanced lineage tracking.
Path: c4h_agents/agents/base_agent.py

Updates:
- Initialize logger with configuration to ensure proper truncation
"""

# Original imports...
from typing import Dict, Any, Optional, List, Tuple
import structlog
import json
from pathlib import Path
from datetime import datetime
import uuid

from c4h_agents.core.project import Project
from c4h_agents.config import create_config_node
from c4h_agents.utils.logging import get_logger  # Keep this import
from .base_lineage import BaseLineage
from .lineage_context import LineageContext
from .types import LogDetail, LLMProvider, LLMMessages, AgentResponse, AgentMetrics
from .base_config import BaseConfig, log_operation
from .base_llm import BaseLLM

# Update to initialize logger with configuration
logger = get_logger()  # Global logger for module-level logging

class BaseAgent(BaseConfig, BaseLLM):
    """Base agent implementation"""
    
    def __init__(self, config: Dict[str, Any] = None, project: Optional[Project] = None):
        # Pass full config to BaseConfig
        super().__init__(config=config, project=project)
        
        # Create configuration node for hierarchical access
        self.config_node = create_config_node(self.config)
        agent_name = self._get_agent_name()
        
        # Generate stable agent instance ID
        self.agent_id = str(uuid.uuid4())
        
        # Ensure system namespace exists
        if "system" not in self.config:
            self.config["system"] = {}
            
        # Store agent ID in system namespace
        self.config["system"]["agent_id"] = self.agent_id

        # --- ADDED: Log the full configuration received by the agent ---
        try:
            # Use default=str for safe serialization of non-standard types like datetime
            config_dump = json.dumps(self.config, indent=2, default=str)
            logger.debug(f"{agent_name}.__init__.received_config", config_json=config_dump)
        except Exception as e:
            logger.error(f"{agent_name}.__init__.config_dump_failed", error=str(e))
        # --- END ADDED ---
            
        # 
        # Resolve provider, model, and temperature using hierarchical lookup
        agent_path = f"llm_config.agents.{agent_name}"
        provider_name = self.config_node.get_value(f"{agent_path}.provider") or self.config_node.get_value("llm_config.default_provider") or "anthropic"
        # --- ADDED: Log resolved provider ---
        logger.debug(f"{agent_name}.__init__.resolved_provider", provider_name=provider_name)
        # --- END ADDED ---
        self.provider = LLMProvider(provider_name)
        
        self.model = self.config_node.get_value(f"{agent_path}.model") or self.config_node.get_value("llm_config.default_model") or "claude-3-opus-20240229"
        self.temperature = self.config_node.get_value(f"{agent_path}.temperature") or 0
        # --- ADDED: Log resolved model ---
        logger.debug(f"{agent_name}.__init__.resolved_model", model_name=self.model)
        # --- END ADDED ---
        
        # Continuation settings
        self.max_continuation_attempts = self.config_node.get_value(f"{agent_path}.max_continuation_attempts") or 5
        self.continuation_token_buffer = self.config_node.get_value(f"{agent_path}.continuation_token_buffer") or 1000
        
        # Initialize metrics
        self.metrics = AgentMetrics(project=self.project.metadata.name if self.project else None)
        
        # Set logging detail level from config
        log_level = self.config_node.get_value("logging.agent_level") or "basic"
        self.log_level = LogDetail.from_str(log_level)
        
        # Build model string and setup LiteLLM
        self.model_str = self._get_model_str()
        self._setup_litellm(self._get_provider_config(self.provider))
        
        # Initialize logger with enhanced context and agent configuration
        log_context = {
            "agent": agent_name,
            "agent_id": self.agent_id,
            "provider": self.provider.serialize(),
            "model": self.model,
            "log_level": str(self.log_level)
        }
        if self.project:
            log_context.update({
                "project_name": self.project.metadata.name,
                "project_version": self.project.metadata.version,
                "project_root": str(self.project.paths.root)
            })
            
        # Update to pass configuration to logger
        self.logger = get_logger(self.config).bind(**log_context)
        
        # Initialize lineage tracking with the full configuration
        self.lineage = None
        try:
            # Log what run_id we're using
            run_id = self._get_workflow_run_id()
            if run_id:
                logger.debug(f"{agent_name}.using_workflow_run_id", 
                            run_id=run_id, 
                            source="config", 
                            config_keys=list(self.config.keys()))
            
            self.lineage = BaseLineage(
                namespace="c4h_agents",
                agent_name=agent_name,
                config=self.config
            )
            
            # Store lineage run ID for consistency
            self.run_id = self.lineage.run_id
            
        except Exception as e:
            logger.error(f"{agent_name}.lineage_init_failed", error=str(e))
            # Generate run ID if lineage fails
            self.run_id = str(uuid.uuid4())

        logger.info(f"{agent_name}.initialized", 
                    run_id=self.run_id,
                    continuation_settings={
                        "max_attempts": self.max_continuation_attempts,
                        "token_buffer": self.continuation_token_buffer
                    },
                    **log_context)
    
    def _get_workflow_run_id(self) -> Optional[str]:
        """Extract workflow run ID from configuration using hierarchical path queries"""
        # Check hierarchical sources in order of priority
        run_id = (
            # 1. Direct context parameter (highest priority)
            self.config_node.get_value("workflow_run_id") or
            # 2. System namespace
            self.config_node.get_value("system.runid") or
            # 3. Runtime configuration (backward compatibility)
            self.config_node.get_value("runtime.workflow_run_id") or
            self.config_node.get_value("runtime.run_id") or
            # 4. Workflow section 
            self.config_node.get_value("runtime.workflow.id")
        )
        
        if run_id:
            return str(run_id)
        return None

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Main process entry point"""
        return self._process(context)

    def _prepare_lineage_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context with appropriate lineage tracking IDs.
        Ensures each execution has proper parent-child relationships.
        """
        # Extract workflow run ID from context or config
        workflow_run_id = context.get("workflow_run_id", self.run_id)
        
        # Extract parent ID if available
        parent_id = context.get("parent_id")
        
        # Check if context already has agent_execution_id
        if "agent_execution_id" in context:
            # Context already has tracking IDs, preserve them
            return context
            
        # No tracking IDs, create them using LineageContext utility
        # If there's a parent ID, we're being called by another agent
        if parent_id:
            # We're a sub-component (skill) being called by an agent
            return LineageContext.create_skill_context(
                agent_id=parent_id,
                skill_type=self._get_agent_name(),
                workflow_run_id=workflow_run_id,
                base_context=context
            )
        else:
            # We're a top-level agent in the workflow
            return LineageContext.create_agent_context(
                workflow_run_id=workflow_run_id,
                agent_type=self._get_agent_name(),
                base_context=context
            )

    def _process(self, context: Dict[str, Any]) -> AgentResponse:
        try:
            if self._should_log(LogDetail.DETAILED):
                logger.info("agent.processing", context_keys=list(context.keys()) if context else None)
            
            # Prepare lineage tracking context
            lineage_context = self._prepare_lineage_context(context)
            agent_execution_id = lineage_context.get("agent_execution_id")
            parent_id = lineage_context.get("parent_id")
            
            logger.debug("agent.lineage_context", 
                        agent_id=self.agent_id,
                        agent_execution_id=agent_execution_id,
                        parent_id=parent_id,
                        workflow_run_id=lineage_context.get("workflow_run_id"))
            
            # Extract data from context
            data = self._get_data(lineage_context)
            
            # Prepare system and user messages
            system_message = self._get_system_message()
            user_message = self._format_request(data)
            
            if self._should_log(LogDetail.DEBUG):
                logger.debug("agent.messages",
                            system_length=len(system_message),
                            user_length=len(user_message),
                            agent_execution_id=agent_execution_id,
                            system=system_message[:10] + "..." if len(system_message) > 10 else system_message,
                            user_message=user_message[:10] + "..." if len(user_message) > 10 else user_message)
                            
            # Create complete messages object for LLM and lineage tracking
            # FIXED: Don't duplicate content between user and formatted_request
            # Only store the formatted_request if it's different from user_message
            messages = LLMMessages(
                system=system_message,
                user=user_message,
                formatted_request="",  # Don't store duplicate content
                raw_context=lineage_context
            )
            
            try:
                # Check if lineage tracking is enabled
                lineage_enabled = hasattr(self, 'lineage') and self.lineage and getattr(self.lineage, 'enabled', False)
                
                # Get completion with automatic continuation handling
                content, raw_response = self._get_completion_with_continuation([
                    {"role": "system", "content": messages.system},
                    {"role": "user", "content": messages.user}
                ])
                
                # Process response
                processed_data = self._process_response(content, raw_response)
                
                # Add execution metadata to processed data
                processed_data["execution_metadata"] = {
                    "agent_execution_id": agent_execution_id,
                    "parent_id": parent_id,
                    "workflow_run_id": lineage_context.get("workflow_run_id"),
                    "agent_id": self.agent_id,
                    "agent_type": self._get_agent_name(),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Calculate metrics
                response_metrics = {"token_usage": getattr(raw_response, 'usage', {})}
                
                # Track lineage if enabled
                if lineage_enabled:
                    try:
                        logger.debug("lineage.tracking_attempt", 
                                    agent=self._get_agent_name(), 
                                    agent_execution_id=agent_execution_id,
                                    parent_id=parent_id,
                                    has_context=bool(lineage_context), 
                                    has_messages=bool(messages), 
                                    has_metrics=hasattr(raw_response, 'usage'))
                                    
                        # Track LLM interaction with full context for event sourcing
                        if hasattr(self.lineage, 'track_llm_interaction'):
                            self.lineage.track_llm_interaction(
                                context=lineage_context,
                                messages=messages,
                                response=raw_response,
                                metrics=response_metrics
                            )
                        logger.info("lineage.tracking_complete", 
                                agent=self._get_agent_name(),
                                agent_execution_id=agent_execution_id)
                    except Exception as e:
                        logger.error("lineage.tracking_failed", 
                                    error=str(e), 
                                    error_type=type(e).__name__, 
                                    agent=self._get_agent_name(),
                                    agent_execution_id=agent_execution_id)
                else:
                    logger.debug("lineage.tracking_skipped",
                            has_lineage=hasattr(self, 'lineage'),
                            lineage_enabled=getattr(self.lineage, 'enabled', False) if hasattr(self, 'lineage') else False,
                            agent=self._get_agent_name())
                
                # Return successful response with lineage tracking metadata
                return AgentResponse(
                    success=True,
                    data=processed_data,
                    error=None,
                    messages=messages,
                    raw_output=raw_response,
                    metrics=response_metrics
                )
            except Exception as e:
                # Handle errors with lineage tracking
                if lineage_enabled and hasattr(self.lineage, 'track_llm_interaction'):
                    try:
                        error_context = {
                            **lineage_context,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                        self.lineage.track_llm_interaction(
                            context=error_context,
                            messages=messages,
                            response={"error": str(e)},
                            metrics={"error": True}
                        )
                    except Exception as lineage_error:
                        logger.error("lineage.failure_tracking_failed", 
                                    error=str(lineage_error),
                                    original_error=str(e))
                
                logger.error("llm.completion_failed", 
                        error=str(e),
                        agent_execution_id=agent_execution_id)
                
                return AgentResponse(
                    success=False, 
                    data={
                        "execution_metadata": {
                            "agent_execution_id": agent_execution_id,
                            "parent_id": parent_id,
                            "workflow_run_id": lineage_context.get("workflow_run_id"),
                            "agent_id": self.agent_id,
                            "agent_type": self._get_agent_name(),
                            "timestamp": datetime.utcnow().isoformat(),
                            "error": str(e)
                        }
                    }, 
                    error=f"LLM completion failed: {str(e)}", 
                    messages=messages
                )
        except Exception as e:
            logger.error("process.failed", error=str(e))
            return AgentResponse(success=False, data={}, error=str(e))
        
    def _get_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if isinstance(context, dict):
                return context
            return {'content': str(context)}
        except Exception as e:
            logger.error("get_data.failed", error=str(e))
            return {}

    def _format_request(self, context: Dict[str, Any]) -> str:
        return str(context)
    
    def _get_llm_content(self, response: Any) -> Any:
        """Extract content from LLM response with robust error handling for different response formats"""
        try:
            # Handle different response types
            if hasattr(response, 'choices') and response.choices:
                # Standard response object
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    content = response.choices[0].message.content
                    if self._should_log(LogDetail.DEBUG):
                        logger.debug("content.extracted_from_model", content_length=len(content) if content else 0)
                    return content
                # Handle delta format (used in streaming)
                elif hasattr(response.choices[0], 'delta') and hasattr(response.choices[0].delta, 'content'):
                    content = response.choices[0].delta.content
                    if self._should_log(LogDetail.DEBUG):
                        logger.debug("content.extracted_from_delta", content_length=len(content) if content else 0)
                    return content
            
            # If we have a simple string content
            if isinstance(response, str):
                return response
                
            # If response is already processed (dict with 'response' key)
            if isinstance(response, dict) and 'response' in response:
                return response['response']
                
            # Last resort fallback - convert to string
            result = str(response)
            logger.warning("content.extraction_fallback", 
                        response_type=type(response).__name__, 
                        content_preview=result[:100] if len(result) > 100 else result)
            return result
        except Exception as e:
            logger.error("content_extraction.failed", error=str(e))
            return str(response)

    def _process_response(self, content: str, raw_response: Any) -> Dict[str, Any]:
        """Process LLM response into standard format without duplicating raw output"""
        try:
            # Extract content using standard helper
            processed_content = self._get_llm_content(content)
            if self._should_log(LogDetail.DEBUG):
                logger.debug("agent.processing_response", 
                            content_length=len(str(processed_content)) if processed_content else 0, 
                            response_type=type(raw_response).__name__)
            
            # Create standard response structure WITHOUT including raw_output
            response = {
                "response": processed_content,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add token usage metrics if available
            if hasattr(raw_response, 'usage'):
                usage = raw_response.usage
                usage_data = {
                    "completion_tokens": getattr(usage, 'completion_tokens', 0),
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                    "total_tokens": getattr(usage, 'total_tokens', 0)
                }
                logger.info("llm.token_usage", **usage_data)
                response["usage"] = usage_data
                
            return response
        except Exception as e:
            logger.error("response_processing.failed", error=str(e))
            return {
                "response": str(content),
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }

    def _get_required_keys(self) -> List[str]:
        return []

    def _get_agent_name(self) -> str:
        return self.__class__.__name__.lower()

    def _get_system_message(self) -> str:
        return self.config.get("llm_config", {}).get("agents", {}).get(self._get_agent_name(), {}).get("prompts", {}).get("system", "")

    def _get_prompt(self, prompt_type: str) -> str:
        prompts = self.config.get("llm_config", {}).get("agents", {}).get(self._get_agent_name(), {}).get("prompts", {})
        if prompt_type not in prompts:
            raise ValueError(f"No prompt template found for type: {prompt_type}")
        return prompts[prompt_type]

    def call_skill(self, skill_name: str, skill_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a skill with proper lineage tracking.
        
        Args:
            skill_name: Name of the skill to call
            skill_context: Context to pass to the skill
            
        Returns:
            Result from the skill
        """
        try:
            # Prepare lineage tracking context for the skill
            lineage_skill_context = LineageContext.create_skill_context(
                agent_id=self.agent_id,
                skill_type=skill_name,
                workflow_run_id=self.run_id,
                base_context=skill_context
            )
            
            logger.debug("agent.calling_skill", 
                    agent_id=self.agent_id,
                    skill=skill_name,
                    skill_execution_id=lineage_skill_context.get("agent_execution_id"))
            
            # Return enhanced context - the skill itself will handle execution
            return lineage_skill_context
        except Exception as e:
            logger.error("agent.skill_context_failed", 
                    error=str(e),
                    skill=skill_name)
            # If lineage context fails, fall back to original context
            return skill_context