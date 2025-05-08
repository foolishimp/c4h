# File: /Users/jim/src/apps/c4h_ai_dev/c4h_agents/agents/base_agent.py
# Corrections for errors related to 'config' and 'agent_name' scope

# Necessary Imports
import json
import uuid
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import re
from c4h_agents.agents.base_config import BaseConfig
from c4h_agents.lineage.event_logger import EventLogger # Keep import for type hints
from c4h_agents.agents.lineage_context import LineageContext # Correct import
from c4h_agents.agents.base_llm import BaseLLM
from c4h_agents.agents.continuation.continuation_handler import ContinuationHandler
from c4h_agents.agents.types import AgentResponse, AgentMetrics, LLMProvider, LLMMessages, LogDetail, SkillResult
from c4h_agents.config import create_config_node # Keep create_config_node
from c4h_agents.core.project import Project
# Import get_logger and log_config_node from the correct utility path
from c4h_agents.utils.logging import get_logger, log_config_node

# Logger instance for this module (if needed, otherwise self.logger is used)
# logger = get_logger() # Removed global logger instance if self.logger is always used

class BaseAgent(BaseConfig, BaseLLM):
    """Base agent implementation"""

    # Corrected __init__ method:
    def __init__(self, full_effective_config: Dict[str, Any], unique_name: str):
        # --- Step 1: Initialize BaseConfig FIRST to set up self.config and self.config_node ---
        # BaseConfig now accepts the full config directly
        BaseConfig.__init__(self, config=full_effective_config)

        # --- Step 2: Initialize BaseLLM ---
        BaseLLM.__init__(self) # Explicitly call BaseLLM init

        # --- Step 3: Store unique_name and agent_id ---
        self.unique_name = unique_name # Store the unique name passed in
        self.agent_id = str(uuid.uuid4())

        # --- Step 4: Define agent_name using the required method ---
        # This ensures _get_agent_name() works correctly later if needed,
        # and provides a variable for logging if preferred over calling the method repeatedly.
        agent_name = self._get_agent_name() # Now returns self.unique_name

        # --- Step 5: Initialize Logger AFTER self.config is set by BaseConfig ---
        # Bind initial context including the unique agent name
        log_context = {"agent": agent_name, "agent_id": self.agent_id}
        # Add project info if project was passed and initialized in BaseConfig
        if self.project:
            log_context.update({
                "project_name": self.project.metadata.name,
                "project_version": self.project.metadata.version,
                "project_root": str(self.project.paths.root)
            })
        # Get logger using self.config which was set in BaseConfig.__init__
        # Ensure get_logger can handle the config format correctly
        self.logger = get_logger(self.config).bind(**log_context)

        # --- Step 6: Log received config using self.config and self.logger ---
        # Use self.config, not a global 'config'
        # Use agent_name variable defined above
        log_config_node(self.logger, self.config, "workorder", log_prefix=f"{agent_name}.init_received")
        log_config_node(self.logger, self.config, "team.llm_config.agents", log_prefix=f"{agent_name}.init_received")
        log_config_node(self.logger, self.config, "llm_config.agents", log_prefix=f"{agent_name}.init_received")

        
        # Important: Log config structure for diagnosis
        if agent_name == "semantic_iterator":
            print(f"DEBUG - Config for semantic_iterator: {self.config}")
            print(f"DEBUG - llm_config keys: {self.config.get('llm_config', {}).keys()}")
            print(f"DEBUG - agents keys: {self.config.get('llm_config', {}).get('agents', {}).keys()}")
            print(f"DEBUG - semantic_iterator config: {self.config.get('llm_config', {}).get('agents', {}).get('semantic_iterator', {})}")
        
        # --- Step 7: Use persona configuration directly ---
        self.config_path = f"llm_config.agents.{agent_name}"
        agent_config = self.config_node.get_value(self.config_path) or {}
        
        # Get persona_key from task config - required for this approach
        self.persona_key = agent_config.get('persona_key')
        
        # Handle skill agents that operate both as skills and agents
        # Look for the agent in skills config if it's not in agents
        if not self.persona_key:
            skill_config = self.config_node.get_value(f"llm_config.skills.{agent_name}")
            if skill_config:
                # This is a skill-based agent, use a default persona
                self.logger.info(f"{agent_name}.init.skill_based_agent", 
                               agent_name=agent_name)
                self.persona_key = self._get_default_persona_key()
                self.logger.info(f"{agent_name}.init.using_default_persona", 
                               agent_name=agent_name,
                               persona_key=self.persona_key)
            else:
                # Not a skill-based agent and no persona key - this is an error
                self.logger.error(f"{agent_name}.init.missing_persona_key", 
                                agent_name=agent_name,
                                config_path=self.config_path)
                raise ValueError(f"Agent {agent_name} is missing required persona_key in configuration")
        
        # Set persona path - this is our primary configuration source
        self.persona_path = f"llm_config.personas.{self.persona_key}"
        
        # Verify persona exists
        persona_config = self.config_node.get_value(self.persona_path)
        if not persona_config:
            self.logger.error(f"{agent_name}.init.persona_not_found", 
                             persona_key=self.persona_key, 
                             persona_path=self.persona_path)
            raise ValueError(f"Persona '{self.persona_key}' not found at {self.persona_path}")
            
        self.logger.info(f"{agent_name}.init.using_persona", 
                       persona_key=self.persona_key, 
                       persona_path=self.persona_path)

        # --- Step 8: Resolve Provider, Model, Temperature primarily from persona ---
        # Use persona configuration as primary source with minimal fallbacks
        self.provider = LLMProvider(
            self.config_node.get_value(f"{self.persona_path}.provider") or \
            self.config_node.get_value("llm_config.default_provider") or \
            "anthropic"  # Ultimate fallback
        )
        
        self.model = (
            self.config_node.get_value(f"{self.persona_path}.model") or \
            self.config_node.get_value(f"llm_config.providers.{self.provider.value}.default_model") or \
            self.config_node.get_value("llm_config.default_model") or \
            "claude-3-opus-20240229"  # Ultimate fallback
        )
        
        # Temperature from persona
        temp_val = self.config_node.get_value(f"{self.persona_path}.temperature")
        self.temperature = float(temp_val) if temp_val is not None else 0.0

        self.logger.debug(f"{agent_name}.init.resolved_settings",
                          provider=self.provider.value, 
                          model=self.model, 
                          temp=self.temperature,
                          persona_key=self.persona_key)

        # --- Step 9: Continuation settings from persona ---
        # Use persona config with defaults only as fallback
        self.max_continuation_attempts = (
            self.config_node.get_value(f"{self.persona_path}.max_continuation_attempts") or 5  # Default value
        )
        
        self.continuation_token_buffer = (
            self.config_node.get_value(f"{self.persona_path}.continuation_token_buffer") or 1000  # Default value
        )

        # --- Step 10: Initialize metrics ---
        self.metrics = AgentMetrics(project=self.project.metadata.name if self.project else None)

        # --- Step 11: Set logging detail level from persona ---
        log_level_str = (
            self.config_node.get_value(f"{self.persona_path}.log_level") or
            self.config_node.get_value("logging.agent_level") or 
            self.config_node.get_value("logging.level") or 
            "basic"
        )
        self.log_level = LogDetail.from_str(log_level_str)

        # --- Step 12: Setup LiteLLM ---
        # _get_model_str uses self.provider/self.model resolved above
        self.model_str = self._get_model_str()
        # _get_provider_config uses self.config_node
        self._setup_litellm(self._get_provider_config(self.provider))

        # --- Step 13: Finalize logger binding with run_id ---
        # Get run_id from the effective config snapshot
        self.run_id = self._get_workflow_run_id() or str(uuid.uuid4())
        log_context.update({
             "provider": self.provider.serialize(),
             "model": self.model,
             "log_level": str(self.log_level),
             "run_id": self.run_id
        })
        self.logger = self.logger.bind(**log_context)

        # --- Step 15: Initialize Continuation Handler ---
        # Ensure it's initialized AFTER all necessary parent attributes (like logger, model_str) are set
        self._continuation_handler: Optional[ContinuationHandler] = ContinuationHandler(self)

        # --- Step 16: Final Log ---
        self.logger.info(f"{agent_name}.initialized",
                     persona_key=self.persona_key,
                     has_persona_config=bool(self.persona_path),
                     continuation_settings={
                         "max_attempts": self.max_continuation_attempts,
                         "token_buffer": self.continuation_token_buffer
                     })


    def _get_workflow_run_id(self) -> Optional[str]:
        """
        Extract workflow run ID from configuration using hierarchical path queries.
        
        IMPORTANT: This method is NOT intended to be overridden in the generic agent model.
        It provides standard configuration path resolution for workflow run IDs.
        The hierarchical search ensures compatibility with various configuration structures.
        
        Returns:
            Workflow run ID or None if not found
        """
        # Check hierarchical sources in order of priority
        # Ensure self.config_node is used
        run_id = (
            self.config_node.get_value("workflow_run_id") or
            self.config_node.get_value("system.runid") or
            self.config_node.get_value("runtime.workflow_run_id") or
            self.config_node.get_value("runtime.run_id") or
            self.config_node.get_value("runtime.workflow.id")
        )

        if run_id:
            return str(run_id)
        return None

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Main process entry point for handling requests.
        
        This is the primary method that MAY be overridden by specific agent implementations.
        It will first check if the agent has an execution plan, and if so, use the 
        ExecutionPlanExecutor to execute it. Otherwise, it will fall back to a simple 
        LLM interaction via _process(context).
        
        Note: Even when overriding this method, implementations should generally use the
        persona-based configuration approach rather than hardcoding prompts or behavior.
        
        Args:
            context: Input context containing data for processing
            
        Returns:
            Standard AgentResponse with results
        """
        # Check if this agent has an execution plan
        if self._has_execution_plan:
            return self._execute_plan(context)
        
        # Fall back to LLM processing
        return self._process(context)
        
    def _execute_plan(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Execute the agent's execution plan using the ExecutionPlanExecutor.
        
        Args:
            context: Input context containing data for processing
            
        Returns:
            Standard AgentResponse with results from the execution plan
        """
        try:
            # Get the execution plan
            execution_plan = self.get_execution_plan()
            if not execution_plan:
                self.logger.error("execute_plan.plan_not_found", 
                               agent_name=self._get_agent_name())
                return AgentResponse(
                    success=False,
                    data={},
                    error="Execution plan not found",
                    messages=None
                )
            
            # Prepare lineage context
            lineage_context = self._prepare_lineage_context(context)
            
            # Add execution metadata
            agent_execution_id = lineage_context.get("agent_execution_id")
            parent_id = lineage_context.get("parent_id")
            
            # Import executor here to avoid circular imports
            from c4h_agents.execution.executor import ExecutionPlanExecutor
            from c4h_agents.skills.registry import SkillRegistry
            
            # Initialize skill registry
            registry = SkillRegistry()
            registry.register_builtin_skills()
            registry.load_skills_from_config(self.config)
            
            # Create executor
            executor = ExecutionPlanExecutor(
                effective_config=self.config,
                skill_registry=registry
            )
            
            # Execute the plan
            self.logger.info("execute_plan.starting", 
                          agent_name=self._get_agent_name(),
                          agent_execution_id=agent_execution_id,
                          steps_count=len(execution_plan.get("steps", [])))
            
            result = executor.execute_plan(execution_plan, lineage_context)
            
            # Create response from execution result
            self.logger.info("execute_plan.completed", 
                          agent_name=self._get_agent_name(),
                          agent_execution_id=agent_execution_id,
                          steps_executed=result.steps_executed,
                          success=result.success)
            
            # Extract response from context or output
            response_content = result.output or result.context.get("response", "")
            
            # Create agent response
            return AgentResponse(
                success=result.success,
                data={
                    "response": response_content,
                    "execution_result": {
                        "steps_executed": result.steps_executed,
                        "success": result.success,
                        "error": result.error
                    },
                    "execution_metadata": {
                        "agent_execution_id": agent_execution_id,
                        "parent_id": parent_id,
                        "workflow_run_id": lineage_context.get("workflow_run_id"),
                        "agent_id": self.agent_id,
                        "agent_type": self._get_agent_name(),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                },
                error=result.error,
                messages=None
            )
            
        except Exception as e:
            self.logger.error("execute_plan.failed", 
                           agent_name=self._get_agent_name(),
                           error=str(e),
                           traceback=traceback.format_exc())
            
            return AgentResponse(
                success=False,
                data={},
                error=f"Error executing plan: {str(e)}",
                messages=None
            )

    def _prepare_lineage_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context with appropriate lineage tracking IDs.
        Ensures each execution has proper parent-child relationships.
        
        IMPORTANT: This method is NOT intended to be overridden in the generic agent model.
        It provides standard lineage context handling that works for all agent types.
        
        Args:
            context: The original context dictionary
            
        Returns:
            Context with added lineage tracking identifiers
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
                agent_id=parent_id, # Use agent's unique instance ID
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
        """
        Core internal processing method for agent functionality.
        
        IMPORTANT: This method is NOT intended to be overridden in the generic agent model.
        The core process logic is standardized for all agents, following these steps:
        1. Prepare lineage context tracking
        2. Extract primary data
        3. Get prompts and messages from configuration
        4. Call LLM with appropriate logging and error handling
        5. Process and return response
        
        Instead of overriding this method, agent implementations should override the public
        'process' method to add specialized pre/post processing while still leveraging this
        core implementation when needed.
        
        Args:
            context: Context dictionary with request information
            
        Returns:
            AgentResponse with results from the LLM interaction
        """
        try:
            if self._should_log(LogDetail.DETAILED): # Use self.logger below
                self.logger.info("agent.processing", context_keys=list(context.keys()) if context else None)

            # Prepare lineage tracking context
            lineage_context = self._prepare_lineage_context(context)
            agent_execution_id = lineage_context.get("agent_execution_id")
            parent_id = lineage_context.get("parent_id")

            self.logger.debug("agent.lineage_context", # Use self.logger
                        agent_id=self.agent_id,
                        agent_execution_id=agent_execution_id,
                        parent_id=parent_id, #
                        workflow_run_id=lineage_context.get("workflow_run_id"))

            # Extract data from context
            data = self._extract_data_from_context(lineage_context)

            # Prepare system and user messages with context for potential overrides
            system_message = self._get_system_message(lineage_context)
            user_message = self._format_request(data, lineage_context)

            if self._should_log(LogDetail.DEBUG): # Use self.logger below
                self.logger.debug("agent.messages",
                            system_length=len(system_message),
                            user_length=len(user_message),
                            user_message=user_message[:10] + "..." if len(user_message) > 10 else user_message)

            # Create complete messages object for LLM and lineage tracking
            messages = LLMMessages(
                system=system_message,
                user=user_message,
                formatted_request="",
                raw_context=lineage_context
            )

            try:
                # Get completion with automatic continuation handling (calls BaseLLM method)
                # Pass lineage_context to support runtime configuration overrides
                content, raw_response = self._get_completion_with_continuation(
                    [
                        {"role": "system", "content": messages.system},
                        {"role": "user", "content": messages.user}
                    ],
                    context=lineage_context
                )

                # Process response - Calls THIS class's _process_response
                processed_data = self._process_response(content, raw_response)

                # Add execution metadata
                processed_data["execution_metadata"] = {
                    "agent_execution_id": agent_execution_id,
                    "parent_id": parent_id,
                    "workflow_run_id": lineage_context.get("workflow_run_id"),
                    "agent_id": self.agent_id,
                    "agent_type": self._get_agent_name(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                # Calculate metrics
                response_metrics = {"token_usage": getattr(raw_response, 'usage', {})}

                # Return successful response with lineage tracking metadatae tracking metadata
                return AgentResponse(
                    success=True,
                    data=processed_data,
                    error=None,
                    messages=messages,
                    raw_output=raw_response, # Keep raw_output in AgentResponse if needed elsewhere
                    metrics=response_metrics
                )
            except Exception as e:
                self.logger.error("llm.completion_failed",
                        error=str(e),
                        agent_execution_id=agent_execution_id,
                        exc_info=True) # Log traceback for completion errors

                return AgentResponse(
                    success=False,
                    data={
                        "execution_metadata": {
                            "agent_execution_id": agent_execution_id,
                            "parent_id": parent_id,
                            "workflow_run_id": lineage_context.get("workflow_run_id"),
                            "agent_id": self.agent_id,
                            "agent_type": self._get_agent_name(),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "error": str(e)
                        }
                    },
                    error=f"LLM completion failed: {str(e)}",
                    messages=messages
                )
        except Exception as e:
            self.logger.error("process.failed", error=str(e), exc_info=True) # Log traceback
            return AgentResponse(success=False, data={}, error=str(e))

    def _get_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts primary data payload, ensuring it's a dictionary."""
        try:
            # Prioritize 'input_data' as the primary container
            if 'input_data' in context and isinstance(context['input_data'], dict):
                self.logger.debug("_get_data: using 'input_data' key")
                return context['input_data']
            # If input_data isn't a dict or doesn't exist, use the context itself if it's a dict
            elif isinstance(context, dict):
                self.logger.debug("_get_data: using context as data (input_data missing or not dict)")
                return context
            # Fallback: create a basic dict if context isn't suitable
            self.logger.warning("_get_data: context is not a suitable dict, creating basic wrapper", context_type=type(context).__name__)
            return {'content': str(context)}
        except Exception as e:
            self.logger.error("_get_data.failed", error=str(e))
            return {}
            
    def _get_system_message(self, context: Dict[str, Any] = None) -> str:
        """
        Get the system message from persona configuration with context-based overrides.
        
        This method checks for system message in the agent's configuration, with overrides
        from the context taking precedence. This allows for runtime customization.
        
        Args:
            context: Optional context that may contain overrides
            
        Returns:
            The system message string to use for the LLM
        """
        try:
            # Try to get the persona key for this agent
            persona_key = self.config_node.get_value(f"orchestration.teams.*.tasks[?name={self.unique_name}].persona_key")
            if not persona_key:
                self.logger.warning("_get_system_message.persona_key_not_found", agent_name=self.unique_name)
                persona_key = "default"
                
            # First check for context override (highest priority)
            if context and 'system_message' in context:
                self.logger.debug("_get_system_message.using_context_override")
                return str(context['system_message'])
                
            # Next try persona configuration (normal path)
            system_message = self.config_node.get_value(f"llm_config.personas.{persona_key}.prompts.system")
            if system_message:
                self.logger.debug("_get_system_message.using_persona_config", persona_key=persona_key)
                return str(system_message)
                
            # Fallback to legacy agent config structure
            system_message = self.config_node.get_value(f"llm_config.agents.{self.unique_name}.prompts.system")
            if system_message:
                self.logger.debug("_get_system_message.using_legacy_agent_config")
                return str(system_message)
                
            # Last resort - default message
            self.logger.warning("_get_system_message.no_system_message_found", persona_key=persona_key, agent_name=self.unique_name)
            return "You are an AI assistant that helps with code analysis and transformation tasks."
            
        except Exception as e:
            self.logger.error("_get_system_message.failed", error=str(e), exc_info=True)
            return "You are an AI assistant that helps with code analysis and transformation tasks."

    def _get_prompt(self, prompt_type: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get a prompt template with support for dynamic overrides.
        
        IMPORTANT: This uses standard configuration paths based on unique_name and persona_key.
        This method is NOT intended to be overridden in the generic agent model.
        All prompt templates are sourced from the persona configuration or runtime overrides,
        NOT from agent class-specific implementations.
        
        Args:
            prompt_type: Type of prompt to retrieve (e.g., 'user')
            context: Optional runtime context that may contain overrides
            
        Returns:
            Prompt template string from context override or persona config
        """
        # Initialize variables
        prompt = None
        prompt_key = prompt_type
        
        # First check for override in context if provided
        if context:
            # Look for override in agent_config_overrides structure
            overrides = context.get('agent_config_overrides', {}).get(self.unique_name, {})
            
            # Direct prompt override for this specific prompt_type
            prompt_override_key = f"{prompt_type}_prompt"
            if prompt_override_key in overrides:
                prompt = overrides[prompt_override_key]
                self.logger.debug("prompt.from_direct_override", 
                                agent_name=self.unique_name,
                                prompt_type=prompt_type,
                                prompt_length=len(prompt) if prompt else 0)
                
            # Or check for prompt_key override
            elif 'prompt_key' in overrides:
                prompt_key = overrides['prompt_key']
                self.logger.debug("prompt.using_key_override", 
                                agent_name=self.unique_name,
                                original_key=prompt_type,
                                override_key=prompt_key)
        
        # If no direct prompt override was found, get from persona using prompt_key
        if prompt is None:
            persona_prompt_path = f"{self.persona_path}.prompts.{prompt_key}"
            prompt = self.config_node.get_value(persona_prompt_path)
            
            if prompt is None:
                # Use self.logger which is initialized
                agent_name = self._get_agent_name()
                self.logger.error("prompt.template_not_found", 
                               prompt_type=prompt_key, 
                               agent=agent_name,
                               persona_path=persona_prompt_path)
                raise ValueError(f"No prompt template '{prompt_key}' found in persona '{self.persona_key}' for agent {agent_name}")
            
        # Ensure prompt is a string before returning
        return str(prompt) if prompt is not None else ""

    def _format_request(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format request data using a template from persona configuration.
        
        Args:
            data: Data to format
            context: Optional context that may contain overrides
            
        Returns:
            Formatted request string
        """
        try:
             # Check for a format template override in context
             template = None
             template_key = "user"  # Default template key
             
             # Check for overrides in context if provided
             if context:
                 overrides = context.get('agent_config_overrides', {}).get(self.unique_name, {})
                 
                 # Direct template override
                 if 'template' in overrides:
                     template = overrides['template']
                     self.logger.debug("format_request.using_direct_template_override", 
                                     agent_name=self.unique_name,
                                     template_length=len(template) if template else 0)
                 
                 # Template key override
                 elif 'template_key' in overrides:
                     template_key = overrides['template_key']
                     self.logger.debug("format_request.using_template_key_override", 
                                     agent_name=self.unique_name,
                                     template_key=template_key)
             
             # If no direct template override, get template from persona
             if template is None:
                 # Get template using updated _get_prompt method (which also handles overrides)
                 template = self._get_prompt(template_key, context)
             
             # Format the template with data if it contains format placeholders
             if isinstance(template, str) and '{' in template:
                 try:
                     return template.format(**data)
                 except KeyError as e:
                     self.logger.warning("format_request.missing_key_in_template", error=str(e))
                     # Fall back to JSON format on template key error
                     return json.dumps(data, indent=2, default=str)
             elif template:
                 # Template exists but doesn't need formatting
                 return str(template)
             else:
                 # No template, use default JSON format
                 self.logger.debug("format_request.using_default_json")
                 return json.dumps(data, indent=2, default=str)
        except Exception as e:
             self.logger.error("format_request.failed", error=str(e))
             return str(data) # Ultimate fallback
             
    def _get_agent_name(self) -> str:
        """
        Return the unique name of this agent.
        Overrides the placeholder in BaseLLM to return the actual agent name.
        """
        return self.unique_name
        
    def _get_default_persona_key(self) -> str:
        """
        Get a default persona key for skill-based agents.
        
        This is used when an agent is also registered as a skill and doesn't have
        a specific persona_key in its configuration.
        
        Returns:
            A valid persona key from the configuration
        """
        # First try to find a default agent persona
        default_persona = "discovery_v1" # Prioritize discovery since it's most general
        
        # Check if the discovery persona exists
        if self.config_node.get_value(f"llm_config.personas.{default_persona}"):
            return default_persona
            
        # Next try to find any valid persona
        persona_keys = self.config_node.get_keys("llm_config.personas") or []
        if persona_keys:
            # Use the first available persona
            return persona_keys[0]
            
        # Last resort fallback
        return "discovery_v1" # This will still fail, but with a clearer error message
        
    def call_skill(self, skill_name: str, skill_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context for a skill call with proper lineage tracking.
        
        Args:
            skill_name: Name of the skill to call
            skill_context: Context to pass to the skill
            
        Returns:
            Enhanced context with lineage tracking information
        """
        try:
            # Prepare skill execution context with appropriate lineage IDs
            skill_exec_context = LineageContext.create_skill_context(
                agent_id=self.agent_id,
                skill_type=skill_name,
                workflow_run_id=getattr(self, 'run_id', None) or skill_context.get('workflow_run_id'),
                base_context=skill_context
            )
            
            self.logger.debug("call_skill.context_prepared",
                            agent_id=self.agent_id,
                            skill=skill_name,
                            skill_execution_id=skill_exec_context.get("agent_execution_id"))
                            
            return skill_exec_context
            
        except Exception as e:
            self.logger.error("call_skill.failed",
                            error=str(e),
                            skill=skill_name,
                            exc_info=True)
            
            # Ensure we return a valid context even if lineage tracking fails
            return skill_context
    
    def _invoke_skill(self, skill_name: str, params: Dict[str, Any], context: Dict[str, Any] = None) -> SkillResult:
        """
        Invoke a registered skill with the specified parameters.
        
        Args:
            skill_name: The name of the skill to invoke (registered in config under llm_config.skills)
            params: Parameters to pass to the skill
            context: Optional context to use for lineage tracking and other metadata
            
        Returns:
            The result from the skill execution
        """
        try:
            # 1. Attempt to locate the skill configuration
            skill_config = self.config_node.get_value(f"llm_config.skills.{skill_name}")
            if not skill_config:
                error_msg = f"Skill '{skill_name}' not found in llm_config.skills configuration"
                self.logger.error("_invoke_skill.skill_not_found", skill_name=skill_name)
                return SkillResult(success=False, error=error_msg)
                
            # 2. Extract module, class, and method information
            module_path = skill_config.get("module")
            class_name = skill_config.get("class")
            method_name = skill_config.get("method", "execute")
            
            if not module_path or not class_name:
                error_msg = f"Skill '{skill_name}' configuration is missing module or class"
                self.logger.error("_invoke_skill.invalid_config", 
                                skill_name=skill_name, 
                                has_module=bool(module_path), 
                                has_class=bool(class_name))
                return SkillResult(success=False, error=error_msg)
                
            # 3. Prepare lineage context for the skill
            if context is None:
                context = {}
                
            skill_exec_context = self.call_skill(skill_name, context)
            
            # 4. Import the skill module and get the class
            self.logger.debug("_invoke_skill.importing", 
                            skill_name=skill_name, 
                            module_path=module_path, 
                            class_name=class_name)
            
            import importlib
            skill_module = importlib.import_module(module_path)
            skill_class = getattr(skill_module, class_name)
            
            # 5. Create configuration for the skill to inherit our config
            # Clone our configuration
            skill_full_config = dict(self.config) if self.config else {}
            
            # Ensure the config has llm_config.agents.{skill_name} with a persona_key
            if 'llm_config' not in skill_full_config:
                skill_full_config['llm_config'] = {}
            if 'agents' not in skill_full_config['llm_config']:
                skill_full_config['llm_config']['agents'] = {}
            
            # Set a persona_key for the skill if it doesn't have one
            agent_config = self.config_node.get_value(f"llm_config.agents.{skill_name}")
            if not agent_config or 'persona_key' not in agent_config:
                self.logger.debug("_invoke_skill.setting_persona_for_skill",
                                skill_name=skill_name,
                                persona_key=self.persona_key)
                
                skill_full_config['llm_config']['agents'][skill_name] = {
                    'persona_key': self.persona_key  # Use our own persona
                }
            
            # 6. Instantiate the skill with our config
            skill_instance = skill_class(skill_full_config)
            
            # 7. Call the skill method with the parameters
            self.logger.info("_invoke_skill.executing", 
                           skill_name=skill_name, 
                           skill_method=method_name,  # Changed from method_name to skill_method to avoid conflict
                           param_keys=list(params.keys()))
            
            skill_method = getattr(skill_instance, method_name)
            result = skill_method(**params)
            
            # 8. Process the result
            self.logger.info("_invoke_skill.completed", 
                           skill_name=skill_name,
                           success=getattr(result, 'success', None))
            
            # Ensure result is of correct type (SkillResult)
            if not isinstance(result, SkillResult):
                self.logger.warning("_invoke_skill.invalid_result_type", 
                                  skill_name=skill_name, 
                                  result_type=type(result).__name__)
                # Try to convert to SkillResult if possible
                if hasattr(result, 'success'):
                    return SkillResult(
                        success=result.success,
                        value=getattr(result, 'value', None) or getattr(result, 'data', None),
                        error=getattr(result, 'error', None)
                    )
                # Default
                return SkillResult(
                    success=True, 
                    value=result
                )
                
            return result
            
        except Exception as e:
            error_msg = f"Error executing skill {skill_name}: {str(e)}"
            self.logger.error("_invoke_skill.failed", 
                            skill_name=skill_name, 
                            error=str(e), 
                            exc_info=True)
            return SkillResult(success=False, error=error_msg)
    
    def _extract_data_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the primary data payload from the context, ensuring it's a dictionary.
        
        Args:
            context: The context dictionary from which to extract data
            
        Returns:
            The extracted data as a dictionary
        """
        try:
            # Prioritize 'input_data' as the primary container
            if 'input_data' in context and isinstance(context['input_data'], dict):
                self.logger.debug("_extract_data_from_context: using 'input_data' key")
                return context['input_data']
                
            # If input_data isn't a dict or doesn't exist, use the context itself
            if isinstance(context, dict):
                self.logger.debug("_extract_data_from_context: using context as data (input_data missing or not dict)")
                return context
                
            # Fallback: create a basic dict if context isn't suitable
            self.logger.warning(
                "_extract_data_from_context: context is not a suitable dict, creating basic wrapper", 
                context_type=type(context).__name__
            )
            return {'content': str(context)}
            
        except Exception as e:
            self.logger.error("_extract_data_from_context.failed", error=str(e))
            return {}
        
    def _should_log(self, detail_level: LogDetail) -> bool:
        """
        Check if the current log level allows logging at the specified detail level.
        
        Args:
            detail_level: The detail level to check against the current log level
            
        Returns:
            True if logging should occur at this detail level, False otherwise
        """
        try:
            # Default to BASIC if not explicitly set
            current_level = getattr(self, 'log_level', LogDetail.BASIC)
            
            # Compare level enum values
            return current_level.value >= detail_level.value
        except Exception:
            # Default to True for BASIC level in case of error
            return detail_level == LogDetail.BASIC
    
    def _process_response(self, content: str, raw_response: Any) -> Dict[str, Any]:
        """
        Process LLM response into standard format.
        
        Extracts and formats the response content into a standardized structure
        with metadata and metrics.
        
        Args:
            content: The extracted content from the LLM response
            raw_response: The full raw response object from the LLM
            
        Returns:
            Structured response dictionary
        """
        try:
            # Get content from response safely
            processed_content = content
            if self._should_log(LogDetail.DEBUG):
                self.logger.debug("agent.processing_response",
                            content_length=len(str(processed_content)) if processed_content else 0,
                            response_type=type(raw_response).__name__)

            # Create standard response structure
            response = {
                "response": processed_content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Add token usage metrics if available
            if hasattr(raw_response, 'usage'):
                usage = raw_response.usage
                usage_data = {
                    "completion_tokens": getattr(usage, 'completion_tokens', 0),
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                    "total_tokens": getattr(usage, 'total_tokens', 0)
                }
                self.logger.info("llm.token_usage", **usage_data)
                response["usage"] = usage_data

            return response
        except Exception as e:
            self.logger.error("response_processing.failed", error=str(e))
            return {
                "response": str(content),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }