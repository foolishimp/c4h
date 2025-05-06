"""
Generic agent implementations using path-addressable configuration.
Path: c4h_agents/agents/generic.py
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json
import os
import subprocess

from c4h_agents.agents.base_agent import BaseAgent, AgentResponse
from c4h_agents.agents.types import LLMMessages, LogDetail, AgentType, SkillResult
from c4h_agents.utils.logging import get_logger

logger = get_logger()

class GenericLLMAgent(BaseAgent):
    """
    Generic LLM agent that performs a single LLM interaction based on configuration.
    This is the primary agent type for general LLM interactions and replaces GenericSingleShotAgent.
    
    Key capabilities:
    - Uses persona-based configuration for prompts and parameters
    - Supports runtime overrides through context
    - Can invoke skills if configured instead of using LLM directly
    - Special handling for discovery agents with project scanning
    """
    
    def __init__(self, full_effective_config: Dict[str, Any], unique_name: str):
        """
        Initialize with full effective configuration snapshot and unique name.
        
        Args:
            full_effective_config: Complete effective configuration after all merging
            unique_name: Unique instance name used for configuration path
        """
        # Initialize base agent with full config
        super().__init__(full_effective_config, unique_name)
        
        # The agent type is stored for introspection
        self.agent_type = AgentType.GENERIC_LLM
        
        # Configure logger with agent-specific fields
        self.logger = self.logger.bind(
            agent_name=self.unique_name, 
            config_path=self.config_path,
            agent_type=self.agent_type.value
        )
        self.logger.info("generic_agent.initialized", agent_type=self.agent_type.value)

    def _get_agent_name(self) -> str:
        """Return the unique name for this agent instance."""
        return self.unique_name

    def _get_system_message(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get system message with support for dynamic overrides.
        
        Args:
            context: Optional runtime context that may contain overrides
            
        Returns:
            System message string from context override or persona config
        """
        # Use BaseAgent's implementation which now includes persona fallback
        return super()._get_system_message(context)

    def _format_request(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format user request based on path-addressable config with support for runtime overrides.
        
        This method:
        1. Checks for overrides in context
        2. Gets the prompt template specified in config or persona
        3. Formats it using data variables
        4. Returns the formatted prompt
        
        Special handling for discovery agent:
        - If this is a discovery agent and tartxt_config is present,
          run the tartxt.py script to gather project information
        
        Args:
            data: Data to format
            context: Optional runtime context that may contain overrides
            
        Returns:
            Formatted request string
        """
        # Make a copy of data to avoid modifying the original
        data_copy = dict(data)
        
        # Check if this is a discovery agent with tartxt config
        if (self.unique_name == "discovery_phase" or 
            self.config_path.endswith(".discovery") or 
            "discovery" in self.unique_name):
            
            # Get tartxt_config from context override or persona
            tartxt_config = None
            
            # First check for override in context
            if context:
                overrides = context.get('agent_config_overrides', {}).get(self.unique_name, {})
                if 'tartxt_config' in overrides and isinstance(overrides['tartxt_config'], dict):
                    tartxt_config = overrides['tartxt_config']
                    self.logger.debug("discovery.tartxt_config_from_override")
            
            # If no override, get from persona
            if not tartxt_config:
                tartxt_config = self.config_node.get_value(f"{self.persona_path}.tartxt_config")
            
            if tartxt_config and isinstance(tartxt_config, dict):
                self.logger.info("discovery.tartxt_config_found", 
                               config_keys=list(tartxt_config.keys()),
                               from_override=bool(context and 'tartxt_config' in context.get('agent_config_overrides', {}).get(self.unique_name, {})))
                
                # Get script path
                script_path = tartxt_config.get("script_path")
                if script_path and os.path.exists(script_path):
                    # Get project path from data/context
                    project_path = data_copy.get("project_path")
                    if not project_path and context:
                        project_path = context.get("project_path")
                    if not project_path:
                        self.logger.warning("discovery.no_project_path_in_context")
                        project_path = os.getcwd()
                    
                    # Get exclusions
                    exclusions = tartxt_config.get("exclusions", [])
                    exclusion_arg = ",".join(exclusions) if exclusions else ""
                    
                    # Create output file path
                    output_file = os.path.join(os.getcwd(), "workspaces", f"project_scan_{self.run_id}.txt")
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    
                    # Build command
                    cmd = [
                        script_path,
                        project_path,
                        "-f", output_file,
                        "-H", "0"  # No history initially for speed
                    ]
                    
                    if exclusion_arg:
                        cmd.extend(["-x", exclusion_arg])
                    
                    # Run tartxt.py
                    self.logger.info("discovery.running_tartxt", cmd=cmd)
                    try:
                        import subprocess
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            check=False
                        )
                        
                        if result.returncode != 0:
                            self.logger.error("discovery.tartxt_failed", 
                                            returncode=result.returncode,
                                            stderr=result.stderr)
                        else:
                            self.logger.info("discovery.tartxt_success", output_file=output_file)
                            
                            # Read output file and add to data
                            try:
                                if os.path.exists(output_file):
                                    with open(output_file, 'r', encoding='utf-8') as f:
                                        project_content = f.read()
                                    
                                    # Update data with project content
                                    data_copy['project_content'] = project_content
                                    data_copy['project_scan_file'] = output_file
                                    self.logger.info("discovery.added_project_content", 
                                                  content_length=len(project_content))
                                else:
                                    self.logger.error("discovery.output_file_not_found", file=output_file)
                            except Exception as e:
                                self.logger.error("discovery.read_output_failed", error=str(e))
                    except Exception as e:
                        self.logger.error("discovery.tartxt_execution_failed", error=str(e))
        
        # Get template key from overrides or persona config
        template_key = "user"  # Default template key
        template = None
        
        # Check for overrides in context
        if context:
            overrides = context.get('agent_config_overrides', {}).get(self.unique_name, {})
            
            # Direct template override
            if 'template' in overrides:
                template = overrides['template']
                self.logger.debug("format_request.using_direct_template_override", 
                               template_length=len(template) if template else 0)
            
            # Or template key override
            elif 'template_key' in overrides:
                template_key = overrides['template_key']
                self.logger.debug("format_request.using_template_key_override", template_key=template_key)
        
        # If no direct template override, check persona config for template_name
        if template is None:
            # First try to get the template_name from persona config
            persona_template_name = self.config_node.get_value(f"{self.persona_path}.prompts.template_name")
            if persona_template_name:
                template_key = persona_template_name
                self.logger.debug("format_request.using_persona_template_name", template_key=template_key)
        
        # If still no template, get it from persona prompts using template_key
        if template is None:
            template = self.config_node.get_value(f"{self.persona_path}.prompts.{template_key}")
            
            if not template:
                self.logger.warning("prompt_template.not_found", 
                                  template_key=template_key,
                                  persona_key=self.persona_key,
                                  persona_path=f"{self.persona_path}.prompts.{template_key}")
                # Fallback to simple JSON string if template not found
                return json.dumps(data_copy, indent=2)
        
        try:
            # Format the template with context variables
            # Handle both string templates with format() and direct values
            if isinstance(template, str) and "{" in template:
                formatted_prompt = template.format(**data_copy)
            else:
                formatted_prompt = str(template)
                
            if self._should_log(LogDetail.DEBUG):
                self.logger.debug("prompt.formatted", 
                                template_key=template_key,
                                prompt_length=len(formatted_prompt))
                
            return formatted_prompt
            
        except KeyError as e:
            self.logger.error("prompt.format_key_error", 
                            error=str(e), 
                            template_key=template_key)
            # Fall back to sending the context directly as JSON
            return json.dumps(data_copy, indent=2)
        except Exception as e:
            self.logger.error("prompt.format_error", 
                            error=str(e), 
                            template_key=template_key)
            return json.dumps(data_copy, indent=2)

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a request using a single LLM interaction or skill invocation.
        Checks for skill configuration before falling back to LLM.
        
        Args:
            context: Input context containing data for processing
            
        Returns:
            Standard AgentResponse with results
        """
        # Check if a skill is configured for this agent
        agent_config = self.config_node.get_value(self.config_path)
        skill_identifier = None
        skill_params = {}
        
        if agent_config and isinstance(agent_config, dict):
            skill_identifier = agent_config.get('skill')
            skill_params = agent_config.get('skill_params', {})
            
        # Also check persona config for skill configuration
        if not skill_identifier and self.persona_key:
            persona_config = self.config_node.get_value(self.persona_path)
            if persona_config and isinstance(persona_config, dict):
                skill_identifier = persona_config.get('skill')
                skill_params = persona_config.get('skill_params', {})
                
        # If a skill is configured, invoke it
        if skill_identifier:
            self.logger.info("agent.using_skill", 
                           skill=skill_identifier, 
                           agent_name=self.unique_name)
                           
            # Prepare skill parameters from context
            # Start with any configured parameters
            execution_params = dict(skill_params) if skill_params else {}
            
            # Add context as input if not already present
            if 'input' not in execution_params:
                execution_params['input'] = context
                
            # Add agent information
            execution_params['agent_name'] = self.unique_name
            execution_params['agent_id'] = self.agent_id
            
            # Invoke the skill
            skill_result = self._invoke_skill(skill_identifier, execution_params)
            
            # Convert to AgentResponse
            if isinstance(skill_result, SkillResult):
                # Use the built-in conversion
                return skill_result.to_agent_response()
            else:
                # Handle unexpected result type
                self.logger.warning("agent.unexpected_skill_result_type", 
                                  type=type(skill_result).__name__)
                
                # Attempt to create a valid AgentResponse
                success = getattr(skill_result, 'success', True)
                error = getattr(skill_result, 'error', None)
                
                return AgentResponse(
                    success=success,
                    data={"skill_result": skill_result},
                    error=error
                )
        
        # No skill configured or skill invocation failed, fall back to LLM
        return super().process(context)

# Legacy class that inherits from GenericLLMAgent for backward compatibility
class GenericSingleShotAgent(GenericLLMAgent):
    """
    DEPRECATED: Use GenericLLMAgent instead.
    
    This class is maintained for backward compatibility only and will be removed in a future release.
    All functionality is now provided by GenericLLMAgent with identical behavior.
    
    Generic agent that performs a single LLM interaction based on configuration.
    Uses path-addressable config to access prompts and settings.
    """
    
    def __init__(self, full_effective_config: Dict[str, Any], unique_name: str):
        """
        Initialize with full effective configuration snapshot and unique name.
        
        Args:
            full_effective_config: Complete effective configuration after all merging
            unique_name: Unique instance name used for configuration path
        """
        # Initialize base agent with full config
        super().__init__(full_effective_config, unique_name)
        
        # Log deprecation warning
        self.logger = self.logger.bind(agent_name=self.unique_name, config_path=self.config_path)
        self.logger.warning("generic_agent.deprecated", 
                           message="GenericSingleShotAgent is deprecated. Use GenericLLMAgent instead.",
                           agent_type="GenericSingleShotAgent")

    def _get_agent_name(self) -> str:
        """Return the unique name for this agent instance."""
        return self.unique_name

    def _get_system_message(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get system message with support for dynamic overrides.
        
        Args:
            context: Optional runtime context that may contain overrides
            
        Returns:
            System message string from context override or persona config
        """
        # Use BaseAgent's implementation which now includes persona fallback
        return super()._get_system_message(context)

    def _format_request(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format user request based on path-addressable config with support for runtime overrides.
        
        This method:
        1. Checks for overrides in context
        2. Gets the prompt template specified in config or persona
        3. Formats it using data variables
        4. Returns the formatted prompt
        
        Special handling for discovery agent:
        - If this is a discovery agent and tartxt_config is present,
          run the tartxt.py script to gather project information
        
        Args:
            data: Data to format
            context: Optional runtime context that may contain overrides
            
        Returns:
            Formatted request string
        """
        # Make a copy of data to avoid modifying the original
        data_copy = dict(data)
        
        # Check if this is a discovery agent with tartxt config
        if (self.unique_name == "discovery_phase" or 
            self.config_path.endswith(".discovery") or 
            "discovery" in self.unique_name):
------------END OVERLAP------------
            
            # Get tartxt_config from context override or persona
            tartxt_config = None
            
            # First check for override in context
            if context:
                overrides = context.get('agent_config_overrides', {}).get(self.unique_name, {})
                if 'tartxt_config' in overrides and isinstance(overrides['tartxt_config'], dict):
                    tartxt_config = overrides['tartxt_config']
                    self.logger.debug("discovery.tartxt_config_from_override")
            
            # If no override, get from persona
            if not tartxt_config:
                tartxt_config = self.config_node.get_value(f"{self.persona_path}.tartxt_config")
            
            if tartxt_config and isinstance(tartxt_config, dict):
                self.logger.info("discovery.tartxt_config_found", 
                               config_keys=list(tartxt_config.keys()),
                               from_override=bool(context and 'tartxt_config' in context.get('agent_config_overrides', {}).get(self.unique_name, {})))
                
                # Get script path
                script_path = tartxt_config.get("script_path")
                if script_path and os.path.exists(script_path):
                    # Get project path from data/context
                    project_path = data_copy.get("project_path")
                    if not project_path and context:
                        project_path = context.get("project_path")
                    if not project_path:
                        self.logger.warning("discovery.no_project_path_in_context")
                        project_path = os.getcwd()
                    
                    # Get exclusions
                    exclusions = tartxt_config.get("exclusions", [])
                    exclusion_arg = ",".join(exclusions) if exclusions else ""
                    
                    # Create output file path
                    output_file = os.path.join(os.getcwd(), "workspaces", f"project_scan_{self.run_id}.txt")
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    
                    # Build command
                    cmd = [
                        script_path,
                        project_path,
                        "-f", output_file,
                        "-H", "0"  # No history initially for speed
                    ]
                    
                    if exclusion_arg:
                        cmd.extend(["-x", exclusion_arg])
                    
                    # Run tartxt.py
                    self.logger.info("discovery.running_tartxt", cmd=cmd)
                    try:
                        import subprocess
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            check=False
                        )
                        
                        if result.returncode != 0:
                            self.logger.error("discovery.tartxt_failed", 
                                            returncode=result.returncode,
                                            stderr=result.stderr)
                        else:
                            self.logger.info("discovery.tartxt_success", output_file=output_file)
                            
                            # Read output file and add to data
                            try:
                                if os.path.exists(output_file):
                                    with open(output_file, 'r', encoding='utf-8') as f:
                                        project_content = f.read()
                                    
                                    # Update data with project content
                                    data_copy['project_content'] = project_content
                                    data_copy['project_scan_file'] = output_file
                                    self.logger.info("discovery.added_project_content", 
                                                  content_length=len(project_content))
                                else:
                                    self.logger.error("discovery.output_file_not_found", file=output_file)
                            except Exception as e:
                                self.logger.error("discovery.read_output_failed", error=str(e))
                    except Exception as e:
                        self.logger.error("discovery.tartxt_execution_failed", error=str(e))
        
        # Get template key from overrides or persona config
        template_key = "user"  # Default template key
        template = None
        
        # Check for overrides in context
        if context:
            overrides = context.get('agent_config_overrides', {}).get(self.unique_name, {})
            
            # Direct template override
            if 'template' in overrides:
                template = overrides['template']
                self.logger.debug("format_request.using_direct_template_override", 
                               template_length=len(template) if template else 0)
            
            # Or template key override
            elif 'template_key' in overrides:
                template_key = overrides['template_key']
                self.logger.debug("format_request.using_template_key_override", template_key=template_key)
        
        # If no direct template override, check persona config for template_name
        if template is None:
            # First try to get the template_name from persona config
            persona_template_name = self.config_node.get_value(f"{self.persona_path}.prompts.template_name")
            if persona_template_name:
                template_key = persona_template_name
                self.logger.debug("format_request.using_persona_template_name", template_key=template_key)
        
        # If still no template, get it from persona prompts using template_key
        if template is None:
            template = self.config_node.get_value(f"{self.persona_path}.prompts.{template_key}")
            
            if not template:
                self.logger.warning("prompt_template.not_found", 
                                  template_key=template_key,
                                  persona_key=self.persona_key,
                                  persona_path=f"{self.persona_path}.prompts.{template_key}")
                # Fallback to simple JSON string if template not found
                return json.dumps(data_copy, indent=2)
        
        try:
            # Format the template with context variables
            # Handle both string templates with format() and direct values
            if isinstance(template, str) and "{" in template:
                formatted_prompt = template.format(**data_copy)
            else:
                formatted_prompt = str(template)
                
            if self._should_log(LogDetail.DEBUG):
                self.logger.debug("prompt.formatted", 
                                template_key=template_key,
                                prompt_length=len(formatted_prompt))
                
            return formatted_prompt
            
        except KeyError as e:
            self.logger.error("prompt.format_key_error", 
                            error=str(e), 
                            template_key=template_key)
            # Fall back to sending the context directly as JSON
            return json.dumps(data_copy, indent=2)
        except Exception as e:
            self.logger.error("prompt.format_error", 
                            error=str(e), 
                            template_key=template_key)
            return json.dumps(data_copy, indent=2)

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a request using a single LLM interaction or skill invocation.
        Checks for skill configuration before falling back to LLM.
        """
        # Check if a skill is configured for this agent
        agent_config = self.config_node.get_value(self.config_path)
        skill_identifier = None
        skill_params = {}
        
        if agent_config and isinstance(agent_config, dict):
            skill_identifier = agent_config.get('skill')
            skill_params = agent_config.get('skill_params', {})
            
        # Also check persona config for skill configuration
        if not skill_identifier and self.persona_key:
            persona_config = self.config_node.get_value(self.persona_path)
            if persona_config and isinstance(persona_config, dict):
                skill_identifier = persona_config.get('skill')
                skill_params = persona_config.get('skill_params', {})
                
        # If a skill is configured, invoke it
        if skill_identifier:
            self.logger.info("agent.using_skill", 
                           skill=skill_identifier, 
                           agent_name=self.unique_name)
                           
            # Prepare skill parameters from context
            # Start with any configured parameters
            execution_params = dict(skill_params) if skill_params else {}
            
            # Add context as input if not already present
            if 'input' not in execution_params:
                execution_params['input'] = context
                
            # Add agent information
            execution_params['agent_name'] = self.unique_name
            execution_params['agent_id'] = self.agent_id
            
            # Invoke the skill
            skill_result = self._invoke_skill(skill_identifier, execution_params)
            
            # Convert to AgentResponse
            if isinstance(skill_result, SkillResult):
                # Use the built-in conversion
                return skill_result.to_agent_response()
            else:
                # Handle unexpected result type
                self.logger.warning("agent.unexpected_skill_result_type", 
                                  type=type(skill_result).__name__)
                
                # Attempt to create a valid AgentResponse
                success = getattr(skill_result, 'success', True)
                error = getattr(skill_result, 'error', None)
                
                return AgentResponse(
                    success=success,
                    data={"skill_result": skill_result},
                    error=error
                )
        
        # No skill configured or skill invocation failed, fall back to LLM
        return super().process(context)


class GenericOrchestratorAgent(BaseAgent):
    """
    Generic agent that orchestrates a series of steps based on an execution plan.
    Uses path-addressable config to access execution plan and settings.
    
    Key capabilities:
    - Executes multi-step workflows defined in configuration
    - Supports conditional execution with branching based on results
    - Manages state and context across steps
    - Integrates with skills for specialized operations
    - Handles errors with configurable recovery strategies
    """
    
    def __init__(self, full_effective_config: Dict[str, Any], unique_name: str):
        """
        Initialize with full effective configuration snapshot and unique name.
        
        Args:
            full_effective_config: Complete effective configuration after all merging
            unique_name: Unique instance name used for configuration path
        """
        # Initialize base agent with full config
        super().__init__(full_effective_config, unique_name)
        
        # Store agent type for introspection
        self.agent_type = AgentType.GENERIC_ORCHESTRATOR
        
        # Configure logger with agent-specific fields
        self.logger = self.logger.bind(
            agent_name=self.unique_name, 
            config_path=self.config_path,
            agent_type=self.agent_type.value
        )
        self.logger.info("generic_agent.initialized", agent_type=self.agent_type.value)
        
        # Get execution plan from persona configuration
        self.execution_plan = self.config_node.get_value(f"{self.persona_path}.execution_plan")
        
        # Also check for execution_plan in the agent's own config
        agent_config = self.config_node.get_value(self.config_path)
        if agent_config and isinstance(agent_config, dict) and "execution_plan" in agent_config:
            self.logger.info("execution_plan.using_agent_config",
                           agent_name=self.unique_name,
                           plan_steps=len(agent_config["execution_plan"].get("steps", [])))
            self.execution_plan = agent_config["execution_plan"]
            
        if not self.execution_plan:
            self.logger.warning("execution_plan.not_found", 
                              persona_key=self.persona_key,
                              persona_path=f"{self.persona_path}.execution_plan")
            self.execution_plan = {"steps": [], "enabled": False}

    def _get_agent_name(self) -> str:
        """Return the unique name for this agent instance."""
        return self.unique_name
        
    def _evaluate_condition(self, condition: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Evaluate a condition against the current state.
        
        Args:
            condition: Condition specification with operator and values
            state: Current execution state
            
        Returns:
            Boolean result of condition evaluation
        """
        if not condition:
            return True  # Empty condition is always true
            
        condition_type = condition.get("type", "simple")
        
        if condition_type == "and":
            # All subconditions must be true
            subconditions = condition.get("conditions", [])
            return all(self._evaluate_condition(cond, state) for cond in subconditions)
            
        elif condition_type == "or":
            # Any subcondition must be true
            subconditions = condition.get("conditions", [])
            return any(self._evaluate_condition(cond, state) for cond in subconditions)
            
        elif condition_type == "not":
            # Negate the subcondition
            subcondition = condition.get("condition", {})
            return not self._evaluate_condition(subcondition, state)
            
        else:
            # Simple condition with field, operator, value
            field = condition.get("field")
            operator = condition.get("operator", "equals")
            expected_value = condition.get("value")
            
            if not field:
                self.logger.warning("condition.missing_field", condition=condition)
                return False
                
            # Get the actual value from state
            parts = field.split(".")
            actual_value = state
            try:
                for part in parts:
                    if isinstance(actual_value, dict) and part in actual_value:
                        actual_value = actual_value[part]
                    else:
                        # Field not found
                        self.logger.debug("condition.field_not_found", 
                                       field=field, part=part, available_keys=list(actual_value.keys()) if isinstance(actual_value, dict) else None)
                        return False
            except Exception as e:
                self.logger.warning("condition.field_access_error", field=field, error=str(e))
                return False
                
            # Evaluate based on operator
            if operator == "equals":
                return actual_value == expected_value
            elif operator == "not_equals":
                return actual_value != expected_value
            elif operator == "contains":
                return expected_value in actual_value if hasattr(actual_value, "__contains__") else False
            elif operator == "greater_than":
                return actual_value > expected_value if hasattr(actual_value, "__gt__") else False
            elif operator == "less_than":
                return actual_value < expected_value if hasattr(actual_value, "__lt__") else False
            elif operator == "exists":
                return actual_value is not None
            elif operator == "is_empty":
                if hasattr(actual_value, "__len__"):
                    return len(actual_value) == 0
                return actual_value is None or actual_value == ""
            else:
                self.logger.warning("condition.unknown_operator", operator=operator)
                return False

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a request by orchestrating steps defined in an execution plan.
        
        The execution plan is loaded from config and defines the steps to execute,
        including skills to call, conditionals, and state management.
        
        Args:
            context: Input context containing data for processing
            
        Returns:
            Standard AgentResponse with results from all executed steps
        """
        self.logger.info("orchestration.start", context_keys=list(context.keys()))
        
        # Check for execution plan override in context
        execution_plan = None
        if context and "agent_config_overrides" in context:
            overrides = context["agent_
------------END OVERLAP------------
config_overrides"].get(self.unique_name, {})
            if "execution_plan" in overrides:
                execution_plan = overrides["execution_plan"]
                self.logger.info("orchestration.using_override_plan", 
                               from_context=True,
                               steps_count=len(execution_plan.get("steps", [])) if isinstance(execution_plan, dict) else 0)
        
        # If no override, use the plan from persona config
        if not execution_plan:
            execution_plan = self.execution_plan
        
        # Check that the plan is valid
        if not execution_plan or not isinstance(execution_plan, dict):
            self.logger.error("execution_plan.invalid", 
                            persona_key=self.persona_key,
                            persona_path=f"{self.persona_path}.execution_plan")
            return AgentResponse(
                success=False,
                data={},
                error=f"Invalid or missing execution plan for '{self.unique_name}'"
            )
            
        # Check if execution plan is enabled
        if not execution_plan.get("enabled", True):
            self.logger.info("execution_plan.disabled")
            return AgentResponse(
                success=True,
                data={"message": "Execution plan is disabled", "context": context},
                error=None
            )
            
        # Get the steps to execute
        steps = execution_plan.get("steps", [])
        if not steps:
            self.logger.warning("execution_plan.empty", persona_key=self.persona_key)
            return AgentResponse(
                success=True,
                data={"message": "Execution plan is empty", "context": context},
                error=None
            )
        
        # Execute the plan
        try:
            # Initialize state with input context
            state = {**context}
            results = []
            
            # Add orchestration metadata to state
            state["orchestration"] = {
                "agent_name": self.unique_name,
                "agent_id": self.agent_id,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "persona_key": self.persona_key
            }
            
            # Process each step in the execution plan
            step_idx = 0
            while step_idx < len(steps):
                step = steps[step_idx]
                step_name = step.get("name", f"step_{step_idx}")
                step_type = step.get("type", "skill")
                
                self.logger.info("step.executing", 
                               step_idx=step_idx, 
                               step_name=step_name,
                               step_type=step_type)
                
                # Check step condition if specified
                condition = step.get("condition")
                if condition and not self._evaluate_condition(condition, state):
                    self.logger.info("step.condition_false", 
                                   step_idx=step_idx, 
                                   step_name=step_name,
                                   condition=condition)
                    
                    # Skip this step and go to next
                    step_idx += 1
                    continue
                
                # Handle different step types
                if step_type == "skill":
                    # Call a skill
                    skill_name = step.get("skill")
                    if not skill_name:
                        self.logger.error("step.missing_skill", step_idx=step_idx, step_name=step_name)
                        return AgentResponse(
                            success=False,
                            data={"results": results, "state": state},
                            error=f"Missing skill name in step {step_name}"
                        )
                    
                    # Prepare skill context
                    skill_params = step.get("params", {})
                    skill_context = {**state}
                    
                    # Add skill parameters to context
                    for param_key, param_value in skill_params.items():
                        skill_context[param_key] = param_value
                    
                    # Add step metadata to context
                    skill_context["step"] = {
                        "index": step_idx,
                        "name": step_name,
                        "type": step_type
                    }
                    
                    # Invoke the skill
                    try:
                        skill_result = self._invoke_skill(skill_name, skill_context)
                        
                        # Add result to results list
                        step_result = {
                            "step": step_idx,
                            "name": step_name,
                            "type": step_type,
                            "skill": skill_name,
                            "success": True,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        
                        # Add skill result data
                        if isinstance(skill_result, SkillResult):
                            step_result["success"] = skill_result.success
                            step_result["error"] = skill_result.error
                            step_result["data"] = skill_result.value
                            
                            # Update state with skill result if output field is specified
                            output_field = step.get("output_field")
                            if output_field and skill_result.success:
                                # Set output value in state
                                parts = output_field.split(".")
                                target = state
                                for i, part in enumerate(parts[:-1]):
                                    if part not in target:
                                        target[part] = {}
                                    target = target[part]
                                target[parts[-1]] = skill_result.value
                        else:
                            step_result["data"] = skill_result
                            
                            # Update state with skill result if output field is specified
                            output_field = step.get("output_field")
                            if output_field:
                                # Set output value in state
                                parts = output_field.split(".")
                                target = state
                                for i, part in enumerate(parts[:-1]):
                                    if part not in target:
                                        target[part] = {}
                                    target = target[part]
                                target[parts[-1]] = skill_result
                        
                        results.append(step_result)
                        
                        # Check for step failure
                        if step.get("stop_on_failure", True) and not step_result["success"]:
                            self.logger.warning("step.failed", 
                                             step_idx=step_idx, 
                                             step_name=step_name,
                                             error=step_result.get("error"))
                            return AgentResponse(
                                success=False,
                                data={"results": results, "state": state},
                                error=f"Step {step_name} failed: {step_result.get('error')}"
                            )
                    
                    except Exception as e:
                        self.logger.error("step.execution_failed", 
                                       step_idx=step_idx, 
                                       step_name=step_name,
                                       skill=skill_name,
                                       error=str(e))
                        
                        # Add failed result
                        results.append({
                            "step": step_idx,
                            "name": step_name,
                            "type": step_type,
                            "skill": skill_name,
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        
                        # Check if we should stop on failure
                        if step.get("stop_on_failure", True):
                            return AgentResponse(
                                success=False,
                                data={"results": results, "state": state},
                                error=f"Error executing skill {skill_name} in step {step_name}: {str(e)}"
                            )
                
                elif step_type == "llm":
                    # Call an LLM with specified configuration
                    prompt = step.get("prompt")
                    if not prompt:
                        self.logger.error("step.missing_prompt", step_idx=step_idx, step_name=step_name)
                        return AgentResponse(
                            success=False,
                            data={"results": results, "state": state},
                            error=f"Missing prompt in step {step_name}"
                        )
                    
                    # Prepare LLM parameters
                    provider = step.get("provider") or self.provider.value
                    model = step.get("model") or self.model
                    temperature = step.get("temperature") if "temperature" in step else self.temperature
                    
                    # Format prompt with state
                    try:
                        formatted_prompt = prompt.format(**state)
                    except KeyError as e:
                        self.logger.error("step.prompt_format_error", 
                                       step_idx=step_idx, 
                                       step_name=step_name,
                                       error=str(e))
                        return AgentResponse(
                            success=False,
                            data={"results": results, "state": state},
                            error=f"Error formatting prompt in step {step_name}: {str(e)}"
                        )
                    
                    # Create LLM messages
                    system_message = step.get("system_message") or self._get_system_message()
                    messages = LLMMessages(
                        system=system_message,
                        user=formatted_prompt
                    )
                    
                    # Call LLM
                    try:
                        llm_response = self._get_completion_with_continuation(
                            messages=messages,
                            provider=provider,
                            model=model,
                            temperature=temperature
                        )
                        
                        # Add result to results list
                        step_result = {
                            "step": step_idx,
                            "name": step_name,
                            "type": step_type,
                            "success": True,
                            "data": llm_response,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        results.append(step_result)
                        
                        # Update state with LLM result if output field is specified
                        output_field = step.get("output_field")
                        if output_field:
                            # Set output value in state
                            parts = output_field.split(".")
                            target = state
                            for i, part in enumerate(parts[:-1]):
                                if part not in target:
                                    target[part] = {}
                                target = target[part]
                            target[parts[-1]] = llm_response
                    
                    except Exception as e:
                        self.logger.error("step.llm_call_failed", 
                                       step_idx=step_idx, 
                                       step_name=step_name,
                                       error=str(e))
                        
                        # Add failed result
                        results.append({
                            "step": step_idx,
                            "name": step_name,
                            "type": step_type,
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        
                        # Check if we should stop on failure
                        if step.get("stop_on_failure", True):
                            return AgentResponse(
                                success=False,
                                data={"results": results, "state": state},
                                error=f"Error calling LLM in step {step_name}: {str(e)}"
                            )
                
                elif step_type == "branch":
                    # Conditional branching
                    branches = step.get("branches", [])
                    if not branches:
                        self.logger.warning("step.empty_branches", step_idx=step_idx, step_name=step_name)
                        step_idx += 1
                        continue
                        
                    # Find the first branch whose condition evaluates to true
                    branch_executed = False
                    for branch_idx, branch in enumerate(branches):
                        condition = branch.get("condition", {})
                        if self._evaluate_condition(condition, state):
                            # Branch condition is true, execute target
                            branch_executed = True
                            
                            # Get target - either step index or name
                            target = branch.get("target")
                            if target is None:
                                self.logger.error("branch.missing_target", 
                                               step_idx=step_idx, 
                                               step_name=step_name,
                                               branch_idx=branch_idx)
                                continue
                                
                            # If target is a string, find the step by name
                            if isinstance(target, str):
                                target_idx = None
                                for i, s in enumerate(steps):
                                    if s.get("name") == target:
                                        target_idx = i
                                        break
                                        
                                if target_idx is None:
                                    self.logger.error("branch.target_not_found", 
                                                   step_idx=step_idx, 
                                                   step_name=step_name,
                                                   branch_idx=branch_idx,
                                                   target=target)
                                    continue
                                    
                                target = target_idx
                            
                            # Check if target is a valid index
                            if not isinstance(target, int) or target < 0 or target >= len(steps):
                                self.logger.error("branch.invalid_target", 
                                               step_idx=step_idx, 
                                               step_name=step_name,
                                               branch_idx=branch_idx,
                                               target=target)
                                continue
                                
                            # Jump to target step
                            self.logger.info("branch.jumping", 
                                          from_step=step_idx, 
                                          to_step=target,
                                          branch_idx=branch_idx)
                                          
                            step_idx = target
                            break
                    
                    # If no branch was executed, go to next step
                    if not branch_executed:
                        self.logger.info("branch.no_condition_true", step_idx=step_idx, step_name=step_name)
                        step_idx += 1
                
                elif step_type == "set_value":
                    # Set a value in the state
                    field = step.get("field")
                    value = step.get("value")
                    
                    if not field:
                        self.logger.error("step.missing_field", step_idx=step_idx, step_name=step_name)
                        return AgentResponse(
                            success=False,
                            data={"results": results, "state": state},
                            error=f"Missing field in step {step_name}"
                        )
                    
                    # Set value in state
                    try:
                        parts = field.split(".")
                        target = state
                        for i, part in enumerate(parts[:-1]):
                            if part not in target:
                                target[part] = {}
                            target = target[part]
                        target[parts[-1]] = value
                        
                        # Add result to results list
                        results.append({
                            "step": step_idx,
                            "name": step_name,
                            "type": step_type,
                            "success": True,
                            "field": field,
                            "value": value,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                    except Exception as e:
                        self.logger.error("step.set_value_failed", 
                                       step_idx=step_idx, 
                                       step_name=step_name,
                                       field=field,
                                       error=str(e))
                        
                        # Add failed result
                        results.append({
                            "step": step_idx,
                            "name": step_name,
                            "type": step_type,
                            "success": False,
                            "field": field,
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        
------------END OVERLAP------------
                        })
                
                else:
                    # Unknown step type
                    self.logger.warning("step.unknown_type", 
                                     step_idx=step_idx, 
                                     step_name=step_name,
                                     step_type=step_type)
                    
                    # Add result to results list
                    results.append({
                        "step": step_idx,
                        "name": step_name,
                        "type": step_type,
                        "success": False,
                        "error": f"Unknown step type: {step_type}",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                # Move to next step unless a branch happened
                if step_type != "branch" or not results[-1]["success"]:
                    step_idx += 1
            
            # Add orchestration completion metadata to state
            if "orchestration" in state:
                state["orchestration"]["end_time"] = datetime.now(timezone.utc).isoformat()
                state["orchestration"]["steps_executed"] = len(results)
            
            # Return results
            return AgentResponse(
                success=True,
                data={
                    "results": results,
                    "state": state,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                error=None
            )
            
        except Exception as e:
            self.logger.error("orchestration.failed", error=str(e))
            return AgentResponse(
                success=False, 
                data={"results": results if 'results' in locals() else [], "state": state if 'state' in locals() else {}}, 
                error=str(e)
            )

# Note: GenericOrchestratingAgent (previously requested in WO-trefac-01) has been renamed to GenericOrchestratorAgent.
# Use GenericOrchestratorAgent directly instead as it provides the same functionality with a more consistent naming pattern.


class GenericSkillAgent(BaseAgent):
    """
    Generic agent optimized for skill-based operations with minimal LLM interaction.
    
    This agent type is designed to:
    - Efficiently invoke predefined skills with proper context preparation
    - Minimize token usage and latency by reducing LLM calls
    - Handle data transformation and processing tasks
    - Support specialized operations like extraction, parsing, and formatting
    - Provide a lightweight alternative to LLM-driven workflows for deterministic tasks
    """
    
    def __init__(self, full_effective_config: Dict[str, Any], unique_name: str):
        """
        Initialize with full effective configuration snapshot and unique name.
        
        Args:
            full_effective_config: Complete effective configuration after all merging
            unique_name: Unique instance name used for configuration path
        """
        # Initialize base agent with full config
        super().__init__(full_effective_config, unique_name)
        
        # Store agent type for introspection
        self.agent_type = AgentType.GENERIC_SKILL
        
        # Configure logger with agent-specific fields
        self.logger = self.logger.bind(
            agent_name=self.unique_name, 
            config_path=self.config_path,
            agent_type=self.agent_type.value
        )
        self.logger.info("generic_agent.initialized", agent_type=self.agent_type.value)
        
        # Get primary skill configuration
        self.primary_skill = self.config_node.get_value(f"{self.persona_path}.skill")
        if not self.primary_skill:
            self.logger.warning("skill_agent.no_primary_skill", 
                              persona_key=self.persona_key,
                              persona_path=f"{self.persona_path}.skill")
            
        # Get skill parameters from configuration
        self.skill_params = self.config_node.get_value(f"{self.persona_path}.skill_params") or {}
        
        # Get LLM fallback configuration
        self.allow_llm_fallback = self.config_node.get_value(f"{self.persona_path}.allow_llm_fallback") or False
        self.fallback_prompt_key = self.config_node.get_value(f"{self.persona_path}.fallback_prompt_key") or "process"

    def _get_agent_name(self) -> str:
        """Return the unique name for this agent instance."""
        return self.unique_name
        
    def _get_primary_skill(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Get the primary skill to invoke, with support for context overrides.
        
        Args:
            context: Runtime context that may contain overrides
            
        Returns:
            Skill identifier or None if not configured
        """
        # Check for skill override in context
        if context and "agent_config_overrides" in context:
            overrides = context["agent_config_overrides"].get(self.unique_name, {})
            if "skill" in overrides:
                skill_override = overrides["skill"]
                self.logger.info("skill_agent.using_override_skill", skill=skill_override)
                return skill_override
                
        # No override, use configured primary skill
        return self.primary_skill
        
    def _get_skill_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get skill parameters, with support for context overrides.
        
        Args:
            context: Runtime context that may contain overrides
            
        Returns:
            Dictionary of skill parameters
        """
        # Start with configured parameters
        params = dict(self.skill_params)
        
        # Check for parameter overrides in context
        if context and "agent_config_overrides" in context:
            overrides = context["agent_config_overrides"].get(self.unique_name, {})
            if "skill_params" in overrides and isinstance(overrides["skill_params"], dict):
                # Merge overrides with base parameters
                override_params = overrides["skill_params"]
                for key, value in override_params.items():
                    params[key] = value
                self.logger.info("skill_agent.using_override_params", 
                              override_keys=list(override_params.keys()))
                
        return params
        
    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a request by invoking the configured skill or falling back to LLM.
        
        This agent primarily uses skills for processing, but can optionally fall back
        to LLM processing if configured and the skill fails or is not available.
        
        Args:
            context: Input context containing data for processing
            
        Returns:
            Standard AgentResponse with results
        """
        self.logger.info("skill_agent.process", context_keys=list(context.keys()))
        
        # Get skill to invoke
        skill_identifier = self._get_primary_skill(context)
        
        if not skill_identifier:
            error_msg = "No skill configured for this agent"
            self.logger.error("skill_agent.no_skill_configured")
            
            # Check if LLM fallback is allowed
            if self.allow_llm_fallback:
                self.logger.info("skill_agent.using_llm_fallback", 
                              fallback_prompt_key=self.fallback_prompt_key)
                # Fall back to LLM processing
                return super().process(context)
            else:
                # No fallback, return error
                return AgentResponse(
                    success=False,
                    data={},
                    error=error_msg
                )
        
        # Get skill parameters from configuration and context
        params = self._get_skill_params(context)
        
        # Prepare skill execution parameters
        execution_params = dict(params)
        
        # Add context as input if not already present
        if "input" not in execution_params:
            execution_params["input"] = context
            
        # Add agent information
        execution_params["agent_name"] = self.unique_name
        execution_params["agent_id"] = self.agent_id
        
        # Add any specific parameters from context
        for key in ["content", "instructions", "format"]:
            if key in context and key not in execution_params:
                execution_params[key] = context[key]
        
        # Invoke the skill
        try:
            self.logger.info("skill_agent.invoking_skill", 
                          skill=skill_identifier, 
                          param_keys=list(execution_params.keys()))
                          
            skill_result = self._invoke_skill(skill_identifier, execution_params)
            
            # Check if skill invocation succeeded
            if isinstance(skill_result, SkillResult):
                # Use the built-in conversion to AgentResponse
                agent_response = skill_result.to_agent_response()
                
                # Add more detail to the response
                agent_response.data["skill_agent"] = {
                    "skill": skill_identifier,
                    "agent_type": self.agent_type.value,
                    "persona_key": self.persona_key
                }
                
                return agent_response
            else:
                # Handle unexpected result type
                self.logger.warning("skill_agent.unexpected_result_type", 
                                  type=type(skill_result).__name__)
                
                # Attempt to create a valid AgentResponse
                success = getattr(skill_result, "success", True)
                error = getattr(skill_result, "error", None)
                
                response = AgentResponse(
                    success=success,
                    data={
                        "skill_result": skill_result,
                        "skill": skill_identifier,
                        "agent_type": self.agent_type.value,
                        "persona_key": self.persona_key
                    },
                    error=error
                )
                
                return response
                
        except Exception as e:
            error_msg = f"Error invoking skill '{skill_identifier}': {str(e)}"
            self.logger.error("skill_agent.skill_invocation_failed", 
                           skill=skill_identifier, 
                           error=str(e))
            
            # Check if LLM fallback is allowed
            if self.allow_llm_fallback:
                self.logger.info("skill_agent.using_llm_fallback_after_error", 
                              fallback_prompt_key=self.fallback_prompt_key,
                              error=str(e))
                # Fall back to LLM processing
                return super().process(context)
            else:
                # No fallback, return error
                return AgentResponse(
                    success=False,
                    data={"skill": skill_identifier},
                    error=error_msg
                )


class GenericFallbackAgent(GenericLLMAgent):
    """
    Generic agent designed for handling failure cases with conservative parameters.
    
    This agent type is designed to:
    - Serve as a fallback when primary agents fail
    - Use more conservative LLM parameters (lower temperature, smaller models)
    - Apply stricter validation and error handling
    - Execute more focused, targeted tasks with explicit constraints
    - Implement appropriate retry and recovery strategies
    """
    
    def __init__(self, full_effective_config: Dict[str, Any], unique_name: str):
        """
        Initialize with full effective configuration snapshot and unique name.
        
        Args:
            full_effective_config: Complete effective configuration after all merging
            unique_name: Unique instance name used for configuration path
        """
        # Initialize GenericLLMAgent with full config
        super().__init__(full_effective_config, unique_name)
        
        # Override agent type for introspection
        self.agent_type = AgentType.GENERIC_FALLBACK
        
        # Configure logger with fallback-specific fields
        self.logger = self.logger.bind(
            agent_name=self.unique_name, 
            config_path=self.config_path,
            agent_type=self.agent_type.value,
            is_fallback=True
        )
        self.logger.info("fallback_agent.initialized", agent_type=self.agent_type.value)
        
        # Get fallback-specific configuration with conservative defaults
        self.max_retries = self.config_node.get_value(f"{self.persona_path}.max_retries") or 2
        self.retry_delay = self.config_node.get_value(f"{self.persona_path}.retry_delay") or 1.0
        self.validation_level = self.config_node.get_value(f"{self.persona_path}.validation_level") or "strict"
        
        # Set conservative default temperature
        temp_value = self.config_node.get_value(f"{self.persona_path}.temperature")
        # Override with a lower temperature if not explicitly set
        if temp_value is None:
            self.temperature = 0.0
            self.logger.info("fallback_agent.using_conservative_temperature", temperature=self.temperature)
        
        # Record previous failure if available in context
        self.previous_error = None

    def _get_agent_name(self) -> str:
        """Return the unique name for this agent instance."""
        return f"fallback_{self.unique_name}"
    
    def _get_system_message(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get system message, adding fallback-specific instructions.
        
        Args:
            context: Optional runtime context that may contain overrides
            
        Returns:
            System message string with fallback instructions
        """
        # Get base system message from persona
        base_system_message = super()._get_system_message(context)
        
        # Add fallback-specific instructions
        fallback_instructions = """
You are operating in fallback mode after a previous approach failed.
Focus on producing a minimal, conservative solution that avoids complexity.
Ensure strict validation and error checking in all operations.
If you can't solve the full problem, solve a simpler subset rather than failing entirely.
"""
        
        # Get previous error information if available
        previous_error = None
        if context and "previous_error" in context:
            previous_error = context["previous_error"]
        elif self.previous_error:
            previous_error = self.previous_error
            
        # Add error information if available
        if previous_error:
            fallback_instructions += f"\nPrevious error: {previous_error}\n"
            fallback_instructions += "Avoid the approach that caused this error.\n"
            
        # Combine base message with fallback instructions
        return f"{base_system_message}\n\n{fallback_instructions}"
        
    def _format_request(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format user request with additional fallback context.
        
        Args:
            data: Data to format
            context: Optional runtime context that may contain overrides
            
        Returns:
            Formatted request string with fallback annotations
        """
        # Make a copy of data to avoid modifying the original
        data_copy = dict(data)
        
        # Add fallback metadata to data
        data_copy["fallback_mode"] = True
        data_copy["agent_type"] = self.agent_type.value
        
        # Store previous error info for reference
        if context and "previous_error" in context:
            self.previous_error = context["previous_error"]
            data_copy["previous_error"] = context["previous_error"]
        
        # Use normal formatting logic from GenericLLMAgent
        return super()._format_request(data_copy, context)
        
    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a request with fallback handling and conservative approach.
        
        This agent focuses on reliable operation with conservative parameters
        and is designed to recover from previous failures.
        
        Args:
            context: Input context containing data for processing
            
        Returns:
            Standard AgentResponse with fallback metadata
        """
        self.logger.info("fallback_agent.process", context_keys=list(context.keys()))
        
        # Store previous error if available
        if "previous_error" in context:
            self.previous_error = context["previous_error"]
            self.logger.info("fallback_agent.handling_previous_error", 
                          error=self.previous_error)
        
        #
------------END OVERLAP------------

        # Check for skill configuration - same as GenericLLMAgent
        agent_config = self.config_node.get_value(self.config_path)
        skill_identifier = None
        skill_params = {}
        
        if agent_config and isinstance(agent_config, dict):
            skill_identifier = agent_config.get('skill')
            skill_params = agent_config.get('skill_params', {})
            
        # Also check persona config for skill configuration
        if not skill_identifier and self.persona_key:
            persona_config = self.config_node.get_value(self.persona_path)
            if persona_config and isinstance(persona_config, dict):
                skill_identifier = persona_config.get('skill')
                skill_params = persona_config.get('skill_params', {})
        
        # If a skill is configured and we're allowed to use skills in fallback mode
        use_skills = self.config_node.get_value(f"{self.persona_path}.use_skills") or False
        if skill_identifier and use_skills:
            self.logger.info("fallback_agent.using_skill", 
                          skill=skill_identifier, 
                          agent_name=self.unique_name)
                          
            # Prepare skill parameters with conservative settings
            execution_params = dict(skill_params) if skill_params else {}
            
            # Add context as input if not already present
            if 'input' not in execution_params:
                execution_params['input'] = context
                
            # Add agent information
            execution_params['agent_name'] = self.unique_name
            execution_params['agent_id'] = self.agent_id
            execution_params['fallback_mode'] = True
            
            # Ensure conservative parameters
            if 'temperature' not in execution_params:
                execution_params['temperature'] = 0.0
                
            if 'validation_level' not in execution_params:
                execution_params['validation_level'] = self.validation_level
            
            # Invoke the skill with retries
            for attempt in range(self.max_retries + 1):
                try:
                    skill_result = self._invoke_skill(skill_identifier, execution_params)
                    
                    # Convert to AgentResponse and add fallback metadata
                    if isinstance(skill_result, SkillResult):
                        response = skill_result.to_agent_response()
                        response.data["fallback_metadata"] = {
                            "agent_type": self.agent_type.value,
                            "attempt": attempt + 1,
                            "max_retries": self.max_retries,
                            "validation_level": self.validation_level,
                            "previous_error": self.previous_error
                        }
                        return response
                    else:
                        # Handle unexpected result type
                        self.logger.warning("fallback_agent.unexpected_skill_result_type", 
                                          type=type(skill_result).__name__)
                        
                        # Attempt to create a valid AgentResponse
                        success = getattr(skill_result, 'success', True)
                        error = getattr(skill_result, 'error', None)
                        
                        return AgentResponse(
                            success=success,
                            data={
                                "skill_result": skill_result,
                                "fallback_metadata": {
                                    "agent_type": self.agent_type.value,
                                    "attempt": attempt + 1,
                                    "max_retries": self.max_retries,
                                    "validation_level": self.validation_level,
                                    "previous_error": self.previous_error
                                }
                            },
                            error=error
                        )
                except Exception as e:
                    # Log the error but retry if attempts remain
                    self.logger.warning("fallback_agent.skill_attempt_failed", 
                                     skill=skill_identifier, 
                                     attempt=attempt + 1,
                                     max_retries=self.max_retries,
                                     error=str(e))
                    
                    # If we have more retries, wait and try again
                    if attempt < self.max_retries:
                        import time
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        # No more retries, fall back to LLM
                        self.logger.info("fallback_agent.skill_retries_exhausted", 
                                      skill=skill_identifier,
                                      falling_back_to="llm_call")
                        # Intentional fall-through to LLM processing
                        break
        
        # Fall back to LLM processing with conservative parameters
        context_with_metadata = dict(context)
        
        # Add fallback metadata to context
        context_with_metadata["fallback_mode"] = True
        context_with_metadata["fallback_metadata"] = {
            "agent_type": self.agent_type.value,
            "validation_level": self.validation_level,
            "previous_error": self.previous_error
        }
        
        # Ensure conservative LLM parameters
        if "llm_params" not in context_with_metadata:
            context_with_metadata["llm_params"] = {}
            
        # Set conservative temperature if not explicitly defined
        if "temperature" not in context_with_metadata["llm_params"]:
            context_with_metadata["llm_params"]["temperature"] = 0.0
            
        # Call the base LLM processing with enhanced context
        response = super().process(context_with_metadata)
        
        # Add fallback metadata to response
        if "fallback_metadata" not in response.data:
            response.data["fallback_metadata"] = {
                "agent_type": self.agent_type.value,
                "validation_level": self.validation_level,
                "previous_error": self.previous_error
            }
            
        return response
}}