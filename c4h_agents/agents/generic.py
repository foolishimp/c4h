"""
Generic agent implementations using path-addressable configuration.
Path: c4h_agents/agents/generic.py
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json

from c4h_agents.agents.base_agent import BaseAgent, AgentResponse
from c4h_agents.agents.types import LLMMessages, LogDetail
from c4h_agents.utils.logging import get_logger

logger = get_logger()

class GenericSingleShotAgent(BaseAgent):
    """
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
        super().__init__(config=full_effective_config)
        
        # Store unique name for path-based config access
        self.unique_name = unique_name
        
        # Determine configuration path
        # First check primary path (team.llm_config) then fallback path
        self.config_path = f"team.llm_config.agents.{unique_name}"
        if not self.config_node.get_value(self.config_path):
            self.config_path = f"llm_config.agents.{unique_name}"
            # If still not found, log the issue but continue (we'll check later when needed)
            if not self.config_node.get_value(self.config_path):
                self.logger.warning("agent.config_not_found", 
                                   unique_name=unique_name, 
                                   primary_path=f"team.llm_config.agents.{unique_name}",
                                   fallback_path=f"llm_config.agents.{unique_name}")
        
        self.logger = self.logger.bind(agent_name=self.unique_name, config_path=self.config_path)
        self.logger.info("generic_agent.initialized", agent_type="GenericSingleShotAgent")

    def _get_agent_name(self) -> str:
        """Return the unique name for this agent instance."""
        return self.unique_name

    def _get_system_message(self) -> str:
        """Get system message from path-addressable config."""
        system_prompt = self.config_node.get_value(f"{self.config_path}.prompts.system")
        if not system_prompt:
            self.logger.warning("system_prompt.not_found", path=f"{self.config_path}.prompts.system")
            return ""
        return system_prompt

    def _format_request(self, context: Dict[str, Any]) -> str:
        """
        Format user request based on path-addressable config.
        
        This method:
        1. Gets the prompt template specified in config
        2. Formats it using context variables
        3. Returns the formatted prompt
        """
        # Get the prompt template name from config or use "user" as default
        template_name = self.config_node.get_value(f"{self.config_path}.prompts.template_name") or "user"
        
        # Get the prompt template
        template = self.config_node.get_value(f"{self.config_path}.prompts.{template_name}")
        if not template:
            self.logger.warning("prompt_template.not_found", 
                              template_name=template_name,
                              path=f"{self.config_path}.prompts.{template_name}")
            # Fallback to simple JSON string if template not found
            return json.dumps(context, indent=2)
        
        try:
            # Format the template with context variables
            # Handle both string templates with format() and direct values
            if isinstance(template, str) and "{" in template:
                formatted_prompt = template.format(**context)
            else:
                formatted_prompt = str(template)
                
            if self._should_log(LogDetail.DEBUG):
                self.logger.debug("prompt.formatted", 
                                template_name=template_name,
                                prompt_length=len(formatted_prompt))
                
            return formatted_prompt
            
        except KeyError as e:
            self.logger.error("prompt.format_key_error", 
                            error=str(e), 
                            template_name=template_name)
            # Fall back to sending the context directly as JSON
            return json.dumps(context, indent=2)
        except Exception as e:
            self.logger.error("prompt.format_error", 
                            error=str(e), 
                            template_name=template_name)
            return json.dumps(context, indent=2)

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a request using a single LLM interaction.
        Leverages BaseAgent's implementation with path-addressable config.
        """
        # Use BaseAgent's process implementation which handles most of the flow
        return super().process(context)


class GenericOrchestratingAgent(BaseAgent):
    """
    Generic agent that orchestrates a series of steps based on an execution plan.
    Uses path-addressable config to access execution plan and settings.
    """
    
    def __init__(self, full_effective_config: Dict[str, Any], unique_name: str):
        """
        Initialize with full effective configuration snapshot and unique name.
        
        Args:
            full_effective_config: Complete effective configuration after all merging
            unique_name: Unique instance name used for configuration path
        """
        # Initialize base agent with full config
        super().__init__(config=full_effective_config)
        
        # Store unique name for path-based config access
        self.unique_name = unique_name
        
        # Determine configuration path
        self.config_path = f"team.llm_config.agents.{unique_name}"
        if not self.config_node.get_value(self.config_path):
            self.config_path = f"llm_config.agents.{unique_name}"
            if not self.config_node.get_value(self.config_path):
                self.logger.warning("agent.config_not_found", 
                                   unique_name=unique_name, 
                                   primary_path=f"team.llm_config.agents.{unique_name}",
                                   fallback_path=f"llm_config.agents.{unique_name}")
        
        self.logger = self.logger.bind(agent_name=self.unique_name, config_path=self.config_path)
        self.logger.info("generic_agent.initialized", agent_type="GenericOrchestratingAgent")

    def _get_agent_name(self) -> str:
        """Return the unique name for this agent instance."""
        return self.unique_name

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a request by orchestrating steps defined in an execution plan.
        
        The execution plan is loaded from config and defines the steps to execute,
        including skills to call, conditionals, and state management.
        """
        self.logger.info("orchestration.start", context_keys=list(context.keys()))
        
        # Get execution plan from config
        execution_plan = self.config_node.get_value(f"{self.config_path}.execution_plan")
        if not execution_plan:
            self.logger.error("execution_plan.not_found", path=f"{self.config_path}.execution_plan")
            return AgentResponse(
                success=False,
                data={},
                error=f"No execution plan found at {self.config_path}.execution_plan"
            )
        
        # Execute the plan
        try:
            # Initialize state with input context
            state = {**context}
            results = []
            
            # Process each step in the execution plan
            for step_idx, step in enumerate(execution_plan):
                self.logger.info("step.executing", step_idx=step_idx, step_type=step.get("type", "unknown"))
                
                # Handle different step types
                if step.get("type") == "skill":
                    # Call a skill
                    skill_name = step.get("skill")
                    skill_context = {**state, **(step.get("params", {}))}
                    
                    # Prepare context for skill with proper lineage tracking
                    skill_context = self.call_skill(skill_name, skill_context)
                    
                    # Execute skill and store result in state
                    # This would be implemented by the skill itself
                    # For now we just log and store the prepared context
                    self.logger.info("skill.prepared", skill=skill_name)
                    results.append({"step": step_idx, "skill": skill_name, "context": skill_context})
                    
                    # In a real implementation, you would call the skill and update state with its result
                
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
            return AgentResponse(success=False, data={}, error=str(e))
