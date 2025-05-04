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
        super().__init__(full_effective_config, unique_name)
        
        # unique_name is already stored by BaseAgent constructor
        
        # Configuration path is already set by BaseAgent constructor
        # BaseAgent handles checking both team.llm_config.agents.{unique_name} and llm_config.agents.{unique_name} paths
        
        self.logger = self.logger.bind(agent_name=self.unique_name, config_path=self.config_path)
        self.logger.info("generic_agent.initialized", agent_type="GenericSingleShotAgent")

    def _get_agent_name(self) -> str:
        """Return the unique name for this agent instance."""
        return self.unique_name

    def _get_system_message(self) -> str:
        """Get system message from path-addressable config with persona fallback."""
        # Use BaseAgent's implementation which now includes persona fallback
        return super()._get_system_message()

    def _format_request(self, context: Dict[str, Any]) -> str:
        """
        Format user request based on path-addressable config with persona fallback.
        
        This method:
        1. Gets the prompt template specified in config or persona
        2. Formats it using context variables
        3. Returns the formatted prompt
        
        Special handling for discovery agent:
        - If this is a discovery agent and tartxt_config is present,
          run the tartxt.py script to gather project information
        """
        # Check if this is a discovery agent with tartxt config
        if (self.unique_name == "discovery_phase" or 
            self.config_path.endswith(".discovery") or 
            "discovery" in self.unique_name):
            
            # Get tartxt_config directly from persona
            tartxt_config = self.config_node.get_value(f"{self.persona_path}.tartxt_config")
            
            if tartxt_config and isinstance(tartxt_config, dict):
                self.logger.info("discovery.tartxt_config_found", 
                               config_keys=list(tartxt_config.keys()),
                               persona_key=self.persona_key)
                
                # Get script path
                script_path = tartxt_config.get("script_path")
                if script_path and os.path.exists(script_path):
                    # Get project path from context
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
                            
                            # Read output file and add to context
                            try:
                                if os.path.exists(output_file):
                                    with open(output_file, 'r', encoding='utf-8') as f:
                                        project_content = f.read()
                                    
                                    # Update context with project content
                                    context['project_content'] = project_content
                                    context['project_scan_file'] = output_file
                                    self.logger.info("discovery.added_project_content", 
                                                  content_length=len(project_content))
                                else:
                                    self.logger.error("discovery.output_file_not_found", file=output_file)
                            except Exception as e:
                                self.logger.error("discovery.read_output_failed", error=str(e))
                    except Exception as e:
                        self.logger.error("discovery.tartxt_execution_failed", error=str(e))
        
        # Continue with standard prompt template handling directly from persona
        # Get the prompt template name from persona configuration or use "user" as default
        template_name = self.config_node.get_value(f"{self.persona_path}.prompts.template_name") or "user"
        
        # Get the prompt template from persona
        template = self.config_node.get_value(f"{self.persona_path}.prompts.{template_name}")
                
        if not template:
            self.logger.warning("prompt_template.not_found", 
                              template_name=template_name,
                              persona_key=self.persona_key,
                              persona_path=f"{self.persona_path}.prompts.{template_name}")
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
        super().__init__(full_effective_config, unique_name)
        
        # unique_name is already stored by BaseAgent constructor
        
        # Configuration path is already set by BaseAgent constructor
        # BaseAgent handles checking both team.llm_config.agents.{unique_name} and llm_config.agents.{unique_name} paths
        
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
        
        # Get execution plan directly from persona config
        execution_plan = self.config_node.get_value(f"{self.persona_path}.execution_plan")
        
        if not execution_plan:
            self.logger.error("execution_plan.not_found", 
                            persona_key=self.persona_key,
                            persona_path=f"{self.persona_path}.execution_plan")
            return AgentResponse(
                success=False,
                data={},
                error=f"No execution plan found in persona '{self.persona_key}'"
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
