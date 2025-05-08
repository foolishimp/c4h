"""
Execution Plan Executor for the C4H Agent System.

This module provides the core execution engine for running execution plans
at various levels of the system (workflows, teams, agents).
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timezone
import copy
import json
import re
import uuid
from enum import Enum
import structlog

from c4h_agents.lineage.event_logger import EventLogger, EventType
from c4h_agents.config import create_config_node, ConfigNode
from c4h_agents.skills.registry import SkillRegistry

# Configure logger
logger = structlog.get_logger()

class StepType(Enum):
    """Types of steps in an execution plan."""
    SKILL_CALL = "skill_call"
    AGENT_CALL = "agent_call"
    TEAM_CALL = "team_call"
    LLM_CALL = "llm_call"
    LOOP = "loop"
    BRANCH = "branch"
    SET_VALUE = "set_value"

class StepAction(Enum):
    """Special actions that can be taken during step execution."""
    EXIT_PLAN_WITH_SUCCESS = "exit_plan_with_success"
    EXIT_PLAN_WITH_FAILURE = "exit_plan_with_failure"
    BREAK_LOOP = "break_loop"
    CONTINUE_LOOP = "continue_loop"

class ExecutionResult:
    """Result of an execution step or plan."""
    
    def __init__(
        self, 
        success: bool = True, 
        context: Optional[Dict[str, Any]] = None,
        output: Optional[Any] = None,
        error: Optional[str] = None,
        steps_executed: int = 0,
        step_name: Optional[str] = None,
        step_type: Optional[str] = None,
        action: Optional[StepAction] = None
    ):
        """
        Initialize an execution result.
        
        Args:
            success: Whether the execution was successful
            context: The execution context after execution
            output: The final output value
            error: Error message if execution failed
            steps_executed: Number of steps executed
            step_name: Name of the step that produced this result
            step_type: Type of the step that produced this result
            action: Special action to take based on this result
        """
        self.success = success
        self.context = context or {}
        self.output = output
        self.error = error
        self.steps_executed = steps_executed
        self.step_name = step_name
        self.step_type = step_type
        self.action = action
        self.timestamp = datetime.now(timezone.utc).isoformat()

class ExecutionPlanExecutor:
    """
    Universal executor for execution plans across the C4H system.
    
    This class is responsible for processing execution plans at all levels
    of the system hierarchy (workflows, teams, agents).
    """
    
    def __init__(
        self, 
        effective_config: Dict[str, Any],
        skill_registry: Optional[SkillRegistry] = None,
        event_logger: Optional[EventLogger] = None
    ):
        """
        Initialize the executor with configuration.
        
        Args:
            effective_config: The effective configuration snapshot
            skill_registry: Optional skill registry to use (creates one if not provided)
            event_logger: Optional event logger to use for lineage tracking
        """
        self.config = effective_config
        self.config_node = create_config_node(effective_config)
        self.skill_registry = skill_registry or SkillRegistry()
        self.event_logger = event_logger
        self.execution_id = str(uuid.uuid4())
        
        # Configure logger
        self.logger = logger.bind(
            executor_id=self.execution_id
        )
        
        self.logger.info("executor.initialized", 
                      skill_registry=bool(skill_registry),
                      event_logger=bool(event_logger))

    def execute_plan(
        self, 
        execution_plan: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute an execution plan with the given context.
        
        Args:
            execution_plan: The execution plan to execute
            context: The initial context for execution
            
        Returns:
            ExecutionResult with the final state and results
        """
        if not execution_plan:
            self.logger.error("execute_plan.invalid_plan", plan=execution_plan)
            return ExecutionResult(
                success=False,
                error="Invalid or empty execution plan"
            )
            
        # Check if the plan is enabled
        if not execution_plan.get("enabled", True):
            self.logger.info("execute_plan.disabled")
            return ExecutionResult(
                success=True,
                context=context,
                output={"message": "Execution plan is disabled"}
            )
            
        # Get the steps to execute
        steps = execution_plan.get("steps", [])
        if not steps:
            self.logger.warning("execute_plan.empty_steps")
            return ExecutionResult(
                success=True,
                context=context,
                output={"message": "Execution plan has no steps"}
            )
            
        # Create a new immutable context to work with
        # Always make a deep copy to ensure immutability
        current_context = copy.deepcopy(context)
        
        # Add execution metadata to context
        if "execution_metadata" not in current_context:
            current_context["execution_metadata"] = {}
        current_context["execution_metadata"]["plan_execution_id"] = self.execution_id
        current_context["execution_metadata"]["start_time"] = datetime.now(timezone.utc).isoformat()
        
        # Log execution start
        self.logger.info("execute_plan.starting", 
                      step_count=len(steps),
                      first_step=steps[0].get("name") if steps else None,
                      context_keys=list(current_context.keys()))
                      
        # Track results and execution path
        results = []
        execution_path = []
        
        # Execute each step in the plan
        step_idx = 0
        while step_idx < len(steps):
            step = steps[step_idx]
            step_name = step.get("name", f"step_{step_idx}")
            step_type = step.get("type")
            
            self.logger.info("step.executing", 
                          step_idx=step_idx, 
                          step_name=step_name,
                          step_type=step_type)
                          
            # Add step to execution path
            execution_path.append(step_name)
            
            # Check step condition if specified
            if "condition" in step:
                condition_result = self._evaluate_condition(step["condition"], current_context)
                if not condition_result:
                    self.logger.info("step.condition_false", 
                                  step_idx=step_idx, 
                                  step_name=step_name)
                    
                    # Skip this step and go to next
                    step_idx += 1
                    continue
            
            # Log step start for lineage tracking
            if self.event_logger:
                self.event_logger.log_event(
                    event_type=EventType.STEP_START,
                    payload={
                        "step_name": step_name,
                        "step_type": step_type,
                        "step_idx": step_idx
                    }
                )
            
            # Execute the step based on its type
            step_result = self._execute_step(step, current_context)
            
            # Store the result
            results.append({
                "step_idx": step_idx,
                "step_name": step_name,
                "step_type": step_type,
                "success": step_result.success,
                "error": step_result.error,
                "timestamp": step_result.timestamp
            })
            
            # Log step completion for lineage tracking
            if self.event_logger:
                self.event_logger.log_event(
                    event_type=EventType.STEP_END,
                    payload={
                        "step_name": step_name,
                        "step_type": step_type,
                        "step_idx": step_idx,
                        "success": step_result.success,
                        "error": step_result.error
                    }
                )
            
            # Handle step result
            if step_result.success:
                # Update the context with the step result context, if available
                if step_result.context:
                    current_context.update(step_result.context)
                    self.logger.debug("step.updated_context_from_result", 
                                   step_name=step_name)
                
                # Also update the context with step result output if output_field is specified
                output_field = step.get("output_field")
                if output_field and step_result.output:
                    # Create a new context with the updated value
                    current_context = self._set_context_value(
                        current_context, 
                        output_field, 
                        step_result.output
                    )
                    self.logger.debug("step.updated_context", 
                                   step_name=step_name,
                                   output_field=output_field)
                
                # Handle special actions from the step result
                if step_result.action:
                    if step_result.action == StepAction.EXIT_PLAN_WITH_SUCCESS:
                        self.logger.info("step.exit_with_success", 
                                      step_name=step_name)
                        return ExecutionResult(
                            success=True,
                            context=current_context,
                            output={
                                "results": results,
                                "execution_path": execution_path
                            }
                        )
                    elif step_result.action == StepAction.EXIT_PLAN_WITH_FAILURE:
                        self.logger.info("step.exit_with_failure", 
                                      step_name=step_name,
                                      error=step_result.error)
                        return ExecutionResult(
                            success=False,
                            context=current_context,
                            output={
                                "results": results,
                                "execution_path": execution_path
                            },
                            error=step_result.error or "Execution plan exited with failure"
                        )
                
                # Check for routing rules
                next_step_idx = self._apply_routing_rules(step, step_result, current_context, step_idx, steps)
                if next_step_idx is not None:
                    step_idx = next_step_idx
                    continue
                
                # Default behavior: move to next step
                step_idx += 1
                
            else:
                # Handle step failure
                self.logger.warning("step.failed", 
                                 step_idx=step_idx, 
                                 step_name=step_name,
                                 error=step_result.error)
                
                # Check if we should stop on failure
                if step.get("stop_on_failure", True):
                    return ExecutionResult(
                        success=False,
                        context=current_context,
                        output={
                            "results": results,
                            "execution_path": execution_path
                        },
                        error=f"Step {step_name} failed: {step_result.error}"
                    )
                
                # Otherwise, continue to next step
                step_idx += 1
        
        # All steps completed successfully
        self.logger.info("execute_plan.completed", 
                      steps_executed=len(execution_path),
                      final_context_keys=list(current_context.keys()))
        
        # Add execution completion metadata
        if "execution_metadata" in current_context:
            current_context["execution_metadata"]["end_time"] = datetime.now(timezone.utc).isoformat()
            current_context["execution_metadata"]["steps_executed"] = len(execution_path)
            
        # Extract response if present
        output = None
        if "response" in current_context:
            output = current_context["response"]
            
        return ExecutionResult(
            success=True,
            context=current_context,
            output=output,
            steps_executed=len(execution_path)
        )
    
    def _execute_step(
        self, 
        step: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a single step in the execution plan.
        
        Args:
            step: The step definition to execute
            context: The current execution context
            
        Returns:
            ExecutionResult with the step's result
        """
        step_name = step.get("name", "unnamed_step")
        step_type_str = step.get("type")
        
        try:
            # Convert step type string to enum
            try:
                step_type = StepType(step_type_str)
            except (ValueError, TypeError):
                return ExecutionResult(
                    success=False,
                    error=f"Invalid step type: {step_type_str}",
                    step_name=step_name,
                    step_type=step_type_str
                )
            
            # Execute based on step type
            if step_type == StepType.SKILL_CALL:
                return self._execute_skill_call(step, context)
            elif step_type == StepType.AGENT_CALL:
                return self._execute_agent_call(step, context)
            elif step_type == StepType.TEAM_CALL:
                return self._execute_team_call(step, context)
            elif step_type == StepType.LLM_CALL:
                return self._execute_llm_call(step, context)
            elif step_type == StepType.LOOP:
                return self._execute_loop(step, context)
            elif step_type == StepType.BRANCH:
                return self._execute_branch(step, context)
            elif step_type == StepType.SET_VALUE:
                return self._execute_set_value(step, context)
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unsupported step type: {step_type}",
                    step_name=step_name,
                    step_type=step_type_str
                )
        except Exception as e:
            self.logger.exception("step.execution_error", 
                               step_name=step_name,
                               step_type=step_type_str,
                               error=str(e))
            return ExecutionResult(
                success=False,
                error=f"Error executing step {step_name}: {str(e)}",
                step_name=step_name,
                step_type=step_type_str
            )
    
    def _execute_skill_call(
        self, 
        step: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a skill_call step by invoking a registered skill.
        
        This method uses the SkillRegistry to look up the skill and then uses
        the Prefect task wrapper (execute_skill_task) to execute it.
        
        Args:
            step: The skill_call step definition
            context: The current execution context
            
        Returns:
            ExecutionResult with the skill execution result
        """
        step_name = step.get("name", "unnamed_skill_call")
        skill_name = step.get("skill")
        
        if not skill_name:
            return ExecutionResult(
                success=False,
                error=f"Missing skill name in step {step_name}",
                step_name=step_name,
                step_type="skill_call"
            )
        
        # Prepare skill parameters
        params = self._prepare_parameters(step.get("params", {}), context)
        
        self.logger.info("skill_call.executing", 
                      step_name=step_name,
                      skill=skill_name,
                      param_keys=list(params.keys()) if params else [])
        
        try:
            # Get the skill configuration from the registry
            skill_config = self.skill_registry.get_skill_config(skill_name)
            if not skill_config:
                return ExecutionResult(
                    success=False,
                    error=f"Skill '{skill_name}' not found in registry",
                    step_name=step_name,
                    step_type="skill_call"
                )
            
            # Create an immutable context for the skill
            skill_context = copy.deepcopy(context)
            
            # Add skill execution metadata
            skill_execution_id = str(uuid.uuid4())
            if "execution_metadata" not in skill_context:
                skill_context["execution_metadata"] = {}
            skill_context["execution_metadata"]["skill_execution_id"] = skill_execution_id
            skill_context["execution_metadata"]["skill_name"] = skill_name
            skill_context["execution_metadata"]["step_name"] = step_name
            
            # Prepare event logger config for the skill execution
            event_logger_config = None
            if self.event_logger:
                event_logger_config = {
                    "parent_execution_id": self.execution_id,
                    "namespace": f"skill_{skill_name}",
                    "enabled": True
                }
            
            # Import the execute_skill_task from the Prefect wrappers
            try:
                from c4h_agents.execution.prefect_wrappers import execute_skill_task
                
                # Call the task directly (not as a flow) to keep execution in the current context
                skill_result_dict = execute_skill_task(
                    skill_name=skill_name,
                    parameters=params,
                    effective_config=self.config,
                    event_logger_config=event_logger_config
                )
                
                # Extract success, output, and error from the result
                success = skill_result_dict.get("success", True)
                data = skill_result_dict.get("result") 
                error = skill_result_dict.get("error")
                
            except ImportError:
                # Fallback to direct execution if Prefect wrappers are not available
                self.logger.warning("skill_call.prefect_wrappers_not_available", 
                                 skill_name=skill_name,
                                 falling_back="direct_execution")
                
                # Instantiate and execute the skill
                skill_instance = self.skill_registry.instantiate_skill(skill_name, self.config)
                
                # Pass parameters as kwargs
                if "input" not in params and context:
                    params["input"] = skill_context
                    
                # Execute the skill method
                method_name = skill_config.get("method", "execute")
                skill_method = getattr(skill_instance, method_name)
                skill_result = skill_method(**params)
                
                # Process the result
                if hasattr(skill_result, "success") and hasattr(skill_result, "value"):
                    # Standard SkillResult type
                    success = skill_result.success
                    data = skill_result.value
                    error = skill_result.error if hasattr(skill_result, "error") else None
                else:
                    # Non-standard result, treat as success with the whole result as data
                    success = True
                    data = skill_result
                    error = None
            
            self.logger.info("skill_call.completed", 
                          step_name=step_name,
                          skill=skill_name,
                          success=success,
                          error=error)
                          
            return ExecutionResult(
                success=success,
                context={},
                output=data,
                error=error,
                step_name=step_name,
                step_type="skill_call"
            )
            
        except Exception as e:
            self.logger.exception("skill_call.failed", 
                               step_name=step_name,
                               skill=skill_name,
                               error=str(e))
            return ExecutionResult(
                success=False,
                error=f"Error executing skill '{skill_name}': {str(e)}",
                step_name=step_name,
                step_type="skill_call"
            )
    
    def _execute_agent_call(
        self, 
        step: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute an agent_call step by invoking an agent instance.
        
        This method performs the following:
        1. Resolves the target agent instance by name
        2. Looks up its configuration/persona
        3. If the agent's persona has an execution_plan, recursively calls execute_plan
        4. Otherwise, invokes the agent's process() method via Prefect task wrapper
        
        Args:
            step: The agent_call step definition
            context: The current execution context
            
        Returns:
            ExecutionResult with the agent execution result
        """
        step_name = step.get("name", "unnamed_agent_call")
        node_name = step.get("node")
        
        if not node_name:
            return ExecutionResult(
                success=False,
                error=f"Missing node name in step {step_name}",
                step_name=step_name,
                step_type="agent_call"
            )
        
        # Prepare agent parameters
        params = self._prepare_parameters(step.get("input_params", {}), context)
        
        self.logger.info("agent_call.executing", 
                      step_name=step_name,
                      node=node_name,
                      param_keys=list(params.keys()) if params else [])
        
        try:
            # Create an immutable context for the agent
            agent_context = copy.deepcopy(context)
            
            # Add agent execution metadata
            agent_execution_id = str(uuid.uuid4())
            if "execution_metadata" not in agent_context:
                agent_context["execution_metadata"] = {}
            agent_context["execution_metadata"]["agent_execution_id"] = agent_execution_id
            agent_context["execution_metadata"]["agent_name"] = node_name
            agent_context["execution_metadata"]["step_name"] = step_name
            
            # Add input parameters to context
            if params:
                agent_context.update(params)
            
            # Look up the agent's configuration and persona
            agent_config = None
            persona_config = None
            
            # First check in the current team's agents list
            team_context = agent_context.get("team_context", {})
            current_team_id = team_context.get("team_id")
            if current_team_id:
                # Look for the agent in the current team's agents list
                team_config = self.config_node.get_value(f"orchestration.teams.{current_team_id}")
                if team_config:
                    # Check both "agents" and "tasks" (for backward compatibility)
                    for agents_key in ["agents", "tasks"]:
                        if agents_key in team_config:
                            for agent in team_config[agents_key]:
                                if agent.get("name") == node_name:
                                    agent_config = agent
                                    
                                    # If agent has persona_key, get the persona config
                                    persona_key = agent.get("persona_key")
                                    if persona_key:
                                        persona_config = self.config_node.get_value(f"llm_config.personas.{persona_key}")
                                    
                                    break
                            
                            # Break outer loop if agent found
                            if agent_config:
                                break
            
            # If not found in team, check global agents configuration
            if not agent_config:
                agent_config = self.config_node.get_value(f"llm_config.agents.{node_name}")
                if agent_config:
                    # If agent has persona_key, get the persona config
                    persona_key = agent_config.get("persona_key")
                    if persona_key:
                        persona_config = self.config_node.get_value(f"llm_config.personas.{persona_key}")
            
            # If still not found, look for the agent in all teams
            if not agent_config:
                teams = self.config_node.get_value("orchestration.teams", {})
                for team_id, team_config in teams.items():
                    # Check both "agents" and "tasks" (for backward compatibility)
                    for agents_key in ["agents", "tasks"]:
                        if agents_key in team_config:
                            for agent in team_config[agents_key]:
                                if agent.get("name") == node_name:
                                    agent_config = agent
                                    
                                    # If agent has persona_key, get the persona config
                                    persona_key = agent.get("persona_key")
                                    if persona_key:
                                        persona_config = self.config_node.get_value(f"llm_config.personas.{persona_key}")
                                    
                                    break
                            
                            # Break outer loop if agent found
                            if agent_config:
                                break
                    
                    # Break teams loop if agent found
                    if agent_config:
                        break
            
            # If agent not found in configuration, log error and return
            if not agent_config:
                self.logger.error("agent_call.agent_not_found", 
                               step_name=step_name,
                               node=node_name)
                return ExecutionResult(
                    success=False,
                    error=f"Agent '{node_name}' not found in configuration",
                    step_name=step_name,
                    step_type="agent_call"
                )
            
            # Determine the agent_type for proper instantiation
            agent_type = agent_config.get("agent_type")
            
            # If no agent_type in agent config, check persona config
            if not agent_type and persona_config:
                agent_type = persona_config.get("agent_type")
                
            # If still no agent_type, use GenericLLMAgent as default
            if not agent_type:
                agent_type = "GenericLLMAgent"
                self.logger.warning("agent_call.agent_type_defaulted", 
                                 step_name=step_name,
                                 node=node_name,
                                 agent_type=agent_type)
            
            # Merge agent_config and persona_config for complete configuration
            # Persona is the base, agent config overrides it
            merged_config = {}
            if persona_config:
                merged_config.update(persona_config)
            if agent_config:
                merged_config.update(agent_config)
            
            # Ensure agent_type is set
            merged_config["agent_type"] = agent_type
                                
            # Check if the agent's persona has an execution_plan
            if "execution_plan" in merged_config and merged_config.get("execution_plan", {}).get("enabled", True):
                self.logger.info("agent_call.executing_plan", 
                              step_name=step_name,
                              node=node_name,
                              agent_type=agent_type)
                
                # Extract the execution plan
                execution_plan = merged_config["execution_plan"]
                
                # Recursively call execute_plan with this plan and the current context
                result = self.execute_plan(execution_plan, agent_context)
                
                # Return the result as the step result
                self.logger.info("agent_call.execution_plan_completed", 
                              step_name=step_name,
                              node=node_name,
                              success=result.success)
                              
                return ExecutionResult(
                    success=result.success,
                    context=result.context,
                    output=result.output,
                    error=result.error,
                    step_name=step_name,
                    step_type="agent_call"
                )
            else:
                # No execution plan, use Prefect task wrapper to execute the agent
                self.logger.info("agent_call.using_prefect_task", 
                              step_name=step_name,
                              node=node_name,
                              agent_type=agent_type)
                
                # Prepare event logger config
                event_logger_config = None
                if self.event_logger:
                    event_logger_config = {
                        "parent_execution_id": self.execution_id,
                        "namespace": f"agent_{node_name}",
                        "enabled": True
                    }
                
                try:
                    # Import the agent execution task
                    from c4h_agents.execution.prefect_wrappers import execute_agent_task
                    
                    # Execute the agent using the Prefect task wrapper
                    agent_result = execute_agent_task(
                        agent_type=agent_type,
                        persona_config=merged_config,
                        input_context=agent_context,
                        effective_config=self.config,
                        event_logger_config=event_logger_config
                    )
                    
                    # Extract results from the task output
                    if isinstance(agent_result, dict):
                        success = agent_result.get("success", True)
                        output = agent_result.get("result") or agent_result.get("output")
                        error = agent_result.get("error")
                        context_updates = agent_result.get("context", {})
                    else:
                        success = True
                        output = agent_result
                        error = None
                        context_updates = {}
                        
                    self.logger.info("agent_call.prefect_task_completed", 
                                  step_name=step_name,
                                  node=node_name,
                                  success=success)
                                  
                    return ExecutionResult(
                        success=success,
                        context=context_updates,
                        output=output,
                        error=error,
                        step_name=step_name,
                        step_type="agent_call"
                    )
                
                except ImportError:
                    # Fallback to direct agent instantiation and execution
                    self.logger.warning("agent_call.prefect_wrappers_not_available", 
                                     step_name=step_name,
                                     node=node_name,
                                     falling_back="direct_execution")
                    
                    # Import the agent type dynamically
                    try:
                        agents_module = importlib.import_module("c4h_agents.agents.generic")
                        agent_class = getattr(agents_module, agent_type)
                    except (ImportError, AttributeError) as e:
                        self.logger.error("agent_call.agent_type_import_failed", 
                                       step_name=step_name,
                                       node=node_name,
                                       agent_type=agent_type,
                                       error=str(e))
                        return ExecutionResult(
                            success=False,
                            error=f"Failed to import agent type '{agent_type}': {str(e)}",
                            step_name=step_name,
                            step_type="agent_call"
                        )
                    
                    # Add effective config to agent configuration
                    merged_config["effective_config"] = self.config
                    
                    # Add event logger if available
                    if self.event_logger:
                        merged_config["event_logger"] = self.event_logger
                    
                    # Instantiate the agent
                    agent_instance = agent_class(merged_config)
                    
                    # Process the input context
                    result = agent_instance.process(agent_context)
                    
                    # Process the result
                    if isinstance(result, dict):
                        success = result.get("success", True)
                        output = result
                        error = result.get("error")
                    else:
                        success = True
                        output = result
                        error = None
                    
                    self.logger.info("agent_call.direct_execution_completed", 
                                  step_name=step_name,
                                  node=node_name,
                                  success=success)
                                  
                    return ExecutionResult(
                        success=success,
                        context={},
                        output=output,
                        error=error,
                        step_name=step_name,
                        step_type="agent_call"
                    )
            
        except Exception as e:
            self.logger.exception("agent_call.failed", 
                               step_name=step_name,
                               node=node_name,
                               error=str(e))
            return ExecutionResult(
                success=False,
                error=f"Error executing agent '{node_name}': {str(e)}",
                step_name=step_name,
                step_type="agent_call"
            )
    
    def _execute_team_call(
        self, 
        step: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a team_call step by invoking a team via its execution plan.
        
        This method performs the following:
        1. Resolves the target team configuration by ID
        2. Creates a team execution context with team-specific metadata
        3. If the team has an execution_plan, recursively calls execute_plan
        4. Otherwise, invokes each of the team's agents sequentially via Prefect task wrapper
        
        Args:
            step: The team_call step definition
            context: The current execution context
            
        Returns:
            ExecutionResult with the team execution result
        """
        step_name = step.get("name", "unnamed_team_call")
        team_id = step.get("target_team")
        
        if not team_id:
            return ExecutionResult(
                success=False,
                error=f"Missing target_team in step {step_name}",
                step_name=step_name,
                step_type="team_call"
            )
        
        # Prepare team parameters
        params = self._prepare_parameters(step.get("params", {}), context)
        
        self.logger.info("team_call.executing", 
                      step_name=step_name,
                      team_id=team_id,
                      param_keys=list(params.keys()) if params else [])
        
        try:
            # Look up the team configuration
            team_config = self.config_node.get_value(f"orchestration.teams.{team_id}")
            if not team_config:
                return ExecutionResult(
                    success=False,
                    error=f"Team '{team_id}' not found in configuration",
                    step_name=step_name,
                    step_type="team_call"
                )
            
            # Create an immutable context for the team
            team_context = copy.deepcopy(context)
            
            # Add team execution metadata
            team_execution_id = str(uuid.uuid4())
            if "execution_metadata" not in team_context:
                team_context["execution_metadata"] = {}
            team_context["execution_metadata"]["team_execution_id"] = team_execution_id
            team_context["execution_metadata"]["team_id"] = team_id
            team_context["execution_metadata"]["step_name"] = step_name
            
            # Add team context if not already present
            if "team_context" not in team_context:
                team_context["team_context"] = {}
            team_context["team_context"]["team_id"] = team_id
            team_context["team_context"]["team_name"] = team_config.get("name", team_id)
            
            # Add input parameters to context
            if params:
                team_context.update(params)
            
            # Check if the team has an execution_plan
            if "execution_plan" in team_config and team_config.get("execution_plan", {}).get("enabled", True):
                self.logger.info("team_call.executing_plan", 
                              step_name=step_name,
                              team_id=team_id)
                
                # Extract the execution plan
                execution_plan = team_config["execution_plan"]
                
                # Recursively call execute_plan with this plan and the current context
                result = self.execute_plan(execution_plan, team_context)
                
                # Return the result as the step result
                self.logger.info("team_call.execution_plan_completed", 
                              step_name=step_name,
                              team_id=team_id,
                              success=result.success)
                              
                return ExecutionResult(
                    success=result.success,
                    context=result.context,
                    output=result.output,
                    error=result.error,
                    step_name=step_name,
                    step_type="team_call"
                )
                
            else:
                # No execution plan, use Prefect task wrapper to execute the team
                self.logger.info("team_call.using_prefect_task", 
                              step_name=step_name,
                              team_id=team_id)
                
                # Prepare event logger config
                event_logger_config = None
                if self.event_logger:
                    event_logger_config = {
                        "parent_execution_id": self.execution_id,
                        "namespace": f"team_{team_id}",
                        "enabled": True
                    }
                
                try:
                    # Import the team execution task
                    from c4h_agents.execution.prefect_wrappers import execute_team_task
                    
                    # Execute the team using the Prefect task wrapper
                    team_result = execute_team_task(
                        team_config=team_config,
                        team_name=team_id,
                        input_context=team_context,
                        effective_config=self.config,
                        event_logger_config=event_logger_config
                    )
                    
                    # Extract results from the task output
                    if isinstance(team_result, dict):
                        success = team_result.get("success", True)
                        output = team_result.get("output")
                        error = team_result.get("error")
                        context_updates = team_result.get("context", {})
                    else:
                        success = True
                        output = team_result
                        error = None
                        context_updates = {}
                        
                    self.logger.info("team_call.prefect_task_completed", 
                                  step_name=step_name,
                                  team_id=team_id,
                                  success=success)
                                  
                    return ExecutionResult(
                        success=success,
                        context=context_updates,
                        output=output,
                        error=error,
                        step_name=step_name,
                        step_type="team_call"
                    )
                    
                except ImportError:
                    # Fallback to direct team execution (execute agents sequentially)
                    self.logger.warning("team_call.prefect_wrappers_not_available", 
                                     step_name=step_name,
                                     team_id=team_id,
                                     falling_back="direct_execution")
                    
                    # Get the team's agents list
                    agents_list = team_config.get("agents", [])
                    if not agents_list:
                        # Check for tasks (legacy configuration)
                        agents_list = team_config.get("tasks", [])
                        
                    if not agents_list:
                        return ExecutionResult(
                            success=False,
                            error=f"Team '{team_id}' has no execution_plan and no agents/tasks list",
                            step_name=step_name,
                            step_type="team_call"
                        )
                    
                    # Execute each agent in sequence
                    agent_results = []
                    current_context = team_context  # Start with the team context
                    
                    for i, agent_config in enumerate(agents_list):
                        agent_name = agent_config.get("name", f"agent_{i}")
                        
                        # Create a step definition for the agent call
                        agent_step = {
                            "name": f"{step_name}_agent_{agent_name}",
                            "type": "agent_call",
                            "node": agent_name,
                            "input_params": {},  # No additional params, using current_context directly
                            "stop_on_failure": agent_config.get("stop_on_failure", True)
                        }
                        
                        # Execute the agent using the agent_call executor
                        self.logger.info("team_call.executing_agent", 
                                      step_name=step_name,
                                      team_id=team_id,
                                      agent_name=agent_name)
                                      
                        agent_result = self._execute_agent_call(agent_step, current_context)
                        
                        # Store the agent result
                        agent_results.append({
                            "agent_name": agent_name,
                            "success": agent_result.success,
                            "error": agent_result.error
                        })
                        
                        # Update context for next agent if successful
                        if agent_result.success and agent_result.context:
                            # Create a new context merging the current with the agent result
                            for key, value in agent_result.context.items():
                                if key not in ["execution_metadata", "team_context"]:
                                    current_context[key] = value
                        
                        # Check if we should stop on failure
                        if not agent_result.success and agent_config.get("stop_on_failure", True):
                            self.logger.warning("team_call.agent_failed_stopping", 
                                             step_name=step_name,
                                             team_id=team_id,
                                             agent_name=agent_name)
                            break
                    
                    # Create final result
                    success = all(result["success"] for result in agent_results)
                    
                    self.logger.info("team_call.direct_execution_completed", 
                                  step_name=step_name,
                                  team_id=team_id,
                                  agents_executed=len(agent_results),
                                  success=success)
                                  
                    return ExecutionResult(
                        success=success,
                        context=current_context,
                        output={
                            "team_id": team_id,
                            "agents_executed": len(agent_results),
                            "agent_results": agent_results
                        },
                        error=None if success else f"Team execution failed: One or more agents failed",
                        step_name=step_name,
                        step_type="team_call"
                    )
            
        except Exception as e:
            self.logger.exception("team_call.failed", 
                               step_name=step_name,
                               team_id=team_id,
                               error=str(e))
            return ExecutionResult(
                success=False,
                error=f"Error executing team '{team_id}': {str(e)}",
                step_name=step_name,
                step_type="team_call"
            )
    
    def _execute_llm_call(
        self, 
        step: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute an llm_call step by directly invoking an LLM.
        
        This method creates a BaseLLM instance to directly interact with
        a language model provider without requiring a full agent instance.
        It supports various LLM providers through configuration.
        
        Args:
            step: The llm_call step definition
            context: The current execution context
            
        Returns:
            ExecutionResult with the LLM call result
        """
        step_name = step.get("name", "unnamed_llm_call")
        prompt = step.get("prompt")
        
        if not prompt:
            return ExecutionResult(
                success=False,
                error=f"Missing prompt in step {step_name}",
                step_name=step_name,
                step_type="llm_call"
            )
        
        # Get LLM parameters
        provider_name = step.get("provider")
        model_name = step.get("model")
        temperature = step.get("temperature", 0.7)  # Default temperature if not specified
        system_message = step.get("system_message")
        
        # Format prompt with context
        formatted_prompt = self._format_template(prompt, context)
        
        # Check for required parameters
        if not provider_name:
            # Look for default provider in config
            provider_name = self.config_node.get_value("llm_config.default_provider")
            if not provider_name:
                return ExecutionResult(
                    success=False,
                    error=f"Missing provider in step {step_name} and no default provider in config",
                    step_name=step_name,
                    step_type="llm_call"
                )
                
        if not model_name:
            # Look for default model in config
            model_name = self.config_node.get_value("llm_config.default_model")
            if not model_name:
                return ExecutionResult(
                    success=False,
                    error=f"Missing model in step {step_name} and no default model in config",
                    step_name=step_name,
                    step_type="llm_call"
                )
        
        self.logger.info("llm_call.executing", 
                      step_name=step_name,
                      provider=provider_name,
                      model=model_name,
                      temperature=temperature,
                      prompt_length=len(formatted_prompt) if formatted_prompt else 0)
        
        try:
            # Import BaseLLM with dynamic import to avoid circular dependencies
            try:
                from c4h_agents.agents.base_llm import BaseLLM, LLMProvider
                import importlib
            except ImportError as e:
                self.logger.error("llm_call.import_failed", 
                               error=str(e))
                return ExecutionResult(
                    success=False,
                    error=f"Failed to import BaseLLM: {str(e)}",
                    step_name=step_name,
                    step_type="llm_call"
                )
            
            # Create messages array with system message if provided
            messages = []
            if system_message:
                formatted_system = self._format_template(system_message, context)
                messages.append({"role": "system", "content": formatted_system})
            
            # Add user message with formatted prompt
            messages.append({"role": "user", "content": formatted_prompt})
            
            # Create BaseLLM instance
            llm = BaseLLM()
            
            # Set provider and model
            try:
                llm.provider = LLMProvider(provider_name)
            except (ValueError, TypeError) as e:
                self.logger.error("llm_call.invalid_provider", 
                               provider=provider_name,
                               error=str(e))
                return ExecutionResult(
                    success=False,
                    error=f"Invalid provider '{provider_name}': {str(e)}",
                    step_name=step_name,
                    step_type="llm_call"
                )
                
            llm.model = model_name
            llm.temperature = float(temperature)
            
            # Get provider configuration from effective config
            provider_config = self.config_node.get_value(f"llm_config.providers.{provider_name}", {})
            
            # Set up BaseLLM configuration
            # Manually configure important attributes that would normally be set in BaseAgent
            llm.config_node = self.config_node
            
            # Set logging
            llm.logger = self.logger.bind(
                llm_call=step_name,
                provider=provider_name,
                model=model_name
            )
            
            # Get the appropriate model string for LiteLLM
            try:
                # Try using _get_model_str directly if it doesn't need config_node
                llm.model_str = f"{provider_name}/{model_name}"
                
                # Set up LiteLLM if needed
                if hasattr(llm, '_setup_litellm'):
                    llm._setup_litellm(provider_config)
            except Exception as config_e:
                self.logger.error("llm_call.config_failed", 
                               error=str(config_e))
                return ExecutionResult(
                    success=False,
                    error=f"Failed to configure LLM: {str(config_e)}",
                    step_name=step_name,
                    step_type="llm_call"
                )
            
            # Make the LLM call
            try:
                # Use litellm directly instead of _get_completion_with_continuation
                # to avoid dependency on ContinuationHandler
                from litellm import completion
                
                # Prepare completion parameters
                completion_params = {
                    "model": llm.model_str,
                    "messages": messages,
                    "temperature": llm.temperature
                }
                
                # Add API base if present in provider config
                if "api_base" in provider_config:
                    completion_params["api_base"] = provider_config["api_base"]
                
                # Add model specific parameters if present
                model_params = provider_config.get("model_params", {})
                for key, value in model_params.items():
                    if key not in completion_params:
                        completion_params[key] = value
                
                # Execute the LLM call
                self.logger.debug("llm_call.completion_params", 
                               params={k: v for k, v in completion_params.items() if k != "messages"})
                               
                response = completion(**completion_params)
                
                # Process the response
                if response and hasattr(response, 'choices') and response.choices:
                    content = llm._get_llm_content(response)
                    
                    # Create result
                    output = {
                        "response": content
                    }
                    
                    # Add usage information if available
                    if hasattr(response, 'usage'):
                        usage = response.usage
                        output["usage"] = {
                            "completion_tokens": getattr(usage, 'completion_tokens', 0),
                            "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                            "total_tokens": getattr(usage, 'total_tokens', 0)
                        }
                        
                        self.logger.info("llm_call.token_usage",
                                      completion_tokens=output["usage"]["completion_tokens"],
                                      prompt_tokens=output["usage"]["prompt_tokens"],
                                      total_tokens=output["usage"]["total_tokens"])
                    
                    self.logger.info("llm_call.completed",
                                  step_name=step_name,
                                  provider=provider_name,
                                  model=model_name,
                                  content_length=len(content) if content else 0)
                                  
                    return ExecutionResult(
                        success=True,
                        context={},
                        output=output,
                        step_name=step_name,
                        step_type="llm_call"
                    )
                else:
                    self.logger.error("llm_call.invalid_response",
                                   step_name=step_name,
                                   provider=provider_name,
                                   model=model_name)
                    return ExecutionResult(
                        success=False,
                        error=f"LLM call returned invalid response structure",
                        step_name=step_name,
                        step_type="llm_call"
                    )
                
            except Exception as llm_e:
                self.logger.exception("llm_call.completion_failed",
                                   step_name=step_name,
                                   provider=provider_name,
                                   model=model_name,
                                   error=str(llm_e))
                return ExecutionResult(
                    success=False,
                    error=f"Error during LLM completion: {str(llm_e)}",
                    step_name=step_name,
                    step_type="llm_call"
                )
            
        except Exception as e:
            self.logger.exception("llm_call.failed", 
                               step_name=step_name,
                               provider=provider_name,
                               model=model_name,
                               error=str(e))
            return ExecutionResult(
                success=False,
                error=f"Error executing LLM call: {str(e)}",
                step_name=step_name,
                step_type="llm_call"
            )
    
    def _execute_loop(
        self, 
        step: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a loop step by iterating over a collection and executing child steps.
        
        This method handles iteration over any iterable collection in the context,
        executing a nested execution plan for each element and collecting results.
        Supports loop control statements (break, continue) and context propagation
        between iterations.
        
        Args:
            step: The loop step definition
            context: The current execution context
            
        Returns:
            ExecutionResult with the loop execution result
        """
        step_name = step.get("name", "unnamed_loop")
        iterate_on = step.get("iterate_on")
        loop_variable = step.get("loop_variable")
        body = step.get("body", [])
        
        # Validate required parameters
        if not iterate_on:
            return ExecutionResult(
                success=False,
                error=f"Missing iterate_on in loop step {step_name}",
                step_name=step_name,
                step_type="loop"
            )
            
        if not loop_variable:
            return ExecutionResult(
                success=False,
                error=f"Missing loop_variable in loop step {step_name}",
                step_name=step_name,
                step_type="loop"
            )
            
        if not body:
            return ExecutionResult(
                success=False,
                error=f"Empty body in loop step {step_name}",
                step_name=step_name,
                step_type="loop"
            )
        
        # Get the collection to iterate over
        # First handle dynamic templates in the iterate_on path
        if "${" in iterate_on and "}" in iterate_on:
            # This is a template that needs resolving
            iterate_on = self._format_template(iterate_on, context)
            
        collection = self._get_context_value(context, iterate_on)
        if collection is None:
            return ExecutionResult(
                success=False,
                error=f"Collection not found at path '{iterate_on}' in loop step {step_name}",
                step_name=step_name,
                step_type="loop"
            )
            
        # Ensure it's iterable
        if not hasattr(collection, "__iter__"):
            return ExecutionResult(
                success=False,
                error=f"Value at path '{iterate_on}' is not iterable in loop step {step_name}",
                step_name=step_name,
                step_type="loop"
            )
        
        self.logger.info("loop.executing", 
                      step_name=step_name,
                      collection_path=iterate_on,
                      collection_length=len(collection) if hasattr(collection, "__len__") else "unknown",
                      body_steps=len(body))
        
        # Initialize aggregated context to collect all updates
        aggregated_context = copy.deepcopy(context)
        
        # Execute the loop
        iteration_results = []
        aggregated_output = []
        loop_items = list(collection)  # Convert to list to ensure consistent iteration
        
        # Get parameters for stopping iteration failures
        stop_on_iteration_failure = step.get("stop_on_iteration_failure", step.get("stop_on_failure", True))
        
        # Additional parameters for ArchDocV3 support
        max_iterations = step.get("max_iterations")
        collection_filter = step.get("filter") # Optional filter to apply to items
        limit_items = len(loop_items)
        if max_iterations is not None and max_iterations > 0:
            limit_items = min(limit_items, int(max_iterations))
            
        for i, item in enumerate(loop_items[:limit_items]):
            # Apply filter if provided
            if collection_filter:
                # Create temporary context with item to evaluate filter condition
                temp_context = copy.deepcopy(context)
                temp_context[loop_variable] = item
                
                # Check if filter condition passes
                filter_result = self._evaluate_condition(collection_filter, temp_context)
                if not filter_result:
                    self.logger.debug("loop.item_filtered_out", 
                                   step_name=step_name, 
                                   iteration=i)
                    continue  # Skip this item
            
            # Create a new context for this iteration
            iteration_context = copy.deepcopy(aggregated_context)
            
            # Set the loop variable in the context
            iteration_context[loop_variable] = item
            
            # Add loop metadata
            if "loop_metadata" not in iteration_context:
                iteration_context["loop_metadata"] = {}
            iteration_context["loop_metadata"]["current_loop"] = step_name
            iteration_context["loop_metadata"]["current_index"] = i
            iteration_context["loop_metadata"]["item_index"] = i
            iteration_context["loop_metadata"]["total_items"] = len(loop_items)
            iteration_context["loop_metadata"]["iteration_count"] = i + 1
            
            # Log loop iteration start
            self.logger.debug("loop.iteration_start", 
                           step_name=step_name,
                           iteration=i,
                           total=len(loop_items))
                           
            if self.event_logger:
                self.event_logger.log_event(
                    event_type=EventType.LOOP_ITERATION_START,
                    payload={
                        "step_name": step_name,
                        "iteration": i,
                        "total": len(loop_items)
                    }
                )
            
            # Create a mini execution plan for the body
            body_plan = {
                "enabled": True,
                "steps": body
            }
            
            # Execute the body as a nested plan
            iteration_result = self.execute_plan(body_plan, iteration_context)
            
            # Process the result data
            item_output = iteration_result.output
            if item_output is None:
                item_output = {}
                
            # Store the iteration result
            iteration_results.append({
                "iteration": i,
                "success": iteration_result.success,
                "error": iteration_result.error,
                "output": item_output 
            })
            
            # Add to aggregated output if this iteration succeeded
            if iteration_result.success:
                aggregated_output.append(item_output)
            
            # Log loop iteration end
            self.logger.debug("loop.iteration_end", 
                           step_name=step_name,
                           iteration=i,
                           success=iteration_result.success)
                           
            if self.event_logger:
                self.event_logger.log_event(
                    event_type=EventType.LOOP_ITERATION_END,
                    payload={
                        "step_name": step_name,
                        "iteration": i,
                        "success": iteration_result.success,
                        "error": iteration_result.error
                    }
                )
            
            # Check for special actions
            if iteration_result.action:
                if iteration_result.action == StepAction.BREAK_LOOP:
                    self.logger.info("loop.break_encountered",
                                  step_name=step_name,
                                  iteration=i)
                    break
                elif iteration_result.action == StepAction.CONTINUE_LOOP:
                    self.logger.info("loop.continue_encountered",
                                  step_name=step_name,
                                  iteration=i)
                    # Use the iteration context up to this point, then continue to next iteration
                    continue
            
            # Check for failure
            if not iteration_result.success and stop_on_iteration_failure:
                self.logger.warning("loop.iteration_failed", 
                                 step_name=step_name,
                                 iteration=i,
                                 error=iteration_result.error)
                
                # Check if we should return failures in the output
                return ExecutionResult(
                    success=False,
                    context=aggregated_context,  # Return the context as it was before the failed iteration
                    output={
                        "iterations": iteration_results,
                        "aggregated_output": aggregated_output
                    },
                    error=f"Loop iteration {i} failed in step {step_name}: {iteration_result.error}",
                    step_name=step_name,
                    step_type="loop"
                )
            
            # Update aggregated context with this iteration's context if successful
            if iteration_result.success and iteration_result.context:
                # But only apply updates to the parent context, don't replace loop-specific keys
                parent_keys_to_update = set(iteration_result.context.keys()) - {loop_variable, "loop_metadata"}
                for key in parent_keys_to_update:
                    aggregated_context[key] = iteration_result.context[key]
        
        # Check if we should aggregate the outputs into a specific structure
        output_format = step.get("output_format", "list")
        output_data = aggregated_output
        
        if output_format == "map" and loop_variable and step.get("key_field"):
            # Convert output list to a map using a key field
            key_field = step.get("key_field")
            output_map = {}
            
            for idx, item_result in enumerate(output_data):
                if isinstance(item_result, dict) and key_field in item_result:
                    key = item_result[key_field]
                    if isinstance(key, (str, int, float, bool)):  # Ensure key is a valid type
                        output_map[str(key)] = item_result
                else:
                    # If key_field not found, use index as key
                    output_map[f"item_{idx}"] = item_result
                    
            output_data = output_map
        
        # Add summary statistics
        success_count = sum(1 for r in iteration_results if r.get("success", False))
        
        self.logger.info("loop.completed", 
                      step_name=step_name,
                      iterations_completed=len(iteration_results),
                      successful_iterations=success_count)
        
        return ExecutionResult(
            success=True,
            context=aggregated_context,
            output={
                "iterations": iteration_results,
                "output": output_data,
                "stats": {
                    "total": len(iteration_results),
                    "success": success_count,
                    "failure": len(iteration_results) - success_count
                }
            },
            step_name=step_name,
            step_type="loop"
        )
    
    def _execute_branch(
        self, 
        step: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a branch step by evaluating conditions and determining routing.
        
        This method evaluates a set of conditional branches and determines which
        execution path to take. It supports complex conditions with AND/OR/NOT logic,
        multiple branches, and explicit actions to take when a condition is true.
        
        Args:
            step: The branch step definition
            context: The current execution context
            
        Returns:
            ExecutionResult with the branch evaluation result
        """
        step_name = step.get("name", "unnamed_branch")
        branches = step.get("branches", [])
        
        if not branches:
            return ExecutionResult(
                success=False,
                error=f"No branches defined in branch step {step_name}",
                step_name=step_name,
                step_type="branch"
            )
        
        # Get default branch if specified
        default_branch = step.get("default")
        
        self.logger.info("branch.executing", 
                      step_name=step_name,
                      branch_count=len(branches),
                      has_default=default_branch is not None)
        
        # Check if we should execute a branch body directly instead of routing
        execute_inline = step.get("execute_inline", False)
        
        # Evaluate each branch condition
        for branch_idx, branch in enumerate(branches):
            branch_name = branch.get("name", f"branch_{branch_idx}")
            condition = branch.get("condition", {})
            target = branch.get("target")
            action = branch.get("action")
            
            # New in ArchDocV3: Support for branch inline execution
            branch_body = branch.get("body", [])
            
            # Validate branch configuration
            if not condition:
                self.logger.warning("branch.missing_condition", 
                                 step_name=step_name,
                                 branch_idx=branch_idx,
                                 branch_name=branch_name)
                continue
                
            # Check if we need a target
            if not execute_inline and target is None and action is None and not branch_body:
                self.logger.warning("branch.missing_target", 
                                 step_name=step_name,
                                 branch_idx=branch_idx,
                                 branch_name=branch_name)
                continue
            
            # Evaluate the condition
            condition_result = self._evaluate_condition(condition, context)
            
            if condition_result:
                self.logger.info("branch.condition_true", 
                              step_name=step_name,
                              branch_idx=branch_idx,
                              branch_name=branch_name,
                              target=target,
                              action=action,
                              has_body=bool(branch_body))
                
                # Log routing event for lineage tracking
                if self.event_logger:
                    self.event_logger.log_event(
                        event_type=EventType.ROUTING_EVALUATION,
                        payload={
                            "step_name": step_name,
                            "branch_name": branch_name,
                            "branch_idx": branch_idx,
                            "condition": condition,
                            "result": True,
                            "target": target,
                            "action": action
                        }
                    )
                
                # Check if we have a branch body to execute inline
                if execute_inline and branch_body:
                    self.logger.info("branch.executing_inline_body", 
                                  step_name=step_name,
                                  branch_name=branch_name,
                                  steps=len(branch_body))
                    
                    # Create a mini execution plan for the branch body
                    branch_plan = {
                        "enabled": True,
                        "steps": branch_body
                    }
                    
                    # Execute the branch body as a nested plan
                    branch_result = self.execute_plan(branch_plan, context)
                    
                    # Return the result with routing information
                    return ExecutionResult(
                        success=branch_result.success,
                        context=branch_result.context,
                        output={
                            "branch_idx": branch_idx,
                            "branch_name": branch_name,
                            "result": branch_result.output,
                            "mode": "inline_execution"
                        },
                        error=branch_result.error,
                        step_name=step_name,
                        step_type="branch",
                        action=branch_result.action  # Propagate any action from the branch body
                    )
                
                # Check if we have a special action to take
                if action:
                    self.logger.info("branch.action_specified", 
                                  step_name=step_name,
                                  branch_name=branch_name,
                                  action=action)
                    
                    # Map the action to StepAction enum
                    action_map = {
                        "exit_plan_with_success": StepAction.EXIT_PLAN_WITH_SUCCESS,
                        "exit_plan_with_failure": StepAction.EXIT_PLAN_WITH_FAILURE,
                        "break_loop": StepAction.BREAK_LOOP,
                        "continue_loop": StepAction.CONTINUE_LOOP
                    }
                    
                    if action in action_map:
                        step_action = action_map[action]
                        
                        # Return with the specified action
                        return ExecutionResult(
                            success=action != "exit_plan_with_failure",
                            context={},
                            output={
                                "branch_idx": branch_idx,
                                "branch_name": branch_name,
                                "action": action
                            },
                            step_name=step_name,
                            step_type="branch",
                            action=step_action
                        )
                    else:
                        self.logger.warning("branch.unknown_action", 
                                         step_name=step_name,
                                         branch_name=branch_name,
                                         action=action)
                
                # Normal routing to a target
                return ExecutionResult(
                    success=True,
                    context={},
                    output={
                        "branch_idx": branch_idx, 
                        "branch_name": branch_name,
                        "target": target,
                        "mode": "routing"
                    },
                    step_name=step_name,
                    step_type="branch"
                )
        
        # No branch condition was true, check for default
        if default_branch is not None:
            self.logger.info("branch.using_default", 
                          step_name=step_name,
                          default=default_branch)
            
            # Return the default branch target
            return ExecutionResult(
                success=True,
                context={},
                output={
                    "branch_name": "default",
                    "target": default_branch,
                    "mode": "default_routing"
                },
                step_name=step_name,
                step_type="branch"
            )
            
        # No branch condition was true and no default
        self.logger.info("branch.no_condition_true", 
                      step_name=step_name)
        
        return ExecutionResult(
            success=True,
            context={},
            output={
                "message": "No branch condition was true",
                "mode": "no_match"
            },
            step_name=step_name,
            step_type="branch"
        )
    
    def _execute_set_value(
        self, 
        step: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a set_value step by setting a value in the context.
        
        This method supports complex value setting, including template resolution
        in both keys and values, recursive resolution of nested structures, and
        special functions for value generation (timestamps, UUIDs, etc.)
        
        Args:
            step: The set_value step definition
            context: The current execution context
            
        Returns:
            ExecutionResult with the set_value result
        """
        step_name = step.get("name", "unnamed_set_value")
        field = step.get("field")
        value = step.get("value")
        value_type = step.get("type")  # New in ArchDocV3: explicit type
        
        # Check for multiple fields/values
        fields = step.get("fields", [])
        
        if not field and not fields:
            return ExecutionResult(
                success=False,
                error=f"Missing field(s) in set_value step {step_name}",
                step_name=step_name,
                step_type="set_value"
            )
        
        try:
            # First, handle single-field case
            if field:
                self.logger.info("set_value.executing.single_field", 
                              step_name=step_name,
                              field=field)
                
                # Format the field path if it's a template
                if isinstance(field, str) and "${" in field and "}" in field:
                    field = self._format_template(field, context)
                    
                # Process value based on type if specified
                formatted_value = self._process_value_by_type(value, value_type, context)
                
                # Create a new context with the updated value
                updated_context = self._set_context_value(context, field, formatted_value)
                
                return ExecutionResult(
                    success=True,
                    context=updated_context,
                    output={"field": field, "value": formatted_value},
                    step_name=step_name,
                    step_type="set_value"
                )
            
            # Handle multi-field case
            if fields:
                self.logger.info("set_value.executing.multi_field", 
                              step_name=step_name,
                              field_count=len(fields))
                
                updated_context = copy.deepcopy(context)
                results = {}
                
                for field_def in fields:
                    field_path = field_def.get("field")
                    field_value = field_def.get("value")
                    field_type = field_def.get("type")
                    
                    if not field_path:
                        self.logger.warning("set_value.missing_field_in_item", 
                                         step_name=step_name)
                        continue
                    
                    # Format the field path if it's a template
                    if isinstance(field_path, str) and "${" in field_path and "}" in field_path:
                        field_path = self._format_template(field_path, updated_context)
                    
                    # Process value based on type if specified
                    processed_value = self._process_value_by_type(field_value, field_type, updated_context)
                    
                    # Update the context
                    updated_context = self._set_context_value(updated_context, field_path, processed_value)
                    
                    # Store the result
                    results[field_path] = processed_value
                    
                    self.logger.debug("set_value.updated_field", 
                                   step_name=step_name,
                                   field=field_path)
                
                return ExecutionResult(
                    success=True,
                    context=updated_context,
                    output={"fields": results},
                    step_name=step_name,
                    step_type="set_value"
                )
            
        except Exception as e:
            self.logger.exception("set_value.failed", 
                               step_name=step_name,
                               field=field if field else "multiple_fields",
                               error=str(e))
            return ExecutionResult(
                success=False,
                error=f"Error in set_value step '{step_name}': {str(e)}",
                step_name=step_name,
                step_type="set_value"
            )
    
    def _process_value_by_type(
        self, 
        value: Any, 
        value_type: Optional[str], 
        context: Dict[str, Any]
    ) -> Any:
        """
        Process a value based on its type and resolve any templates.
        
        Args:
            value: The value to process
            value_type: Optional type information to guide processing
            context: The current execution context for template resolution
            
        Returns:
            The processed value
        """
        # Handle template strings
        if isinstance(value, str):
            # Check for "{{ }}" style templates
            if "{{" in value and "}}" in value:
                value = self._format_template(value, context)
            # Check for "${ }" style templates
            elif "${" in value and "}" in value:
                value = self._format_template(value, context)
        
        # Handle specific types if specified
        if value_type:
            if value_type == "timestamp":
                from datetime import datetime, timezone
                return datetime.now(timezone.utc).isoformat()
            
            elif value_type == "uuid":
                import uuid
                return str(uuid.uuid4())
            
            elif value_type == "integer":
                if isinstance(value, str):
                    return int(value)
                return int(value)
            
            elif value_type == "float":
                if isinstance(value, str):
                    return float(value)
                return float(value)
            
            elif value_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ["true", "yes", "1", "y"]
                return bool(value)
            
            elif value_type == "string":
                return str(value)
            
            elif value_type == "json":
                if isinstance(value, str):
                    import json
                    return json.loads(value)
                return value
            
            elif value_type == "template":
                # Already processed above for strings
                return value
            
            elif value_type == "path":
                # For values that reference paths in the context
                if isinstance(value, str):
                    return self._get_context_value(context, value)
                return value
        
        # Handle dictionaries (recursively resolve any template strings in values)
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                # Process the key if it's a template
                if isinstance(k, str) and "${" in k and "}" in k:
                    k = self._format_template(k, context)
                
                # Recursively process the value
                result[k] = self._process_value_by_type(v, None, context)
            return result
            
        # Handle lists (recursively resolve any template strings in items)
        if isinstance(value, list):
            return [self._process_value_by_type(item, None, context) for item in value]
        
        # Default case: return the value as is
        return value
    
    def _evaluate_condition(
        self, 
        condition: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a condition against the current context.
        
        Args:
            condition: The condition to evaluate
            context: The current execution context
            
        Returns:
            Boolean result of the condition evaluation
        """
        if not condition:
            return True  # Empty condition is always true
            
        condition_type = condition.get("type", "simple")
        
        if condition_type == "and":
            # All subconditions must be true
            subconditions = condition.get("conditions", [])
            return all(self._evaluate_condition(cond, context) for cond in subconditions)
            
        elif condition_type == "or":
            # Any subcondition must be true
            subconditions = condition.get("conditions", [])
            return any(self._evaluate_condition(cond, context) for cond in subconditions)
            
        elif condition_type == "not":
            # Negate the subcondition
            subcondition = condition.get("condition", {})
            return not self._evaluate_condition(subcondition, context)
            
        else:
            # Simple condition with field, operator, value
            field = condition.get("field")
            operator = condition.get("operator", "equals")
            expected_value = condition.get("value")
            
            if not field:
                self.logger.warning("condition.missing_field", condition=condition)
                return False
                
            # Get the actual value from context
            actual_value = self._get_context_value(context, field)
            
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
    
    def _apply_routing_rules(
        self, 
        step: Dict[str, Any], 
        step_result: ExecutionResult,
        context: Dict[str, Any],
        current_idx: int,
        steps: List[Dict[str, Any]]
    ) -> Optional[int]:
        """
        Apply routing rules to determine the next step.
        
        Args:
            step: The current step
            step_result: The result of the current step
            context: The current execution context
            current_idx: The current step index
            steps: The list of all steps in the plan
            
        Returns:
            The index of the next step to execute, or None to use default behavior
        """
        rules = step.get("rules", [])
        if not rules:
            return None
            
        step_name = step.get("name", f"step_{current_idx}")
        
        self.logger.debug("routing.evaluating_rules", 
                       step_name=step_name,
                       rule_count=len(rules))
        
        for rule_idx, rule in enumerate(rules):
            condition = rule.get("condition", {})
            
            # Evaluate the condition
            condition_result = self._evaluate_condition(condition, context)
            
            if condition_result:
                self.logger.info("routing.rule_matched", 
                              step_name=step_name,
                              rule_idx=rule_idx)
                
                # Check for action
                action = rule.get("action")
                if action:
                    # Handle special actions
                    if action == "exit_plan_with_success":
                        self.logger.info("routing.exit_with_success", 
                                      step_name=step_name,
                                      rule_idx=rule_idx)
                        step_result.action = StepAction.EXIT_PLAN_WITH_SUCCESS
                        return None
                    elif action == "exit_plan_with_failure":
                        self.logger.info("routing.exit_with_failure", 
                                      step_name=step_name,
                                      rule_idx=rule_idx)
                        step_result.action = StepAction.EXIT_PLAN_WITH_FAILURE
                        return None
                    elif action == "break_loop":
                        self.logger.info("routing.break_loop", 
                                      step_name=step_name,
                                      rule_idx=rule_idx)
                        step_result.action = StepAction.BREAK_LOOP
                        return None
                    elif action == "continue_loop":
                        self.logger.info("routing.continue_loop", 
                                      step_name=step_name,
                                      rule_idx=rule_idx)
                        step_result.action = StepAction.CONTINUE_LOOP
                        return None
                
                # Get the target step
                route_to_step = rule.get("route_to_step")
                if route_to_step is None:
                    self.logger.warning("routing.missing_target", 
                                     step_name=step_name,
                                     rule_idx=rule_idx)
                    continue
                
                # If target is a string, find the step by name
                if isinstance(route_to_step, str):
                    target_idx = None
                    for i, s in enumerate(steps):
                        if s.get("name") == route_to_step:
                            target_idx = i
                            break
                            
                    if target_idx is None:
                        self.logger.error("routing.target_not_found", 
                                       step_name=step_name,
                                       rule_idx=rule_idx,
                                       target=route_to_step)
                        continue
                        
                    route_to_step = target_idx
                
                # Check if target is a valid index
                if not isinstance(route_to_step, int) or route_to_step < 0 or route_to_step >= len(steps):
                    self.logger.error("routing.invalid_target", 
                                   step_name=step_name,
                                   rule_idx=rule_idx,
                                   target=route_to_step)
                    continue
                    
                # Log routing decision for lineage tracking
                if self.event_logger:
                    self.event_logger.log_event(
                        event_type=EventType.ROUTING_EVALUATION,
                        payload={
                            "step_name": step_name,
                            "rule_idx": rule_idx,
                            "condition": condition,
                            "result": True,
                            "from_step": current_idx,
                            "to_step": route_to_step
                        }
                    )
                
                # Return the target index
                self.logger.info("routing.jumping", 
                              from_step=current_idx, 
                              to_step=route_to_step,
                              rule_idx=rule_idx)
                return route_to_step
        
        # No rule matched, use default behavior
        return None
    
    def _get_context_value(
        self, 
        context: Dict[str, Any], 
        path: str
    ) -> Any:
        """
        Get a value from the context using a dot-notation path.
        
        Args:
            context: The execution context
            path: Dot-notation path to the value
            
        Returns:
            The value at the path, or None if not found
        """
        parts = path.split(".")
        current = context
        
        try:
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    # Path not found
                    return None
            return current
        except Exception:
            return None
    
    def _set_context_value(
        self, 
        context: Dict[str, Any], 
        path: str, 
        value: Any
    ) -> Dict[str, Any]:
        """
        Set a value in the context using a dot-notation path.
        Creates a new context rather than modifying the existing one.
        
        Args:
            context: The execution context
            path: Dot-notation path to set
            value: The value to set
            
        Returns:
            A new context with the value set
        """
        parts = path.split(".")
        
        # Create a deep copy to ensure immutability
        result = copy.deepcopy(context)
        
        # Handle empty path
        if not parts:
            return result
        
        # Navigate to the target location
        current = result
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value at the final location
        current[parts[-1]] = value
        
        return result
    
    def _prepare_parameters(
        self, 
        params: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare parameters for a step by substituting context values.
        
        Args:
            params: The raw parameters from the step definition
            context: The current execution context
            
        Returns:
            Parameters with context values substituted
        """
        if not params:
            return {}
            
        # Create a deep copy to avoid modifying the original
        result = copy.deepcopy(params)
        
        # Process each parameter
        for key, value in result.items():
            if isinstance(value, str):
                # Apply template substitution for strings
                result[key] = self._format_template(value, context)
            elif isinstance(value, dict):
                # Recursively process dictionaries
                result[key] = self._prepare_parameters(value, context)
            elif isinstance(value, list):
                # Process list items
                result[key] = [
                    self._format_template(item, context) if isinstance(item, str)
                    else self._prepare_parameters(item, context) if isinstance(item, dict)
                    else item
                    for item in value
                ]
        
        return result
    
    def _format_template(
        self, 
        template: str, 
        context: Dict[str, Any]
    ) -> str:
        """
        Format a template string by substituting context values.
        
        Args:
            template: The template string with {{context.path.to.value}} placeholders
            context: The current execution context
            
        Returns:
            The formatted string
        """
        if not template or not isinstance(template, str):
            return template
            
        # Find all context variable references
        pattern = r'\{\{(context\.[^}]+)\}\}'
        matches = re.findall(pattern, template)
        
        result = template
        for match in matches:
            # Extract the path
            path = match.replace("context.", "")
            
            # Get the value from context
            value = self._get_context_value(context, path)
            
            # Convert value to string for substitution
            if value is None:
                value_str = ""
            elif isinstance(value, (dict, list)):
                try:
                    value_str = json.dumps(value)
                except Exception:
                    value_str = str(value)
            else:
                value_str = str(value)
            
            # Replace in the template
            result = result.replace(f"{{{{{match}}}}}", value_str)
        
        return result