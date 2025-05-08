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
            
            # This step would normally use the Prefect task to run the agent
            # For now, we'll delegate to the agent factory/task implementation
            # that would be invoked via Prefect
            
            # Here, we'll just return a stub result for now
            # The actual implementation in the Prefect tasks will handle this
            self.logger.info("agent_call.delegated_to_prefect_task", 
                          step_name=step_name,
                          node=node_name)
                          
            return ExecutionResult(
                success=True,
                context={},
                output={
                    "message": "Agent call delegated to Prefect task execution",
                    "agent_name": node_name,
                    "agent_execution_id": agent_execution_id
                },
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
            # Create an immutable context for the team
            team_context = copy.deepcopy(context)
            
            # Add team execution metadata
            team_execution_id = str(uuid.uuid4())
            if "execution_metadata" not in team_context:
                team_context["execution_metadata"] = {}
            team_context["execution_metadata"]["team_execution_id"] = team_execution_id
            team_context["execution_metadata"]["team_id"] = team_id
            team_context["execution_metadata"]["step_name"] = step_name
            
            # Add input parameters to context
            if params:
                team_context.update(params)
            
            # This step would normally use the Prefect flow to run the team
            # For now, we'll delegate to the team execution flow that 
            # would be invoked via Prefect
            
            # Here, we'll just return a stub result for now
            # The actual implementation in the Prefect workflows will handle this
            self.logger.info("team_call.delegated_to_prefect_flow", 
                          step_name=step_name,
                          team_id=team_id)
                          
            return ExecutionResult(
                success=True,
                context={},
                output={
                    "message": "Team call delegated to Prefect flow execution",
                    "team_id": team_id,
                    "team_execution_id": team_execution_id
                },
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
        provider = step.get("provider")
        model = step.get("model")
        temperature = step.get("temperature")
        system_message = step.get("system_message")
        
        # Format prompt with context
        formatted_prompt = self._format_template(prompt, context)
        
        self.logger.info("llm_call.executing", 
                      step_name=step_name,
                      provider=provider,
                      model=model,
                      temperature=temperature,
                      prompt_length=len(formatted_prompt) if formatted_prompt else 0)
        
        try:
            # This would normally use an LLM client to make the API call
            # For now, we'll just delegate to a BaseLLM implementation
            # that would be instantiated with the specified provider and model
            
            # Here, we'll just return a stub result for now
            # The actual implementation will handle this
            self.logger.info("llm_call.delegated_to_llm_client", 
                          step_name=step_name,
                          provider=provider,
                          model=model)
                          
            return ExecutionResult(
                success=True,
                context={},
                output={
                    "message": "LLM call delegated to LLM client",
                    "provider": provider,
                    "model": model,
                    "temperature": temperature
                },
                step_name=step_name,
                step_type="llm_call"
            )
            
        except Exception as e:
            self.logger.exception("llm_call.failed", 
                               step_name=step_name,
                               provider=provider,
                               model=model,
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
        
        # Execute the loop
        iteration_results = []
        loop_items = list(collection)  # Convert to list to ensure consistent iteration
        
        for i, item in enumerate(loop_items):
            # Create a new context for this iteration
            iteration_context = copy.deepcopy(context)
            
            # Set the loop variable in the context
            iteration_context[loop_variable] = item
            
            # Add loop metadata
            if "loop_metadata" not in iteration_context:
                iteration_context["loop_metadata"] = {}
            iteration_context["loop_metadata"]["current_loop"] = step_name
            iteration_context["loop_metadata"]["current_index"] = i
            iteration_context["loop_metadata"]["total_items"] = len(loop_items)
            
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
            
            # Store the result
            iteration_results.append({
                "iteration": i,
                "success": iteration_result.success,
                "error": iteration_result.error,
                "data": iteration_result.data
            })
            
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
            
            # Check for failure
            if not iteration_result.success and step.get("stop_on_iteration_failure", True):
                self.logger.warning("loop.iteration_failed", 
                                 step_name=step_name,
                                 iteration=i,
                                 error=iteration_result.error)
                return ExecutionResult(
                    success=False,
                    context={},
                    output={"iterations": iteration_results},
                    error=f"Loop iteration {i} failed in step {step_name}: {iteration_result.error}",
                    step_name=step_name,
                    step_type="loop"
                )
            
            # Use the final context from the iteration as input to the next iteration
            if iteration_result.success and iteration_result.context:
                # But only apply updates to the parent context, don't replace whole context
                parent_keys_to_update = set(iteration_result.context.keys()) - {loop_variable, "loop_metadata"}
                for key in parent_keys_to_update:
                    context[key] = iteration_result.context[key]
        
        self.logger.info("loop.completed", 
                      step_name=step_name,
                      iterations_completed=len(iteration_results),
                      successful_iterations=sum(1 for r in iteration_results if r["success"]))
        
        return ExecutionResult(
            success=True,
            context={},
            output={"iterations": iteration_results},
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
        
        self.logger.info("branch.executing", 
                      step_name=step_name,
                      branch_count=len(branches))
        
        # Evaluate each branch condition
        for branch_idx, branch in enumerate(branches):
            condition = branch.get("condition", {})
            target = branch.get("target")
            
            if not condition:
                self.logger.warning("branch.missing_condition", 
                                 step_name=step_name,
                                 branch_idx=branch_idx)
                continue
                
            if target is None:
                self.logger.warning("branch.missing_target", 
                                 step_name=step_name,
                                 branch_idx=branch_idx)
                continue
            
            # Evaluate the condition
            condition_result = self._evaluate_condition(condition, context)
            
            if condition_result:
                self.logger.info("branch.condition_true", 
                              step_name=step_name,
                              branch_idx=branch_idx,
                              target=target)
                
                # Log routing event for lineage tracking
                if self.event_logger:
                    self.event_logger.log_event(
                        event_type=EventType.ROUTING_EVALUATION,
                        payload={
                            "step_name": step_name,
                            "branch_idx": branch_idx,
                            "condition": condition,
                            "result": True,
                            "target": target
                        }
                    )
                
                return ExecutionResult(
                    success=True,
                    context={},
                    output={"branch_idx": branch_idx, "target": target},
                    step_name=step_name,
                    step_type="branch"
                )
        
        # No branch condition was true
        self.logger.info("branch.no_condition_true", 
                      step_name=step_name)
        
        return ExecutionResult(
            success=True,
            context={},
            output={"message": "No branch condition was true"},
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
        
        Args:
            step: The set_value step definition
            context: The current execution context
            
        Returns:
            ExecutionResult with the set_value result
        """
        step_name = step.get("name", "unnamed_set_value")
        field = step.get("field")
        value = step.get("value")
        
        if not field:
            return ExecutionResult(
                success=False,
                error=f"Missing field in set_value step {step_name}",
                step_name=step_name,
                step_type="set_value"
            )
        
        self.logger.info("set_value.executing", 
                      step_name=step_name,
                      field=field)
        
        try:
            # Format the value if it's a template string
            if isinstance(value, str) and "{{" in value and "}}" in value:
                formatted_value = self._format_template(value, context)
            else:
                formatted_value = value
            
            # Create a new context with the updated value
            updated_context = self._set_context_value(context, field, formatted_value)
            
            return ExecutionResult(
                success=True,
                context=updated_context,
                output={"field": field, "value": formatted_value},
                step_name=step_name,
                step_type="set_value"
            )
            
        except Exception as e:
            self.logger.exception("set_value.failed", 
                               step_name=step_name,
                               field=field,
                               error=str(e))
            return ExecutionResult(
                success=False,
                error=f"Error setting value at '{field}': {str(e)}",
                step_name=step_name,
                step_type="set_value"
            )
    
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