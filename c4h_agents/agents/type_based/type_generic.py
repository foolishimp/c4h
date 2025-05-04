"""
Generic type-based agent implementations.

This module provides the implementation for the core type-based agents:
- GenericLLMAgent: Generic agent that processes input using an LLM
- GenericOrchestratorAgent: Orchestrates execution of skills according to a plan
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from c4h_agents.context.execution_context import ExecutionContext
from c4h_agents.context.template import TemplateResolver
from c4h_agents.agents.type_based.type_base_agent import BaseTypeAgent, AgentResponse

logger = logging.getLogger(__name__)


class GenericLLMAgent(BaseTypeAgent):
    """Generic agent that processes input using an LLM."""
    
    def process(self, **kwargs) -> AgentResponse:
        """
        Process input using LLM with persona configuration.
        
        Args:
            **kwargs: Input parameters, expected to include at least 'input'
            
        Returns:
            AgentResponse: Structured response from LLM processing
        """
        # Get required inputs
        input_content = kwargs.get('input', '')
        if not input_content:
            raise ValueError("Input content is required for LLM processing")
        
        # Prepare LLM configuration
        llm_config = self._prepare_llm_config()
        
        # Create prompt from template
        prompt = self._create_prompt(input_content)
        
        # Call LLM
        llm_response = self._call_llm(prompt, llm_config)
        
        # Process response
        processed_response = self._process_llm_response(llm_response)
        
        # Create structured agent response
        result = AgentResponse(
            content=processed_response.get('content', ''),
            metadata={
                'prompt': prompt,
                'llm_config': self._sanitize_config(llm_config),
                'model_used': llm_config.get('model', 'unknown'),
                'processing_metadata': processed_response.get('metadata', {})
            },
            context_updates=processed_response.get('context_updates', {})
        )
        
        return result
    
    def _prepare_llm_config(self) -> Dict[str, Any]:
        """Prepare LLM configuration from persona config."""
        # Start with default LLM settings from system config
        llm_config = self.context.get('system.llm_config', {})
        
        # Override with persona-specific LLM settings
        persona_llm_config = self.persona_config.get('llm_configuration', {})
        if 'prompt_parameters' in persona_llm_config:
            llm_config.update(persona_llm_config['prompt_parameters'])
        
        # Apply any runtime overrides
        runtime_overrides = self.context.get('runtime.llm_overrides', {})
        llm_config.update(runtime_overrides)
        
        return llm_config
    
    def _create_prompt(self, input_content: str) -> str:
        """Create a prompt from template and input content."""
        # Get system prompt from persona config
        system_prompt = self.persona_config.get('llm_configuration', {}).get('system_prompt', '')
        
        # Get user prompt template from persona config
        user_prompt_template = self.persona_config.get('llm_configuration', {}).get('user_prompt_template', '{input}')
        
        # Create a context with input for template resolution
        template_context = self.context.set('input', input_content)
        
        # Resolve template variables in user prompt
        user_prompt = TemplateResolver.resolve(user_prompt_template, template_context)
        
        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        return full_prompt
    
    def _call_llm(self, prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call LLM with prompt and configuration.
        
        This is a placeholder implementation with lineage tracking.
        In a real implementation, this would use an appropriate LLM client
        based on the provider.
        """
        provider = config.get('provider', 'unknown')
        model = config.get('model', 'unknown')
        
        logger.info(f"Calling LLM provider '{provider}' with model '{model}'")
        
        # Create messages object for lineage tracking
        from c4h_agents.agents.types import LLMMessages
        messages = LLMMessages(
            system=self.persona_config.get('llm_configuration', {}).get('system_prompt', ''),
            user=prompt
        )
        
        # Create context for lineage tracking
        lineage_context = {
            "workflow_run_id": self.run_id,
            "system": {"runid": self.run_id},
            "agent_execution_id": self.agent_id,
            "provider": provider,
            "model": model,
            "config": config
        }
        
        # Include configuration snapshot information in lineage context
        base_context = self.context.to_dict()
        if "config_snapshot_path" in base_context:
            lineage_context["config_snapshot_path"] = base_context["config_snapshot_path"]
        if "config_hash" in base_context:
            lineage_context["config_hash"] = base_context["config_hash"]
        
        # Include runtime config metadata if present
        if "runtime" in base_context and isinstance(base_context["runtime"], dict) and "config_metadata" in base_context["runtime"]:
            lineage_context["runtime"] = {"config_metadata": base_context["runtime"]["config_metadata"]}
        
        # This is where you would call the actual LLM
        # For now, just create a mock response
        mock_response = {
            'content': f"This is a mock response from {provider} {model}.",
            'usage': {
                'prompt_tokens': len(prompt.split()),
                'completion_tokens': 20,
                'total_tokens': len(prompt.split()) + 20
            }
        }
        
        # Track the interaction with lineage
        metrics = {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": 20,
            "total_tokens": len(prompt.split()) + 20
        }
        
        # Record lineage if enabled
        self._track_llm_interaction(
            context=lineage_context,
            messages=messages,
            response=mock_response,
            metrics=metrics
        )
        
        return mock_response
    
    def _process_llm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process the LLM response into a structured format."""
        # Extract content from response
        content = response.get('content', '')
        
        # Extract or compute metadata
        metadata = {
            'usage': response.get('usage', {}),
            'finish_reason': response.get('finish_reason', 'unknown')
        }
        
        # By default, no context updates
        context_updates = {}
        
        # Parse any special context update instructions in content
        # This is a simple implementation - in practice, you might use
        # a more sophisticated extraction mechanism
        if '{{CONTEXT_UPDATE:' in content:
            try:
                # Extract context updates
                start_marker = '{{CONTEXT_UPDATE:'
                end_marker = '}}'
                start_idx = content.find(start_marker)
                end_idx = content.find(end_marker, start_idx + len(start_marker))
                
                if start_idx >= 0 and end_idx >= 0:
                    # Extract and parse the JSON
                    updates_str = content[start_idx + len(start_marker):end_idx].strip()
                    context_updates = json.loads(updates_str)
                    
                    # Remove the marker from content
                    content = content[:start_idx] + content[end_idx + len(end_marker):]
            except Exception as e:
                logger.warning(f"Failed to parse context updates: {e}")
        
        return {
            'content': content.strip(),
            'metadata': metadata,
            'context_updates': context_updates
        }
    
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from config."""
        # Create a copy to avoid modifying the original
        sanitized = dict(config)
        
        # Remove any API keys or sensitive information
        sensitive_keys = ['api_key', 'secret', 'token', 'password']
        for key in sensitive_keys:
            if key in sanitized:
                sanitized[key] = '<redacted>'
        
        return sanitized


class GenericOrchestratorAgent(BaseTypeAgent):
    """Orchestrates execution of skills according to a predefined execution plan."""
    
    def process(self, **kwargs) -> AgentResponse:
        """
        Execute an orchestration plan.
        
        Args:
            **kwargs: Input parameters
                plan: Optional specific plan to execute
                
        Returns:
            AgentResponse: Structured response from orchestration
        """
        # Get execution plan from kwargs or persona config
        plan = kwargs.get('plan')
        if not plan:
            plan = self.persona_config.get('execution_plan')
        
        if not plan or not plan.get('enabled', False):
            raise ValueError("No valid execution plan provided")
        
        # Execute the plan
        snapshot_id = f"orchestration-{self.agent_id}"
        self.context.create_snapshot(snapshot_id)
        
        try:
            result = self._execute_plan(plan, self.context)
            
            # Create structured agent response
            return AgentResponse(
                content=result.get('output', ''),
                metadata={
                    'execution_steps': result.get('steps', []),
                    'execution_errors': result.get('errors', []),
                },
                context_updates=result.get('context_updates', {})
            )
        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            # Restore context snapshot
            self.context = self.context.restore_snapshot(snapshot_id)
            
            # Return error response
            return AgentResponse(
                content=f"Error executing orchestration plan: {str(e)}",
                metadata={
                    'error': str(e),
                    'error_type': type(e).__name__
                }
            )
    
    def _execute_plan(self, plan: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Execute a plan or subplan with given context."""
        steps = plan.get('steps', [])
        results = {
            'steps': [],
            'output': '',
            'errors': [],
            'context_updates': {}
        }
        
        for step in steps:
            step_name = step.get('name', 'unnamed')
            step_type = step.get('type')
            
            try:
                logger.info(f"Executing step '{step_name}' of type '{step_type}'")
                
                # Execute step based on its type
                if step_type == 'skill':
                    step_result = self._execute_skill_step(step, context)
                elif step_type == 'loop':
                    step_result = self._execute_loop_step(step, context)
                elif step_type == 'conditional':
                    step_result = self._execute_conditional_step(step, context)
                elif step_type == 'parallel':
                    step_result = self._execute_parallel_step(step, context)
                elif step_type == 'try_catch':
                    step_result = self._execute_try_catch_step(step, context)
                else:
                    raise ValueError(f"Unknown step type: {step_type}")
                
                # Apply context updates
                if 'context_updates' in step_result:
                    for path, value in step_result['context_updates'].items():
                        context = context.set(path, value)
                        results['context_updates'][path] = value
                
                # Record step result
                step_info = {
                    'name': step_name,
                    'type': step_type,
                    'status': 'success',
                    'output': step_result.get('output', '')
                }
                results['steps'].append(step_info)
                
                # If this step produces a final output, record it
                if step.get('is_output', False):
                    results['output'] = step_result.get('output', '')
            
            except Exception as e:
                logger.error(f"Error executing step '{step_name}': {e}")
                
                # Record error
                results['errors'].append({
                    'step': step_name,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                
                # Record step failure
                step_info = {
                    'name': step_name,
                    'type': step_type,
                    'status': 'error',
                    'error': str(e)
                }
                results['steps'].append(step_info)
        
        return results
    
    def _execute_skill_step(self, step: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Execute a skill step."""
        skill_key = step.get('skill')
        if not skill_key:
            raise ValueError("Skill step missing 'skill' key")
        
        # Prepare parameters from context
        parameters = step.get('parameters', {})
        resolved_params = {}
        
        for param_name, param_value in parameters.items():
            resolved_params[param_name] = TemplateResolver.resolve(param_value, context)
        
        # Invoke skill
        skill_result = self.skill_registry.invoke_skill(skill_key, **resolved_params)
        
        # Process outputs
        outputs = step.get('outputs', {})
        context_updates = {}
        
        for output_name, context_path in outputs.items():
            if output_name in skill_result.outputs:
                context_updates[context_path] = skill_result.outputs[output_name]
        
        return {
            'output': skill_result.result,
            'context_updates': context_updates
        }
    
    def _execute_loop_step(self, step: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Execute a loop step."""
        iterate_on = step.get('iterate_on')
        if not iterate_on:
            raise ValueError("Loop step missing 'iterate_on' key")
        
        # Get the collection to iterate on
        collection = context.get(iterate_on, [])
        if not isinstance(collection, (list, tuple)):
            raise ValueError(f"Cannot iterate on non-iterable: {iterate_on}")
        
        # Get variable name for current item
        variable_name = step.get('as_variable', 'item')
        
        # Get subplan steps
        substeps = step.get('steps', [])
        if not substeps:
            raise ValueError("Loop step has no substeps")
        
        # Prepare subplan
        subplan = {'steps': substeps}
        
        # Execute loop
        all_results = []
        context_updates = {}
        
        for i, item in enumerate(collection):
            # Create a new context for this iteration
            iter_context = context.set(variable_name, item)
            iter_context = iter_context.set('_loop_index', i)
            
            # Execute subplan
            iter_result = self._execute_plan(subplan, iter_context)
            
            # Collect results if requested
            if step.get('collect_results', False):
                all_results.append(iter_result.get('output', ''))
            
            # Apply any context updates from this iteration
            if 'context_updates' in iter_result:
                for path, value in iter_result['context_updates'].items():
                    # Avoid overwriting loop control variables
                    if path != variable_name and path != '_loop_index':
                        context = context.set(path, value)
                        context_updates[path] = value
        
        # Store collected results if requested
        if step.get('collect_results', False) and step.get('results_variable'):
            results_var = step.get('results_variable')
            context_updates[results_var] = all_results
        
        return {
            'output': '\n'.join(all_results) if all_results else '',
            'context_updates': context_updates
        }
    
    def _execute_conditional_step(self, step: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Execute a conditional step."""
        condition = step.get('condition')
        if not condition:
            raise ValueError("Conditional step missing 'condition' key")
        
        # Evaluate condition (this is a simplified implementation)
        # In a real implementation, you might use a proper expression evaluator
        condition_value = TemplateResolver.resolve(condition, context)
        
        # Determine which branch to execute
        if condition_value:
            if 'then' not in step:
                return {'output': '', 'context_updates': {}}
            
            # Execute 'then' branch
            subplan = {'steps': step.get('then', [])}
            return self._execute_plan(subplan, context)
        else:
            if 'else' not in step:
                return {'output': '', 'context_updates': {}}
            
            # Execute 'else' branch
            subplan = {'steps': step.get('else', [])}
            return self._execute_plan(subplan, context)
    
    def _execute_parallel_step(self, step: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """
        Execute steps in parallel.
        
        Note: This is a simplified implementation that executes steps sequentially.
        In a real implementation, you would use a proper parallel execution mechanism.
        """
        substeps = step.get('steps', [])
        if not substeps:
            raise ValueError("Parallel step has no substeps")
        
        # Prepare subplan for each step
        results = []
        all_context_updates = {}
        
        for substep in substeps:
            # Create a subplan with a single step
            subplan = {'steps': [substep]}
            
            # Execute subplan
            result = self._execute_plan(subplan, context)
            results.append(result.get('output', ''))
            
            # Collect context updates
            if 'context_updates' in result:
                all_context_updates.update(result['context_updates'])
        
        return {
            'output': '\n'.join(results),
            'context_updates': all_context_updates
        }
    
    def _execute_try_catch_step(self, step: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Execute a try-catch step."""
        try_steps = step.get('try', [])
        catch_steps = step.get('catch', [])
        finally_steps = step.get('finally', [])
        
        if not try_steps:
            raise ValueError("Try-catch step missing 'try' steps")
        
        # Create a snapshot for rollback
        snapshot_id = f"try-catch-{self.agent_id}"
        context.create_snapshot(snapshot_id)
        
        try:
            # Execute 'try' steps
            try_plan = {'steps': try_steps}
            result = self._execute_plan(try_plan, context)
            
            # Return result from try block
            return result
        except Exception as e:
            logger.warning(f"Exception in try block, executing catch: {e}")
            
            # Restore context from snapshot
            context = context.restore_snapshot(snapshot_id)
            
            # Add error information to context
            context = context.set('_error', {
                'message': str(e),
                'type': type(e).__name__
            })
            
            # If no catch block, re-raise
            if not catch_steps:
                raise
            
            # Execute 'catch' steps
            catch_plan = {'steps': catch_steps}
            return self._execute_plan(catch_plan, context)
        finally:
            # Execute 'finally' steps if provided
            if finally_steps:
                finally_plan = {'steps': finally_steps}
                self._execute_plan(finally_plan, context)