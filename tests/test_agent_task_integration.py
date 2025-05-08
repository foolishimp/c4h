"""
Test script for agent task integration with ExecutionPlanExecutor.
Path: tests/test_agent_task_integration.py

This script tests the updated agent task integration with the ExecutionPlanExecutor,
including the ability to detect and execute agents with embedded execution plans.
"""

import os
import sys
import json
from pathlib import Path
import yaml
import logging
import structlog
from typing import Dict, Any, List, Optional
import unittest
import pytest
from unittest import mock

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory()
)

logger = structlog.get_logger()

def create_test_config() -> Dict[str, Any]:
    """Create a test configuration with an agent that has an embedded execution plan"""
    test_config = {
        "llm_config": {
            "default_provider": "anthropic",
            "default_model": "claude-3-opus-20240229",
            "providers": {
                "anthropic": {
                    "default_model": "claude-3-opus-20240229",
                    "valid_models": [
                        "claude-3-opus-20240229",
                        "claude-3-sonnet-20240229"
                    ]
                }
            },
            "agents": {
                "test_agent": {
                    "agent_type": "GenericLLMAgent",
                    "name": "test_agent",
                    "persona_key": "test_persona",
                    "execution_plan": {
                        "description": "Test execution plan for agent",
                        "steps": [
                            {
                                "name": "set_initial_value",
                                "type": "set_value",
                                "field": "test_value",
                                "value": "This is a test value from agent execution plan"
                            },
                            {
                                "name": "set_response",
                                "type": "set_value",
                                "field": "response",
                                "value": "Agent execution plan completed successfully"
                            }
                        ]
                    }
                },
                "factory_agent": {
                    "agent_type": "GenericLLMAgent",
                    "name": "factory_agent",
                    "persona_key": "test_persona"
                }
            },
            "personas": {
                "test_persona": {
                    "provider": "anthropic",
                    "model": "claude-3-opus-20240229",
                    "temperature": 0.0,
                    "prompts": {
                        "system": "You are a helpful AI assistant.",
                        "user": "Hello, how can I help you today?"
                    }
                }
            }
        }
    }
    return test_config

def test_agent_with_execution_plan():
    """Test that the agent task can detect and execute an agent with an embedded execution plan"""
    from c4h_services.src.intent.impl.prefect.tasks import run_agent_task
    from c4h_agents.execution.executor import ExecutionPlanExecutor, ExecutionResult
    
    # Create test configuration
    config = create_test_config()
    
    # Create mock execution result
    mock_result = ExecutionResult(
        success=True,
        context={"test_value": "This is a test value from agent execution plan", 
                "response": "Agent execution plan completed successfully"},
        output={"message": "Execution plan test"},
        steps_executed=2
    )
    
    # Mock the ExecutionPlanExecutor
    mock_executor = mock.MagicMock()
    mock_executor.execution_id = "test_executor_id"
    mock_executor.execute_plan.return_value = mock_result
    
    # Mock the ExecutionPlanExecutor class
    with mock.patch('c4h_agents.execution.executor.ExecutionPlanExecutor', return_value=mock_executor):
        # Mock Prefect's get_run_logger
        with mock.patch('prefect.get_run_logger') as mock_logger:
            mock_logger.return_value = structlog.get_logger()
            
            # Mock flow_run for ID generation
            with mock.patch('prefect.runtime.flow_run.get_id') as mock_get_id:
                mock_get_id.return_value = "test_run_id"
                
                # Create task config for agent with execution plan
                task_config = {
                    "agent_type": "GenericLLMAgent",
                    "name": "test_agent",
                    "persona_key": "test_persona"
                }
                
                # Mock the SkillRegistry
                with mock.patch('c4h_agents.skills.registry.SkillRegistry') as mock_registry_class:
                    mock_registry = mock.MagicMock()
                    mock_registry_class.return_value = mock_registry
                    
                    # Mock the EventLogger
                    with mock.patch('c4h_agents.lineage.event_logger.EventLogger') as mock_event_logger_class:
                        # Execute agent task
                        with mock.patch('c4h_services.src.orchestration.factory.AgentFactory.create_agent') as mock_create_agent:
                            # We should never call create_agent when using execution plan
                            result = run_agent_task(
                                task_config=task_config,
                                context={},
                                effective_config=config
                            )
                            
                            # Verify that we didn't use the factory
                            assert not mock_create_agent.called, "Should not have used the factory when agent has execution plan"
                            
                            # Verify result structure
                            assert result is not None, "Result should not be None"
                            assert isinstance(result, dict), "Result should be a dictionary"
                            assert "success" in result, "Result should have success field"
                            assert result["success"] is True, "Result should be successful"
                            assert "execution_type" in result, "Result should have execution_type field"
                            assert result["execution_type"] == "execution_plan", "execution_type should be 'execution_plan'"
                            assert "context" in result, "Result should have context field"
                            assert "test_value" in result["context"], "Context should have test_value"
                            assert result["context"]["test_value"] == "This is a test value from agent execution plan", "test_value should be set correctly"
                            assert "response" in result["context"], "Context should have response"
                            assert result["context"]["response"] == "Agent execution plan completed successfully", "response should be set correctly"
                            assert "steps_executed" in result, "Result should have steps_executed field"
                            assert result["steps_executed"] == 2, "Should have executed 2 steps"
                            
                            logger.info("Agent with execution plan test passed successfully!")

def test_agent_with_factory():
    """Test that the agent task falls back to factory-based execution for agents without execution plans"""
    from c4h_services.src.intent.impl.prefect.tasks import run_agent_task
    from unittest.mock import MagicMock, patch
    from c4h_agents.agents.base_agent import BaseAgent
    
    # Create test configuration
    config = create_test_config()
    
    # Create mock agent process result
    class MockAgentResult:
        def __init__(self):
            self.success = True
            self.data = {"message": "Agent executed via factory"}
            self.error = None
            self.messages = MagicMock()
            self.messages.to_dict.return_value = {"role": "assistant", "content": "Hello"}
            self.raw_output = "Agent output"
            self.metrics = {"tokens": 100}
            
    # Create mock agent
    mock_agent = MagicMock(spec=BaseAgent)
    mock_agent.process.return_value = MockAgentResult()
    
    # Mock the AgentFactory and its create_agent method
    mock_factory = MagicMock()
    mock_factory.create_agent.return_value = mock_agent
    
    with patch('c4h_services.src.intent.impl.prefect.tasks.AgentFactory', return_value=mock_factory):
        # Mock Prefect's get_run_logger
        with patch('prefect.get_run_logger') as mock_logger:
            mock_logger.return_value = structlog.get_logger()
            
            # Mock flow_run for ID generation
            with patch('prefect.runtime.flow_run.get_id') as mock_get_id:
                mock_get_id.return_value = "test_run_id"
                
                # Create task config for factory-based agent
                task_config = {
                    "agent_type": "GenericLLMAgent",
                    "name": "factory_agent", 
                    "persona_key": "test_persona"
                }
                
                # Mock other required dependencies to simplify testing
                with patch('c4h_agents.config.create_config_node') as mock_create_config_node:
                    mock_config_node = MagicMock()
                    mock_config_node.get_value.return_value = {}
                    mock_create_config_node.return_value = mock_config_node
                    
                    # Execute agent task
                    result = run_agent_task(
                        task_config=task_config,
                        context={},
                        effective_config=config
                    )
                    
                    # Verify that we used the factory
                    assert mock_factory.create_agent.called, "Should have used the factory for agent without execution plan"
                    
                    # Verify result structure
                    assert result is not None, "Result should not be None"
                    assert isinstance(result, dict), "Result should be a dictionary"
                    assert "success" in result, "Result should have success field"
                    assert result["success"] is True, "Result should be successful"
                    assert "execution_type" in result, "Result should have execution_type field"
                    assert result["execution_type"] == "factory_based", "execution_type should be 'factory_based'"
                    assert "result_data" in result, "Result should have result_data field"
                    assert "message" in result["result_data"], "result_data should have message field"
                    assert result["result_data"]["message"] == "Agent executed via factory", "message should be set correctly"
                    
                    logger.info("Agent with factory test passed successfully!")

def main():
    """Run all tests"""
    logger.info("Starting agent task integration tests")
    
    try:
        # Test agent with execution plan
        test_agent_with_execution_plan()
        
        # Test agent with factory
        test_agent_with_factory()
        
        logger.info("All tests passed!")
        
    except Exception as e:
        logger.error("Tests failed", error=str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()