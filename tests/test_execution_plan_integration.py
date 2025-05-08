#!/usr/bin/env python
"""
Test script for ExecutionPlanExecutor integration with agent and team tasks.
Tests the refactored code that requires execution plans for all components.
"""

import sys
import os
import json
import uuid
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4h_agents.execution.executor import ExecutionPlanExecutor
from c4h_agents.skills.registry import SkillRegistry
from c4h_agents.lineage.event_logger import EventLogger
from c4h_services.src.intent.impl.prefect.tasks import run_agent_task
from c4h_services.src.intent.impl.prefect.workflows import execute_team_subflow

def test_agent_with_execution_plan():
    """Test an agent with an execution plan using the ExecutionPlanExecutor directly."""
    print("\n--- Testing Agent with Execution Plan ---")
    
    # Create a sample execution plan for the agent
    execution_plan = {
        "enabled": True,
        "steps": [
            {
                "name": "set_response",
                "type": "set_value",
                "field": "response",
                "value": {
                    "message": "This is a test response from an execution plan"
                }
            }
        ]
    }
    
    # Create a run ID
    run_id = f"test_{uuid.uuid4()}"
    
    # Create a context with all required fields
    context = {
        "workflow_run_id": run_id,
        "test_data": "Sample test data",
        "system": {"runid": run_id}
    }
    
    # Create effective config with all required fields
    effective_config = {
        "llm_config": {
            "agents": {
                "lineage": {
                    "enabled": False
                }
            }
        },
        "runtime": {
            "workflow": {
                "id": run_id
            }
        }
    }
    
    try:
        print("Testing execution plan directly with ExecutionPlanExecutor...")
        # Initialize skill registry
        registry = SkillRegistry()
        registry.register_builtin_skills()
        
        # Create the ExecutionPlanExecutor
        executor = ExecutionPlanExecutor(
            effective_config=effective_config,
            skill_registry=registry
        )
        
        # Execute the plan
        result = executor.execute_plan(execution_plan, context)
        
        # Print the result
        print(f"Execution Result:")
        print(f"  Success: {result.success}")
        print(f"  Steps Executed: {result.steps_executed}")
        print(f"  Output: {result.output}")
        print(f"  Error: {result.error or 'None'}")
        
        # Assert expected results
        assert result.success == True, "Execution failed"
        assert result.steps_executed == 1, "Did not execute the expected number of steps"
        assert result.output and "message" in result.output, "Missing expected message in output"
        
        print("✅ Agent execution plan test passed")
        return True
    except Exception as e:
        print(f"❌ Agent execution plan test failed: {str(e)}")
        return False

def test_team_with_execution_plan():
    """Test a team with an execution plan using execute_team_subflow."""
    print("\n--- Testing Team with Execution Plan ---")
    
    # Create a sample execution plan for the team
    execution_plan = {
        "enabled": True,
        "steps": [
            {
                "name": "step1",
                "type": "set_value",
                "field": "processed_data",
                "value": "Processed by team execution plan"
            },
            {
                "name": "step2",
                "type": "set_value",
                "field": "response",
                "value": {
                    "message": "Team execution complete",
                    "timestamp": datetime.now().isoformat()
                }
            }
        ]
    }
    
    # Create the team config in effective config with necessary additional config
    effective_config = {
        "llm_config": {
            "agents": {
                "lineage": {
                    "enabled": False
                }
            }
        },
        "orchestration": {
            "teams": {
                "test_team": {
                    "name": "Test Team",
                    "execution_plan": execution_plan
                }
            }
        },
        # Adding runtime namespace to avoid errors
        "runtime": {
            "workflow": {
                "id": f"test_{uuid.uuid4()}"
            }
        }
    }
    
    # Create context with all required fields
    run_id = f"test_{uuid.uuid4()}"
    context = {
        "workflow_run_id": run_id,
        "input_data": "Test input data",
        "system": {"runid": run_id}  # System namespace required
    }
    
    try:
        # Execute the team
        result = execute_team_subflow(
            team_id="test_team",
            effective_config=effective_config,
            current_context=context
        )
        
        # Print the result
        print(f"Team Execution Result:")
        print(f"  Success: {result.get('success', False)}")
        print(f"  Execution Type: {result.get('execution_type', 'unknown')}")
        print(f"  Output: {result.get('output', {})}")
        print(f"  Steps Executed: {result.get('steps_executed', 0)}")
        print(f"  Error: {result.get('error', 'none')}")
        
        # Assert expected results
        assert result.get("success") == True, "Team execution failed"
        assert result.get("execution_type") == "execution_plan", "Did not use execution plan"
        assert result.get("steps_executed") == 2, "Did not execute expected number of steps"
        assert "message" in result.get("output", {}), "Missing expected message in output"
        
        # Check context updates
        context_updates = result.get("context", {})
        assert "processed_data" in context_updates, "Missing processed_data in context updates"
        
        print("✅ Team execution test passed")
        return True
    except Exception as e:
        print(f"❌ Team execution test failed: {str(e)}")
        return False

def test_agent_missing_execution_plan():
    """Test that an agent without an execution plan fails as expected."""
    print("\n--- Testing Agent Without Execution Plan ---")
    
    # Create a run ID
    run_id = f"test_{uuid.uuid4()}"
    
    # Create a context with all required fields
    context = {
        "workflow_run_id": run_id,
        "test_data": "Sample test data",
        "system": {"runid": run_id}
    }
    
    # Create effective config with all required fields
    effective_config = {
        "llm_config": {
            "agents": {
                "lineage": {
                    "enabled": False
                }
            }
        },
        "runtime": {
            "workflow": {
                "id": run_id
            }
        }
    }
    
    try:
        print("Testing ExecutionPlanExecutor with missing execution plan...")
        # Initialize skill registry
        registry = SkillRegistry()
        registry.register_builtin_skills()
        
        # Create the ExecutionPlanExecutor
        executor = ExecutionPlanExecutor(
            effective_config=effective_config,
            skill_registry=registry
        )
        
        # Create an empty execution plan (no steps)
        missing_execution_plan = None
        
        # Execute with missing plan - should return a failed result
        # Try to execute with None as the execution plan
        result = executor.execute_plan(missing_execution_plan, context)
        
        # Should return a result object with success=False
        print(f"Execution with missing plan result:")
        print(f"  Success: {result.success}")
        print(f"  Error: {result.error}")
        
        # Verify the result shows a failure
        assert not result.success, "Execution with None plan should have failed"
        assert "plan" in result.error.lower(), "Error should mention plan"
        print(f"✅ Execution correctly failed with error: {result.error}")
            
        # Try with an empty dict
        # Try to execute with empty dict as the execution plan
        empty_plan = {}
        result = executor.execute_plan(empty_plan, context)
        
        # Should fail with appropriate error
        print(f"Execution with empty plan result:")
        print(f"  Success: {result.success}")
        print(f"  Error: {result.error}")
        
        # Verify the result shows a failure
        assert not result.success, "Execution with empty plan should have failed"
        assert "plan" in result.error.lower(), "Error should mention plan"
        print(f"✅ Execution with empty plan correctly failed with error: {result.error}")
        
        print("✅ Agent missing execution plan test passed - correctly failed")
        return True
    except Exception as e:
        print(f"❌ Agent missing execution plan test failed: {str(e)}")
        return False

def test_team_missing_execution_plan():
    """Test that a team without an execution plan fails as expected."""
    print("\n--- Testing Team Without Execution Plan ---")
    
    # Create the team config in effective config without execution plan
    effective_config = {
        "llm_config": {
            "agents": {
                "lineage": {
                    "enabled": False
                }
            }
        },
        "orchestration": {
            "teams": {
                "test_team_no_plan": {
                    "name": "Test Team No Plan",
                    "tasks": [
                        {
                            "name": "task1",
                            "agent_type": "GenericLLMAgent"
                        }
                    ]
                }
            }
        }
    }
    
    # Create context
    context = {
        "workflow_run_id": f"test_{uuid.uuid4()}",
        "input_data": "Test input data"
    }
    
    try:
        # Execute the team - should fail with missing execution plan
        result = execute_team_subflow(
            team_id="test_team_no_plan",
            effective_config=effective_config,
            current_context=context
        )
        
        # Print the result
        print(f"Team Execution Result:")
        print(f"  Success: {result.get('success')}")
        print(f"  Error: {result.get('error')}")
        
        # Assert expected results - should have failed
        assert result.get("success") == False, "Execution unexpectedly succeeded"
        assert "does not have an execution_plan" in result.get("error", ""), "Missing expected error message"
        
        print("✅ Team missing execution plan test passed - correctly failed")
        return True
    except Exception as e:
        print(f"❌ Team missing execution plan test failed: {str(e)}")
        return False

def run_tests():
    """Run all tests and return overall success."""
    tests = [
        test_agent_with_execution_plan,
        test_team_with_execution_plan,
        test_agent_missing_execution_plan,
        test_team_missing_execution_plan
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n--- Test Summary ---")
    print(f"Tests Run: {len(results)}")
    print(f"Tests Passed: {results.count(True)}")
    print(f"Tests Failed: {results.count(False)}")
    
    return all(results)

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)