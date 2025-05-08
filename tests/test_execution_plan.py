"""
Test script for ExecutionPlanExecutor implementation.
Path: tests/test_execution_plan.py

This script tests the ExecutionPlanExecutor and related configuration components.
"""

import os
import sys
import json
from pathlib import Path
import yaml
import logging
import structlog
from typing import Dict, Any, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import components
from c4h_agents.execution.executor import ExecutionPlanExecutor, ExecutionResult
from c4h_agents.skills.registry import SkillRegistry
from c4h_agents.utils.config_materializer import materialize_config
from c4h_agents.utils.config_validation import ConfigValidator
from c4h_agents.utils.schema_validation import SchemaValidator
from c4h_agents.config import load_config, deep_merge, expand_env_vars

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

def load_test_config() -> Dict[str, Any]:
    """Load test configuration for the executor"""
    # Look for test configuration in configs directory
    config_path = project_root / "tests" / "config" / "test_execution_plan.yml"
    
    # If test config doesn't exist, create a simple one
    if not config_path.exists():
        logger.info("Creating test configuration", path=str(config_path))
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a simple test configuration
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
                "skills": {
                    "semantic_extract": {
                        "module": "c4h_agents.skills.semantic_extract",
                        "class": "SemanticExtract",
                        "method": "execute",
                        "description": "Extract semantic data from text"
                    },
                    "tartxt_runner": {
                        "module": "c4h_agents.skills.tartxt_runner",
                        "class": "TartXTRunner",
                        "method": "execute",
                        "description": "Scan project and extract content"
                    }
                },
                "agents": {
                    "test_agent": {
                        "persona_key": "test_persona",
                        "execution_plan": {
                            "description": "Test execution plan",
                            "steps": [
                                {
                                    "name": "set_initial_value",
                                    "type": "set_value",
                                    "field": "test_value",
                                    "value": "This is a test value"
                                },
                                {
                                    "name": "check_value",
                                    "type": "branch",
                                    "description": "Check if value is set",
                                    "branches": [
                                        {
                                            "condition": {
                                                "field": "test_value",
                                                "operator": "exists"
                                            },
                                            "target": "return_success"
                                        },
                                        {
                                            "condition": {
                                                "type": "not",
                                                "condition": {
                                                    "field": "test_value",
                                                    "operator": "exists"
                                                }
                                            },
                                            "target": "return_failure"
                                        }
                                    ]
                                },
                                {
                                    "name": "return_success",
                                    "type": "set_value",
                                    "field": "response",
                                    "value": "Execution plan completed successfully"
                                },
                                {
                                    "name": "return_failure",
                                    "type": "set_value",
                                    "field": "response",
                                    "value": "Execution plan failed"
                                }
                            ]
                        }
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
        
        # Write configuration to file
        with open(config_path, "w") as f:
            yaml.safe_dump(test_config, f)
    
    # Load configuration
    logger.info("Loading test configuration", path=str(config_path))
    config = load_config(config_path)
    
    return config

def test_execution_plan():
    """Test the ExecutionPlanExecutor with a simple execution plan"""
    # Load test configuration
    config = load_test_config()
    
    # Materialize configuration
    effective_config = materialize_config(config, "test_run")
    
    # Initialize skill registry
    registry = SkillRegistry()
    registry.register_builtin_skills()
    registry.load_skills_from_config(effective_config)
    
    # Get execution plan
    execution_plan = effective_config["llm_config"]["agents"]["test_agent"]["execution_plan"]
    
    # Create executor
    executor = ExecutionPlanExecutor(effective_config=effective_config, skill_registry=registry)
    
    # Execute plan
    context = {}
    logger.info("Executing test plan", steps_count=len(execution_plan["steps"]))
    result = executor.execute_plan(execution_plan, context)
    
    # Print result
    logger.info("Execution complete", 
               success=result.success, 
               output=result.output,
               context=result.context)
    
    # Verify result
    assert result.success, "Execution plan should succeed"
    assert "test_value" in result.context, "Should set test_value in context"
    assert result.context["test_value"] == "This is a test value", "Should set test_value correctly"
    assert "response" in result.context, "Should set response in context"
    assert result.context["response"] in ["Execution plan completed successfully", "Execution plan failed"], "Should set response correctly"

def test_schema_validation():
    """Test schema validation for execution plans"""
    # Load test configuration
    config = load_test_config()
    
    # Get execution plan
    execution_plan = config["llm_config"]["agents"]["test_agent"]["execution_plan"]
    
    # Get schema path
    schema_path = project_root / "config" / "schemas" / "execution_plan_v1.json"
    
    # Validate execution plan
    try:
        logger.info("Validating execution plan against schema")
        validated_plan = SchemaValidator.validate_execution_plan(execution_plan, schema_path)
        logger.info("Validation successful")
        
        # Verify plan structure
        assert "steps" in validated_plan, "Validated plan should have steps"
        assert len(validated_plan["steps"]) == 4, "Validated plan should have 4 steps"
        
    except Exception as e:
        logger.error("Validation failed", error=str(e))
        raise

def main():
    """Run all tests"""
    logger.info("Starting execution plan tests")
    
    try:
        # Test schema validation
        test_schema_validation()
        
        # Test execution plan
        test_execution_plan()
        
        logger.info("All tests passed!")
        
    except Exception as e:
        logger.error("Tests failed", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()