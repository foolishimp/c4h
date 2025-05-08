"""
Validation script for agent task integration with ExecutionPlanExecutor.

This script checks the code changes to verify that the proper execution plan 
integration has been added to the agent tasks.
"""

import sys
import re
from pathlib import Path
import logging
import structlog

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

def check_agent_task_code():
    """Check that the agent task code includes the necessary execution plan integration"""
    # Path to the agent task implementation
    task_path = project_root / "c4h_services" / "src" / "intent" / "impl" / "prefect" / "tasks.py"
    
    if not task_path.exists():
        logger.error("Agent task file not found", path=str(task_path))
        return False
    
    # Read the file content
    with open(task_path, "r") as f:
        content = f.read()
    
    # Check for key code patterns to validate the implementation
    checks = [
        {
            "name": "Check for merged config",
            "pattern": r'merged_config\.update\(task_config\)',
            "required": True
        },
        {
            "name": "ExecutionPlanExecutor import",
            "pattern": r'from c4h_agents\.execution\.executor import ExecutionPlanExecutor',
            "required": True
        },
        {
            "name": "Skill registry use",
            "pattern": r'SkillRegistry\(\)',
            "required": True
        },
        {
            "name": "Executor instantiation",
            "pattern": r'executor = ExecutionPlanExecutor\(',
            "required": True
        },
        {
            "name": "Plan execution call",
            "pattern": r'execution_result = executor\.execute_plan\(',
            "required": True
        },
        {
            "name": "Execution type field",
            "pattern": r'"execution_type": "execution_plan"',
            "required": True
        },
        {
            "name": "Factory-based execution type",
            "pattern": r'"execution_type": "factory_based"',
            "required": True
        },
        {
            "name": "Event logger integration",
            "pattern": r'event_logger = EventLogger\(',
            "required": True
        },
        {
            "name": "Duration tracking",
            "pattern": r'execution_start_time = datetime\.now\(timezone\.utc\)',
            "required": True
        },
        {
            "name": "Duration calculation",
            "pattern": r'duration_seconds = \(execution_end_time - execution_start_time\)\.total_seconds\(\)',
            "required": True
        }
    ]
    
    all_passed = True
    for check in checks:
        match = re.search(check["pattern"], content, re.DOTALL)
        if match and check["required"]:
            logger.info(f"✅ Found required pattern: {check['name']}")
        elif not match and check["required"]:
            logger.error(f"❌ Missing required pattern: {check['name']}")
            all_passed = False
        elif match and not check["required"]:
            logger.warning(f"⚠️ Found prohibited pattern: {check['name']}")
            all_passed = False
        else:
            logger.info(f"✅ Correctly did not find prohibited pattern: {check['name']}")
    
    return all_passed

def check_prefect_workflows_code():
    """Check that the Prefect workflows code includes necessary ExecutionPlanExecutor integration"""
    # Path to the Prefect workflows implementation
    workflows_path = project_root / "c4h_services" / "src" / "intent" / "impl" / "prefect" / "workflows.py"
    
    if not workflows_path.exists():
        logger.error("Workflows file not found", path=str(workflows_path))
        return False
    
    # Read the file content
    with open(workflows_path, "r") as f:
        content = f.read()
    
    # Check for key code patterns to validate the implementation
    checks = [
        {
            "name": "ExecutionPlanExecutor import in workflows",
            "pattern": r'from c4h_agents\.execution\.executor import ExecutionPlanExecutor',
            "required": True
        },
        {
            "name": "Check for execution plan in team",
            "pattern": r'if "execution_plan" in team_config:',
            "required": True
        },
        {
            "name": "ExecutionPlanExecutor initialization in workflows",
            "pattern": r'executor = ExecutionPlanExecutor\(',
            "required": True
        },
        {
            "name": "Execution plan execution in workflows",
            "pattern": r'execution_result = executor\.execute_plan\(',
            "required": True
        },
        {
            "name": "Convert execution result to team result format",
            "pattern": r'# Convert ExecutionResult to team result format',
            "required": True
        }
    ]
    
    all_passed = True
    for check in checks:
        match = re.search(check["pattern"], content, re.DOTALL)
        if match and check["required"]:
            logger.info(f"✅ Found required pattern: {check['name']}")
        elif not match and check["required"]:
            logger.error(f"❌ Missing required pattern: {check['name']}")
            all_passed = False
        elif match and not check["required"]:
            logger.warning(f"⚠️ Found prohibited pattern: {check['name']}")
            all_passed = False
        else:
            logger.info(f"✅ Correctly did not find prohibited pattern: {check['name']}")
    
    return all_passed

def main():
    """Run all code validation checks"""
    logger.info("Starting agent task code validation")
    
    try:
        # Check agent task code
        agent_task_valid = check_agent_task_code()
        
        # Check Prefect workflows code
        workflows_valid = check_prefect_workflows_code()
        
        # Check if all validations passed
        if agent_task_valid and workflows_valid:
            logger.info("All code validations passed! The implementation looks correct.")
            print("\n✅ CODE VALIDATION PASSED: Agent task integration with ExecutionPlanExecutor looks good!")
            return 0
        else:
            logger.error("Some code validations failed. Please check the implementation.")
            print("\n❌ CODE VALIDATION FAILED: Some required patterns were missing.")
            return 1
        
    except Exception as e:
        logger.error("Code validation failed unexpectedly", error=str(e), exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())