"""
Test script for the Coder agent implementation.
Directly tests the Coder agent's ability to process code changes.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from c4h_agents.agents.coder import Coder
import yaml

def test_coder_agent():
    """Test the Coder agent with sample change blocks"""
    
    # Reset test projects
    os.system(f"{parent_dir}/tests/setup/setup_test_projects.sh")
    print("Test projects reset to initial state.")
    
    # Load configuration directly
    config_path = os.path.join(parent_dir, "config", "system_config.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add necessary configuration
    if 'llm_config' not in config:
        config['llm_config'] = {}
    if 'skills' not in config['llm_config']:
        config['llm_config']['skills'] = {}
    if 'agents' not in config['llm_config']:
        config['llm_config']['agents'] = {}
    if 'personas' not in config['llm_config']:
        config['llm_config']['personas'] = {}
    
    # Add skills configuration
    config['llm_config']['skills']['semantic_iterator'] = {
        'module': 'c4h_agents.skills.semantic_iterator',
        'class': 'SemanticIterator',
        'description': 'Extracts structured information from text in a consistent format',
        'method': 'execute'
    }
    
    config['llm_config']['skills']['asset_manager'] = {
        'module': 'c4h_agents.skills.asset_manager',
        'class': 'AssetManager',
        'description': 'Manages file creation, modification, and deletion with safety features',
        'method': 'execute'
    }
    
    # Add coder agent configuration
    config['llm_config']['agents']['coder'] = {
        'persona_key': 'coder_v1'
    }
    
    # Add coder persona if it doesn't exist
    if 'coder_v1' not in config['llm_config']['personas']:
        config['llm_config']['personas']['coder_v1'] = {
            'provider': 'anthropic',
            'model': 'claude-3-opus-20240229',
            'temperature': 0,
            'prompts': {
                'system': 'You are an expert code modification agent.'
            }
        }
    
    # Create Coder agent
    coder = Coder(full_effective_config=config)
    
    # Create sample solution design with change blocks
    solution_design = """
===CHANGE_BEGIN===
FILE: /Users/jim/src/apps/c4h_ai_dev/tests/test_projects/project1/sample.py
TYPE: modify
DESCRIPTION: Add logging capability to sample.py
DIFF:
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sample.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def greet(name):
    logger.info(f"Greeting {name}")
    print(f"Hello, {name}!")

def calculate_sum(numbers):
    logger.info(f"Calculating sum of {len(numbers)} numbers")
    result = sum(numbers)
    logger.info(f"Sum calculated: {result}")
    return result

if __name__ == "__main__":
    logger.info("Starting sample.py")
    greet("World")
    print(calculate_sum([1, 2, 3, 4, 5]))
    logger.info("Finished sample.py")
```
===CHANGE_END===

===CHANGE_BEGIN===
FILE: /Users/jim/src/apps/c4h_ai_dev/tests/test_projects/project2/utils.py
TYPE: modify
DESCRIPTION: Add logging capability to utils.py
DIFF:
```python
import logging

# Configure logging
logger = logging.getLogger(__name__)

def format_name(name):
    # Format name by stripping whitespace and converting to title case
    logger.info(f"Formatting name: {name}")
    formatted = name.strip().title()
    logger.debug(f"Name formatted: {formatted}")
    return formatted

def validate_age(age):
    # Validate age is an integer between 0 and 150
    logger.info(f"Validating age: {age}")
    if not isinstance(age, int):
        logger.error(f"Invalid age type: {type(age)}")
        raise TypeError("Age must be an integer")
    if age < 0 or age > 150:
        logger.error(f"Age out of range: {age}")
        raise ValueError("Age must be between 0 and 150")
    logger.debug(f"Age validated: {age}")
    return age
```
===CHANGE_END===

===CHANGE_BEGIN===
FILE: /Users/jim/src/apps/c4h_ai_dev/tests/test_projects/project2/main.py
TYPE: modify
DESCRIPTION: Add logging capability to main.py
DIFF:
```python
import logging
from utils import format_name, validate_age

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("application.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_user(user_data):
    # Process user data and return formatted string
    logger.info(f"Processing user data: {user_data}")
    name = format_name(user_data["name"])
    age = validate_age(user_data["age"])
    result = f"{name} is {age} years old"
    logger.info(f"User processed: {result}")
    return result

if __name__ == "__main__":
    logger.info("Application starting")
    test_data = {
        "name": "john doe",
        "age": 25
    }
    logger.info(f"Test data: {test_data}")
    result = process_user(test_data)
    print(result)
    logger.info("Application finished")
```
===CHANGE_END===
"""
    
    # Process the solution design
    response = coder.process({"response": solution_design})
    
    # Print results
    print(f"Success: {response.success}")
    if response.error:
        print(f"Error: {response.error}")
    
    metrics = response.data.get("metrics", {})
    print(f"Metrics: {metrics}")
    
    changes = response.data.get("changes", [])
    print(f"Total changes: {len(changes)}")
    for i, change in enumerate(changes):
        print(f"  Change {i + 1}: {change.get('path')} - {'Success' if change.get('success') else 'Failed'}")
        if not change.get('success') and change.get('error'):
            print(f"    Error: {change.get('error')}")
    
    # Verify the changes were applied
    sample_py_path = os.path.join(parent_dir, "tests", "test_projects", "project1", "sample.py")
    utils_py_path = os.path.join(parent_dir, "tests", "test_projects", "project2", "utils.py")
    main_py_path = os.path.join(parent_dir, "tests", "test_projects", "project2", "main.py")
    
    print("\nVerifying changes were applied:")
    
    if os.path.exists(sample_py_path):
        with open(sample_py_path, 'r') as f:
            content = f.read()
        print(f"  sample.py: {'Logging added' if 'import logging' in content else 'No logging found'}")
    
    if os.path.exists(utils_py_path):
        with open(utils_py_path, 'r') as f:
            content = f.read()
        print(f"  utils.py: {'Logging added' if 'import logging' in content else 'No logging found'}")
    
    if os.path.exists(main_py_path):
        with open(main_py_path, 'r') as f:
            content = f.read()
        print(f"  main.py: {'Logging added' if 'import logging' in content else 'No logging found'}")

if __name__ == "__main__":
    test_coder_agent()