"""
Script to apply logging changes to test projects directly.
This bypasses the complex orchestration mechanism for testing.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from c4h_agents.skills.asset_manager import AssetManager

def apply_logging_changes():
    """Apply logging changes to test project files"""
    
    # Initialize asset manager
    config = {
        'project': {
            'path': str(Path(parent_dir) / 'tests' / 'test_projects'),
            'workspace_root': 'workspaces'
        }
    }
    asset_manager = AssetManager(config)
    
    # Update sample.py
    sample_py_path = str(Path(parent_dir) / 'tests' / 'test_projects' / 'project1' / 'sample.py')
    sample_py_content = """import logging

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
"""
    result = asset_manager.execute(action='write', path=sample_py_path, content=sample_py_content)
    print(f"Sample.py update: {result.success}")
    
    # Update utils.py
    utils_py_path = str(Path(parent_dir) / 'tests' / 'test_projects' / 'project2' / 'utils.py')
    utils_py_content = """import logging

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
"""
    result = asset_manager.execute(action='write', path=utils_py_path, content=utils_py_content)
    print(f"Utils.py update: {result.success}")
    
    # Update main.py
    main_py_path = str(Path(parent_dir) / 'tests' / 'test_projects' / 'project2' / 'main.py')
    main_py_content = """import logging
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
"""
    result = asset_manager.execute(action='write', path=main_py_path, content=main_py_content)
    print(f"Main.py update: {result.success}")

if __name__ == "__main__":
    apply_logging_changes()