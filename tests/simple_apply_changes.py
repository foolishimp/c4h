"""
Simple script to directly apply logging changes to test projects.
This bypasses the complex agent orchestration for testing.
"""

import os
import sys
from pathlib import Path

def apply_changes():
    """Apply logging changes directly"""
    
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_projects'))
    
    # Update sample.py
    sample_py_path = os.path.join(project_dir, 'project1', 'sample.py')
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
    with open(sample_py_path, 'w') as f:
        f.write(sample_py_content)
    print(f"Updated {sample_py_path}")
    
    # Update utils.py
    utils_py_path = os.path.join(project_dir, 'project2', 'utils.py')
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
    with open(utils_py_path, 'w') as f:
        f.write(utils_py_content)
    print(f"Updated {utils_py_path}")
    
    # Update main.py
    main_py_path = os.path.join(project_dir, 'project2', 'main.py')
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
    with open(main_py_path, 'w') as f:
        f.write(main_py_content)
    print(f"Updated {main_py_path}")

if __name__ == "__main__":
    apply_changes()