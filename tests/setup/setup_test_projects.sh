#!/bin/bash
rm -rfd tests/test_projects
# Create project1 directory and files
mkdir -p tests/test_projects/project1
cat > tests/test_projects/project1/sample.py << 'EOF'
def greet(name):
    print(f"Hello, {name}!")

def calculate_sum(numbers):
    return sum(numbers)

if __name__ == "__main__":
    greet("World")
    print(calculate_sum([1, 2, 3, 4, 5]))
EOF

# Create project2 directory and files
mkdir -p tests/test_projects/project2
cat > tests/test_projects/project2/main.py << 'EOF'
from utils import format_name, validate_age

def process_user(user_data):
    """Process user data and return formatted string"""
    name = format_name(user_data["name"])
    age = validate_age(user_data["age"])
    return f"{name} is {age} years old"

if __name__ == "__main__":
    test_data = {
        "name": "john doe",
        "age": 25
    }
    print(process_user(test_data))
EOF

cat > tests/test_projects/project2/utils.py << 'EOF'
def format_name(name):
    """Format name by stripping whitespace and converting to title case"""
    return name.strip().title()

def validate_age(age):
    """Validate age is an integer between 0 and 150"""
    if not isinstance(age, int):
        raise TypeError("Age must be an integer")
    if age < 0 or age > 150:
        raise ValueError("Age must be between 0 and 150")
    return age
EOF