# Skill Tests

This directory contains tests for the skill components of the C4H Agent System.

## CommandLineRunner

`test_command_line_runner.py` contains tests for the new CommandLineRunner skill, which provides a generic interface for executing command-line tools and Python modules. It replaces the specialized TartXTRunner skill for a more flexible and maintainable approach.

### Migration from TartXTRunner

If you're using the TartXTRunner in your code, you should migrate to the CommandLineRunner:

```python
# Old approach with TartXTRunner
result = tartxt_runner.execute(
    project_path=project_dir,
    exclusions=["**/node_modules/**", "**/__pycache__/**"],
    output_file="workspaces/project_scan.txt",
)

# New approach with CommandLineRunner
result = command_line_runner.execute(
    command_name="tartxt",  # Use the predefined tartxt command
    command_args={
        "exclude": "**/node_modules/**,**/__pycache__/**",
        "file": "workspaces/project_scan.txt",
        "positional_args": [project_dir]
    }
)
```

The CommandLineRunner provides several advantages:
1. Can run any command-line tool, not just tartxt.py
2. Provides both direct Python module import and subprocess execution
3. Handles arguments correctly for different command types
4. Supports positional arguments and option flags
5. Can capture output to files or memory