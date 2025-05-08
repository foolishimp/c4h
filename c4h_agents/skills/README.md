# C4H Agent Skills

This directory contains skills that can be used by the various agents in the C4H system.

## CommandLineRunner

The CommandLineRunner skill (`command_line_runner.py`) provides a generic interface for executing command-line tools and Python modules. It replaces the specialized TartXTRunner skill with a more flexible and maintainable approach.

### Usage

```python
from c4h_agents.skills.command_line_runner import CommandLineRunner

# Initialize with configuration
runner = CommandLineRunner({
    "command_configs": {
        "tartxt": {
            "type": "shell_command",
            "command": [sys.executable, "-m", "c4h_agents.skills.tartxt"],
            "description": "Project scanning and content extraction"
        }
    }
})

# Execute a named command
result = runner.execute(
    command_name="tartxt",
    command_args={
        "exclude": "**/node_modules/**,**/__pycache__/**",
        "file": "workspaces/project_scan.txt",
        "positional_args": [project_dir]
    }
)

# Execute a direct command line
result = runner.execute(
    command_line="python -m c4h_agents.skills.tartxt -f output.txt -x '*.pyc' /path/to/project"
)
```

### Migration from TartXTRunner

The TartXTRunner has been deprecated in favor of the CommandLineRunner. The tartxt_runner.py file is maintained for backward compatibility but will be removed in a future release.

If you're currently using TartXTRunner:

```python
# Old approach with TartXTRunner
result = tartxt_runner.execute(
    project_path=project_dir,
    exclusions=["**/node_modules/**", "**/__pycache__/**"],
    output_file="workspaces/project_scan.txt",
)
```

You should update your code to use CommandLineRunner:

```python
# New approach with CommandLineRunner
result = command_line_runner.execute(
    command_name="tartxt",
    command_args={
        "exclude": "**/node_modules/**,**/__pycache__/**",
        "file": "workspaces/project_scan.txt",
        "positional_args": [project_dir]
    }
)
```