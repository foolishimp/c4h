"""
Test for the CommandLineRunner skill with git parameter support.

This test verifies that the CommandLineRunner can handle tartxt with git parameters.
"""

import os
import sys
import uuid
from pathlib import Path

# Add project root to PYTHONPATH if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from c4h_agents.skills.command_line_runner import CommandLineRunner
from c4h_agents.agents.types import SkillResult

def test_tartxt_git_params():
    """Test that tartxt can be executed with git parameters"""
    print("Testing tartxt with git parameters")
    
    # Since we might not have a git repo to test with,
    # this test just verifies the command construction without failing execution
    
    # Configure with git support
    config = {
        "command_configs": {
            "tartxt": {
                "type": "shell_command",
                "command": [sys.executable, "-m", "c4h_agents.skills.tartxt"],
                "description": "Project scanning and content extraction",
                "default_args": {
                    "exclude": "**/node_modules/**,**/dist/**,**/.venv/**,**/__pycache__/**,**/*.pyc"
                }
            }
        }
    }
    
    runner = CommandLineRunner(config)
    
    # Create output file path
    output_file = os.path.join(os.getcwd(), "workspaces", f"test_tartxt_git_{uuid.uuid4()}.txt")
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create arguments including git command
    args = {
        "file": output_file,
        "git": "diff HEAD~3 HEAD",
        "history": 1
    }
    
    # Execute command (using _resolve_command directly to avoid running the command)
    cmd, cmd_type, script_path = runner._resolve_command("tartxt", None)
    prepared_args = runner._prepare_command_args(cmd_type, args, {}, command_name="tartxt")
    
    print(f"Resolved command: {cmd}")
    print(f"Command type: {cmd_type}")
    print(f"Prepared args: {prepared_args}")
    
    # Verify that git and history parameters are in the prepared args
    assert "git" in prepared_args
    assert prepared_args["git"] == "diff HEAD~3 HEAD"
    assert "history" in prepared_args
    assert prepared_args["history"] == 1
    assert "exclude" in prepared_args  # Default arg should be included
    
    print("Test passed!")
    
if __name__ == "__main__":
    test_tartxt_git_params()