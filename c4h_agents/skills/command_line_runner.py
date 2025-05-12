"""
Command Line Runner Skill for C4H Agent System.

This skill provides a generic interface for running command line tools and scripts,
including the tartxt.py project scanner. It abstracts the details of command execution
and output handling while supporting both Python modules and standalone executables.
"""

import os
import sys
import subprocess
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import structlog
import json
import importlib
import inspect

from c4h_agents.agents.types import SkillResult

# Configure logger
logger = structlog.get_logger()

class CommandLineRunner:
    """
    Skill for running command line tools and scripts.
    
    This skill provides a generic interface for executing command line tools,
    supporting both direct execution and module imports for Python scripts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the skill with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.skill_id = str(uuid.uuid4())[:8]
        
        # Configure logger with skill-specific context
        self.logger = logger.bind(
            skill="command_line_runner",
            skill_id=self.skill_id
        )
        
        # Get command configuration from skill config
        self.command_configs = self.config.get("command_configs", {})
        
        self.logger.info("command_line_runner.initialized", 
                      commands=list(self.command_configs.keys()) if self.command_configs else [])
    
    def execute(
        self, 
        input: Dict[str, Any] = None,
        command_name: Optional[str] = None,
        command_args: Optional[Dict[str, Any]] = None,
        command_line: Optional[str] = None,
        working_dir: Optional[str] = None,
        output_to_file: bool = False,
        **kwargs
    ) -> SkillResult:
        """
        Execute a command line tool or script.
        
        This skill supports two execution modes:
        1. Named command: Uses a pre-configured command from the skill config
        2. Direct command: Executes the specified command_line directly
        
        Following the Total Functions principle (1.4), this method handles all expected error
        conditions internally and always returns a structured SkillResult, even in error cases.
        
        Args:
            input: Input context
            command_name: Name of pre-configured command to execute
            command_args: Arguments to pass to the command
            command_line: Direct command line to execute (alternative to command_name)
            working_dir: Directory to execute command in
            output_to_file: Whether to write command output to a file
            
        Returns:
            SkillResult with command output and metadata (never raises exceptions for expected failures)
        """
        # Normalize input to prevent NoneType errors
        input = input or {}
        command_args = command_args or {}
        
        # 1. Verify command specification
        if not command_name and not command_line:
            self.logger.error("command_line_runner.no_command_specified")
            return SkillResult(
                success=False,
                error="No command specified. Provide either command_name or command_line.",
                value={
                    "provided_args": {
                        "command_name": command_name,
                        "command_line": command_line
                    }
                }
            )
            
        # 2. Check if named command exists in configuration
        if command_name and command_name not in self.command_configs:
            self.logger.error("command_line_runner.command_not_found", command_name=command_name)
            available_commands = list(self.command_configs.keys())
            return SkillResult(
                success=False,
                error=f"Command '{command_name}' not found in configuration. Available commands: {available_commands}",
                value={
                    "available_commands": available_commands
                }
            )
            
        try:
            # 3. Resolve command to execute
            cmd, cmd_type, script_path = self._resolve_command(command_name, command_line)
            if not cmd:
                # This shouldn't happen given the earlier checks, but handling as a precaution
                return SkillResult(
                    success=False,
                    error="Failed to resolve command. Command resolution returned empty command.",
                    value={
                        "command_name": command_name,
                        "command_line": command_line
                    }
                )
            
            # 4. Resolve working directory with error handling
            try:
                working_dir = self._resolve_working_dir(working_dir, input)
                # Verify working directory exists
                if not os.path.isdir(working_dir):
                    self.logger.error("command_line_runner.invalid_working_dir", 
                                   working_dir=working_dir)
                    return SkillResult(
                        success=False,
                        error=f"Working directory does not exist: {working_dir}",
                        value={
                            "command": cmd,
                            "working_dir": working_dir
                        }
                    )
            except Exception as e:
                self.logger.error("command_line_runner.working_dir_error", 
                               error=str(e),
                               working_dir=working_dir)
                return SkillResult(
                    success=False,
                    error=f"Error resolving working directory: {str(e)}",
                    value={
                        "command": cmd,
                        "working_dir_input": working_dir
                    }
                )
            
            # 5. Prepare command arguments, including command_name for defaults
            try:
                cmd_args = self._prepare_command_args(
                    cmd_type, 
                    command_args, 
                    input,
                    command_name=command_name
                )
            except Exception as e:
                self.logger.error("command_line_runner.args_preparation_error", 
                               error=str(e),
                               command_args=command_args)
                return SkillResult(
                    success=False,
                    error=f"Error preparing command arguments: {str(e)}",
                    value={
                        "command": cmd,
                        "command_args": command_args
                    }
                )
            
            # 6. Generate output file path if needed
            output_file = None
            if output_to_file:
                try:
                    run_id = input.get("workflow_run_id") or input.get("execution_id") or str(uuid.uuid4())
                    output_dir = os.path.join(os.getcwd(), "workspaces")
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, f"cmd_output_{run_id}.txt")
                    self.logger.debug("command_line_runner.output_file_created", 
                                   output_file=output_file)
                except Exception as e:
                    self.logger.error("command_line_runner.output_file_error", 
                                   error=str(e))
                    # Continue without output file if it fails
                    output_to_file = False
                    output_file = None
            
            # 7. Execute the command based on its type
            if cmd_type == "python_module":
                # Validate script path
                if not script_path:
                    return SkillResult(
                        success=False,
                        error="Missing script path for Python module execution",
                        value={
                            "command": cmd,
                            "cmd_type": cmd_type
                        }
                    )
                result = self._execute_python_module(script_path, cmd_args, working_dir, output_file)
            else:
                result = self._execute_shell_command(cmd, cmd_args, working_dir, output_file)
            
            return result
            
        except Exception as e:
            # Final fallback for any unhandled exceptions to ensure we never raise
            self.logger.exception("command_line_runner.unhandled_error", 
                              error=str(e),
                              error_type=type(e).__name__)
            return SkillResult(
                success=False,
                error=f"Unhandled error executing command: {str(e)}",
                value={
                    "command_name": command_name,
                    "command_line": command_line,
                    "error_type": type(e).__name__
                }
            )
    
    def _resolve_command(
        self, 
        command_name: Optional[str], 
        command_line: Optional[str]
    ) -> tuple[List[str], str, Optional[str]]:
        """
        Resolve the command to execute based on command_name or command_line.
        
        This method identifies the command to execute and determines whether it's
        a shell command or Python module, extracting the appropriate parameters for each.
        
        Args:
            command_name: Name of pre-configured command
            command_line: Direct command line to execute
            
        Returns:
            Tuple of (command, command_type, script_path) where:
            - command is the resolved command as a list of strings
            - command_type is one of "shell_command" or "python_module"
            - script_path is the path to the Python script (for python_module type only)
            
        Note: 
            If command resolution fails, the caller should handle the empty command list
            by returning an appropriate SkillResult, not by raising an exception.
        """
        # Case 1: Resolve from named command in configuration
        if command_name:
            # Look up command in configuration (validation was done in the execute method)
            if command_name not in self.command_configs:
                self.logger.error("command_line_runner.command_not_found", command_name=command_name)
                return [], "", None
                
            cmd_config = self.command_configs[command_name]
            cmd_type = cmd_config.get("type", "shell_command")
            
            # Handle Python module type command
            if cmd_type == "python_module":
                script_path = cmd_config.get("module_path")
                if not script_path:
                    self.logger.error("command_line_runner.missing_module_path", 
                                   command_name=command_name)
                    # Return empty command to signal failure
                    return [], "python_module", None
                    
                # For Python modules, we'll execute through Python interpreter
                # Check if it's a module path (contains dots) or a file path
                if "." in script_path and not script_path.endswith(".py"):
                    # It's a module path like "a.b.c", use -m flag
                    cmd = [sys.executable, "-m", script_path]
                else:
                    # It's a file path, execute directly
                    cmd = [sys.executable, script_path]
                    
                self.logger.debug("command_line_runner.using_python_module", 
                               module=script_path,
                               command=cmd)
                return cmd, cmd_type, script_path
            
            # Handle shell command type
            else:
                # Get the command template
                cmd_template = cmd_config.get("command", "")
                if not cmd_template:
                    self.logger.error("command_line_runner.missing_command_template", 
                                   command_name=command_name)
                    # Return empty command to signal failure
                    return [], "shell_command", None
                    
                # Convert command template to list if it's a string
                if isinstance(cmd_template, str):
                    cmd = cmd_template.split()
                else:
                    cmd = list(cmd_template)
                    
                self.logger.debug("command_line_runner.using_shell_command", 
                               command=cmd,
                               command_name=command_name)
                return cmd, "shell_command", None
        
        # Case 2: Resolve from direct command line        
        elif command_line:
            try:
                # Convert command line to command list
                if isinstance(command_line, str):
                    cmd = command_line.split()
                else:
                    cmd = list(command_line)
                
                # Check if command list is valid
                if not cmd or not any(cmd):
                    self.logger.error("command_line_runner.empty_command_line")
                    return [], "shell_command", None
                    
                # Check if it's a Python script execution
                if len(cmd) > 0:
                    # Case: python script.py
                    if cmd[0] == sys.executable and len(cmd) > 1 and cmd[1].endswith(".py"):
                        script_path = cmd[1]
                        cmd_type = "python_module"
                    # Case: script.py (convert to python script.py)
                    elif cmd[0].endswith(".py"):
                        script_path = cmd[0]
                        cmd = [sys.executable, script_path] + cmd[1:]
                        cmd_type = "python_module"
                    # Case: Other executable
                    else:
                        script_path = None
                        cmd_type = "shell_command"
                else:
                    script_path = None
                    cmd_type = "shell_command"
                    
                self.logger.debug("command_line_runner.using_direct_command", 
                               command=cmd,
                               command_type=cmd_type)
                return cmd, cmd_type, script_path
            except Exception as e:
                # Handle any errors parsing the command line
                self.logger.error("command_line_runner.command_line_parse_error",
                              error=str(e),
                              command_line=command_line)
                return [], "shell_command", None
            
        # Case 3: No command specified (should be caught by execute method)
        else:
            self.logger.error("command_line_runner.no_command_specified")
            return [], "", None
    
    def _resolve_working_dir(self, working_dir: Optional[str], input: Dict[str, Any]) -> str:
        """
        Resolve the working directory for command execution.
        
        Args:
            working_dir: Explicit working directory
            input: Input context
            
        Returns:
            Resolved working directory path
        """
        if working_dir:
            dir_path = Path(working_dir).resolve()
            self.logger.debug("command_line_runner.using_explicit_working_dir", 
                           working_dir=str(dir_path))
            return str(dir_path)
            
        # Check for project_path in input
        if "project_path" in input:
            dir_path = Path(input["project_path"]).resolve()
            self.logger.debug("command_line_runner.using_project_path", 
                           working_dir=str(dir_path))
            return str(dir_path)
            
        # Default to current directory
        dir_path = os.getcwd()
        self.logger.debug("command_line_runner.using_current_dir", 
                       working_dir=str(dir_path))
        return dir_path
    
    def _prepare_command_args(
        self, 
        cmd_type: str, 
        cmd_args: Dict[str, Any], 
        input: Dict[str, Any],
        command_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare command arguments from input and explicitly provided args.
        
        Args:
            cmd_type: Type of command ("shell_command" or "python_module")
            cmd_args: Explicitly provided command arguments
            input: Input context
            command_name: Optional name of pre-configured command
            
        Returns:
            Prepared command arguments
        """
        # First, check if we have default arguments from command configuration
        default_args = {}
        if command_name and command_name in self.command_configs:
            cmd_config = self.command_configs[command_name]
            default_args = cmd_config.get("default_args", {})
            if default_args:
                self.logger.debug("command_line_runner.using_default_args",
                               command_name=command_name,
                               default_arg_keys=list(default_args.keys()))
        
        # Start with default args, then override with explicitly provided args
        args = {}
        if default_args:
            args.update(default_args)
        args.update(cmd_args)  # Override defaults with explicitly provided args
        
        # Add common args from input
        common_args = ["project_path", "input_paths", "exclusions", "output_file"]
        for arg in common_args:
            if arg in input and arg not in args:
                args[arg] = input[arg]
                
        # Special case for tartxt-style commands
        if "project_scan" in input and "project_scan" not in args:
            args["project_scan"] = input["project_scan"]
        
        # Handle git command for tartxt
        if command_name == "tartxt" and "git_command" in input and "git" not in args:
            args["git"] = input["git_command"]
            
        # Handle history for tartxt
        if command_name == "tartxt" and "history" in input and "history" not in args:
            args["history"] = input["history"]
            
        # For Python modules, we may need to convert args to the right format
        if cmd_type == "python_module":
            # No special handling needed yet, but could be added here
            pass
            
        self.logger.debug("command_line_runner.prepared_args", 
                       arg_keys=list(args.keys()))
        return args
    
    def _execute_python_module(
        self, 
        script_path: str, 
        args: Dict[str, Any], 
        working_dir: str, 
        output_file: Optional[str]
    ) -> SkillResult:
        """
        Execute a Python module or script with the provided arguments.
        
        This method tries two approaches:
        1. Import and call the module directly (if possible)
        2. Fall back to subprocess execution if direct import doesn't work
        
        Args:
            script_path: Path to Python script or module
            args: Arguments to pass to the script
            working_dir: Directory to execute in
            output_file: Path to write output to
            
        Returns:
            SkillResult with execution result
        """
        # Try to import the module first
        try:
            # Handle both module paths (a.b.c) and file paths (path/to/script.py)
            if os.path.exists(script_path):
                # It's a file path, try to import it
                module_name = os.path.splitext(os.path.basename(script_path))[0]
                spec = importlib.util.spec_from_file_location(module_name, script_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.logger.info("command_line_runner.imported_module", 
                                  script_path=script_path,
                                  module_name=module_name)
                else:
                    self.logger.warning("command_line_runner.import_failed", 
                                     script_path=script_path)
                    # Fall back to subprocess execution
                    return self._execute_shell_command(
                        [sys.executable, script_path], 
                        args, 
                        working_dir, 
                        output_file
                    )
            else:
                # It's a module path, try to import it
                module = importlib.import_module(script_path)
                self.logger.info("command_line_runner.imported_module", 
                              module_name=script_path)
            
            # Check if the module has a main function
            if hasattr(module, "main"):
                # Set up output capture if needed
                old_stdout = sys.stdout
                output_capture = None
                
                try:
                    if output_file:
                        output_dir = os.path.dirname(output_file)
                        os.makedirs(output_dir, exist_ok=True)
                        sys.stdout = open(output_file, "w")
                    else:
                        output_capture = tempfile.NamedTemporaryFile(mode="w+", delete=False)
                        sys.stdout = output_capture
                    
                    # We'll try direct module execution only if we have a simple case
                    # with no arguments or just a dictionary - otherwise, fall back to subprocess
                    if len(args) == 0 or (len(args) == 1 and 'input' in args):
                        # Call the main function with args
                        main_func = getattr(module, "main")
                        
                        # Call main with minimal args
                        self.logger.debug("command_line_runner.calling_main_minimal", 
                                       arg_keys=list(args.keys()))
                        result = main_func()
                        
                        # Get output
                        if output_file:
                            output_path = output_file
                        else:
                            output_capture.flush()
                            output_capture.seek(0)
                            output = output_capture.read()
                            output_path = output_capture.name
                        
                        # Restore stdout
                        sys.stdout.close()
                        sys.stdout = old_stdout
                        
                        # Process result
                        self.logger.info("command_line_runner.module_execution_complete", 
                                      script_path=script_path,
                                      result_type=type(result).__name__ if result is not None else "None")
                        
                        if output_file:
                            # Read the output file
                            with open(output_file, "r") as f:
                                output = f.read()
                        
                        return SkillResult(
                            success=True,
                            value={
                                "output": output,
                                "output_file": output_path,
                                "result": result,
                                "script_path": script_path
                            }
                        )
                    else:
                        # Complex arguments - fall back to subprocess  
                        sys.stdout.close()
                        sys.stdout = old_stdout
                        self.logger.info("command_line_runner.complex_args_fallback", 
                                      arg_count=len(args))
                        # Don't try to import and call directly - use subprocess
                        return self._execute_shell_command(
                            [sys.executable, script_path], 
                            args, 
                            working_dir, 
                            output_file
                        )
                except SystemExit as e:
                    # Main called sys.exit - clean up and fall back to subprocess
                    if sys.stdout != old_stdout:
                        sys.stdout.close()
                    sys.stdout = old_stdout
                    self.logger.warning("command_line_runner.module_exit", 
                                     script_path=script_path,
                                     exit_code=e.code)
                    # Use subprocess to avoid sys.exit issues
                    return self._execute_shell_command(
                        [sys.executable, script_path], 
                        args, 
                        working_dir, 
                        output_file
                    )
                    
                finally:
                    # Clean up and restore stdout
                    if sys.stdout != old_stdout:
                        sys.stdout.close()
                    sys.stdout = old_stdout
                    
                    # Clean up temp file
                    if output_capture and not output_file:
                        output_capture.close()
            
            # No main function, fall back to subprocess
            self.logger.warning("command_line_runner.no_main_function", 
                             script_path=script_path)
            return self._execute_shell_command(
                [sys.executable, script_path], 
                args, 
                working_dir, 
                output_file
            )
            
        except ImportError:
            # Module import failed, fall back to subprocess
            self.logger.warning("command_line_runner.import_failed", 
                             script_path=script_path)
            return self._execute_shell_command(
                [sys.executable, script_path], 
                args, 
                working_dir, 
                output_file
            )
            
        except Exception as e:
            # Other errors in module execution
            self.logger.error("command_line_runner.module_execution_failed", 
                           script_path=script_path,
                           error=str(e))
            return SkillResult(
                success=False,
                error=f"Error executing Python module {script_path}: {str(e)}"
            )
    
    def _execute_shell_command(
        self, 
        cmd: List[str], 
        args: Dict[str, Any], 
        working_dir: str, 
        output_file: Optional[str]
    ) -> SkillResult:
        """
        Execute a shell command with the provided arguments.
        
        Args:
            cmd: Base command to execute as a list of strings
            args: Arguments to pass to the command
            working_dir: Directory to execute in
            output_file: Path to write output to
            
        Returns:
            SkillResult with execution result
        """
        # Convert args to command line arguments
        cmd_with_args = list(cmd)
        
        # Extract positional arguments first if present
        pos_args = []
        
        # Special handling for positional arguments - they should be added last
        if "positional_args" in args:
            pos_args = args["positional_args"]
            # Remove from args so we don't process it as a regular option
            args = {k: v for k, v in args.items() if k != "positional_args"}
        
        # Handle different argument styles based on the base command
        if len(cmd) > 0 and (cmd[0].endswith(".py") or cmd[0] == sys.executable):
            # Python script - use -- style args
            for k, v in args.items():
                if isinstance(v, bool) and v:
                    # Boolean flags
                    cmd_with_args.append(f"--{k}")
                elif isinstance(v, (list, tuple)):
                    # Lists get expanded
                    # If it's a normal argument with list value
                    if k != "positional_args":
                        for item in v:
                            if k.startswith('_'):  # Special prefix for raw args
                                cmd_with_args.append(str(item))
                            else:
                                cmd_with_args.append(f"--{k}")
                                cmd_with_args.append(str(item))
                else:
                    # Other values
                    cmd_with_args.append(f"--{k}")
                    if v is not None:
                        cmd_with_args.append(str(v))
        else:
            # Generic command - try to follow convention
            for k, v in args.items():
                if k.startswith('_'):  # Special prefix for raw args
                    # Just add as is, without flags
                    if isinstance(v, (list, tuple)):
                        cmd_with_args.extend([str(item) for item in v])
                    else:
                        cmd_with_args.append(str(v))
                elif len(k) == 1:
                    # Short option (-a)
                    if isinstance(v, bool) and v:
                        cmd_with_args.append(f"-{k}")
                    elif isinstance(v, (list, tuple)):
                        for item in v:
                            cmd_with_args.append(f"-{k}")
                            cmd_with_args.append(str(item))
                    else:
                        cmd_with_args.append(f"-{k}")
                        if v is not None:
                            cmd_with_args.append(str(v))
                else:
                    # Long option (--arg)
                    if isinstance(v, bool) and v:
                        cmd_with_args.append(f"--{k}")
                    elif isinstance(v, (list, tuple)):
                        for item in v:
                            cmd_with_args.append(f"--{k}")
                            cmd_with_args.append(str(item))
                    else:
                        cmd_with_args.append(f"--{k}")
                        if v is not None:
                            cmd_with_args.append(str(v))
        
        # Add positional arguments after all options
        if pos_args:
            if isinstance(pos_args, list):
                cmd_with_args.extend([str(a) for a in pos_args])
            else:
                cmd_with_args.append(str(pos_args))
                
        # Log the full command
        self.logger.info("command_line_runner.executing_command", 
                      command=cmd_with_args,
                      working_dir=working_dir)
        
        try:
            # Determine output handling
            if output_file:
                # Write output to file
                with open(output_file, "w") as f:
                    result = subprocess.run(
                        cmd_with_args,
                        stdout=f,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        cwd=working_dir
                    )
                # Read the output file
                with open(output_file, "r") as f:
                    stdout = f.read()
            else:
                # Capture output in memory
                result = subprocess.run(
                    cmd_with_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    cwd=working_dir
                )
                stdout = result.stdout
                
            # Check result
            if result.returncode != 0:
                self.logger.error("command_line_runner.command_failed", 
                               returncode=result.returncode,
                               stderr=result.stderr)
                return SkillResult(
                    success=False,
                    error=f"Command failed with exit code {result.returncode}: {result.stderr}",
                    value={
                        "returncode": result.returncode,
                        "stderr": result.stderr,
                        "stdout": stdout,
                        "output_file": output_file
                    }
                )
                
            # Command succeeded
            self.logger.info("command_line_runner.command_succeeded", 
                          returncode=result.returncode,
                          output_length=len(stdout))
            
            # Parse output if it looks like JSON
            output_data = None
            try:
                if stdout.strip().startswith(("{", "[")):
                    output_data = json.loads(stdout)
            except json.JSONDecodeError:
                # Not JSON, that's fine
                pass
                
            return SkillResult(
                success=True,
                value={
                    "output": stdout,
                    "parsed_output": output_data,
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                    "output_file": output_file
                }
            )
            
        except Exception as e:
            self.logger.error("command_line_runner.execution_error", 
                           command=cmd_with_args,
                           error=str(e))
            return SkillResult(
                success=False,
                error=f"Error executing command: {str(e)}"
            )