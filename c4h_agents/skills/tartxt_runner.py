"""
TartXT Runner Skill for C4H Agent System.

DEPRECATED: This skill has been deprecated in favor of the CommandLineRunner skill.
Please update your code to use CommandLineRunner instead. See the skills/README.md
for migration instructions.

This skill provides a wrapper around the tartxt.py script for project scanning and
content extraction. It abstracts the details of invoking tartxt and handling its
output, making it available as a standard C4H Skill.
"""

import os
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import structlog
import json

from c4h_agents.agents.types import SkillResult

# Import the tartxt module if available, otherwise provide fallback path
try:
    from c4h_agents.skills import tartxt
    TARTXT_PATH = str(Path(tartxt.__file__).resolve())
except ImportError:
    TARTXT_PATH = "c4h_agents/skills/tartxt.py"

# Configure logger
logger = structlog.get_logger()

class TartXTRunner:
    """
    Skill for scanning projects and extracting content using tartxt.py script.
    
    This skill wraps the functionality of the tartxt.py script, allowing it to be
    invoked as a standard C4H Skill rather than directly within agent implementations.
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
            skill="tartxt_runner",
            skill_id=self.skill_id
        )
        
        # Log deprecation warning
        self.logger.warning("tartxt_runner.deprecated", 
                         message="TartXTRunner is deprecated. Please use CommandLineRunner instead.")
        
        # Get tartxt script path from config or use default
        self.script_path = self._resolve_script_path()
        
        self.logger.info("tartxt_runner.initialized", 
                      script_path=self.script_path)
    
    def _resolve_script_path(self) -> str:
        """
        Resolve the path to the tartxt.py script.
        
        Checks for script_path in multiple locations in the following order:
        1. Explicit path in skill params
        2. Explicit path in config
        3. Imported module path
        4. Default relative path
        
        Returns:
            Resolved path to tartxt.py script
        """
        # First, check for path in the effective config
        config_node = self.config.get("tartxt_config", {})
        if isinstance(config_node, dict) and "script_path" in config_node:
            path = config_node.get("script_path")
            if path and os.path.exists(path):
                self.logger.debug("tartxt_runner.using_config_path", path=path)
                return path
        
        # Next, try imported module path
        if os.path.exists(TARTXT_PATH):
            self.logger.debug("tartxt_runner.using_module_path", path=TARTXT_PATH)
            return TARTXT_PATH
        
        # Finally, fall back to default relative path
        default_path = "c4h_agents/skills/tartxt.py"
        self.logger.warning("tartxt_runner.using_default_path", 
                         path=default_path,
                         config_path_tried=config_node.get("script_path"))
        return default_path
    
    def execute(
        self, 
        input: Dict[str, Any] = None,
        project_path: Optional[str] = None,
        input_paths: Optional[List[str]] = None,
        exclusions: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        history_days: int = 0,
        **kwargs
    ) -> SkillResult:
        """
        Execute tartxt.py to scan project content.
        
        Args:
            input: Input context (can contain project_path)
            project_path: Path to project root
            input_paths: List of paths to include in scan
            exclusions: List of patterns to exclude
            output_file: Path to write output (generated if not provided)
            history_days: Number of days of history to include (0 for none)
            
        Returns:
            SkillResult with project content
        """
        try:
            # Normalize input arguments
            input = input or {}
            
            # Get project path (prioritize explicit arg, fall back to input context)
            if not project_path:
                project_path = input.get("project_path")
                if not project_path:
                    self.logger.warning("tartxt_runner.no_project_path_provided")
                    project_path = os.getcwd()
            
            # Get input paths (prioritize explicit arg, fall back to config)
            if not input_paths:
                config_node = self.config.get("tartxt_config", {})
                input_paths = input.get("input_paths") or config_node.get("input_paths") or ["."]
            
            # Get exclusions (prioritize explicit arg, fall back to config)
            if exclusions is None:
                config_node = self.config.get("tartxt_config", {})
                exclusions = input.get("exclusions") or config_node.get("exclusions") or []
            
            # Generate output file path if not provided
            if not output_file:
                run_id = input.get("workflow_run_id") or input.get("execution_id") or str(uuid.uuid4())
                output_dir = os.path.join(os.getcwd(), "workspaces")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"project_scan_{run_id}.txt")
            
            # Prepare command
            exclusion_arg = ",".join(exclusions) if exclusions else ""
            cmd = [
                self.script_path,
                project_path,
                "-f", output_file,
                "-H", str(history_days)
            ]
            
            if exclusion_arg:
                cmd.extend(["-x", exclusion_arg])
            
            # Log command
            self.logger.info("tartxt_runner.executing", 
                          project_path=project_path,
                          input_paths=input_paths,
                          exclusions=exclusions,
                          output_file=output_file,
                          cmd=cmd)
            
            # Run tartxt.py
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Check for errors
            if result.returncode != 0:
                self.logger.error("tartxt_runner.command_failed", 
                               returncode=result.returncode,
                               stderr=result.stderr)
                return SkillResult(
                    success=False,
                    error=f"tartxt.py failed with code {result.returncode}: {result.stderr}"
                )
            
            # Read output file
            if not os.path.exists(output_file):
                self.logger.error("tartxt_runner.output_file_not_found", file=output_file)
                return SkillResult(
                    success=False,
                    error=f"Output file not created: {output_file}"
                )
            
            # Read the output file
            with open(output_file, 'r', encoding='utf-8') as f:
                project_content = f.read()
            
            self.logger.info("tartxt_runner.completed", 
                          content_length=len(project_content),
                          output_file=output_file)
            
            # Return result
            return SkillResult(
                success=True,
                value={
                    "project_content": project_content,
                    "project_scan_file": output_file,
                    "project_path": project_path,
                    "exclusions": exclusions,
                    "content_length": len(project_content)
                }
            )
            
        except Exception as e:
            self.logger.exception("tartxt_runner.failed", error=str(e))
            return SkillResult(
                success=False,
                error=f"Error executing tartxt runner: {str(e)}"
            )