"""
Claude Code runner skill for invoking Claude Code to implement code changes.
Path: c4h_agents/skills/claude_code_runner.py
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import json
import tempfile
import os
from c4h_agents.skills.base_skill import BaseSkill
from c4h_agents.agents.types import SkillResult
from c4h_agents.utils.logging import get_logger

logger = get_logger()

class ClaudeCodeRunner(BaseSkill):
    """Skill to invoke Claude Code for code implementation"""
    
    def __init__(self, config: Dict[str, Any], skill_name: str = "claude_code_runner"):
        """Initialize with configuration and skill name"""
        super().__init__(config, skill_name)
        
        # Get skill-specific configuration
        skill_config = self.config.get('llm_config', {}).get('skills', {}).get('claude_code_runner', {})
        
        # Configuration options
        self.mode = skill_config.get('mode', 'cli')  # cli, api, or direct
        self.cli_path = skill_config.get('cli_path', 'claude')  # Default to 'claude' in PATH
        self.api_endpoint = skill_config.get('api_endpoint', 'http://localhost:8000/api/v1')
        self.api_key = skill_config.get('api_key', os.environ.get('CLAUDE_CODE_API_KEY'))
        self.timeout = skill_config.get('timeout', 300)
        self.retry_attempts = skill_config.get('retry_attempts', 2)
        
        logger.info("claude_code_runner.initialized", 
                   mode=self.mode,
                   cli_path=self.cli_path if self.mode == 'cli' else None,
                   api_endpoint=self.api_endpoint if self.mode == 'api' else None)
    
    def execute(self, **kwargs) -> SkillResult:
        """
        Execute Claude Code to implement code changes.
        
        Args:
            working_directory: Directory to run Claude Code in
            solution_design: The solution design from solution designer
            intent: Original intent description
            mode: Implementation mode (implementation, review, etc.)
            
        Returns:
            SkillResult with implementation results
        """
        working_dir = kwargs.get('working_directory', '.')
        if not working_dir:
            working_dir = '.'  # Default to current directory if empty
        solution_design = kwargs.get('solution_design', '')
        intent = kwargs.get('intent', '')
        mode = kwargs.get('mode', 'implementation')
        
        logger.info("claude_code_runner.execute",
                   mode=self.mode,
                   working_dir=working_dir,
                   has_solution_design=bool(solution_design),
                   has_intent=bool(intent))
        
        try:
            if self.mode == 'cli':
                return self._execute_cli(working_dir, solution_design, intent, mode)
            elif self.mode == 'api':
                return self._execute_api(working_dir, solution_design, intent, mode)
            elif self.mode == 'direct':
                return self._execute_direct(working_dir, solution_design, intent, mode)
            else:
                return SkillResult(
                    success=False,
                    error=f"Unsupported mode: {self.mode}"
                )
        except Exception as e:
            logger.error("claude_code_runner.execute_failed",
                        error=str(e),
                        mode=self.mode)
            return SkillResult(
                success=False,
                error=f"Claude Code execution failed: {str(e)}"
            )
    
    def _execute_cli(self, working_dir: str, solution_design: str, 
                     intent: str, mode: str) -> SkillResult:
        """Execute Claude Code via CLI"""
        
        # Create the prompt content
        prompt = self._create_claude_code_prompt(solution_design, intent, mode)
        
        try:
            # Build Claude Code command - pass prompt via stdin
            cmd = [
                self.cli_path,
                'code',  # Use claude code subcommand
                '--dangerously-skip-permissions',  # Allow file writes without prompting
                '--print',  # Non-interactive mode
            ]
            
            logger.debug("claude_code_runner.cli.executing",
                        command=' '.join(cmd),
                        working_dir=working_dir,
                        prompt_length=len(prompt))
            
            # Execute Claude Code with prompt via stdin
            result = subprocess.run(
                cmd,
                input=prompt,  # Pass prompt via stdin
                capture_output=True,
                text=True,
                cwd=working_dir,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                # Parse output to extract file changes
                files_modified = self._parse_cli_output(result.stdout)
                
                logger.info("claude_code_runner.cli.success",
                           files_modified_count=len(files_modified),
                           stdout_length=len(result.stdout),
                           stderr_length=len(result.stderr))
                
                # Log first 500 chars of output for debugging
                logger.debug("claude_code_runner.cli.output_preview",
                            preview=result.stdout[:500] if result.stdout else "No output")
                
                return SkillResult(
                    success=True,
                    value={
                        'files_modified': files_modified,
                        'summary': self._extract_summary(result.stdout),
                        'full_output': result.stdout,
                        'mode': 'cli'
                    }
                )
            else:
                logger.error("claude_code_runner.cli.failed",
                           returncode=result.returncode,
                           stderr=result.stderr)
                return SkillResult(
                    success=False,
                    error=f"Claude Code CLI failed: {result.stderr}"
                )
        except Exception as e:
            logger.error("claude_code_runner.cli.exception",
                        error=str(e),
                        error_type=type(e).__name__)
            return SkillResult(
                success=False,
                error=f"Claude Code CLI execution error: {str(e)}"
            )
    
    def _execute_api(self, working_dir: str, solution_design: str,
                     intent: str, mode: str) -> SkillResult:
        """Execute Claude Code via API (placeholder for future implementation)"""
        # This would be implemented when Claude Code API is available
        return SkillResult(
            success=False,
            error="API mode not yet implemented"
        )
    
    def _execute_direct(self, working_dir: str, solution_design: str,
                        intent: str, mode: str) -> SkillResult:
        """Execute using direct Claude API with tools (placeholder)"""
        # This would use Claude's native tools when available
        return SkillResult(
            success=False,
            error="Direct mode not yet implemented"
        )
    
    def _create_claude_code_prompt(self, solution_design: str, 
                                   intent: str, mode: str) -> str:
        """Create a prompt for Claude Code based on solution design"""
        
        prompt = f"""# Code Implementation Request

## Intent
{intent}

## Implementation Mode
{mode}

## Solution Design
The following solution has been designed by the solution designer. Please implement these changes exactly as specified:

{solution_design}

## Requirements
1. Apply each change block exactly as specified in the solution design
2. The solution design contains change blocks in this format:
   ===CHANGE_BEGIN===
   FILE: <file_path>
   TYPE: <create|modify|delete>
   DESCRIPTION: <change description>
   DIFF:
   <unified diff format>
   ===CHANGE_END===
3. For each change block:
   - If TYPE is "create": Create the new file with the content from the diff
   - If TYPE is "modify": Apply the diff to the existing file
   - If TYPE is "delete": Remove the file
4. Ensure all changes compile/run correctly
5. Follow the existing code style and conventions
6. Do not make any changes beyond what is specified in the solution design

## Important Notes
- This is an automated code modification system
- Only implement the exact changes specified
- If you encounter any issues, report them clearly
- After implementation, validate that the changes work correctly
"""
        return prompt
    
    def _parse_cli_output(self, output: str) -> List[Dict[str, str]]:
        """Parse Claude Code CLI output to extract file modifications"""
        files_modified = []
        
        # Look for common patterns in Claude Code output
        # Check for patterns like "Created main.py", "Modified: main.py", etc.
        lines = output.split('\n')
        for line in lines:
            # Look for file operation patterns
            if any(pattern in line.lower() for pattern in ['created', 'modified', 'updated', 'wrote', 'deleted']):
                # Try to extract file paths (common patterns)
                # Pattern 1: "Created file.py"
                # Pattern 2: "Modified: file.py"
                # Pattern 3: "Wrote to file.py"
                import re
                
                # Match patterns like "Created main.py", "Modified: main.py", "Wrote to main.py"
                file_pattern = r'(?:created|modified|updated|wrote to|deleted)(?:\:)?\s+([^\s]+\.py|[^\s]+\.txt|[^\s]+\.md|main\.py|[^\s]+)'
                match = re.search(file_pattern, line, re.IGNORECASE)
                if match:
                    file_path = match.group(1)
                    action = 'created' if 'created' in line.lower() or 'wrote' in line.lower() else 'modified'
                    if 'deleted' in line.lower():
                        action = 'deleted'
                    files_modified.append({
                        'action': action,
                        'path': file_path
                    })
        
        # If no files detected from output parsing, check if main.py was requested
        if not files_modified and 'main.py' in output:
            # Assume it was created if mentioned
            files_modified.append({
                'action': 'created',
                'path': 'main.py'
            })
        
        return files_modified
    
    def _extract_summary(self, output: str) -> str:
        """Extract a summary from Claude Code output"""
        # Simple extraction - look for summary section or use first few lines
        lines = output.split('\n')
        summary_lines = []
        
        in_summary = False
        for line in lines:
            if 'summary' in line.lower():
                in_summary = True
                continue
            if in_summary and line.strip():
                summary_lines.append(line.strip())
            if in_summary and not line.strip():
                break
        
        if summary_lines:
            return '\n'.join(summary_lines[:5])  # First 5 lines of summary
        else:
            # Return first non-empty line as summary
            for line in lines:
                if line.strip():
                    return line.strip()
            return "Code changes applied"