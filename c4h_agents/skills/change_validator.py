"""
Change validator skill for validating code changes after implementation.
Path: c4h_agents/skills/change_validator.py
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
from c4h_agents.skills.base_skill import BaseSkill
from c4h_agents.agents.types import SkillResult
from c4h_agents.utils.logging import get_logger

logger = get_logger()

class ChangeValidator(BaseSkill):
    """Skill to validate code changes after implementation"""
    
    def __init__(self, config: Dict[str, Any], skill_name: str = "change_validator"):
        """Initialize with configuration and skill name"""
        super().__init__(config, skill_name)
        
        # Get skill-specific configuration
        skill_config = self.config.get('llm_config', {}).get('skills', {}).get('change_validator', {})
        
        # Validation steps to perform
        self.validation_steps = skill_config.get('validation_steps', [
            'syntax_check',
            'file_exists',
            'basic_validation'
        ])
        
        # Test command configuration
        self.test_commands = skill_config.get('test_commands', {})
        self.lint_commands = skill_config.get('lint_commands', {})
        
        logger.info("change_validator.initialized",
                   validation_steps=self.validation_steps)
    
    def execute(self, **kwargs) -> SkillResult:
        """
        Validate code changes.
        
        Args:
            changes: List of changes made (files modified, created, deleted)
            project_path: Path to the project root
            run_tests: Whether to run tests (default: True)
            run_lint: Whether to run linting (default: True)
            
        Returns:
            SkillResult with validation results
        """
        changes = kwargs.get('changes', [])
        project_path = Path(kwargs.get('project_path', '.'))
        run_tests = kwargs.get('run_tests', True)
        run_lint = kwargs.get('run_lint', True)
        
        logger.info("change_validator.execute",
                   changes_count=len(changes),
                   project_path=str(project_path),
                   run_tests=run_tests,
                   run_lint=run_lint)
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks_performed': []
        }
        
        try:
            # Validate file existence
            if 'file_exists' in self.validation_steps:
                self._validate_file_existence(changes, project_path, validation_results)
            
            # Basic syntax validation
            if 'syntax_check' in self.validation_steps:
                self._validate_syntax(changes, project_path, validation_results)
            
            # Run linting if requested
            if run_lint and 'linting' in self.validation_steps:
                self._run_linting(project_path, validation_results)
            
            # Run tests if requested
            if run_tests and 'test_execution' in self.validation_steps:
                self._run_tests(project_path, validation_results)
            
            # Determine overall validity
            validation_results['valid'] = len(validation_results['errors']) == 0
            
            return SkillResult(
                success=True,
                value=validation_results
            )
            
        except Exception as e:
            logger.error("change_validator.execute_failed",
                        error=str(e))
            return SkillResult(
                success=False,
                error=f"Validation failed: {str(e)}"
            )
    
    def _validate_file_existence(self, changes: List[Dict], project_path: Path,
                                results: Dict[str, Any]):
        """Validate that files exist or don't exist as expected"""
        logger.debug("change_validator.validating_file_existence")
        results['checks_performed'].append('file_exists')
        
        for change in changes:
            if isinstance(change, dict):
                file_path = change.get('path') or change.get('file_path')
                action = change.get('action') or change.get('type', 'modify')
                
                if file_path:
                    full_path = project_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
                    
                    if action in ['create', 'modify', 'write']:
                        if not full_path.exists():
                            results['errors'].append(f"Expected file does not exist: {file_path}")
                        elif not full_path.is_file():
                            results['errors'].append(f"Expected file is not a file: {file_path}")
                    elif action == 'delete':
                        if full_path.exists():
                            results['errors'].append(f"File should have been deleted: {file_path}")
    
    def _validate_syntax(self, changes: List[Dict], project_path: Path,
                        results: Dict[str, Any]):
        """Validate syntax of changed files"""
        logger.debug("change_validator.validating_syntax")
        results['checks_performed'].append('syntax_check')
        
        for change in changes:
            if isinstance(change, dict):
                file_path = change.get('path') or change.get('file_path')
                action = change.get('action') or change.get('type', 'modify')
                
                if file_path and action != 'delete':
                    full_path = project_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
                    
                    if full_path.exists() and full_path.suffix == '.py':
                        # Python syntax check
                        result = subprocess.run(
                            ['python', '-m', 'py_compile', str(full_path)],
                            capture_output=True,
                            text=True
                        )
                        if result.returncode != 0:
                            results['errors'].append(f"Python syntax error in {file_path}: {result.stderr}")
                    
                    # Add more language-specific syntax checks here
    
    def _run_linting(self, project_path: Path, results: Dict[str, Any]):
        """Run linting on the project"""
        logger.debug("change_validator.running_linting")
        results['checks_performed'].append('linting')
        
        # Try different linters based on what's available
        linters_tried = []
        
        # Try ruff first (fast Python linter)
        if self._try_linter('ruff', ['ruff', 'check', str(project_path)], project_path, results):
            linters_tried.append('ruff')
        
        # Try flake8
        elif self._try_linter('flake8', ['flake8', str(project_path)], project_path, results):
            linters_tried.append('flake8')
        
        # Try pylint
        elif self._try_linter('pylint', ['pylint', str(project_path)], project_path, results):
            linters_tried.append('pylint')
        
        if not linters_tried:
            results['warnings'].append("No Python linters found (tried: ruff, flake8, pylint)")
    
    def _try_linter(self, name: str, command: List[str], project_path: Path,
                    results: Dict[str, Any]) -> bool:
        """Try to run a specific linter"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=str(project_path),
                timeout=30
            )
            
            if result.returncode != 0:
                # Linting issues found
                results['warnings'].append(f"{name} found issues:\n{result.stdout}")
            
            return True  # Linter ran successfully
            
        except (subprocess.SubprocessError, FileNotFoundError):
            return False  # Linter not available
    
    def _run_tests(self, project_path: Path, results: Dict[str, Any]):
        """Run tests on the project"""
        logger.debug("change_validator.running_tests")
        results['checks_performed'].append('test_execution')
        
        # Look for test command in configuration or common patterns
        test_commands = [
            ['python', '-m', 'pytest'],
            ['pytest'],
            ['python', '-m', 'unittest', 'discover'],
            ['python', 'setup.py', 'test'],
            ['npm', 'test'],
            ['yarn', 'test'],
            ['make', 'test']
        ]
        
        # Add custom test commands from config
        if self.test_commands:
            test_commands = self.test_commands + test_commands
        
        test_ran = False
        for cmd in test_commands:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(project_path),
                    timeout=60  # 1 minute timeout for tests
                )
                
                if result.returncode == 0:
                    results['warnings'].append(f"Tests passed using: {' '.join(cmd)}")
                else:
                    results['errors'].append(f"Tests failed using {' '.join(cmd)}:\n{result.stdout}\n{result.stderr}")
                
                test_ran = True
                break  # Stop after first test command that runs
                
            except (subprocess.SubprocessError, FileNotFoundError):
                continue  # Try next test command
        
        if not test_ran:
            results['warnings'].append("No test runner found or tests could not be executed")