"""
Asset management with minimal processing and LLM-first design.
Path: c4h_agents/skills/asset_manager.py
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
import shutil
import os
import json
from datetime import datetime

from c4h_agents.skills.base_skill import BaseSkill
from c4h_agents.agents.types import SkillResult
from c4h_agents.skills.semantic_merge import SemanticMerge
from c4h_agents.utils.logging import get_logger

logger = get_logger()

@dataclass
class AssetPath:
    """Describes a file path with relevant metadata"""
    path: Path
    exists: bool = False
    is_file: bool = False
    is_dir: bool = False
    size: Optional[int] = None
    modified: Optional[str] = None
    
    @classmethod
    def from_path(cls, path: Path) -> 'AssetPath':
        """Create AssetPath from a Path object with populated metadata"""
        exists = path.exists()
        is_file = path.is_file()
        is_dir = path.is_dir()
        size = path.stat().st_size if exists and is_file else None
        modified = datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else None
        
        return cls(
            path=path,
            exists=exists,
            is_file=is_file,
            is_dir=is_dir,
            size=size,
            modified=modified
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "path": str(self.path),
            "exists": self.exists,
            "is_file": self.is_file,
            "is_dir": self.is_dir,
            "size": self.size,
            "modified": self.modified
        }

@dataclass
class AssetOperationResult:
    """Detailed result of an asset operation"""
    operation: str
    success: bool
    source_path: Optional[Path] = None
    target_path: Optional[Path] = None
    backup_path: Optional[Path] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "operation": self.operation,
            "success": self.success,
            "source_path": str(self.source_path) if self.source_path else None,
            "target_path": str(self.target_path) if self.target_path else None,
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "error": self.error
        }

class AssetManager(BaseSkill):
    """Manages file operations with improved backup support"""
    
    def __init__(self, config: Dict[str, Any], skill_name: str = "asset_manager"):
        """Initialize with configuration and skill name"""
        super().__init__(config, skill_name)
        
        # Get project path from config if available
        self.project_path = None
        if 'project' in self.config:
            project_config = self.config['project']
            if isinstance(project_config, dict):
                self.project_path = Path(project_config.get('path', '.')).resolve()
                workspace_root = project_config.get('workspace_root', 'workspaces')
                if not Path(workspace_root).is_absolute():
                    workspace_root = str(self.project_path / workspace_root)
                self.backup_dir = Path(workspace_root) / "backups"
            else:
                # Handle Project instance if present
                try:
                    self.project_path = getattr(project_config, 'paths', {}).root
                    self.backup_dir = getattr(project_config, 'paths', {}).workspace / "backups"
                except (AttributeError, TypeError):
                    self.logger.warning("asset_manager.project_config_invalid", 
                                      type=type(project_config).__name__)
                    self.project_path = Path('.').resolve()
                    self.backup_dir = Path('workspaces/backups')
        else:
            # Default paths if no project config
            self.project_path = Path('.').resolve()
            self.backup_dir = Path('workspaces/backups')
            
        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.debug("asset_manager.initialized", 
                        project_path=str(self.project_path),
                        backup_dir=str(self.backup_dir))

    def execute(self, **kwargs) -> SkillResult:
        """
        Execute the asset management operation.
        
        Args:
            action: The action to perform (read, write, backup, etc.)
            path: Path to the file or directory
            content: Content to write (for write operations)
            **kwargs: Additional parameters for specific actions
            
        Returns:
            SkillResult with the operation outcome
        """
        # Extract parameters
        action = kwargs.get('action', 'info')
        path_str = kwargs.get('path')
        content = kwargs.get('content')
        
        # Log request
        self.logger.info("asset_manager.execute", 
                       action=action, 
                       path=path_str, 
                       has_content=content is not None)
                       
        # Validate path
        if not path_str:
            return SkillResult(
                success=False,
                error="Path parameter is required"
            )
        
        # Normalize path
        path = Path(path_str)
        if not path.is_absolute() and self.project_path:
            path = self.project_path / path
            
        # Resolve path
        try:
            path = path.resolve()
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"Failed to resolve path: {str(e)}"
            )
            
        # Dispatch to appropriate method based on action
        try:
            if action == 'read':
                return self._handle_errors(self._read_file, path)
            elif action == 'write':
                return self._handle_errors(self._write_file, path, content)
            elif action == 'backup':
                return self._handle_errors(self._backup_file, path)
            elif action == 'delete':
                return self._handle_errors(self._delete_file, path)
            elif action == 'info':
                return self._handle_errors(self._get_file_info, path)
            elif action == 'list':
                return self._handle_errors(self._list_directory, path)
            else:
                return SkillResult(
                    success=False,
                    error=f"Unsupported action: {action}"
                )
        except Exception as e:
            self.logger.error("asset_manager.execute_failed", 
                           action=action, 
                           path=str(path), 
                           error=str(e))
            return SkillResult(
                success=False,
                error=f"Operation failed: {str(e)}"
            )
            
    def _get_file_info(self, path: Path) -> SkillResult:
        """Get information about a file or directory"""
        asset_path = AssetPath.from_path(path)
        return SkillResult(
            success=True,
            value=asset_path.to_dict()
        )
        
    def _read_file(self, path: Path) -> SkillResult:
        """Read a file and return its contents"""
        if not path.exists():
            return SkillResult(
                success=False,
                error=f"File not found: {path}"
            )
            
        if not path.is_file():
            return SkillResult(
                success=False,
                error=f"Not a file: {path}"
            )
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return SkillResult(
                success=True,
                value={
                    "content": content,
                    "path": str(path),
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                }
            )
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"Failed to read file: {str(e)}"
            )
            
    def _write_file(self, path: Path, content: str) -> SkillResult:
        """Write content to a file with automatic backup"""
        if path.exists() and path.is_file():
            # Backup existing file
            backup_result = self._backup_file(path)
            if not backup_result.success:
                return SkillResult(
                    success=False,
                    error=f"Failed to backup file before writing: {backup_result.error}",
                    value=backup_result.value
                )
                
        # Create parent directories if they don't exist
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"Failed to create directory structure: {str(e)}"
            )
            
        # Write the file
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            return SkillResult(
                success=True,
                value={
                    "path": str(path),
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                }
            )
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"Failed to write file: {str(e)}"
            )
            
    def _backup_file(self, path: Path) -> SkillResult:
        """Create a backup of a file"""
        if not path.exists():
            return SkillResult(
                success=False,
                error=f"File not found: {path}"
            )
            
        if not path.is_file():
            return SkillResult(
                success=False,
                error=f"Not a file: {path}"
            )
            
        try:
            # Create timestamped backup name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            relative_path = path.relative_to(self.project_path) if self.project_path and str(path).startswith(str(self.project_path)) else path.name
            backup_path = self.backup_dir / f"{timestamp}_{relative_path}"
            
            # Create parent directories
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(path, backup_path)
            
            return SkillResult(
                success=True,
                value={
                    "original_path": str(path),
                    "backup_path": str(backup_path),
                    "timestamp": timestamp
                }
            )
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"Failed to create backup: {str(e)}"
            )
            
    def _delete_file(self, path: Path) -> SkillResult:
        """Delete a file with automatic backup"""
        if not path.exists():
            return SkillResult(
                success=False,
                error=f"File not found: {path}"
            )
            
        if not path.is_file():
            return SkillResult(
                success=False,
                error=f"Not a file: {path}"
            )
            
        # Backup before deletion
        backup_result = self._backup_file(path)
        if not backup_result.success:
            return SkillResult(
                success=False,
                error=f"Failed to backup file before deletion: {backup_result.error}",
                value=backup_result.value
            )
            
        try:
            # Delete the file
            path.unlink()
            
            return SkillResult(
                success=True,
                value={
                    "deleted_path": str(path),
                    "backup_info": backup_result.value
                }
            )
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"Failed to delete file: {str(e)}",
                value={"backup_info": backup_result.value}
            )
            
    def _list_directory(self, path: Path) -> SkillResult:
        """List contents of a directory"""
        if not path.exists():
            return SkillResult(
                success=False,
                error=f"Directory not found: {path}"
            )
            
        if not path.is_dir():
            return SkillResult(
                success=False,
                error=f"Not a directory: {path}"
            )
            
        try:
            items = []
            for item in path.iterdir():
                items.append(AssetPath.from_path(item).to_dict())
                
            return SkillResult(
                success=True,
                value={
                    "path": str(path),
                    "items": items,
                    "count": len(items)
                }
            )
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"Failed to list directory: {str(e)}"
            )
            
    def process_action(self, action_data: Dict[str, Any]) -> SkillResult:
        """
        Process an asset management action (legacy compatibility method)
        
        Args:
            action_data: Dictionary with action parameters
            
        Returns:
            SkillResult with the operation outcome
        """
        return self.execute(**action_data)