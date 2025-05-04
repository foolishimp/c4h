"""
Implementation of ExecutionContext for the type-based architecture.

The ExecutionContext is a core component of the type-based architecture,
providing an immutable context with structured access patterns, snapshots,
and serialization capabilities.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Set, List, Tuple


class ExecutionContext:
    """Immutable execution context with structured access patterns."""
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize with optional data."""
        self._data = data or {
            "metadata": {
                "context_id": f"ctx-{uuid.uuid4()}",
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        self._snapshots = {}
        self._sensitive_keys = set()
        
    def get(self, path: str, default: Any = None) -> Any:
        """Get a value from the context using a dot-notation path."""
        parts = path.split('.')
        current = self._data
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
            
        return current
    
    def set(self, path: str, value: Any) -> 'ExecutionContext':
        """
        Set a value in the context using a dot-notation path.
        Returns a new ExecutionContext instance (immutable pattern).
        """
        parts = path.split('.')
        new_data = self._deep_copy(self._data)
        current = new_data
        
        # Navigate to the appropriate nesting level
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value at the final level
        current[parts[-1]] = value
        
        # Create a new context with the updated data
        new_context = ExecutionContext(new_data)
        new_context._snapshots = self._deep_copy(self._snapshots)
        new_context._sensitive_keys = self._sensitive_keys.copy()
        
        return new_context
    
    def create_snapshot(self, snapshot_id: str) -> str:
        """Create a snapshot of the current context state."""
        self._snapshots[snapshot_id] = self._deep_copy(self._data)
        return snapshot_id
    
    def restore_snapshot(self, snapshot_id: str) -> 'ExecutionContext':
        """Restore from a snapshot, returning a new ExecutionContext."""
        if snapshot_id not in self._snapshots:
            raise ValueError(f"Snapshot {snapshot_id} not found")
        
        new_context = ExecutionContext(self._deep_copy(self._snapshots[snapshot_id]))
        new_context._snapshots = self._deep_copy(self._snapshots)
        new_context._sensitive_keys = self._sensitive_keys.copy()
        
        return new_context
    
    def list_snapshots(self) -> List[str]:
        """List available snapshot IDs."""
        return list(self._snapshots.keys())
    
    def mark_sensitive(self, path: str) -> 'ExecutionContext':
        """Mark a path as containing sensitive information."""
        new_context = ExecutionContext(self._deep_copy(self._data))
        new_context._snapshots = self._deep_copy(self._snapshots)
        new_context._sensitive_keys = self._sensitive_keys.copy()
        new_context._sensitive_keys.add(path)
        
        return new_context
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert context to a dictionary, optionally excluding sensitive data."""
        if include_sensitive:
            return self._deep_copy(self._data)
        
        result = self._deep_copy(self._data)
        for path in self._sensitive_keys:
            self._remove_path(result, path)
        
        return result
    
    def merge(self, other: 'ExecutionContext') -> 'ExecutionContext':
        """Merge another context into this one, returning a new context."""
        new_data = self._deep_copy(self._data)
        self._deep_merge(new_data, other._data)
        
        new_context = ExecutionContext(new_data)
        # Merge snapshots and sensitive keys
        new_context._snapshots = self._deep_copy(self._snapshots)
        for k, v in other._snapshots.items():
            if k not in new_context._snapshots:
                new_context._snapshots[k] = self._deep_copy(v)
        
        new_context._sensitive_keys = self._sensitive_keys.union(other._sensitive_keys)
        
        return new_context
    
    def _deep_copy(self, obj: Any) -> Any:
        """Create a deep copy of an object."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        elif isinstance(obj, set):
            return {self._deep_copy(item) for item in obj}
        elif isinstance(obj, tuple):
            return tuple(self._deep_copy(item) for item in obj)
        else:
            return obj
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source into target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = self._deep_copy(value)
    
    def _remove_path(self, data: Dict[str, Any], path: str) -> None:
        """Remove a path from a dictionary."""
        parts = path.split('.')
        current = data
        
        # Navigate to the parent of the target
        for part in parts[:-1]:
            if not isinstance(current, dict) or part not in current:
                return
            current = current[part]
        
        # Remove the target if it exists
        if isinstance(current, dict) and parts[-1] in current:
            del current[parts[-1]]