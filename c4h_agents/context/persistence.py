"""
Persistence mechanisms for ExecutionContext.

This module provides functionality for serializing and deserializing
ExecutionContext instances, supporting both file-based and database storage.
"""

import json
import os
from typing import Any, Dict, Optional

from .execution_context import ExecutionContext


class ContextPersistence:
    """Handles persistence of execution contexts."""
    
    @staticmethod
    def save_to_file(context: ExecutionContext, file_path: str, include_sensitive: bool = False) -> None:
        """Save context to a file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert context to dict and save as JSON
        with open(file_path, 'w') as f:
            json.dump(context.to_dict(include_sensitive=include_sensitive), f, indent=2)
    
    @staticmethod
    def load_from_file(file_path: str) -> ExecutionContext:
        """Load context from a file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return ExecutionContext(data)
    
    @staticmethod
    def to_json(context: ExecutionContext, include_sensitive: bool = False) -> str:
        """Convert context to a JSON string."""
        return json.dumps(context.to_dict(include_sensitive=include_sensitive), indent=2)
    
    @staticmethod
    def from_json(json_str: str) -> ExecutionContext:
        """Create context from a JSON string."""
        data = json.loads(json_str)
        return ExecutionContext(data)