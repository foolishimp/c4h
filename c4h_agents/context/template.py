"""
Template resolution for configuration values.

This module provides functionality for resolving templates in configuration
values, using the ExecutionContext as a source of values.
"""

import re
from typing import Any, Dict, Pattern, Union

from .execution_context import ExecutionContext


class TemplateResolver:
    """Resolves templates in configuration values."""
    
    # Template pattern: ${path.to.value} or ${path.to.value:default}
    _TEMPLATE_PATTERN: Pattern = re.compile(r'\${([^:}]+)(?::([^}]+))?}')
    
    @staticmethod
    def resolve(value: Any, context: ExecutionContext) -> Any:
        """
        Resolve templates in a value, using the context as a source of values.
        
        Supports:
        - String templates: "Hello, ${user.name:Anonymous}!"
        - Dict templates: {"name": "${user.name:Anonymous}"}
        - List templates: ["${items[0]}", "${items[1]}"]
        """
        if isinstance(value, str):
            return TemplateResolver._resolve_string(value, context)
        elif isinstance(value, dict):
            return {k: TemplateResolver.resolve(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [TemplateResolver.resolve(item, context) for item in value]
        else:
            return value
    
    @staticmethod
    def _resolve_string(value: str, context: ExecutionContext) -> Union[str, Any]:
        """Resolve templates in a string value."""
        # Check if the entire string is a template
        if value.startswith('${') and value.endswith('}') and value.count('${') == 1:
            # Extract path and default
            match = TemplateResolver._TEMPLATE_PATTERN.match(value)
            if match:
                path, default = match.groups()
                # Get the value from context
                result = context.get(path, default)
                # If the result is None and a default was provided, use the default
                if result is None and default is not None:
                    return default
                return result
        
        # Otherwise, replace all templates in the string
        def replace(match):
            path, default = match.groups()
            result = context.get(path)
            if result is None and default is not None:
                return default
            return str(result) if result is not None else ''
        
        return TemplateResolver._TEMPLATE_PATTERN.sub(replace, value)