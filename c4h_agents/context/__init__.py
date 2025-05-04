"""
Context module for the type-based architecture.

This module contains components for immutable context management, persistence,
and template resolution.
"""

from .execution_context import ExecutionContext
from .persistence import ContextPersistence
from .template import TemplateResolver

__all__ = ['ExecutionContext', 'ContextPersistence', 'TemplateResolver']