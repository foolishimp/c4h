"""
Execution module for the C4H Agent System.

This package provides the components necessary for executing plans
at various levels of the system hierarchy (workflows, teams, agents).
"""

from .executor import ExecutionPlanExecutor

__all__ = ['ExecutionPlanExecutor']