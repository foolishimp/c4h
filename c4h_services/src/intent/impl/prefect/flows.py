# File: /Users/jim/src/apps/c4h_ai_dev/c4h_services/src/intent/impl/prefect/flows.py
"""
Flow implementations for prefect-based workflow orchestration.
This module contains the recovery workflow implementation.
The primary workflows have been moved to workflows.py.
"""

from prefect import flow, task, get_run_logger
from prefect.states import Completed, Failed, Pending
from prefect.context import get_flow_context, FlowRunContext
from prefect.utilities.annotations import unmapped
from typing import Dict, Any, Optional
from c4h_services.src.utils.logging import get_logger
from pathlib import Path
import json

from .tasks import run_agent_task

# Use the imported get_logger
logger = get_logger()


@flow(name="intent_recovery")
def run_recovery_workflow(
    flow_run_id: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recovery workflow for handling failed runs using Prefect states.
    (This flow might need further refactoring depending on how recovery
     should work with the declarative model and snapshots)
    """
    # Use get_run_logger() inside the flow for Prefect context awareness
    run_logger = get_run_logger()
    try:
        # Get failed flow run state
        # Ensure get_flow_context() provides the necessary client or adjust
        flow_ctx = get_flow_context()
        if not flow_ctx or not hasattr(flow_ctx, 'client'):
             run_logger.error("Prefect client not found in context for recovery.")
             # Return a Failed state's data directly
             return Failed(message="Prefect client not found", result={"status": "error", "error": "Client unavailable"}).data

        client = flow_ctx.client
        flow_run = client.read_flow_run(flow_run_id)

        if not flow_run.state.is_failed():
            # Return a Completed state's data directly
            return Completed(
                message="Flow run is not in failed state",
                result={"status": "error", "error": "Flow is not failed"}
            ).data # Return data for consistency if needed

        # Extract failure point and data
        failed_result = flow_run.state.result(fetch=True) # Fetch the actual result data
        failed_stage = failed_result.get("stage") if isinstance(failed_result, dict) else None

        if not failed_stage:
             # Return a Failed state's data directly
            return Failed(
                message="Could not determine failure point from result",
                result={"status": "error", "error": "Unknown failure point"}
            ).data # Return data

        # Resume from failed stage
        # TODO: Implement stage-specific recovery logic for the declarative model
        # This would likely involve reloading the snapshot and re-running `run_declarative_workflow`
        # with context indicating the recovery point.
        run_logger.warning("Recovery logic needs implementation for declarative workflows.")
        # Return a Pending state's data directly
        return Pending(message="Recovery not yet implemented for declarative workflows").data # Return data

    except Exception as e:
        error_msg = str(e)
        run_logger.error("recovery_workflow.failed", error=error_msg, exc_info=True) # Log traceback
        # Return a Failed state's data directly
        return Failed(
            message=f"Recovery failed: {error_msg}",
            result={"status": "error", "error": error_msg}
        ).data # Return data
