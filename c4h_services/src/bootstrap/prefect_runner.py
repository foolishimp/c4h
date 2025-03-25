#!/usr/bin/env python3
"""
Streamlined runner focused exclusively on team-based workflow execution and API service.
Path: c4h_services/src/bootstrap/prefect_runner.py
"""

from pathlib import Path
import sys
import os
import uuid  # Add missing import
import requests
import time

# Add the project root to the Python path
# We need to go up enough levels to include both c4h_services and c4h_agents
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent  # Go up to the project root
sys.path.append(str(project_root))

import uvicorn
from c4h_services.src.utils.logging import get_logger
import argparse
from enum import Enum
import yaml
from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timezone

# Now imports should work correctly
from c4h_services.src.api.service import create_app
from c4h_agents.config import deep_merge
from c4h_services.src.orchestration.orchestrator import Orchestrator

logger = get_logger()

class LogMode(str, Enum):
    """Logging modes supported by runner"""
    DEBUG = "debug"
    NORMAL = "normal"

def load_configs(app_config_path: Optional[str] = None, system_config_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load and merge configurations in proper order"""
    try:
        app_config = {}
        if app_config_path:
            with open(app_config_path) as f:
                app_config = yaml.safe_load(f) or {}
                
            logger.info("config.content.loaded",
                      # Update logger with config after loading
                      logger = get_logger(app_config),
                      app_config_keys=list(app_config.keys()),
                      project_path=app_config.get('project', {}).get('path'),
                      has_intent=('intent' in app_config))
        
        merged_config = {}
        
        if system_config_paths:
            for sys_path in system_config_paths:
                path = Path(sys_path)
                if not path.exists():
                    logger.warning("config.system_config.not_found", path=str(path))
                    continue
                with open(path) as f:
                    sys_config = yaml.safe_load(f) or {}
                    logger.debug("config.merge.system_config",
                                path=str(path),
                                config_keys=list(sys_config.keys()))
                    merged_config = deep_merge(merged_config, sys_config)
                    
        elif not merged_config and app_config_path:  # Only look for default config if app config is provided
            default_paths = [
                Path("config/system_config.yml"),
                Path("../config/system_config.yml"),
                project_root / "config" / "system_config.yml"
            ]
            logger.info("config.paths.search", 
                cwd=str(Path.cwd()),
                root_dir=str(project_root),
                sys_paths=[str(p) for p in default_paths],
                config_path=app_config_path
            )
            for path in default_paths:
                if path.exists():
                    with open(path) as f:
                        sys_config = yaml.safe_load(f) or {}
                        logger.debug("config.merge.default_system",
                                    path=str(path),
                                    config_keys=list(sys_config.keys()))
                        merged_config = deep_merge(merged_config, sys_config)
                    break
                    
        final_config = deep_merge(merged_config, app_config)
        
        # Ensure minimal config structure exists
        if app_config_path and 'llm_config' not in final_config:
            logger.warning("config.no_llm_config_found",
                         final_keys=list(final_config.keys()))
            final_config['llm_config'] = {}
            
        # Ensure orchestration is enabled
        if 'orchestration' not in final_config:
            final_config['orchestration'] = {'enabled': True}
        else:
            final_config['orchestration']['enabled'] = True
            
        return final_config

    except Exception as e:
        logger.error("config.load_failed", error=str(e))
        if app_config_path:  # Only raise if a config file was explicitly requested
            raise
        return {}

def run_workflow(project_path: Optional[str], intent_desc: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a team-based workflow with the provided configuration"""
    try:
        logger.info("workflow.starting",
                   project_path=project_path,
                   intent_keys=list(intent_desc.keys()) if intent_desc else {},
                   config_keys=list(config.keys()) if config else {})
                
        # Create orchestrator and execute workflow
        orchestrator = Orchestrator(config)
        
        # Initialize workflow configuration and context
        prepared_config, context = orchestrator.initialize_workflow(
            project_path=project_path,
            intent_desc=intent_desc,
            config=config
        )
         
        # Get entry team from config or use default
        entry_team = prepared_config.get("orchestration", {}).get("entry_team", "discovery")
         
        # Execute workflow
        result = orchestrator.execute_workflow(
            entry_team=entry_team,
            context=context
        )
         
        logger.info("workflow.completed",
                   workflow_id=result.get("workflow_run_id", "unknown"),
                   status=result.get("status", "unknown"))
                 
        return result
            
    except Exception as e:
        error_msg = str(e)
        logger.error("workflow.failed", error=error_msg, project_path=project_path)
        return {
            "status": "error",
            "error": error_msg,
            "project_path": project_path,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

def send_workflow_request(host: str, port: int, project_path: str, intent_desc: Dict[str, Any],
                          app_config: Optional[Dict[str, Any]] = None,
                          system_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Send workflow request to server and return the response.
    
    Args:
        host: Server hostname or IP address
        port: Server port number
        project_path: Path to the project
        intent_desc: Intent description dictionary
        app_config: Application configuration (optional)
        system_config: System configuration (optional)
        
    Returns:
        Response data from the server
    """
    url = f"http://{host}:{port}/api/v1/workflow"
    
    # Prepare request data
    request_data = {
        "project_path": project_path,
        "intent": intent_desc,
        "app_config": app_config,
        "system_config": system_config
    }
    
    # Log request details
    logger.info("client.sending_request",
               url=url,
               project_path=project_path,
               has_intent=bool(intent_desc),
               has_app_config=bool(app_config),
               has_system_config=bool(system_config))
    
    # Send request
    try:
        response = requests.post(url, json=request_data)
        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()
        
        logger.info("client.request_success",
                  workflow_id=result.get("workflow_id"),
                  status=result.get("status"))
                  
        return result
    except requests.RequestException as e:
        logger.error("client.request_failed", error=str(e))
        return {
            "status": "error",
            "error": f"Request failed: {str(e)}",
            "workflow_id": None
        }

def get_workflow_status(host: str, port: int, workflow_id: str) -> Dict[str, Any]:
    """
    Get status of a workflow from the server.
    
    Args:
        host: Server hostname or IP address
        port: Server port number
        workflow_id: ID of the workflow to check
        
    Returns:
        Status data from the server
    """
    url = f"http://{host}:{port}/api/v1/workflow/{workflow_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        
        logger.debug("client.status_check",
                   workflow_id=workflow_id,
                   status=result.get("status"))
                   
        return result
    except requests.RequestException as e:
        logger.error("client.status_check_failed",
                   workflow_id=workflow_id,
                   error=str(e))
        return {
            "status": "error",
            "error": f"Status check failed: {str(e)}"
        }

def poll_workflow_status(host: str, port: int, workflow_id: str, poll_interval: int = 5, max_polls: int = 60) -> Dict[str, Any]:
    """
    Poll workflow status until completion or timeout.
    
    Args:
        host: Server hostname or IP address
        port: Server port number
        workflow_id: ID of the workflow to check
        poll_interval: Seconds between status checks
        max_polls: Maximum number of polls before timeout
        
    Returns:
        Final status from the server
    """
    for polls in range(max_polls):
        result = get_workflow_status(host, port, workflow_id)
        if result.get("status") in ["success", "error", "complete", "failed"]:
            return result
        time.sleep(poll_interval)
    return {"status": "timeout", "error": f"Polling timed out after {max_polls} attempts", "workflow_id": workflow_id}

def main():
    parser = argparse.ArgumentParser(description="Streamlined team-based workflow runner and API service")
    parser.add_argument("mode", type=str, nargs="?", choices=["workflow", "service", "client"],
                        default="service", help="Run mode (workflow, service, or client)")
    parser.add_argument("-P", "--port", type=int, default=8000, help="Port number for API service mode or client communication")
    
    # Config parameters
    parser.add_argument("--config", help="Path to application config file")
    parser.add_argument("--system-configs", nargs="+", help="Optional system config files in merge order")
    
    # Workflow parameters
    parser.add_argument("--project-path", help="Path to the project (optional if defined in config)")
    parser.add_argument("--intent-file", help="Path to intent JSON file (optional if intent defined in config)")
    
    # Client parameters
    parser.add_argument("--host", default="localhost", help="Host for client mode")
    parser.add_argument("--poll", action="store_true", help="Poll for workflow completion in client mode")
    parser.add_argument("--poll-interval", type=int, default=5, help="Seconds between status checks in client mode")
    parser.add_argument("--max-polls", type=int, default=60, help="Maximum number of status checks in client mode")
    
    parser.add_argument(
        "--log",
        type=LogMode,
        choices=list(LogMode),
        default=LogMode.NORMAL,
        help="Logging level"
    )

    args = parser.parse_args()

    # Service mode handling
    if args.mode == "service":
        # Create FastAPI app with empty default config
        config = load_configs(args.config, args.system_configs)
        app = create_app(default_config=config)
        print(f"Service mode enabled, running on port {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return
    
    # Client mode handling
    elif args.mode == "client":
        # Load and merge configurations
        config = load_configs(args.config, args.system_configs)
        
        # Check if project path is available
        if not args.project_path and not config.get('project', {}).get('path'):
            parser.error("--project-path is required when not defined in config")
            
        # Use project path from args or config
        project_path = args.project_path or config.get('project', {}).get('path')
        
        # Get intent from file or config
        intent_desc = {}
        
        if args.intent_file:
            try:
                with open(args.intent_file) as f:
                    if args.intent_file.endswith('.json'):
                        intent_desc = json.load(f)
                    else:
                        intent_desc = yaml.safe_load(f)
            except Exception as e:
                parser.error(f"Failed to load intent file: {str(e)}")
        elif 'intent' in config:
            intent_desc = config.get('intent', {})
            logger.info("client.using_intent_from_config",
                        intent_keys=list(intent_desc.keys()) if intent_desc else {})
        else:
            parser.error("--intent-file is required when intent is not defined in config")
            
        # Send workflow request to server
        result = send_workflow_request(
            host=args.host,
            port=args.port,
            project_path=project_path,
            intent_desc=intent_desc,
            app_config=config
        )
        
        # Check result and display
        if result.get("status") == "error":
            print(f"Error: {result.get('error')}")
            sys.exit(1)
            
        workflow_id = result.get("workflow_id")
        print(f"Workflow submitted successfully. Workflow ID: {workflow_id}")
        print(f"Initial status: {result.get('status')}")
        
        # Poll for completion if requested
        if args.poll and workflow_id:
            print(f"Polling for completion every {args.poll_interval} seconds (max {args.max_polls} polls)...")
            status = poll_workflow_status(args.host, args.port, workflow_id, args.poll_interval, args.max_polls)
            print(f"Final status: {status.get('status', 'unknown')}")
            if status.get("status") != "success":
                sys.exit(1)
        sys.exit(0)

    # Load and merge configurations first to potentially get project path and intent
    config = load_configs(args.config, args.system_configs)
    
    # Check if project path is available in config when not provided as argument
    if not args.project_path and not config.get('project', {}).get('path'):
        parser.error("--project-path is required when not defined in config")
    
    # Get intent from file or config
    intent_desc = {}
    
    if args.intent_file:
        try:
            with open(args.intent_file) as f:
                if args.intent_file.endswith('.json'):
                    intent_desc = json.load(f)
                else:
                    intent_desc = yaml.safe_load(f)
        except Exception as e:
            parser.error(f"Failed to load intent file: {str(e)}")
    elif 'intent' in config:
        intent_desc = config.get('intent', {})
        logger.info("workflow.using_intent_from_config",
                    intent_keys=list(intent_desc.keys()) if intent_desc else {})
    else:
        parser.error("--intent-file is required when intent is not defined in config")

    # Run workflow
    result = run_workflow(
        project_path=args.project_path,
        intent_desc=intent_desc,
        config=config
    )

    if result.get("status") != "success":
        sys.exit(1)

if __name__ == "__main__":
    main()