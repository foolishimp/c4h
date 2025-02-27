#!/usr/bin/env python3
"""
Streamlined Prefect runner focusing exclusively on workflow execution and API service.
Path: c4h_services/examples/prefect_runner.py
"""

from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import uvicorn
import structlog
import argparse
from enum import Enum
import yaml
from typing import Dict, Any, Optional, List
import json

# Import API components
from c4h_services.src.api.service import create_app

# Import workflow components
from c4h_agents.config import deep_merge
from c4h_services.src.intent.impl.prefect.workflows import run_basic_workflow

logger = structlog.get_logger()

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
                root_dir / "config" / "system_config.yml"
            ]
            logger.info("config.paths.search", 
                cwd=str(Path.cwd()),
                root_dir=str(root_dir),
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
            
        return final_config

    except Exception as e:
        logger.error("config.load_failed", error=str(e))
        if app_config_path:  # Only raise if a config file was explicitly requested
            raise
        return {}

def run_workflow(project_path: Optional[str], intent_desc: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a workflow with the provided configuration"""
    try:
        # Get project path from config if not provided as argument
        if not project_path:
            project_path = config.get('project', {}).get('path')
            if project_path:
                logger.info("workflow.using_project_from_config", project_path=project_path)
            else:
                raise ValueError("No project path specified in arguments or config")
                
        logger.info("workflow.starting",
                   project_path=project_path,
                   intent_keys=list(intent_desc.keys()) if intent_desc else {},
                   config_keys=list(config.keys()) if config else {})
                
        # Process project path
        if 'project' not in config:
            config['project'] = {}
        config['project']['path'] = project_path
                
        # Execute workflow using Prefect's run_basic_workflow
        result = run_basic_workflow(
            project_path=Path(project_path),
            intent_desc=intent_desc,
            config=config
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
            "timestamp": None
        }

def format_output(result: Dict[str, Any]) -> None:
    """Format workflow results for display"""
    print("\n=== Workflow Results ===\n")
    
    def print_value(value: Any, indent: int = 0) -> None:
        spaces = " " * indent
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ['stages', 'metrics']:
                    print(f"{spaces}{k}: {{...}}")  # Summarize large nested structures
                else:
                    print(f"{spaces}{k}:")
                    print_value(v, indent + 4)
        elif isinstance(value, list):
            if len(value) > 3:
                print(f"{spaces}[{len(value)} items]")
            else:
                for item in value:
                    print_value(item, indent + 4)
        else:
            print(f"{spaces}{value}")
    
    try:
        status = result.get("status", "unknown")
        print(f"Status: {status}")
        
        if status == "error":
            print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"Workflow ID: {result.get('workflow_run_id', 'unknown')}")
            print(f"Project: {result.get('project_path', 'unknown')}")
            print(f"Changes: {len(result.get('changes', []))} modifications")
            
            if 'execution_metadata' in result:
                print("\nExecution Information:")
                metadata = result['execution_metadata']
                print(f"  Steps: {metadata.get('step_sequence', 0)}")
                print(f"  Agents: {len(metadata.get('agent_sequence', []))}")
                
        print("\nComplete result structure:")
        print_value(result, 2)
            
    except Exception as e:
        logger.error("output.format_failed", error=str(e))
        print(f"Error formatting output: {str(e)}")
        print("Raw data:")
        print(str(result))

def main():
    parser = argparse.ArgumentParser(description="Streamlined Prefect runner for workflows and API service")
    parser.add_argument("mode", type=str, nargs="?", choices=["workflow", "service"], 
                       default="service", help="Run mode (workflow or service)")
    parser.add_argument("-P", "--port", type=int, default=8000, help="Port number for API service mode")
    
    # Config parameters
    parser.add_argument("--config", help="Path to application config file")
    parser.add_argument("--system-configs", nargs="+", help="Optional system config files in merge order")
    
    # Workflow parameters
    parser.add_argument("--project-path", help="Path to the project (optional if defined in config)")
    parser.add_argument("--intent-file", help="Path to intent JSON file (optional if intent defined in config)")
    
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

##    format_output(result)

    if result.get("status") != "success":
        sys.exit(1)

if __name__ == "__main__":
    main()