#!/usr/bin/env python3
"""
Streamlined runner focused exclusively on the jobs API service and client interactions.
Path: c4h_services/src/bootstrap/prefect_runner.py # cite: 1175
REMOVED legacy /workflow client mode. # cite: 1176
"""

# Standard Library Imports
import sys
import os
import uuid
import time
import json
import yaml
import argparse
import logging
from pathlib import Path
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# External Library Imports
try:
    import requests
except ImportError:
    print("Error: 'requests' library not found. Please install it: pip install requests")
    sys.exit(1)
try:
    import uvicorn
except ImportError:
    uvicorn = None

# Add the project root # cite: 1177
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.append(str(project_root))

# Logger Setup (Fallback included)
try:
    from c4h_services.src.utils.logging import get_logger
    logger = get_logger()
    USING_C4H_LOGGER = True
except ImportError:
    logger = logging.getLogger("prefect_runner_fallback")
    log_level = logging.INFO
    if not logger.hasHandlers():
         handler = logging.StreamHandler()
         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # cite: 1178
         handler.setFormatter(formatter)
         logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.warning("Could not import C4H logger, using basic fallback logger.")
    USING_C4H_LOGGER = False

# --- LogMode Enum ---
class LogMode(str, Enum): # cite: 1179
    DEBUG = "debug"
    NORMAL = "normal"

# --- Config Loading ---
def load_single_config(config_path: str) -> Dict[str, Any]: # cite: 1180
    """Loads a single YAML configuration file with robust logging."""
    path_obj = Path(config_path)
    try:
        if not path_obj.exists():
            log_message = f"Configuration file not found: {str(path_obj)}"
            logger.error(f"config.load.file_not_found: {log_message}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(path_obj) as f:
            config = yaml.safe_load(f) or {}
        log_message = f"Successfully loaded config from {str(path_obj)}. Keys: {list(config.keys())}"
        logger.info(f"config.load.success: {log_message}")
        return config
    except yaml.YAMLError as e:
        log_message = f"Invalid YAML in config file: {str(path_obj)}. Error: {str(e)}"
        logger.error(f"config.load.yaml_error: {log_message}") # cite: 1181
        raise ValueError(f"Invalid YAML in config file: {e}")
    except Exception as e:
        log_message = f"Failed to load config file: {str(path_obj)}. Error: {str(e)}" # cite: 1182
        logger.error(f"config.load.failed: {log_message}") # cite: 1182
        raise

# --- API Client Functions ---
def send_job_request(host: str, port: int, config_payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]: # cite: 1183
    """ Send job request (expects {"configs": [...]}) to server and return the response. """
    url = f"http://{host}:{port}/api/v1/jobs"
    # Logging request details
    log_payload_summary = "Payload structure seems valid for MultiConfigJobRequest"
    if isinstance(config_payload, dict) and 'configs' in config_payload and isinstance(config_payload['configs'], list):
         list_keys = [list(d.keys())[0] for d in config_payload['configs'] if isinstance(d, dict) and d]
         logger.info(f"client.sending_job_request (MultiConfig format): url={url} config_count={len(config_payload['configs'])} list_keys={list_keys}")
    else:
        # Log if the format is unexpected now
        logger.warning(f"client.sending_job_request (Unexpected format): url={url} payload_type={type(config_payload).__name__}")
        log_payload_summary = "Payload structure might not match expected MultiConfigJobRequest"

    # Send request # cite: 1184
    try:
        logger.debug(f"Sending payload: {json.dumps(config_payload, indent=2)}") # Log full payload on debug
        response = requests.post(url, json=config_payload)
        logger.debug(f"client.job_response_received: status_code={response.status_code} content_length={len(response.content)}")
        response.raise_for_status()
        result = response.json()
        logger.info(f"client.job_request_success: job_id={result.get('job_id')} status={result.get('status')}")
        return result
    except requests.HTTPError as e:
        err_msg = str(e)
        error_detail_text = ""
        try:
            error_data = e.response.json() # cite: 1185
            error_detail_text = json.dumps(error_data) # Get full detail
            if "detail" in error_data:
                # Handle potential nested detail
                 err_detail = error_data['detail']
                 if isinstance(err_detail, list) and err_detail: # Handle FastAPI validation errors
                     err_msg = f"{e.response.status_code}: {json.dumps(err_detail)}"
                 elif isinstance(err_detail, str):
                      err_msg = f"{e.response.status_code}: {err_detail}"
                 else:
                      err_msg = f"{e.response.status_code}: {json.dumps(err_detail)}"

        except json.JSONDecodeError:
            error_detail_text = e.response.text # Get raw text if not JSON
            err_msg = f"{e.response.status_code}: {error_detail_text[:500]}..." # Truncate long raw errors

        logger.error(f"client.job_request_http_error: status_code={e.response.status_code if hasattr(e, 'response') else 'unknown'} error='{err_msg}' raw_response='{error_detail_text[:500]}'")
        # Return error structure consistent with API model if possible
        return { "status": "error", "error": err_msg, "job_id": None, "detail": error_detail_text}
    except requests.RequestException as e:
        logger.error(f"client.job_request_failed: error='{str(e)}'")
        return { "status": "error", "error": f"Request failed: {str(e)}", "job_id": None } # cite: 1186

# --- get_status and poll_status ---
def get_status(host: str, port: int, url_path: str, id_value: str) -> Dict[str, Any]: # cite: 1187
    """ Get status of a job from the server. (Path is now always 'jobs') """
    url = f"http://{host}:{port}/api/v1/{url_path}/{id_value}" # url_path will be 'jobs'
    try:
        logger.debug(f"client.status_checking: id={id_value} url={url}")
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"client.status_check: id={id_value} status={result.get('status')}")
        return result
    except requests.HTTPError as e:
        err_msg = str(e)
        try:
            error_data = e.response.json() # cite: 1188
            if "detail" in error_data: err_msg = f"{e.response.status_code}: {error_data['detail']}"
        except: pass
        logger.error(f"client.status_http_error: id={id_value} status_code={e.response.status_code if hasattr(e, 'response') else 'unknown'} error='{err_msg}'")
        return { "status": "error", "error": f"HTTP error: {err_msg}" }
    except requests.RequestException as e:
        logger.error(f"client.status_check_failed: id={id_value} error='{str(e)}'")
        return { "status": "error", "error": f"Status check failed: {str(e)}" } # cite: 1189

def get_job_status(host: str, port: int, job_id: str) -> Dict[str, Any]:
    """Get status of a job from the server."""
    return get_status(host, port, "jobs", job_id) # Hardcode path to 'jobs'

def poll_status(host: str, port: int, url_path: str, id_value: str, # cite: 1190
               poll_interval: int = 5, max_polls: int = 60) -> Dict[str, Any]:
    """ Poll status until completion or timeout. (Path is now always 'jobs') """
    logger.info(f"client.polling_status: id={id_value} url_path={url_path} poll_interval={poll_interval} max_polls={max_polls}")
    terminal_statuses = ["success", "error", "complete", "failed"] # Statuses from JobStatus model
    last_result = {}
    for poll_count in range(max_polls):
        last_result = get_status(host, port, url_path, id_value)
        status = last_result.get("status")
        if status in terminal_statuses:
            logger.info(f"client.polling_complete: id={id_value} status={status} polls={poll_count+1}")
            return last_result
        if poll_count % 5 == 0 or poll_count < 2: # cite: 1191
            logger.info(f"client.polling: id={id_value} status={status} poll_count={poll_count+1} max_polls={max_polls}")
        time.sleep(poll_interval)
    logger.warning(f"client.polling_timeout: id={id_value} polls={max_polls} last_status={last_result.get('status')}")
    return { "status": "timeout", "error": f"Polling timed out after {max_polls} attempts", "job_id": id_value } # Changed key 'id' to 'job_id'

def poll_job_status(host: str, port: int, job_id: str, poll_interval: int = 5, max_polls: int = 60) -> Dict[str, Any]:
    """Poll job status until completion or timeout."""
    return poll_status(host, port, "jobs", job_id, poll_interval, max_polls) # Hardcode path to 'jobs'


# --- REMOVED build_job_config function ---


# --- Mode Handlers (REVISED handle_jobs_mode, REMOVED handle_client_mode) ---
def handle_jobs_mode(args: argparse.Namespace) -> None: # cite: 1193
    """Handle jobs mode for the updated multi-config jobs API"""
    try:
        if not args.config:
            print("Error: --config file path is required for jobs mode.")
            sys.exit(1) # cite: 1194

        # Load the single config file specified by the user
        loaded_config = load_single_config(args.config)

        # Construct the list payload by splitting the loaded config
        config_fragments = []
        if 'workorder' in loaded_config:
            config_fragments.append({"workorder": loaded_config['workorder']})
            logger.debug("jobs_mode.added_workorder_fragment")
        if 'team' in loaded_config:
            config_fragments.append({"team": loaded_config['team']})
            logger.debug("jobs_mode.added_team_fragment")
        if 'runtime' in loaded_config:
             config_fragments.append({"runtime": loaded_config['runtime']})
             logger.debug("jobs_mode.added_runtime_fragment")

        # Check if any main fragments were added
        if not config_fragments:
             logger.warning("jobs_mode.no_fragments_extracted", config_keys=list(loaded_config.keys()))
             # Fallback: send the whole config as one item if no standard keys found
             config_fragments = [loaded_config]

        # Handle CLI overrides for project_path and lineage/stage
        override_config = {} # cite: 1195
        if args.project_path:
            if 'workorder' not in override_config: override_config['workorder'] = {}
            if 'project' not in override_config['workorder']: override_config['workorder']['project'] = {}
            override_config['workorder']['project']['path'] = args.project_path
            logger.info(f"jobs_mode.adding_cli_override: project_path={args.project_path}")

        if args.lineage_file or args.stage or not args.keep_runid:
            if 'runtime' not in override_config: override_config['runtime'] = {}
            if 'runtime' not in override_config['runtime']: override_config['runtime']['runtime'] = {} # cite: 1196
            if args.lineage_file: override_config['runtime']['runtime']['lineage_file'] = args.lineage_file
            if args.stage: override_config['runtime']['runtime']['stage'] = args.stage
            if not args.keep_runid: override_config['runtime']['runtime']['keep_runid'] = False
            logger.info(f"jobs_mode.adding_cli_override: lineage/stage args present")

        # Prepend the override config if it contains anything
        if override_config: # cite: 1197
            config_fragments.insert(0, override_config)
            logger.info("jobs_mode.prepended_override_config")

        # Wrap the list of fragments in a dictionary with the key "configs"
        final_payload = {"configs": config_fragments}
        logger.debug("jobs_mode.wrapped_fragments_in_configs_key")


        # Send the final payload (which is now a dictionary)
        result = send_job_request(
            host=args.host,
            port=args.port,
            config_payload=final_payload # Send the dictionary containing the list
        )

        # Result handling # cite: 1198
        if result.get("status") == "error":
            error_detail = result.get('error', 'Unknown error')
            print(f"Error: {error_detail}")
            # Try to print detailed error if available
            raw_detail = result.get('detail', '')
            if raw_detail:
                print(f"Raw Error Detail: {raw_detail}")
            sys.exit(1)
        job_id = result.get("job_id")
        if not job_id:
            print("Error: No job ID returned from server")
            sys.exit(1)
        print(f"Job submitted successfully. Job ID: {job_id}") # cite: 1199
        print(f"Initial status: {result.get('status')}")

        # Polling
        if args.poll and job_id:
            print(f"Polling for completion every {args.poll_interval} seconds (max {args.max_polls} polls)...")
            status_result = poll_job_status(args.host, args.port, job_id, args.poll_interval, args.max_polls)
            print(f"\nFinal status: {status_result.get('status', 'unknown')}") # cite: 1200
            if status_result.get("changes"): print(f"Changes: {json.dumps(status_result.get('changes'), indent=2)}") # Safe access
            if status_result.get("error"): print(f"\nError: {status_result.get('error')}")
            if status_result.get("status") not in ["success", "complete"]: sys.exit(1)

        sys.exit(0)
    # Error handling
    except FileNotFoundError as e: print(f"Error: {str(e)}"); sys.exit(1)
    except ValueError as e: print(f"Error: {str(e)}"); sys.exit(1)
    except Exception as e: print(f"Unexpected error in jobs mode: {str(e)}"); sys.exit(1) # cite: 1201


# --- handle_service_mode (Unchanged) ---
def handle_service_mode(args: argparse.Namespace) -> None:
    """Handle service mode to run API server"""
    if uvicorn is None:
        print("Error: 'uvicorn' is required to run in service mode. Install it: pip install uvicorn[standard]")
        sys.exit(1)
    try:
        # Ensure service components are importable from the execution context
        from c4h_services.src.api.service import create_app
        # Load config only if --config is specified
        config = load_single_config(args.config) if args.config else {}
        app = create_app(default_config=config) # Pass loaded config or empty dict
        print(f"Service mode enabled, running on http://0.0.0.0:{args.port}") # cite: 1202
        uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False) # reload=False recommended
    except ImportError as e:
        print(f"Error importing service components: {e}. Make sure packages are installed correctly and PYTHONPATH includes the project.")
        sys.exit(1)
    except Exception as e:
        print(f"Service startup failed: {str(e)}"); sys.exit(1) # cite: 1203

# --- REMOVED handle_client_mode function ---

# --- Main Execution (REVISED arguments and mode handling) ---
def main():
    parser = argparse.ArgumentParser(description="API service runner and jobs client for C4H operations")
    # REMOVED 'client' choice
    parser.add_argument("mode", type=str, nargs="?", choices=["service", "jobs"], default="service", help="Run mode (service or jobs)")
    parser.add_argument("-P", "--port", type=int, default=5500, help="Port number for API service or client communication (default: 5500)")
    # Clarified --config usage based on mode
    parser.add_argument("--config", help="Path to config file (YAML). Required for 'jobs' mode. Optional base config for 'service' mode.") # cite: 1204
    parser.add_argument("--host", default="localhost", help="Host for jobs mode client (default: localhost)")
    # Kept arguments relevant to 'jobs' mode overrides
    parser.add_argument("--project-path", help="Path to the project (overrides config file if provided, inserted into workorder.project)")
    parser.add_argument("--poll", action="store_true", help="Poll for completion status in jobs mode")
    parser.add_argument("--poll-interval", type=int, default=5, help="Seconds between status checks")
    parser.add_argument("--max-polls", type=int, default=120, help="Maximum number of status checks (default 120 = 10 mins)")
    parser.add_argument("--lineage-file", help="Path to lineage file for workflow continuation (overrides config, inserted into runtime.runtime)")
    parser.add_argument("--stage", choices=["discovery", "solution_designer", "coder"], help="Stage to execute from lineage (overrides config, inserted into runtime.runtime)")
    parser.add_argument("--keep-runid", action=argparse.BooleanOptionalAction, default=True, help="Keep original run ID from lineage file (use --no-keep-runid to generate new, inserted into runtime.runtime)") # cite: 1205
    parser.add_argument("--log", type=LogMode, choices=list(LogMode), default=LogMode.NORMAL, help="Logging level (default: normal)")
    args = parser.parse_args()

    # Configure logging level
    if USING_C4H_LOGGER and hasattr(logger, 'setLevel'):
        log_level_std = logging.DEBUG if args.log == LogMode.DEBUG else logging.INFO
        logger.setLevel(log_level_std)
        logger.info(f"Setting log level to {args.log.value.upper()}")
    elif not USING_C4H_LOGGER:
        log_level_std = logging.DEBUG if args.log == LogMode.DEBUG else logging.INFO # cite: 1206
        logger.setLevel(log_level_std)
        # Also set level for root logger if using fallback to capture deps logging
        logging.getLogger().setLevel(log_level_std)
        logger.info(f"(Fallback Logger) Setting log level to {args.log.value.upper()}")

    # Execute selected mode
    try:
        if args.mode == "service":
             # Check if --config was provided for service mode (recommended but optional)
             if not args.config:
                 logger.warning("Service mode started without --config flag. Using empty default config. Base system config might not be loaded initially.")
             handle_service_mode(args)
        # REMOVED client mode handling
        elif args.mode == "jobs":
             # Check if --config was provided for jobs mode (required)
             if not args.config:
                 parser.error("--config argument is required for 'jobs' mode.")
             handle_jobs_mode(args)
        else: print(f"Error: Unsupported mode: {args.mode}"); sys.exit(1) # cite: 1207
    except Exception as e:
        print(f"Unexpected error in main: {str(e)}"); sys.exit(1) # cite: 1208

if __name__ == "__main__":
    main()