#!/usr/bin/env python3
"""
Streamlined runner focused exclusively on the jobs API service and client interactions.
Includes 'jobs' mode for standard multi-config job submission and 'apply_diff' mode
for directly applying raw diff content via the coder stage.
Path: c4h_services/src/bootstrap/prefect_runner.py
"""

# Standard Library Imports
import sys
import os
import uuid
import time
import json
import yaml
import argparse
import logging # Keep standard logging for fallback
from pathlib import Path
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone # Added timezone
import tempfile # Added for apply_diff mode

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

# Add the project root
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.append(str(project_root))

# Logger Setup (Fallback included)
try:
    # Import get_logger from the correct path
    from c4h_agents.utils.logging import get_logger, initialize_logging_config
    logger = get_logger() # Initialize logger early
    USING_C4H_LOGGER = True
except ImportError:
    # --- Fallback Logger Setup ---
    logger = logging.getLogger("prefect_runner_fallback")
    log_level = logging.INFO # Default level for fallback
    if not logger.hasHandlers():
         handler = logging.StreamHandler()
         # Use a format string compatible with standard logging
         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         handler.setFormatter(formatter)
         logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.warning("Could not import C4H logger, using basic fallback logger.")
    USING_C4H_LOGGER = False
    # Define initialize_logging_config as no-op if import failed
    def initialize_logging_config(config): pass
    # --- End Fallback Logger Setup ---


# --- LogMode Enum ---
class LogMode(str, Enum):
    DEBUG = "debug"
    NORMAL = "normal"

# --- Config Loading ---
def load_single_config(config_path: str) -> Dict[str, Any]:
    """Loads a single YAML configuration file with robust logging."""
    path_obj = Path(config_path)
    try:
        if not path_obj.exists():
            log_message = f"Configuration file not found: {str(path_obj)}"
            # Use appropriate logging style based on logger type
            if USING_C4H_LOGGER:
                logger.error("config.load.file_not_found", path=str(path_obj), error=log_message)
            else:
                logger.error(f"config.load.file_not_found: path={str(path_obj)} error='{log_message}'") # f-string for fallback
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path_obj) as f:
            config = yaml.safe_load(f) or {}
        # Ensure config is a dict even if YAML is empty
        if config is None:
             config = {}
        log_message = f"Successfully loaded config from {str(path_obj)}. Keys: {list(config.keys())}"
        # Use appropriate logging style based on logger type
        if USING_C4H_LOGGER:
            logger.info("config.load.success", path=str(path_obj), keys=list(config.keys()))
        else:
            logger.info(f"config.load.success: path={str(path_obj)} keys={list(config.keys())}") # f-string for fallback
        return config
    except yaml.YAMLError as e:
        log_message = f"Invalid YAML in config file: {str(path_obj)}. Error: {str(e)}"
        # Use appropriate logging style based on logger type
        if USING_C4H_LOGGER:
            logger.error("config.load.yaml_error", path=str(path_obj), error=str(e))
        else:
            logger.error(f"config.load.yaml_error: path={str(path_obj)} error='{str(e)}'") # f-string for fallback
        raise ValueError(f"Invalid YAML in config file: {e}")
    except Exception as e:
        log_message = f"Failed to load config file: {str(path_obj)}. Error: {str(e)}"
        # Use appropriate logging style based on logger type
        if USING_C4H_LOGGER:
            logger.error("config.load.failed", path=str(path_obj), error=str(e))
        else:
            logger.error(f"config.load.failed: path={str(path_obj)} error='{str(e)}'") # f-string for fallback
        raise

# --- API Client Functions ---
# Assuming these functions use logger correctly (key=value for structlog)
def send_job_request(host: str, port: int, config_payload: Dict[str, Any]) -> Dict[str, Any]:
    """ Send job request (expects {"configs": [...]}) to server and return the response. """
    if not isinstance(config_payload, dict) or 'configs' not in config_payload or not isinstance(config_payload.get('configs'), list):
         error_msg = "Invalid config_payload structure: Must be a dict with a 'configs' list."
         logger.error("client.send_job_request.invalid_payload", error=error_msg)
         return {"status": "error", "error": error_msg, "job_id": None}

    url = f"http://{host}:{port}/api/v1/jobs"
    list_keys = [list(d.keys())[0] for d in config_payload['configs'] if isinstance(d, dict) and d]
    logger.info("client.sending_job_request", type="MultiConfig", url=url, config_count=len(config_payload['configs']), list_keys=list_keys)

    try:
        response = requests.post(url, json=config_payload)
        logger.debug("client.job_response_received", status_code=response.status_code, content_length=len(response.content))
        response.raise_for_status()
        result = response.json()
        logger.info("client.job_request_success", job_id=result.get('job_id'), status=result.get('status'))
        return result
    except requests.HTTPError as e:
        err_msg = str(e)
        error_detail_text = ""
        try:
            error_data = e.response.json()
            error_detail_text = json.dumps(error_data)
            if "detail" in error_data:
                 err_detail = error_data['detail']
                 if isinstance(err_detail, list) and err_detail:
                     err_msg = f"{e.response.status_code}: {json.dumps(err_detail)}"
                 elif isinstance(err_detail, str):
                      err_msg = f"{e.response.status_code}: {err_detail}"
                 else:
                      err_msg = f"{e.response.status_code}: {json.dumps(err_detail)}"
        except json.JSONDecodeError:
            error_detail_text = e.response.text
            err_msg = f"{e.response.status_code}: {error_detail_text[:500]}..."
        logger.error("client.job_request_http_error", status_code=e.response.status_code if hasattr(e, 'response') else 'unknown', error=err_msg, raw_response=error_detail_text[:500])
        return { "status": "error", "error": err_msg, "job_id": None, "detail": error_detail_text}
    except requests.RequestException as e:
        logger.error("client.job_request_failed", error=str(e))
        return { "status": "error", "error": f"Request failed: {str(e)}", "job_id": None }

def get_status(host: str, port: int, url_path: str, id_value: str) -> Dict[str, Any]:
    """ Get status of a job from the server. (Path is now always 'jobs') """
    url = f"http://{host}:{port}/api/v1/{url_path}/{id_value}"
    try:
        logger.debug("client.status_checking", id=id_value, url=url)
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        logger.debug("client.status_check", id=id_value, status=result.get('status'))
        return result
    except requests.HTTPError as e:
        err_msg = str(e)
        try:
            error_data = e.response.json()
            if "detail" in error_data: err_msg = f"{e.response.status_code}: {error_data['detail']}"
        except: pass
        logger.error("client.status_http_error", id=id_value, status_code=e.response.status_code if hasattr(e, 'response') else 'unknown', error=err_msg)
        return { "status": "error", "error": f"HTTP error: {err_msg}" }
    except requests.RequestException as e:
        logger.error("client.status_check_failed", id=id_value, error=str(e))
        return { "status": "error", "error": f"Status check failed: {str(e)}" }

def get_job_status(host: str, port: int, job_id: str) -> Dict[str, Any]:
    """Get status of a job from the server."""
    return get_status(host, port, "jobs", job_id)

def poll_status(host: str, port: int, url_path: str, id_value: str,
               poll_interval: int = 5, max_polls: int = 60) -> Dict[str, Any]:
    """ Poll status until completion or timeout. (Path is now always 'jobs') """
    logger.info("client.polling_status", id=id_value, url_path=url_path, poll_interval=poll_interval, max_polls=max_polls)
    terminal_statuses = ["success", "error", "complete", "failed"]
    last_result = {}
    for poll_count in range(max_polls):
        last_result = get_status(host, port, url_path, id_value)
        status = last_result.get("status")
        if status in terminal_statuses:
            logger.info("client.polling_complete", id=id_value, status=status, polls=poll_count+1)
            return last_result
        if poll_count % 5 == 0 or poll_count < 2:
            logger.info("client.polling", id=id_value, status=status, poll_count=poll_count+1, max_polls=max_polls)
        time.sleep(poll_interval)
    logger.warning("client.polling_timeout", id=id_value, polls=max_polls, last_status=last_result.get('status'))
    return { "status": "timeout", "error": f"Polling timed out after {max_polls} attempts", "job_id": id_value }

def poll_job_status(host: str, port: int, job_id: str, poll_interval: int = 5, max_polls: int = 60) -> Dict[str, Any]:
    """Poll job status until completion or timeout."""
    return poll_status(host, port, "jobs", job_id, poll_interval, max_polls)


# --- Mode Handlers ---
def handle_jobs_mode(args: argparse.Namespace) -> None:
    try:
        if not args.config:
            print("Error: --config file path is required for jobs mode.")
            sys.exit(1)

        # Load the single config file specified by the user
        loaded_config = load_single_config(args.config)
        # Initialize logging with the specific config for this job run
        if USING_C4H_LOGGER:
             initialize_logging_config(loaded_config)
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
        # Add other top-level keys if needed (e.g., llm_config, backup, logging)
        for key in ['llm_config', 'backup', 'logging', 'orchestration']:
             if key in loaded_config and key not in ['workorder', 'team', 'runtime']:
                  config_fragments.append({key: loaded_config[key]})
                  logger.debug(f"jobs_mode.added_{key}_fragment")

        if not config_fragments:
             logger.warning("jobs_mode.no_fragments_extracted", config_keys=list(loaded_config.keys()))
             config_fragments = [loaded_config]

        # Handle CLI overrides for project_path and lineage/stage
        override_config = {}
        if args.project_path:
            if 'workorder' not in override_config: override_config['workorder'] = {}
            if 'project' not in override_config['workorder']: override_config['workorder']['project'] = {}
            override_config['workorder']['project']['path'] = args.project_path
            logger.info("jobs_mode.adding_cli_override", project_path=args.project_path)

        if args.lineage_file or args.stage or not args.keep_runid:
            if 'runtime' not in override_config: override_config['runtime'] = {}
            if 'runtime' not in override_config['runtime']: override_config['runtime']['runtime'] = {}
            if args.lineage_file: override_config['runtime']['runtime']['lineage_file'] = args.lineage_file
            if args.stage: override_config['runtime']['runtime']['stage'] = args.stage
            if args.keep_runid is False:
                 override_config['runtime']['runtime']['keep_runid'] = False
            logger.info("jobs_mode.adding_cli_override", lineage_file=args.lineage_file, stage=args.stage, keep_runid=args.keep_runid)

        if override_config:
            config_fragments.insert(0, {"cli_overrides": override_config})
            logger.info("jobs_mode.prepended_override_config")

        final_payload = {"configs": config_fragments}
        logger.debug("jobs_mode.payload_constructed")

        result = send_job_request(
            host=args.host, port=args.port, config_payload=final_payload
        )

        if result.get("status") == "error":
            error_detail = result.get('error', 'Unknown error')
            print(f"Error: {error_detail}")
            raw_detail = result.get('detail', '')
            if raw_detail: print(f"Raw Error Detail: {raw_detail}")
            sys.exit(1)

        job_id = result.get("job_id")
        if not job_id:
            print("Error: No job ID returned from server")
            sys.exit(1)
        print(f"Job submitted successfully. Job ID: {job_id}")
        print(f"Initial status: {result.get('status')}")

        if args.poll and job_id:
            print(f"Polling for completion every {args.poll_interval} seconds (max {args.max_polls} polls)...")
            status_result = poll_job_status(args.host, args.port, job_id, args.poll_interval, args.max_polls)
            print(f"\nFinal status: {status_result.get('status', 'unknown')}")
            if status_result.get("changes"): print(f"Changes: {json.dumps(status_result.get('changes'), indent=2)}")
            if status_result.get("error"): print(f"\nError: {status_result.get('error')}")
            if status_result.get("status") not in ["success", "complete"]: sys.exit(1)

        sys.exit(0)
    except FileNotFoundError as e: print(f"Error: {str(e)}"); sys.exit(1)
    except ValueError as e: print(f"Error: {str(e)}"); sys.exit(1)
    except Exception as e: print(f"Unexpected error in jobs mode: {str(e)}"); logger.exception("jobs_mode.unexpected_error"); sys.exit(1)


def handle_service_mode(args: argparse.Namespace) -> None:
    """Handle service mode to run API server"""
    if uvicorn is None:
        print("Error: 'uvicorn' is required to run in service mode. Install it: pip install uvicorn[standard]")
        sys.exit(1)
    try:
        from c4h_services.src.api.service import create_app, system_config_path_default

        config_to_load_path = args.config if args.config else system_config_path_default
        logger.info(f"service_mode: Attempting to load configuration from: {config_to_load_path}")

        try:
             loaded_config = load_single_config(str(config_to_load_path))
             if USING_C4H_LOGGER:
                  initialize_logging_config(loaded_config)
             logger.info(f"service_mode: Configuration loaded successfully from {config_to_load_path}")
        except (FileNotFoundError, ValueError, Exception) as e:
             logger.error(f"service_mode: Failed to load configuration from {config_to_load_path}: {e}. Exiting.")
             print(f"Error: Could not load configuration file '{config_to_load_path}'. Please ensure the file exists and is valid YAML.")
             sys.exit(1)

        app = create_app(config=loaded_config)

        print(f"Service mode enabled, running on http://0.0.0.0:{args.port}")
        logger.info(f"service_mode: Starting Uvicorn server on port {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)

    except ImportError as e:
        import_error_msg = f"Error importing service components: {e}. Make sure packages are installed correctly and PYTHONPATH includes the project root."
        logger.error(f"service_mode: {import_error_msg}")
        print(import_error_msg)
        sys.exit(1)
    except Exception as e:
        startup_error_msg = f"Service startup failed: {str(e)}"
        logger.error(f"service_mode: {startup_error_msg}", exc_info=True)
        print(startup_error_msg)
        sys.exit(1)


def handle_apply_diff_mode(args: argparse.Namespace) -> None:
    temp_lineage_file = None
    try:
        if not args.project_path:
            print("Error: --project-path is required for apply_diff mode.")
            sys.exit(1)
        if not args.diff_file:
            print("Error: --diff-file is required for apply_diff mode.")
            sys.exit(1)

        diff_file_path = Path(args.diff_file)
        if not diff_file_path.is_file():
            print(f"Error: Diff file not found at {args.diff_file}")
            sys.exit(1)

        project_path_abs = str(Path(args.project_path).resolve())
        logger.info("apply_diff_mode.started", project=project_path_abs, diff_file=str(diff_file_path))

        try:
            diff_content = diff_file_path.read_text()
            if "===CHANGE_BEGIN===" not in diff_content:
                 logger.warning("Diff file content does not contain expected '===CHANGE_BEGIN===' marker.", diff_file=str(diff_file_path))
        except Exception as e:
            print(f"Error reading diff file {args.diff_file}: {e}")
            sys.exit(1)

        lineage_content = {
          "event_id": f"apply-diff-event-{str(uuid.uuid4())[:8]}",
          "timestamp": datetime.now(timezone.utc).isoformat(),
          "agent": { "name": "apply_diff_wrapper", "type": "solution_designer" },
          "workflow": { "run_id": f"apply-diff-run-{str(uuid.uuid4())[:8]}",
                        "parent_id": None, "step": None, "execution_path": ["apply_diff_wrapper:manual"] },
          "llm_input": { "note": f"Direct input for coder stage from file: {diff_file_path.name}" },
          "llm_output": { "content": diff_content, "finish_reason": "stop", "model": "manual_input", "usage": None },
          "metrics": None,
          "error": None
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tf:
            json.dump(lineage_content, tf)
            temp_lineage_file = tf.name
        logger.info("apply_diff_mode.temp_lineage_created", path=temp_lineage_file)

        base_config_path = args.config if args.config else "config/system_config.yml"
        base_config = {}
        try:
            if Path(base_config_path).exists():
                base_config = load_single_config(base_config_path)
            else:
                logger.warning(f"Base config file not found at {base_config_path}. Proceeding without base config.", mode="apply_diff")
        except Exception as e:
            logger.warning(f"Failed to load base config {base_config_path}: {e}. Proceeding without base config.", mode="apply_diff")

        if USING_C4H_LOGGER:
             initialize_logging_config(base_config)

        config_fragments = []
        config_fragments.append({
            "runtime": {
                "runtime": {
                    "stage": "coder",
                    "lineage_file": temp_lineage_file,
                    "keep_runid": False
                }
            }
        })
        config_fragments.append({
            "workorder": {
                "project": { "path": project_path_abs },
                "intent": { "description": f"Apply direct diff from {diff_file_path.name}" }
            }
        })
        for key in ['llm_config', 'runtime', 'backup', 'logging', 'orchestration']:
             if key in base_config:
                  config_fragments.append({key: base_config[key]})
                  logger.debug(f"apply_diff_mode.added_{key}_fragment_from_base")

        final_payload = {"configs": config_fragments}

        result = send_job_request(
            host=args.host, port=args.port, config_payload=final_payload
        )

        if result.get("status") == "error":
            error_detail = result.get('error', 'Unknown error')
            print(f"Error: {error_detail}")
            raw_detail = result.get('detail', '')
            if raw_detail: print(f"Raw Error Detail: {raw_detail}")
            sys.exit(1)

        job_id = result.get("job_id")
        if not job_id:
            print("Error: No job ID returned from server")
            sys.exit(1)
        print(f"Apply diff job submitted successfully. Job ID: {job_id}")
        print(f"Initial status: {result.get('status')}")

        if args.poll and job_id:
            print(f"Polling for completion every {args.poll_interval} seconds (max {args.max_polls} polls)...")
            status_result = poll_job_status(args.host, args.port, job_id, args.poll_interval, args.max_polls)
            print(f"\nFinal status: {status_result.get('status', 'unknown')}")
            if status_result.get("changes"): print(f"Changes: {json.dumps(status_result.get('changes'), indent=2)}")
            if status_result.get("error"): print(f"\nError: {status_result.get('error')}")
            if status_result.get("status") not in ["success", "complete"]: sys.exit(1)

        sys.exit(0)

    except Exception as e:
        print(f"Unexpected error in apply_diff mode: {str(e)}")
        if hasattr(logger, 'exception'): logger.exception("apply_diff_mode.unexpected_error")
        sys.exit(1)
    finally:
        if temp_lineage_file and Path(temp_lineage_file).exists():
            try:
                os.remove(temp_lineage_file)
                logger.info("apply_diff_mode.temp_lineage_cleaned", path=temp_lineage_file)
            except Exception as e:
                logger.warning("apply_diff_mode.temp_lineage_cleanup_failed", path=temp_lineage_file, error=str(e))


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="API service runner and jobs client for C4H operations")
    parser.add_argument("mode", type=str, nargs="?", choices=["service", "jobs", "apply_diff"], default="service", help="Run mode (service, jobs, apply_diff)")
    parser.add_argument("-P", "--port", type=int, default=5500, help="Port number for API service or client communication (default: 5500)")
    parser.add_argument("--config", help="Path to config file (YAML). Used by 'jobs' mode. Used as base/override for 'service'/'apply_diff' mode.")
    parser.add_argument("--host", default="localhost", help="Host for jobs/apply_diff client (default: localhost)")
    parser.add_argument("--project-path", help="Path to the project. Required for 'apply_diff' mode. Overrides config for 'jobs' mode.")
    parser.add_argument("--diff-file", help="Path to file containing raw diff content. Required for 'apply_diff' mode.")
    parser.add_argument("--poll", action="store_true", help="Poll for completion status in jobs/apply_diff mode")
    parser.add_argument("--poll-interval", type=int, default=5, help="Seconds between status checks")
    parser.add_argument("--max-polls", type=int, default=120, help="Maximum number of status checks (default 120 = 10 mins)")
    parser.add_argument("--lineage-file", help="('jobs' mode only) Path to lineage file for workflow continuation.")
    parser.add_argument("--stage", choices=["discovery", "solution_designer", "coder"], help="('jobs' mode only) Stage to execute from lineage.")
    parser.add_argument("--keep-runid", action=argparse.BooleanOptionalAction, default=True, help="('jobs' mode only) Keep original run ID from lineage file.")
    parser.add_argument("--log", type=LogMode, choices=list(LogMode), default=LogMode.NORMAL, help="Logging level (default: normal)")
    args = parser.parse_args()

    # Configure logging level early based on args
    # REMOVED incorrect logger.setLevel calls
    log_level_requested = args.log.value.upper()
    # Log the requested level, but actual level is controlled by config loaded later
    # Corrected logging call: Use key=value for structlog, f-string for fallback
    if USING_C4H_LOGGER:
        logger.info("cli.log_level_requested", requested_level=log_level_requested, note="Actual level depends on config.")
    else:
        logger.info(f"cli.log_level_requested: level={log_level_requested}, note='Actual level depends on config.'")


    # Execute selected mode
    try:
        if args.mode == "service":
             handle_service_mode(args) # handle_service_mode now loads config
        elif args.mode == "jobs":
             if not args.config:
                 parser.error("--config argument is required for 'jobs' mode.")
             handle_jobs_mode(args)
        elif args.mode == "apply_diff":
             handle_apply_diff_mode(args)
        else: print(f"Error: Unsupported mode: {args.mode}"); sys.exit(1)
    except Exception as e:
        print(f"Unexpected error in main: {str(e)}");
        if hasattr(logger, 'exception'): logger.exception("main.unexpected_error") # Log traceback
        sys.exit(1)

if __name__ == "__main__":
    main()
