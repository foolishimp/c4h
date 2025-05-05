python c4h_services/examples/prefect_runner.py agent --agent semantic_iterator --config c4h_agents/examples/configs/iter_config_01.yml --log debug
## python c4h_services/examples/prefect_runner.py workflow --config c4h_services/examples/config/workflow_coder_self_01.yml --log debug
## python c4h_services/examples/prefect_runner.py workflow --config c4h_services/examples/config/workflow_coder_01.yml

# Test EventLogger with workflow 
# Load environment variables from .env
set -a
source .env
set +a
python c4h_services/src/bootstrap/prefect_runner.py jobs --config temp_workorder.yml --system-config config_teams_0502/system_config.yml --host localhost --port 5501 --poll --log debug