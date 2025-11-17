# C4H Project Context

## Overview
C4H (Coder for Hire) is an intelligent code refactoring system that leverages Large Language Models (LLMs) to automate code modifications based on natural language intents. It uses a multi-agent architecture with specialized teams for discovery, solution design, and implementation.

## Project Structure
```
/Users/jim/src/apps/c4h_ai_dev/
├── c4h_agents/         # Core agent library
│   ├── agents/         # Agent implementations (BaseAgent, GenericLLMAgent)
│   ├── skills/         # Reusable skills (semantic iteration, merging, asset management)
│   ├── context/        # Execution context tools
│   ├── core/           # Project domain model
│   └── utils/          # Utilities (logging, config)
├── c4h_services/       # Service layer
│   └── src/
│       ├── api/        # REST API endpoints
│       ├── intent/     # Intent processing
│       ├── orchestration/ # Workflow orchestration
│       └── bootstrap/  # Entry point (prefect_runner.py)
├── config/             # System configuration
│   ├── personas/       # Agent persona definitions
│   └── schemas/        # JSON schema validation
├── conf/               # Hydra configuration
│   ├── config.yaml     # Main config with defaults
│   ├── provider/       # LLM provider configs
│   ├── team/           # Team definitions
│   ├── persona/        # Persona configs
│   └── skills/         # Skills configuration
├── tests/              # Test suite
│   ├── setup/          # Test project setup scripts
│   └── examples/       # Example configurations
├── workspaces/         # Runtime data
│   ├── workflows/      # Workflow storage
│   ├── lineage/        # Operation tracking
│   └── backups/        # File backups
└── ai_dev/            # AI development documentation

## Key Concepts

### 1. Agent Architecture
- **BaseAgent**: Abstract base class for all agents
- **GenericLLMAgent**: Configurable agent that uses personas
- Agents implement `process(context)` returning `AgentResponse`
- Minimal logic in agents - delegate to LLMs

### 2. Workflow Teams
- **Discovery Team**: Analyzes project structure, finds relevant files
- **Solution Design Team**: Plans code changes based on intent
- **Coder Team**: Implements the actual code modifications
- **Fallback Team**: Handles edge cases and errors

### 3. Configuration System
- Uses Hydra for configuration management
- Hierarchical config with defaults and overrides
- Key paths:
  - `llm_config.providers.<name>`: LLM provider settings
  - `llm_config.agents.<name>`: Agent configurations
  - `llm_config.personas.<name>`: Persona definitions
  - `orchestration.teams.<name>`: Team configurations

### 4. Execution Modes
- **Service Mode**: API server (default port 5500)
- **Jobs Mode**: Client for submitting jobs to service
- **Apply Diff Mode**: Direct diff application
- **Workflow Mode**: Direct workflow execution (deprecated)

### 5. Skills Framework
- **SemanticIterator**: Extracts structured info from text
- **SemanticMerge**: Intelligently merges code changes
- **AssetManager**: Safe file operations with backups
- **ClaudeCodeRunner**: Integration with Claude for coding
- **ChangeValidator**: Validates code modifications

## Common Commands

```bash
# Setup environment
python -m venv venv
source venv/bin/activate
pip install -e c4h_agents
pip install -e c4h_services

# Run tests
./tests/run_server_client_test.sh

# Start service
python -m c4h_services.src.bootstrap.prefect_runner service -P 5500 --config-path conf

# Submit job
python -m c4h_services.src.bootstrap.prefect_runner jobs -P 5500 --config job.yml --poll

# Apply diff
python -m c4h_services.src.bootstrap.prefect_runner apply_diff --project-path /path --diff-file changes.diff
```

## Workflow Process
1. User submits intent (e.g., "add logging to all functions")
2. Discovery team finds relevant files
3. Solution designer creates implementation plan
4. Coder team implements changes
5. Results saved with full lineage tracking

## Key Files
- `c4h_services/src/bootstrap/prefect_runner.py`: Main entry point
- `c4h_agents/agents/base_agent.py`: Base agent class
- `c4h_agents/agents/generic_llm_agent.py`: Configurable LLM agent
- `c4h_services/src/orchestration/orchestrator.py`: Workflow orchestration
- `conf/config.yaml`: Main Hydra configuration

## Environment Variables
- `ANTHROPIC_API_KEY`: For Claude models
- `OPENAI_API_KEY`: For OpenAI models
- `GOOGLE_API_KEY`: For Gemini models
- `XAI_API_KEY`: For xAI models

## Testing
- Test projects in `tests/test_projects/`
- Example configs in `tests/examples/config/`
- Server-client test: `tests/run_server_client_test.sh`

## Important Notes
- Always use personas for agent configuration
- Lineage tracking enabled by default
- Backup system creates copies before modifications
- Structured logging throughout the system
- Hydra config allows flexible composition