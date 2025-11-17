# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Load Project Context
Please read the C4H_CONTEXT.md file in the project root for comprehensive project context and architecture details.

## Repository Overview

C4H (Coder for Hire) is an intelligent code refactoring system that leverages Large Language Models (LLMs) to automate code modifications based on natural language intents. The system consists of three main components:

1. **c4h_agents**: Core library containing specialized agents for code analysis, solution design, and implementation
2. **c4h_services**: Service layer providing workflow orchestration, API endpoints, and execution management
3. **c4h_ai_dev**: AI development branch with additional features and enhancements

## Project Structure

The repository is organized into several packages:

```
/Users/jim/src/apps/c4h_ai_dev/
├── c4h_agents/         # Core agent library
│   ├── agents/         # Agent implementations
│   ├── skills/         # Reusable skills
│   ├── context/        # Execution context tools
│   ├── core/           # Project domain model
│   ├── execution/      # Executor components
│   ├── lineage/        # Event logging and tracking
│   ├── messages/       # Message models
│   └── utils/          # Utility functions
├── c4h_services/       # Service layer
│   └── src/
│       ├── api/        # REST API endpoints
│       ├── intent/     # Intent processing
│       ├── orchestration/ # Workflow orchestration
│       ├── bootstrap/  # Entry points (prefect_runner.py)
│       └── utils/      # Service utilities
├── conf/               # Hydra configuration directory
│   ├── config.yaml     # Main configuration with defaults
│   ├── provider/       # LLM provider configurations
│   ├── team/           # Team definitions
│   ├── persona/        # Agent persona configs
│   ├── skills/         # Skills configuration
│   ├── llm_config/     # LLM-specific configs
│   └── agent/          # Agent-specific configs
├── workspaces/         # Runtime workspaces
│   ├── lineage/        # Workflow execution history
│   ├── backups/        # File backups
│   └── workflows/      # Workflow storage
├── ai_dev/             # AI development resources
│   ├── ai_docs/        # Documentation and master_docs
│   └── ai_scripts/     # Utility scripts
├── tests/              # Test suite
│   ├── setup/          # Test project setup
│   ├── examples/       # Example configurations
│   └── test_projects/  # Test project files
├── tools/              # Utility tools and scripts
└── workorders/         # Hydra work order documentation
```

The `ai_dev` folder contains:
- `ai_docs/`: Master documentation, coding guidelines, and constraints
- `ai_scripts/`: Utility scripts (temporary scripts created by Claude should be placed here)

## Common Development Commands

### Setup and Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install c4h_agents in development mode
cd c4h_agents
pip install -e .
cd ..

# Install c4h_services with dependencies
cd c4h_services
pip install -e .
cd ..

# Run tests to verify installation
python -m pytest tests/
```

### Running Tests

```bash
# Setup test cases
python tests/setup/setup_test_projects.sh

# Run tests with specific configuration
python -m c4h_services.src.bootstrap.prefect_runner workflow \
    --config tests/examples/config/workflow_coder_01.yml

# Run server-client test
./tests/run_server_client_test.sh
```

### Running the Backend Service

```bash
# Start in workflow mode (deprecated - use service/jobs mode instead)
# python -m c4h_services.src.bootstrap.prefect_runner workflow --config path/to/config.yml

# Start in service mode (API server)
python -m c4h_services.src.bootstrap.prefect_runner service \
    --port 5500 \
    --config-path conf

# Use in jobs mode (API client)
python -m c4h_services.src.bootstrap.prefect_runner jobs \
    --host localhost \
    --port 5500 \
    --config path/to/job_config.yml \
    --poll
```

## Key Architecture Concepts

### Agent System

The agent system is built around generic agents that perform specific tasks when configured with a persona:

- **Discovery Agent**: Analyzes project structure to identify relevant files
- **Solution Designer**: Plans code changes based on intent and discovered files
- **Coder Agent**: Implements code changes based on the solution design

Each agent follows these design principles:
1. **LLM-First Processing**: Offload logic and decision-making to the LLM
2. **Minimal Agent Logic**: Keep agent code focused on infrastructure concerns
3. **Single Responsibility**: Each agent has one clear, focused task
4. **Stateless Operation**: Agents work in a stateless fashion

### Workflow Orchestration

The system uses a team-based workflow approach:
1. User submits an intent (e.g., "Add logging to all functions")
2. The orchestrator initializes a workflow
3. The Discovery Team analyzes the project structure
4. The Solution Design Team creates a plan for implementing the intent
5. The Coder Team implements the changes
6. The workflow completes and returns results

### Configuration System

The configuration system is hierarchical:
- System-level settings in `config/system_config.yml`
- Agent-specific settings under `llm_config.agents.<agent_name>`
- Provider settings under `llm_config.providers`
- Workflow orchestration under `orchestration`
- Persona definitions under `config/personas/`

### Skills Framework

Skills are reusable components that agents can utilize:
- **SemanticIterator**: Extracts structured information from text
- **AssetManager**: Manages file operations with safety features
- **SemanticMerge**: Intelligently merges changes into existing files

## Working with the Codebase

When working on this codebase:

1. **Understanding BaseAgent**: All agents extend `BaseAgent` and implement a `process(context)` method

2. **Configuration Paths**: Agents retrieve settings via:
   ```python
   config_node.get_value("llm_config.agents.<agent_name>.<parameter>")
   ```

3. **Workflow Execution**: The main entry point is:
   ```python
   orchestrator.execute_workflow(entry_team="discovery", context=context)
   ```

4. **Persona-Based Configuration**: Agents use persona configs for behavior:
   ```python
   # Agent links to persona via persona_key
   persona_key = "discovery_v1"
   persona_path = f"llm_config.personas.{persona_key}"
   ```

5. **Lineage Tracking**: The system tracks operations in lineage files:
   ```python
   # Lineage is stored under:
   "workspaces/lineage/{workflow_run_id}"
   ```

## Skills System

Key skills available:
- **SemanticIterator**: Extracts structured information from text/code
- **SemanticMerge**: Intelligently merges code changes
- **AssetManager**: Safe file operations with backup support
- **ClaudeCodeRunner**: Integration with Claude for coding tasks
- **ChangeValidator**: Validates code modifications

## Logging System

The project uses structured logging:
- Custom logger from `c4h_agents.utils.logging`
- Structured key-value logging format
- Log levels configurable per component
- Agent-specific logging contexts

## Testing Strategy

- Unit tests for individual components
- Integration tests for workflow execution
- Server-client tests for API functionality
- Test projects provide consistent test cases