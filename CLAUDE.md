# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

C4H (Coder for Hire) is an intelligent code refactoring system that leverages Large Language Models (LLMs) to automate code modifications based on natural language intents. The system consists of two main components:

1. **c4h_agents**: Core library containing specialized agents for code analysis, solution design, and implementation
2. **c4h_services**: Service layer providing workflow orchestration, API endpoints, and execution management

## Project Structure

The repository is organized into two main packages:

```
c4h/
├── c4h_agents/     # Core agent library
│   ├── agents/     # Agent implementations
│   ├── skills/     # Reusable skills
│   └── core/       # Project domain model
├── c4h_services/   # Service layer
│   └── src/
│       ├── api/    # REST API
│       ├── intent/ # Intent processing
│       └── orchestration/ # Workflow orchestration
├── config/         # System configuration
└── workspaces/     # Runtime workspaces for code processing
```

## Common Development Commands

### Setup and Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install c4h_agents in development mode with test dependencies
cd c4h_agents
pip install -e ".[test]"
cd ..

# Install c4h_services with test and prefect dependencies
cd c4h_services
pip install -e ".[test,prefect]"
cd ..
```

### Running Tests

```bash
# Setup test cases
python tests/setup/setup_test_projects.sh

# Run tests with specific configuration
python -m c4h_services.src.bootstrap.prefect_runner workflow \
    --config tests/examples/config/workflow_coder_01.yml
```

### Running the Service

```bash
# Start in workflow mode (direct execution)
python -m c4h_services.src.bootstrap.prefect_runner workflow \
    --project-path /path/to/project \
    --intent-file intent.json

# Start in service mode (API server)
python -m c4h_services.src.bootstrap.prefect_runner service \
    --port 8000 \
    --config config.yml

# Use in client mode (API client)
python -m c4h_services.src.bootstrap.prefect_runner client \
    --host localhost \
    --port 8000 \
    --project-path /path/to/project \
    --intent-file intent.json \
    --poll
```

## Key Architecture Concepts

### Agent System

The agent system is built around specialized agents that perform specific tasks:

- **Discovery Agent**: Analyzes project structure to identify relevant files
- **Solution Designer**: Plans code changes based on intent and discovered files
- **Coder Agent**: Implements code changes based on the solution design

Each agent follows these design principles:
1. **LLM-First Processing**: Offload logic and decision-making to the LLM
2. **Minimal Agent Logic**: Keep agent code focused on infrastructure concerns
3. **Single Responsibility**: Each agent has one clear, focused task
4. **Stateless Operation**: Most agents work in a stateless fashion

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

## Working with the Codebase

When working on this codebase:

1. **Understanding Agents**: Each agent extends `BaseAgent` and implements a `process(context)` method that takes a context dictionary and returns an `AgentResponse`

2. **Skills vs. Agents**: 
   - Agents are responsible for orchestration and control flow
   - Skills encapsulate reusable functionality like semantic extraction, merging, etc.

3. **Configuration Paths**:
   - Agents retrieve their settings via hierarchical queries like:
   ```python
   config_node.get_value("llm_config.agents.<agent_name>.<parameter>")
   ```

4. **Workflow Testing**:
   - Use the prefect runner to test workflows with configuration files
   - Configuration files are in `tests/examples/config/`

5. **Lineage Tracking**:
   - The system tracks agent operations and data flow in lineage files
   - Stored in the `workspaces/lineage/` directory
   - Used for debugging and resuming workflows