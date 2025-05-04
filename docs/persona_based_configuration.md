# Persona-Based Configuration System

## Architecture Overview

```mermaid
flowchart TD
    subgraph ConfigFiles["Configuration Files"]
        A[system_config.yml] --> |includes| B[personas definition]
        C[persona_1.yml] --> |loaded by key| D[persona collection]
        E[persona_2.yml] --> |loaded by key| D
        F[persona_3.yml] --> |loaded by key| D
    end

    subgraph ConfigProcessing["Configuration Processing"]
        G[materialise_config] --> |loads| H[system config]
        G --> |loads| I[persona config]
        G --> |deep merges| J[effective config]
        I --> |validates against| K[persona.json schema]
    end

    subgraph AgentCreation["Agent Creation"]
        L[AgentFactory] --> |creates| M[GenericSingleShotAgent]
        L --> |creates| N[GenericOrchestratingAgent]
        L --> |passes| O[full_effective_config]
    end

    subgraph AgentExecution["Agent Execution"]
        P[BaseAgent init] --> |stores| Q[unique_name]
        P --> |resolves| R[persona_key]
        P --> |looks up| S[llm_config.personas.{persona_key}]
        P --> |gets| T[prompts.system]
        P --> |gets| U[provider/model/temp]
    end

    D --> |referenced by| V[Task Definitions]
    J --> |passed to| L
    O --> |passed to| P
    V --> |persona_key| I
    
    style ConfigFiles fill:#f9f,stroke:#333,stroke-width:2px
    style ConfigProcessing fill:#bbf,stroke:#333,stroke-width:2px
    style AgentCreation fill:#bfb,stroke:#333,stroke-width:2px
    style AgentExecution fill:#fbf,stroke:#333,stroke-width:2px
```

## Implementation Components

### 1. Configuration Structure
- System config defines base structure at `llm_config.personas`
- Individual persona files contain specific agent behavior configurations
- Task definitions reference personas via `persona_key`

### 2. Configuration Loading and Merging
- `materialise_config` in `c4h_services/src/intent/impl/prefect/tasks.py` loads system config
- Detects `persona_key` in tasks and loads corresponding persona files
- Merges all config fragments in priority order: system → personas → job config

### 3. Agent Factory
- `AgentFactory` in `c4h_services/src/orchestration/factory.py` takes the effective config snapshot
- Task config specifies `agent_type` and `name` (used as unique identifier)
- Factory creates appropriate agent class with the effective config

### 4. Persona-Based Agent Initialization
- `BaseAgent.__init__` in `c4h_agents/agents/base_agent.py` takes `full_effective_config` and `unique_name`
- Gets agent config at `llm_config.agents.{unique_name}` 
- Extracts `persona_key` from agent config
- Looks up persona config at `llm_config.personas.{persona_key}`
- Resolves provider/model/temperature primarily from persona config

### 5. Generic Agents
- `GenericSingleShotAgent` in `c4h_agents/agents/generic.py` inherits from `BaseAgent`
- Gets system prompt and request template from persona config
- Special handling for discovery agent using tartxt_config from persona
- `GenericOrchestratingAgent` gets execution_plan from persona config

## Persona Configuration Structure
- Provider and model settings
- Temperature and other LLM parameters
- Prompts (system, user templates)
- Agent-specific configurations (tartxt_config, execution_plan)

## Workflow
1. Team/task definition specifies a `persona_key`
2. Configuration processor loads persona files based on keys
3. Agent Factory creates agent instances with effective config
4. Agents resolve their behavior from persona config
5. Agent behavior is completely determined by configuration

## Files and Responsibilities

| File | Responsibility |
|------|----------------|
| `config.py` | Core configuration utilities, load_persona_config, ConfigNode, merge logic |
| `tasks.py` | materialise_config detects persona_key and loads corresponding files |
| `factory.py` | Maps agent_type to agent classes, instantiates with effective config |
| `base_agent.py` | Extracts persona_key, resolves configuration from persona path |
| `generic.py` | Generic agents that use persona-based configuration |
| `persona_*.yml` | Individual persona configuration files |
| `system_config.yml` | Base configuration and persona structure definition |