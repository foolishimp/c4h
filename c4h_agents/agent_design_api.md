# C4H Agents Library Overview and Configuration Guide

This document provides a high-level overview of the **c4h_agents** library. It explains the system configuration, the purpose and interfaces of each agent, and the available skills. The goal is to allow an LLM to understand how to use the agents without requiring the entire codebase.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Design Principles](#design-principles)
3. [System Configuration](#system-configuration)
4. [Library Architecture](#library-architecture)
   - [Project Module](#project-module)
   - [Configuration Module](#configuration-module)
5. [Agents Overview](#agents-overview)
   - [Base Agent and Config](#base-agent-and-config)
   - [Generic Type-Based Agents](#generic-type-based-agents)
   - [Discovery Agent](#discovery-agent)
   - [Coder Agent](#coder-agent)
   - [Assurance Agent](#assurance-agent)
   - [Solution Designer Agent](#solution-designer-agent)
6. [Skills Overview](#skills-overview)
7. [Common Interfaces](#common-interfaces)
8. [Advanced Orchestration](#advanced-orchestration)
   - [Conditional Routing](#conditional-routing)
   - [Loop-Based Iteration](#loop-based-iteration)
   - [Context-Aware Recursion](#context-aware-recursion)
   - [Configuration Snapshots](#configuration-snapshots)
9. [Usage Example](#usage-example)
10. [Conclusion](#conclusion)

---

## Introduction

**c4h_agents** is a modern Python library designed for LLM-based code refactoring and project operations. It provides a set of agents that perform tasks such as project discovery, code modification, and validation. The library follows clear design principles to ensure each component has a single responsibility and interfaces are simple and stateless.

---

## Design Principles

The library is built around several key principles:

- **LLM-First Processing:** Agents leverage large language models (LLMs) for generating and refining code changes.
- **Minimal Agent Logic:** Each agent encapsulates minimal core logic, delegating semantic processing to specialized skills.
- **Clear Boundaries:** Separation between system configuration, agent behavior, and reusable skills.
- **Single Responsibility:** Components are designed to perform one task effectively.
- **Stateless Operation:** Most agents work in a stateless fashion, relying on configuration and provided context.

---

## System Configuration

The configuration system is built around a hierarchical and node-based approach. Key aspects include:

- **ConfigNode Class:** Enables hierarchical, dot-delimited access (with support for wildcards) to configuration parameters.
- **Hierarchical Lookup:** Agent-specific settings are stored under `llm_config.agents.<agent_name>`, and provider defaults under `llm_config.providers`.
- **Merging & Overrides:** The library supports deep merging of system and application configurations, ensuring runtime values and defaults are combined appropriately.
- **Logging and Metrics:** Configuration also controls logging detail (from minimal to debug) and metrics collection across agent operations.

---

## Library Architecture

### Project Module

- **Project and ProjectPaths:** Define the domain model for a project, including paths for source code, configuration, workspaces, and output. Projects are initialized from a configuration dictionary, ensuring proper directory setup and metadata tracking.

### Configuration Module

- **Config Functions:** Utility functions such as `get_value`, `get_by_path`, and `deep_merge` provide robust access to nested configuration values.
- **Dynamic Lookup:** Supports both dot and slash notations for accessing nested configuration settings, making it flexible for different use cases.

---

## Agents Overview

The library provides two categories of agents: generic type-based agents for general functionality and specialized domain-specific agents.

### Base Agent and Config

- **BaseAgent:** All agents inherit from `BaseAgent`, which combines configuration management (via `BaseConfig`) and LLM interfacing (via `BaseLLM`).
- **Agent Interfaces:** Every agent implements a `process(context)` method that takes a context dictionary and returns an `AgentResponse` (which includes success status, data, error messages, and metrics).
- **Agent Types:** The library defines standard agent types via the `AgentType` enum in `agents/types.py`.

### Generic Type-Based Agents

These agents provide general-purpose functionality based on their type:

#### GenericLLMAgent

- **Purpose:** General-purpose agent for single LLM interactions (replaces GenericSingleShotAgent).
- **Key Features:**
  - Uses persona-based configuration for prompts and parameters
  - Supports runtime overrides through context
  - Can invoke skills if configured instead of using LLM directly
  - Special handling for discovery agents with project scanning
- **Configuration:** Configured via persona keys and can use either LLM or skill invocation.

#### GenericOrchestratorAgent

- **Purpose:** Coordinates multi-step processes based on an execution plan.
- **Key Features:**
  - Executes multi-step workflows defined in configuration
  - Supports conditional execution with branching based on results
  - Manages state and context across steps
  - Integrates with skills for specialized operations
  - Handles errors with configurable recovery strategies
- **Configuration:** Uses an `execution_plan` with steps, conditions, and branches.

#### GenericSkillAgent

- **Purpose:** Optimized for skill-based operations with minimal LLM interaction.
- **Key Features:**
  - Efficiently invokes predefined skills with proper context preparation
  - Minimizes token usage and latency by reducing LLM calls
  - Handles data transformation and processing tasks
  - Supports specialized operations like extraction, parsing, and formatting
  - Optional LLM fallback if skills fail
- **Configuration:** Requires a `skill` identifier and optional `skill_params`.

#### GenericFallbackAgent

- **Purpose:** Designed for handling failure cases with conservative parameters.
- **Key Features:**
  - Serves as a fallback when primary agents fail
  - Uses more conservative LLM parameters (lower temperature, smaller models)
  - Applies stricter validation and error handling
  - Executes more focused, targeted tasks with explicit constraints
  - Implements appropriate retry and recovery strategies
- **Configuration:** Includes additional parameters like `max_retries`, `retry_delay`, and `validation_level`.

### Specialized Agents

These domain-specific agents handle particular workflows:

#### Discovery Agent

- **Purpose:** Scans a project directory to identify source files and generate a manifest using an external tool (tartxt).
- **Key Functions:** 
  - Resolves input paths relative to the project root.
  - Executes the tartxt script with proper exclusions.
  - Parses output to create a file manifest.
- **Configuration:** Uses a dedicated `tartxt_config` to define script paths, input paths, exclusions, and output formatting.
- **Implementation:** Now typically implemented using GenericLLMAgent with discovery persona.

#### Coder Agent

- **Purpose:** Manages code modifications through semantic processing. It uses skills such as semantic extraction, merging, and iteration.
- **Key Functions:**
  - Retrieves and processes input code.
  - Uses an iterator (via the SemanticIterator skill) to extract changes.
  - Applies changes via the SemanticMerge skill and manages backups using the AssetManager.
- **Metrics:** Collects detailed metrics on code changes, including counts of successful and failed modifications.
- **Implementation:** Now typically implemented using GenericLLMAgent with coder persona.

#### Assurance Agent

- **Purpose:** Executes validation tests to ensure code changes meet requirements.
- **Key Functions:**
  - Runs tests using tools like pytest.
  - Optionally executes validation scripts.
  - Parses output to report validation success or failure.
- **Cleanup:** Manages workspace cleanup after validations.
- **Implementation:** Can be implemented using GenericOrchestratorAgent with an execution plan for validation.

#### Solution Designer Agent

- **Purpose:** Assists in planning or designing solutions based on input requirements.
- **Configuration:** Located under its own agent section in the configuration, ensuring tailored prompts and operational parameters.
- **Implementation:** Now typically implemented using GenericLLMAgent with solution_designer persona.

---

## Skills Overview

Skills are reusable components that encapsulate common functionalities used by the agents:

- **Semantic Extraction:** Tools like `semantic_extract` extract meaningful code segments or change suggestions.
- **Semantic Merging:** `semantic_merge` integrates modifications into the codebase intelligently.
- **Semantic Iteration:** `semantic_iterator` iterates over code elements to identify potential changes.
- **Asset Management:** `asset_manager` ensures that backups and file modifications are safely managed.
- **Formatting and Fast/Slow Processing:** Modules like `semantic_formatter`, `_semantic_fast`, and `_semantic_slow` provide optimized text processing for different scenarios.
- **Shared Utilities:** Reusable utilities (e.g., markdown utilities and type definitions) are shared across skills.

---

## Common Interfaces

All agents and skills adhere to a consistent set of interfaces:

- **Process Method:**  
  ```python
  def process(context: Dict[str, Any]) -> AgentResponse:
      ...
  ```
  – Accepts a context dictionary and returns an `AgentResponse` with keys like `success`, `data`, `error`, and `metrics`.

- **Configuration Lookup:**  
  Agents retrieve their settings via hierarchical queries such as:
  ```python
  config_node.get_value("llm_config.agents.<agent_name>.<parameter>")
  ```

- **Logging and Metrics:**  
  Detailed logging is implemented via `structlog`, and performance/usage metrics are tracked and reported in each agent’s response.

- **LLM Integration:**  
  The agents interface with LLMs (via the `litellm` provider) using standardized model parameters (provider, model, temperature) drawn from configuration.

---

## Usage Example

### Direct Agent Usage

1. **Initialize Configuration:**

   Create a configuration dictionary (typically loaded from YAML) that defines:
   - Global settings under `llm_config`
   - Agent-specific settings under `llm_config.agents.<agent_name>`
   - Persona configurations under `llm_config.personas.<persona_key>`
   - Provider details under `llm_config.providers`

2. **Create a Project Instance:**

   Use the `Project.from_config(config)` method to set up project paths and metadata.

3. **Instantiate an Agent:**

   For example, to run a discovery operation using GenericLLMAgent:
   ```python
   from c4h_agents.agents.generic import GenericLLMAgent
   
   # Create agent with effective config and unique name
   discovery = GenericLLMAgent(full_effective_config=config, unique_name="discovery_agent")
   
   # Process with context
   response = discovery.process({
       "project_path": "/path/to/project",
       "agent_config_overrides": {
           "discovery_agent": {
               "temperature": 0.2
           }
       }
   })
   ```

4. **Process the Response:**

   Each agent returns an `AgentResponse` object. Inspect `response.data`, `response.error`, and `response.metrics` to handle outcomes accordingly.

### Factory-Based Agent Creation

The preferred approach uses the AgentFactory for managing agent creation:

1. **Initialize the Factory:**

   ```python
   from c4h_services.src.orchestration.factory import AgentFactory
   
   # Create factory with effective configuration
   factory = AgentFactory(effective_config_snapshot=config)
   ```

2. **Prepare Task Configuration:**

   ```python
   task_config = {
       "name": "discovery_phase",
       "agent_type": "GenericLLMAgent",
       "persona_key": "discovery",
       "description": "Analyze project structure and files"
   }
   ```

3. **Create and Use the Agent:**

   ```python
   # Create agent from task config
   agent = factory.create_agent(task_config)
   
   # Process with context
   response = agent.process({"project_path": "/path/to/project"})
   ```

### Skill-Based Processing

For specialized processing using skills:

```python
from c4h_agents.agents.generic import GenericSkillAgent

# Create skill-based agent
extractor = GenericSkillAgent(full_effective_config=config, unique_name="extractor")

# Process with skill invocation context
response = extractor.process({
    "content": "Text content to extract from",
    "instructions": "Extract all function names",
    "format": "json"
})
```

### Orchestration Example

For complex multi-step processes:

```python
from c4h_agents.agents.generic import GenericOrchestratorAgent

# Create orchestrator agent
orchestrator = GenericOrchestratorAgent(full_effective_config=config, unique_name="workflow_orchestrator")

# Define execution plan in context (or use from config)
context = {
    "input_data": {...},
    "agent_config_overrides": {
        "workflow_orchestrator": {
            "execution_plan": {
                "enabled": True,
                "steps": [
                    {"name": "step1", "type": "skill", "skill": "extraction_skill"},
                    {"name": "step2", "type": "llm", "prompt": "Process {extraction_result}"}
                ]
            }
        }
    }
}

# Process with orchestration context
response = orchestrator.process(context)
```

---

## Advanced Orchestration

The c4h_agents library provides powerful orchestration capabilities to handle complex workflows, iterative processing, and conditional execution patterns.

### Conditional Routing

- **Enhanced YAML DSL:** The system supports a rich declarative syntax for defining complex routing conditions:
  ```yaml
  routing:
    rules:
      - condition:
          type: "and"
          conditions:
            - type: "exists"
              field: "analysis_result"
            - type: "equals"
              field: "analysis_result.has_errors"
              value: true
        target: "error_handler"
  ```

- **Configuration Examples:** For comprehensive examples, see:
  - The `advanced_routing` team definition in `/config/system_config.yml` (lines 797-877)
  - The `enhanced_routing` section in `/config/system_config.yml` (lines 989-1011)

- **Logical Operators:** Supports nested `and`, `or`, and `not` operators for complex boolean logic.
  
- **Rich Operator Functions:** Includes a comprehensive set of operators:
  - **Comparison:** `equals`, `not_equals`, `greater_than`, `less_than`, etc.
  - **Existence:** `exists`, `not_exists` to check field presence
  - **String Matching:** `contains`, `starts_with`, `ends_with`, `matches_regex`
  - **Collection:** `in`, `not_in`, `has_any`, `has_all`
  - **Nested Access:** Supports dot notation for accessing nested fields

- **Error Handling:** Includes fallback paths and error reporting for routing issues

### Loop-Based Iteration

- **Declarative Iteration:** Teams can be configured with `type: "loop"` to process collections:
  ```yaml
  teams:
    collection_processor:
      type: "loop"
      collection_field: "items"
      item_field: "current_item"
      tasks:
        - name: "process_item"
          agent_type: "GenericLLMAgent"
          persona_key: "processor"
  ```

- **Configuration Examples:** For comprehensive examples, see:
  - The `batch_processor` and `item_processor` team definitions in `/config/system_config.yml` (lines 698-734)
  - The `file_processor` loop example in `/config/system_config.yml` (lines 736-746)
  - The `loop_configurations` section in `/config/system_config.yml` (lines 1014-1035)

- **Collection Sources:** Can iterate over:
  - Fields in context containing arrays
  - Dynamically generated collections from previous tasks
  - File lists and directory contents
  
- **Processing Modes:** Supports sequential, parallel, and batched processing

- **Accumulation:** Collects results in configurable result fields, with options for aggregation strategies

- **Loop Control:** Includes support for early termination, conditionals, and maximum iteration limits

### Context-Aware Recursion

- **Recursion Strategies:** Multiple strategies for team recursion:
  - `new_snapshot`: Creates a new immutable configuration snapshot 
  - `child_context`: Propagates parent context with nondestructive updates
  - `shared_context`: Maintains a single context across recursive calls
  
- **Configuration Examples:** For comprehensive examples, see:
  - The `recursive_team` definition in `/config/system_config.yml` (lines 880-917)
  - The `recursion` section in `/config/system_config.yml` (lines 1038-1063)
  
- **Recursion Control:** Configuration options include:
  ```yaml
  recursion:
    strategy: "new_snapshot"
    max_depth: 3
    propagate_fields: ["project_data", "analysis_result"]
    preserve_lineage: true
  ```

- **Lineage Tracking:** Maintains parent-child relationships across recursive executions

- **Depth Limiting:** Configurable maximum recursion depth with early termination

- **Context Evolution:** Supports both immutable and evolving context patterns

### Configuration Snapshots

- **Immutable Snapshots:** Configuration is captured as immutable snapshots at execution time:
  ```yaml
  config_snapshots:
    enabled: true
    store_path: "{project_path}/snapshots"
    include_metadata: true
    hash_algorithm: "sha256"
  ```
  
- **Configuration Examples:** For comprehensive examples, see:
  - The `config_snapshots` section in `/config/system_config.yml` (lines 967-986)

- **Validation:** Config fragments are validated against JSON schemas before snapshot creation

- **Persistence:** Snapshots are persisted to disk with metadata including:
  - Timestamp and run ID
  - Hash for deterministic identification
  - Fragment metadata for traceability
  
- **Lineage Integration:** Snapshots are recorded in lineage events for complete provenance tracking

- **Reproducibility:** Enables exact reproduction of conditions for a specific agent execution

### Integration of Advanced Features

The advanced orchestration features in the c4h_agents library can be combined for powerful, flexible workflows:

- **Combine Loops with Conditional Routing**: Use loop outputs to make branching decisions
- **Apply Recursion with Snapshots**: Create immutable records for each level of recursion
- **Chain Specialized Teams**: Build complex processing pipelines with specialized teams
- **Context-Aware Decision Making**: Pass enriched context between teams for intelligent routing
- **Lineage Tracking**: Maintain complete provenance for each agent execution

These capabilities enable sophisticated workflows while maintaining full traceability, making the system ideal for complex multi-stage processing tasks with robust error handling and recovery.

## Conclusion

This document outlines the structure and interfaces of the **c4h_agents** library. By understanding the configuration system, the responsibilities of each agent, and the purpose of various skills, an LLM (or developer) can effectively use the library for tasks such as code refactoring, project discovery, and validation—without needing access to the complete codebase.