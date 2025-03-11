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
   - [Discovery Agent](#discovery-agent)
   - [Coder Agent](#coder-agent)
   - [Assurance Agent](#assurance-agent)
   - [Solution Designer Agent](#solution-designer-agent)
6. [Skills Overview](#skills-overview)
7. [Common Interfaces](#common-interfaces)
8. [Usage Example](#usage-example)
9. [Conclusion](#conclusion)

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

The library provides several agents that extend common base classes to perform specific tasks:

### Base Agent and Config

- **BaseAgent:** All agents inherit from `BaseAgent`, which combines configuration management (via `BaseConfig`) and LLM interfacing (via `BaseLLM`).
- **Agent Interfaces:** Every agent implements a `process(context)` method that takes a context dictionary and returns an `AgentResponse` (which includes success status, data, error messages, and metrics).

### Discovery Agent

- **Purpose:** Scans a project directory to identify source files and generate a manifest using an external tool (tartxt).
- **Key Functions:** 
  - Resolves input paths relative to the project root.
  - Executes the tartxt script with proper exclusions.
  - Parses output to create a file manifest.
- **Configuration:** Uses a dedicated `tartxt_config` to define script paths, input paths, exclusions, and output formatting.

### Coder Agent

- **Purpose:** Manages code modifications through semantic processing. It uses skills such as semantic extraction, merging, and iteration.
- **Key Functions:**
  - Retrieves and processes input code.
  - Uses an iterator (via the SemanticIterator skill) to extract changes.
  - Applies changes via the SemanticMerge skill and manages backups using the AssetManager.
- **Metrics:** Collects detailed metrics on code changes, including counts of successful and failed modifications.

### Assurance Agent

- **Purpose:** Executes validation tests to ensure code changes meet requirements.
- **Key Functions:**
  - Runs tests using tools like pytest.
  - Optionally executes validation scripts.
  - Parses output to report validation success or failure.
- **Cleanup:** Manages workspace cleanup after validations.

### Solution Designer Agent

- **Purpose:** (Typically) assists in planning or designing solutions based on input requirements. Although details may vary, it follows similar patterns as other agents with specialized prompts and processing.
- **Configuration:** Located under its own agent section in the configuration, ensuring tailored prompts and operational parameters.

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

1. **Initialize Configuration:**

   Create a configuration dictionary (typically loaded from YAML) that defines:
   - Global settings under `llm_config`
   - Agent-specific settings under `llm_config.agents.discovery`, `llm_config.agents.coder`, etc.
   - Provider details under `llm_config.providers`

2. **Create a Project Instance:**

   Use the `Project.from_config(config)` method to set up project paths and metadata.

3. **Instantiate an Agent:**

   For example, to run a discovery operation:
   ```python
   from c4h_agents.agents.discovery import DiscoveryAgent
   discovery = DiscoveryAgent(config=config)
   response = discovery.process({"project_path": "/path/to/project"})
   ```

4. **Process the Response:**

   Each agent returns an `AgentResponse` object. Inspect `response.data`, `response.error`, and `response.metrics` to handle outcomes accordingly.

---

## Conclusion

This document outlines the structure and interfaces of the **c4h_agents** library. By understanding the configuration system, the responsibilities of each agent, and the purpose of various skills, an LLM (or developer) can effectively use the library for tasks such as code refactoring, project discovery, and validation—without needing access to the complete codebase.