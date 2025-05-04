# C4H Agent System Refactoring: Factory Model & Enhanced Orchestration
**Version: 1.3**
**Date: 2025-05-01**

## 1. Introduction

This document outlines the requirements and design for refactoring the C4H Agent System. The primary goal is to mandate a move away from the legacy model, where agent behavior was tightly coupled to specific agent classes, towards a unified and flexible factory pattern. In this required model, all agent instances *must* be created based on configuration specifying a generic agent capability (`agent_type`), a specific set of behaviors and settings (`persona_key`), and a unique instance name (`unique_name`).

A key aspect of this refactoring is the introduction of **effective configuration snapshots**. For each job run, the system will generate and persist a complete snapshot of the configuration *after* all merging, persona injection, and environment variable expansion has occurred. Agent execution will be based *solely* on this immutable snapshot, ensuring deterministic replayability and auditability.

A secondary, related goal is to enhance the workflow orchestration capabilities (Prefect) to support more complex control flow, including conditional branching and context-aware recursion, driven by a declarative DSL.

This refactoring removes all backward compatibility, eliminates technical debt, and enforces a clean, configuration-as-code approach. It builds upon `Agent_Design_Principles_v2.md` and `Config_Design_Principles.md`, leveraging LLM-First, Minimal Agent Logic, Stateless Operation, and Hierarchical Configuration principles.

## 2. Requirements

**R0: Clean Design Enforcement & Tech Debt Removal:** The refactoring *must* eliminate support for the legacy `agent_class` configuration model. All agent instantiations *must* use the new factory/persona model (`agent_type`, `persona_key`, `unique_name`). Existing configurations and workflows *must* be migrated to this new pattern.

**R1: Generic Agent Capabilities & Path-Addressable Config:** Introduce generic agent base classes representing core capabilities (e.g., single-shot execution, internal orchestration). Agents *must* be initialized with the full effective configuration snapshot and their designated configuration path within it. Agents access their settings via standard path lookups within the full snapshot. (See Section 3.1)

**R2: Persona-Driven Configuration:** Define agent behavior (prompts, LLM settings, `execution_plan`, etc.) in reusable "persona" configurations. Personas can reside in separate files (e.g., `config/personas/`) and are referenced by `persona_key`. The merge process *must* inject the specified persona configuration before generating the effective config snapshot. (See Section 3.2, 3.4.3)

**R3: Factory-Based Instantiation:** Implement an Agent Factory responsible for creating agent instances based *only* on configuration specifying `agent_type`, `persona_key`, and `unique_name`, using the effective configuration snapshot. (See Section 3.3)

**R4: Updated Orchestration Configuration:** Modify the `system_config.yml` structure for defining team tasks to *exclusively* support the new factory model (`agent_type`, `persona_key`, `unique_name`). Task definitions may include a `persona:` key to trigger persona file injection. (See Section 3.2)

**R5: Declarative Orchestration DSL:** Enhance the YAML DSL for `orchestration.teams.<team_id>.routing.rules` to support expressive conditional branching based on task outcomes and context variables within the effective config snapshot. (See Section 3.4.1)

**R6: Context-Aware Recursion:** Implement a mechanism within the orchestration logic to allow teams to be re-executed with modified context (potentially leading to a new snapshot for the recursive run), enabling effective recursion. (See Section 3.4.2)

**R7: Prefect-Based DSL Execution:** Refactor the Prefect implementation (`c4h_services`) to act as an execution engine for the declarative YAML DSL. The engine first generates the effective config snapshot and then executes tasks based *only* on that snapshot. (See Section 3.4.3)

**R8: ~~Backward Compatibility~~ (REMOVED):** Requirement removed.

**R9: New Team Support:** The refactored system must support the definition and execution of new teams using the factory/persona model, snapshot-based execution, and enhanced orchestration.

**R10: Clarified Agent/Skill Roles:** Maintain a clear conceptual distinction between Agents (orchestrated workflow participants defined by Personas/snapshots) and Skills (reusable operational capabilities). (See Section 3.6)

**R11: Recursion Safety Controls:** Implement mechanisms within the Prefect orchestration flow to prevent infinite recursion, including Maximum Total Team Executions and Maximum Team Recursion Depth per team instance. (See Section 3.4.4)

**R12: Effective Configuration Snapshots (New):** For every job run, the system *must* generate and persist an immutable snapshot of the effective configuration *after* all merging, persona injection, and environment variable expansion. Agent execution *must* be based solely on this snapshot. The snapshot path must be recorded in the run's lineage/metadata. (See Section 3.4.3)

**R13: Configuration Schema Validation (New):** Before merging configuration fragments, each fragment *must* be validated against a predefined JSON Schema to ensure structural correctness and presence of required fields. (See Section 3.4.3)

## 3. Design

### 3.1. Generic Agent Classes (`c4h_agents/agents/generic.py` - New)

* Agents inherit from `BaseAgent`.
* **Initialization:** `__init__(self, full_effective_config: Dict, unique_name: str)`:
    * Accepts the *complete* effective configuration (loaded from the snapshot) and the agent's unique instance name.
    * Stores the full config and the unique name.
    * Determines its specific configuration path within the `full_effective_config` (e.g., based on `unique_name` or a convention mapped from `agent_type`). Stores this path prefix (e.g., `self.config_path = "llm_config.agents.specific_instance_name"`).
* **Configuration Access:** Agents use standard methods inherited from `BaseConfig` (like `self.lookup(relative_path)`) or direct dictionary access, applying their `self.config_path` prefix to query the `full_effective_config`. **No** overrides of base config access methods are needed.
* **`_get_agent_name(self) -> str`:** Returns the `unique_name` provided during instantiation.
* **`GenericSingleShotAgent(BaseAgent)`:**
    * Purpose: Single LLM interaction.
    * `process`: Reads prompts, settings etc., using its config path from the full config, formats the request, calls the LLM, and processes the response.
* **`GenericOrchestratingAgent(BaseAgent)`:**
    * Purpose: Internal orchestration based on `execution_plan`.
    * `process`: Reads its `execution_plan` using its config path from the full config. Interprets the plan, calls Skills (passing full config or relevant sub-sections), manages state, aggregates results.

### 3.2. Configuration Enhancements (`config/system_config.yml`, `personas/`)

* **`llm_config.personas` Section (Structure Definition):**
    * Defines the *structure* and *default values* for personas within `system_config.yml`.
* **Persona Files (e.g., `config/personas/solution_designer_v1.yml` - New Location):**
    * Individual persona configurations are stored as separate YAML files.
    * These files contain the specific settings (provider, model, prompts, `execution_plan`, etc.) for that persona.
    * **Example (`config/personas/solution_designer_v1.yml`):**

            # Inherits structure from llm_config.personas in system_config.yml
            provider: "anthropic"
            model: "claude-3-7-sonnet-20250219"
            temperature: 1
            prompts:
              system: "You are a solution designer..."
              solution: "..."

* **`orchestration.teams.tasks` Structure Update (Mandatory):**
    * Tasks *must* use the factory model: `name`, `agent_type`, `persona_key`.
    * The `persona_key` (e.g., `"solution_designer_v1"`) implicitly tells the pre-merge process to load and inject `config/personas/solution_designer_v1.yml`.
    * `config`: Optional dictionary for task-specific overrides applied *after* persona injection but *before* snapshot generation.
    * Legacy `agent_class` is **not supported**.
* **Orchestration Safety Limits:** (Unchanged from v1.2) Add `max_total_teams`, `max_recursion_depth` globally and optionally per-team. Example structure:

        orchestration:
          enabled: true
          entry_team: "discovery"
          max_total_teams: 30
          max_recursion_depth: 5
          teams:
            engineering_team:
              name: "Engineering Team"
              max_recursion_depth: 3 # Override global limit
              # ... tasks, routing ...

### 3.3. Agent Factory (`c4h_services/src/orchestration/factory.py` - Updated)

* **`AgentFactory` Class:**
    * `__init__(self, effective_config_snapshot: Dict)`: Stores the *complete, loaded* effective configuration snapshot dictionary.
    * `_get_agent_class(self, agent_type: str) -> Type[BaseAgent]:` (Unchanged) Maps `agent_type` string to class.
    * `create_agent(self, task_config: Dict) -> BaseAgent:`:
        1.  Extract `name` (unique instance name), `agent_type` from `task_config`.
        2.  Call `_get_agent_class(agent_type)`.
        3.  Instantiate the agent, passing the *full effective config snapshot* and the `unique_name`: `agent_class(full_effective_config=self.effective_config_snapshot, unique_name=name)`.
        4.  Return the instantiated agent.
    * *Note:* The factory no longer performs merging; that happens *before* the snapshot is created. It simply uses the final snapshot.

### 3.4. Orchestration Enhancements (Prefect-Based)

#### 3.4.1. Declarative Routing DSL (`system_config.yml`)

Enhance the `orchestration.teams.<team_id>.routing.rules.condition` syntax. A structured dictionary format is preferred. Example structure under `routing:`:

    rules:
      # Example 1: Branch based on specific task output field
      - condition:
          task: "impl_evaluator"      # Target task by name
          output_field: "status"      # Path in AgentResponse.data
          operator: "equals"
          value: "FAIL"
        next_team: "correction_wo_generator_team"

      # Example 2: Combine task status and context variable (AND logic)
      - condition:
          - task: "impl_evaluator"
            status: "success"         # Check AgentResponse.success
          - context_field: "review_outcome.needs_rework" # Path in context
            operator: "equals"
            value: false
        next_team: null # End workflow

      # Example 3: Simple task success check
      - condition:
          task: "impl_evaluator"
          status: "success"
        next_team: "manual_approval"

    default: "handle_evaluation_error" # Fallback team

* Supported `operator` values should include `equals`, `not_equals`, `contains`, `greater_than`, `less_than`, `exists`, `is_empty`, etc.
* Accessing task output requires the `AgentResponse.data` to be structured predictably.

#### 3.4.2. Context-Aware Recursion Mechanism

(Unchanged from v1.2 - routing task returns `next_team_id` and `context_updates`, main flow merges updates before next team execution).

#### 3.4.3. Prefect Implementation Refactoring (`c4h_services` - Updated)

* **`render_config` Utility (New):**
    * A utility function (e.g., in `utils/config.py`) responsible for:
        1.  Taking a list of configuration fragments (system, environment, job, injected persona).
        2.  Performing JSON Schema validation on each fragment (R13).
        3.  Performing the `deep_merge` operation.
        4.  Expanding environment variable placeholders (e.g., `${VAR}`).
        5.  Expanding secrets placeholders (e.g., `secret://path/to/secret`).
        6.  Serializing the final dictionary to YAML (`yaml.safe_dump(..., sort_keys=True)`).
        7.  Saving the YAML content to a uniquely named file (e.g., `effective_config_{hash}.yaml`) within the run's workspace directory (`workspaces/{run_id}/`).
        8.  Returning the `Path` to the saved snapshot file.
* **`materialise_config (@task)` (New Task):**
    * Called early in `run_declarative_workflow`.
    * Collects all relevant config fragments (base system config, environment overlays, job-specific fragments from context).
    * Checks for a `persona:` key in fragments; if found, loads the corresponding persona file (`personas/<key>.yml`) and prepends it to the fragment list.
    * Calls the `render_config` utility.
    * Returns the *path* to the generated `effective_config.yaml` snapshot.
* **`run_declarative_workflow (@flow)`:**
    * Takes initial context (containing job fragments) and base system config path.
    * Calls `materialise_config` task first to get the `effective_config_path`.
    * Loads the effective config from the snapshot path.
    * Initializes safety counters.
    * Main loop:
        * Checks safety limits.
        * Calls `execute_team_subflow`, passing the loaded `effective_config` dictionary and `current_context`.
        * Calls `evaluate_routing_task`, passing results and the loaded `effective_config`.
        * Applies `context_updates`.
        * Updates `current_team_id`.
    * Handles termination.
* **`execute_team_subflow (@flow)`:**
    * Takes `team_id`, loaded `effective_config`, `current_context`.
    * Looks up team tasks in `effective_config`.
    * Iterates through tasks, calling `run_agent_task`, passing the `effective_config` and `current_context`.
    * Collects results.
    * Returns results.
* **`run_agent_task (@task)`:**
    * Takes `task_config` (from effective config), loaded `effective_config`, `current_context`.
    * Instantiates the `AgentFactory` with the loaded `effective_config` dictionary.
    * Calls `factory.create_agent(task_config)` to get the agent instance (factory passes the full effective config to the agent).
    * Calls agent's `process` method with `current_context`.
    * Returns `AgentResponse`.
* **`evaluate_routing_task (@task)`:**
    * Takes `team_results`, `current_context`, `routing_config` (from effective config).
    * Evaluates conditions against results and context (potentially reading from the effective config if needed).
    * Returns `{'next_team_id': '...', 'context_updates': {...}}`.

#### 3.4.4. Recursion Safety Controls (New Section)

(Unchanged from v1.2 - implement `max_total_teams` and `max_recursion_depth` checks in `run_declarative_workflow`).

### 3.5. ~~Backward Compatibility Strategy~~ (REMOVED)

(Unchanged from v1.2 - Section removed).

### 3.6. Agent vs. Skill Roles (Clarified)

(Unchanged from v1.2 - Agents are orchestrated participants defined by Personas/snapshots; Skills are reusable operational capabilities).

This refined design incorporates the snapshotting and path-addressable config suggestions, leading to a more robust, reproducible, and maintainable system while strictly adhering to the clean design principles.
