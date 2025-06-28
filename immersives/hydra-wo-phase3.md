workorder:
  project:
    path: /Users/jim/src/apps/c4h_ai_dev
    workspace_root: workspaces
  intent:
    description: |
      # WO-HYDRA-3: Refactor Agent Code and Deprecate Custom Config System

      ## Goal
      To complete the Hydra integration by refactoring all agent classes to consume the `OmegaConf.DictConfig` object directly, and subsequently deleting the now-redundant custom configuration system (`config.py`, `ConfigNode`, etc.). This will finalize the migration, resulting in a cleaner, more maintainable codebase that fully leverages the Hydra framework.

      ## Current Implementation Context
      - Per `WO-HYDRA-2`, the API service now uses Hydra's Compose API to create a job-specific `OmegaConf.DictConfig` object.
      - However, the core agent classes (`BaseAgent`, `GenericLLMAgent`, etc.) still expect a standard Python `dict` and use a custom `ConfigNode` utility with a `get_value("path.to.key")` method to access configuration values.
      - The custom configuration modules (`c4h_agents/config.py`, `c4h_agents/utils/config_validation.py`, `c4h_agents/utils/config_materializer.py`) are still present in the codebase, even though their primary consumer (the API service's merge logic) has been replaced.

      ## Required Changes
      This work order involves a direct refactoring of the agent-layer Python code, followed by the deletion of legacy files.

      1.  **Refactor `BaseAgent` and `BaseConfig`:**
          * **Files to Modify:** `c4h_agents/agents/base_config.py` and `c4h_agents/agents/base_agent.py`.
          * **Action:** The `__init__` methods MUST be updated to accept and store the `OmegaConf.DictConfig` object directly in `self.config`.
          * The `self.config_node` attribute and its creation via `create_config_node` MUST be removed.

      2.  **Update Configuration Access Pattern in All Agents:**
          * **Files to Modify:** All agent classes inheriting from `BaseAgent` (`generic.py`, `base_llm.py`, skills like `semantic_iterator.py`, etc.).
          * **Action:** All method calls used to access configuration values MUST be replaced with Hydra's native dot-path attribute access.
          * **Migration Pattern:**
              * **BEFORE:** `self.config_node.get_value("llm_config.default_provider")`
              * **AFTER:** `self.config.llm_config.default_provider`
          * This change must be applied consistently across all agents and skills that read from the configuration.

      3.  **Deprecate and Delete Custom Configuration System:**
          * **Action:** Once all agent code has been refactored to use the new access pattern, the following files and their associated logic are obsolete and MUST be deleted from the project:
              * `c4h_agents/config.py`
              * `c4h_agents/utils/config_validation.py`
              * `c4h_agents/utils/schema_validation.py`
              * `c4h_agents/utils/config_materializer.py`
          * **Action:** The Prefect task `materialise_config` in `c4h_services/src/intent/impl/prefect/tasks.py` MUST be deleted.

      ## Design Principles & Context
      - **MANDATORY:** This refactoring MUST strictly adhere to the requirements outlined in the **Hydra Integration Design & Requirements** document, specifically **Functional Requirements FR-3** (Agent Consumption) and **FR-4** (Deprecation).
      - This final step realizes the full benefit of the Hydra migration by removing technical debt and simplifying the agent codebase.
      - The direct use of the `OmegaConf.DictConfig` object provides type-safety hints and a more pythonic way to access configuration.

      ## Files to Modify/Create
      - **Modify:** `c4h_agents/agents/base_config.py`
      - **Modify:** `c4h_agents/agents/base_agent.py`
      - **Modify:** `c4h_agents/agents/base_llm.py`
      - **Modify:** `c4h_agents/agents/generic.py`
      - **Modify:** All skills that inherit from `BaseSkill`/`BaseConfig` and access config.
      - **Delete:** `c4h_agents/config.py`
      - **Delete:** `c4h_agents/utils/config_validation.py`, `schema_validation.py`, `config_materializer.py`
      - **Modify:** `c4h_services/src/intent/impl/prefect/tasks.py` (to remove `materialise_config`)
llm_config:
  agents:
    discovery:
      tartxt_config:
        input_paths:
          - /Users/jim/src/apps/c4h_ai_dev/c4h_agents/agents/
          - /Users/jim/src/apps/c4h_ai_dev/c4h_agents/skills/
          - /Users/jim/src/apps/c4h_ai_dev/c4h_agents/utils/
          - /Users/jim/src/apps/c4h_ai_dev/c4h_services/src/intent/impl/prefect/tasks.py
          - /Users/jim/src/apps/c4h_ai_dev/c4h_agents/config.py
          - /Users/jim/src/apps/c4h_ai_dev/immersives/hydra-refactoring-requirements-v1.md
          - /Users/jim/src/apps/c4h_ai_dev/immersives/hydra-wo-phase2.md
        exclusions:
          - '**/__pycache__/**'
          - '**/.git/**'
