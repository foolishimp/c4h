workorder:
  project:
    path: /Users/jim/src/apps/c4h_ai_dev
    workspace_root: workspaces
  intent:
    description: |
      # WO-HYDRA-1: Configuration Restructuring for Hydra Integration

      ## Goal
      To refactor the C4H project's configuration file structure by decomposing the monolithic `system_config.yml` and reorganizing all YAML files into a modular, Hydra-compliant `conf/` directory structure. This is the foundational first phase of migrating the system to use the Hydra framework.

      ## Current Implementation Context
      The project currently uses a single, large `config/system_config.yml` file to define most of the system's behavior, including LLM providers, orchestration teams, and skill definitions. Agent-specific "personas" are located in a separate `config/personas/` directory. This structure is not composable and makes configuration management complex.

      ## Required Changes
      This work order involves only file system operations: moving, creating, and deleting YAML configuration files. No Python code should be altered.

      1.  **Create New Directory Structure:**
          * A new top-level directory named `conf` MUST be created at the project root (`/Users/jim/src/apps/c4h_ai_dev/conf`).
          * Inside `conf`, the following subdirectories MUST be created to serve as Hydra "Config Groups":
              * `conf/team/`
              * `conf/persona/`
              * `conf/provider/`

      2.  **Decompose `system_config.yml`:**
          * The contents of `config/system_config.yml` MUST be broken apart and moved into the new structure.
          * **Teams:** Each team defined under the `orchestration.teams` key MUST be moved into its own YAML file inside `conf/team/`. For example, the `discovery` team's configuration should be moved to `conf/team/discovery.yaml`.
          * **Providers:** Each provider defined under `llm_config.providers` (e.g., `anthropic`, `openai`) MUST be moved into its own YAML file inside `conf/provider/`. For example, the `anthropic` provider configuration goes into `conf/provider/anthropic.yaml`.

      3.  **Relocate Persona Files:**
          * All existing persona files from `config/personas/` (e.g., `coder_v1.yml`, `discovery_v1.yml`) MUST be moved to the new `conf/persona/` directory.

      4.  **Create Main `config.yaml`:**
          * A new primary Hydra configuration file MUST be created at `conf/config.yaml`.
          * This file will define the default composition of the system. It MUST contain a `defaults` list to specify the default team, persona, and provider to load.
          * It MUST also establish the top-level structure of the configuration, using Hydra's interpolation syntax (`${...}`) to reference the decomposed files.
          * **Template for `conf/config.yaml`:**
              ```yaml
              # Default composition for the C4H system
              defaults:
                - team: discovery
                - persona: discovery_v1
                - provider: anthropic
                - _self_

              # Placeholder for runtime-injected project path and intent
              project:
                path: ???

              intent:
                description: ???

              # Global settings moved from the old system_config.yml
              logging:
                level: "debug"
                format: "structured"

              runtime:
                # ... runtime settings ...

              backup:
                # ... backup settings ...

              llm_config:
                default_provider: "anthropic"
                default_model: "claude-3-5-sonnet-20241022"
                providers:
                  # Use interpolation to include the provider config
                  anthropic: ${provider.anthropic}

              orchestration:
                enabled: true
                entry_team: "discovery"
                teams:
                  # Use interpolation to compose teams
                  discovery: ${team.discovery}
                  solution: ${team.solution_designer} # Example
                  coder: ${team.coder}         # Example
              ```

      5.  **Cleanup Old Configuration:**
          * After all contents have been successfully moved, the entire original `config/` directory, including `system_config.yml` and the `personas/` subdirectory, MUST be deleted.

      ## Design Principles & Context
      - **MANDATORY:** This refactoring MUST strictly adhere to the requirements outlined in the **Hydra Integration Design & Requirements** document, specifically **Functional Requirement FR-1**.
      - The new structure directly maps to Hydra's Config Group pattern, where each subdirectory under `conf/` represents a group of swappable components.
      - The `defaults` list in `conf/config.yaml` is the declarative replacement for the old system's manual merging logic.

      ## Files to Modify/Create
      - **Create:** `conf/config.yaml`
      - **Create:** `conf/team/discovery.yaml`, `conf/team/solution.yaml`, etc.
      - **Create:** `conf/provider/anthropic.yaml`, `conf/provider/openai.yaml`, etc.
      - **Move:** All files from `config/personas/*.yml` to `conf/persona/`.
      - **Delete:** The entire `config/` directory.
llm_config:
  agents:
    discovery:
      tartxt_config:
        input_paths:
          - /Users/jim/src/apps/c4h_ai_dev/config/system_config.yml
          - /Users/jim/src/apps/c4h_ai_dev/config/personas/
          - /Users/jim/src/apps/c4h_projects/docs/design_docs/c4h_unified_architecture_doc.md
          - /Users/jim/src/apps/c4h_ai_dev/immersives/hydra-refactoring-requirements-v1.md
        exclusions:
          - '**/__pycache__/**'
          - '**/.git/**'
