workorder:
  project:
    path: /Users/jim/src/apps/c4h_ai_dev
    workspace_root: workspaces
  intent:
    description: |
      # WO-HYDRA-2: Refactor API Service to Use Hydra Compose API

      ## Goal
      To refactor the C4H FastAPI service (`c4h_services/src/api/service.py`) to replace its custom configuration merging logic with Hydra's programmatic **Compose API**. This change is critical for enabling dynamic, per-request configuration composition in a robust and standardized way.

      ## Current Implementation Context
      - Per `WO-HYDRA-1`, all YAML configuration files have been relocated to a modular `conf/` directory structure.
      - The `create_job` endpoint in `c4h_services/src/api/service.py` currently receives a `MultiConfigJobRequest`. It manually iterates through the provided configuration fragments and uses a custom `deep_merge` function (from `c4h_agents/config.py`) to create a job-specific configuration.
      - This manual process is to be replaced entirely.

      ## Required Changes
      This work order focuses on modifying the Python code of the API service.

      1.  **Add New Dependencies:**
          * The project (or the specific `c4h_api_service` library if modularized) MUST add `hydra-core` and `omegaconf` to its dependencies.

      2.  **Refactor `create_job` Endpoint:**
          * **File to Modify:** `c4h_services/src/api/service.py`
          * **Action:** The core logic inside the `create_job` function MUST be replaced.
          * **Remove Custom Logic:** The `for` loop that iterates through `request.configs` and calls `deep_merge` MUST be deleted.
          * **Implement Hydra Compose API:**
              1.  The function MUST use a `with hydra.initialize(...)` block to set the configuration path to the `conf/` directory.
              2.  Inside the block, it MUST first call `hydra.compose(config_name="config")` to load the default configuration, creating a base `OmegaConf.DictConfig` object.
              3.  It MUST then iterate through the fragments in the incoming `request.configs`. For each fragment, it will create a new `OmegaConf` object and merge it onto the base configuration using `OmegaConf.merge()`.
              4.  The final, merged `OmegaConf.DictConfig` object is the "Effective Configuration Snapshot" for the job.
              5.  This object MUST be converted to a standard Python dictionary using `OmegaConf.to_container(cfg, resolve=True)` before being passed to the `Orchestrator`.

      3.  **Update `create_app` Function (in `service.py`):**
          * The `create_app` function initializes the `Orchestrator` with a static, default configuration. This is now incorrect, as each job will have a unique configuration.
          * **Action:** The `Orchestrator` instantiation MUST be moved *inside* the `create_job` endpoint. The `app.state.orchestrator` should be removed, as the orchestrator is no longer a static, shared resource but is created dynamically for each job with its unique configuration.

      ## Design Principles & Context
      - **MANDATORY:** This refactoring MUST strictly adhere to the requirements outlined in the **Hydra Integration Design & Requirements** document, specifically **Functional Requirement FR-2**.
      - The use of the **Compose API** is non-negotiable as it preserves the essential dynamic nature of the API service, allowing each job request to define its own unique configuration.
      - This change effectively decouples the API service from the mechanics of configuration merging.

      ## Files to Modify/Create
      - **Modify:** `c4h_services/src/api/service.py`
      - **Context Reference:** The custom `deep_merge` logic to be replaced is located in `c4h_agents/config.py`.
llm_config:
  agents:
    discovery:
      tartxt_config:
        input_paths:
          - /Users/jim/src/apps/c4h_ai_dev/c4h_services/src/api/service.py
          - /Users/jim/src/apps/c4h_ai_dev/c4h_agents/config.py
          - /Users/jim/src/apps/c4h_ai_dev/immersives/hydra-refactoring-requirements-v1.md
          - /Users/jim/src/apps/c4h_ai_dev/immersives/hydra-wo-phase1.md
        exclusions:
          - '**/__pycache__/**'
          - '**/.git/**'
