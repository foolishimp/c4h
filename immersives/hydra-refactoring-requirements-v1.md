### ---

**Refactoring Design & Requirements: C4H Hydra Integration**

#### **1\. Goal & Objective**

The primary goal of this refactoring is to **replace the C4H project's custom-built configuration system with the Hydra framework**.

This initiative will formalize the project's configuration-driven philosophy by adopting an industry-standard tool, leading to a more robust, maintainable, and flexible system. The refactoring must preserve all existing application and API functionality, particularly the service's ability to dynamically compose configurations for each job request.

#### **2\. Current Implementation Context**

The current system relies on a set of custom components to manage its configuration:

* **Manual Loading:** A script (prefect\_runner.py) or an API service (api/service.py) manually loads one or more YAML files.  
* **Custom Merging:** A bespoke deep\_merge function in c4h\_agents/config.py is used to combine these files into an "Effective Configuration Snapshot".  
* **Custom Access Layer:** A ConfigNode class provides dot-path traversal and access to the configuration dictionary.  
* **Dynamic API Composition:** The /api/v1/jobs endpoint accepts a list of configuration fragments (MultiConfigJobRequest) and merges them at request time to create a unique configuration for each job1111.

While functional, this custom implementation increases the maintenance burden and lacks the advanced features of a dedicated framework.

#### **3\. Architectural Principles for the Refactoring**

The refactoring must adhere to the following principles:

1. **Preserve All Functionality:** The refactored system must expose the same capabilities as the original, especially the dynamic, per-request configuration merging in the API service.  
2. **Embrace Hydra's Structure:** The configuration files must be reorganized to fit Hydra's conventional conf/ directory structure with modular config groups.  
3. **Adopt the Compose API for Services:** The FastAPI service must use Hydra's programmatic **Compose API** to handle dynamic job requests, not the @hydra.main decorator.  
4. **Simplify Agent Code:** Agent classes should be simplified by removing the custom ConfigNode and directly using the OmegaConf.DictConfig object provided by Hydra.  
5. **Maintain Library Modularity:** The refactoring should respect the planned separation of concerns into c4h\_core, c4h\_prefect\_integration, and c4h\_api\_service libraries.

### ---

**4\. Required Changes (Phased for Work Orders)**

This section details the specific changes required, organized into phases that can be executed as sequential work orders.

#### **Phase 1: Restructure YAML Configuration**

*Goal: Reorganize all YAML files into a Hydra-compliant conf/ directory structure.*

* **Work Order 1.1: Create Hydra Directory Structure**  
  * **Action:** Create a new top-level conf/ directory. Inside it, create subdirectories for each configuration group: conf/team/, conf/persona/, and conf/provider/.  
* **Work Order 1.2: Decompose system\_config.yml**  
  * **Action:** Take the monolithic system\_config.yml and break its contents into smaller, modular files.  
    * Each team definition under orchestration.teams becomes a separate file (e.g., conf/team/discovery.yaml, conf/team/solution.yaml).  
    * Each provider definition under llm\_config.providers becomes a separate file (e.g., conf/provider/anthropic.yaml).  
    * Global settings (logging, runtime, backup) can remain in the main config.yaml or be moved to their own files (e.g., conf/runtime/default.yaml).  
* **Work Order 1.3: Relocate Persona Files**  
  * **Action:** Move all existing persona YAML files from config/personas/ to the new conf/persona/ directory.  
* **Work Order 1.4: Create the Main config.yaml**  
  * **Action:** Create the primary Hydra configuration file, conf/config.yaml, which will define the default composition of the system.  
  * **Content Template:**  
    YAML  
    defaults:  
      \- team: discovery  
      \- persona: discovery\_v1  
      \- provider: anthropic  
      \- \_self\_

    \# Project and Intent are expected to be overridden at runtime  
    project:  
      path: ???  
    intent:  
      description: ???

    \# Other base configs  
    logging:  
      level: "debug"  
      format: "structured"

    llm\_config:  
      default\_provider: "anthropic"  
      default\_model: "claude-3-5-sonnet-20241022"  
      providers:  
        anthropic: ${provider.anthropic} \# Use Hydra's interpolation

    orchestration:  
      enabled: true  
      entry\_team: "discovery"  
      teams:  
        discovery: ${team.discovery} \# Interpolate team configs  
        solution: ${team.solution\_designer}  
        coder: ${team.coder}

#### **Phase 2: Refactor Code to Integrate Hydra**

*Goal: Remove all custom configuration-handling code and replace it with calls to the Hydra framework.*

* **Work Order 2.1: Refactor the API Service Endpoint**  
  * **File to Modify:** c4h\_api\_service/api/service.py  
  * **Action:** In the create\_job endpoint, replace the manual deep\_merge loop with Hydra's Compose API.  
  * **Logic:**  
    1. Receive the MultiConfigJobRequest as before.  
    2. Inside a with hydra.initialize(...) block, call hydra.compose(config\_name="config") to get the base configuration.  
    3. Iterate through the request.configs list. For each fragment, create an OmegaConf object and merge it into the base config using OmegaConf.merge().  
    4. The resulting cfg object is the job-specific "Effective Configuration Snapshot." Pass this object (as a dict) to the Orchestrator.  
* **Work Order 2.2: Refactor Agent Configuration Access**  
  * **Files to Modify:** c4h\_core/agents/base\_agent.py, c4h\_core/agents/generic.py, and all other agent classes.  
  * **Action:** Replace the custom ConfigNode with direct access on the OmegaConf.DictConfig object.  
  * **Logic:**  
    1. Modify BaseConfig and BaseAgent to accept and store the OmegaConf.DictConfig object in self.config.  
    2. The self.config\_node attribute should be removed.  
    3. All calls like self.config\_node.get\_value("path.to.key") must be replaced with direct attribute access: self.config.path.to.key. This change will propagate to all agent classes inheriting from BaseAgent.  
* **Work Order 2.3: Deprecate and Delete Custom Logic**  
  * **Action:** Once all consumers are updated, the following components are no longer needed and must be deleted:  
    1. The file c4h\_agents/config.py (containing ConfigNode, deep\_merge, load\_config, etc.).  
    2. The Prefect task materialise\_config in c4h\_prefect\_integration/tasks.py. Its function is now performed by Hydra at the start of every run.

---

This document provides the complete context and requirements needed to perform the refactoring. By breaking it down into the phases and work orders described above, an LLM can be tasked with generating the precise code modifications for each step of the migration to Hydra.

