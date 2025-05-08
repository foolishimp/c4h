# Implementation Plan for Work Order WO-trefac.2-02

## Work Order Reference
- **ID**: WO-trefac.2-02
- **Date**: May 8, 2025
- **File Path**: `/Users/jim/src/apps/c4h_data/repositories/workorders/WO-trefac.2-02.json`

## Overview
Work Order WO-trefac.2-02 requires refactoring the Discovery agent functionality to align with the C4H Unified Architecture (Version 3.0). This involves removing special handling from GenericLLMAgent, implementing tartxt as a Skill, creating a new Persona for GenericSkillAgent, and updating the default configuration.

## Current State Analysis

1. **TASK-AGENT-R-001: Refactor GenericLLMAgent**
   - **Status**: COMPLETE
   - **Evidence**: The code in `c4h_agents/agents/generic.py` shows comments indicating that the tartxt execution logic has been removed:
     ```python
     # REMOVED: Special handling for discovery agent with tartxt.py execution.
     # This functionality is now available via the tartxt_runner skill.
     ```

2. **TASK-AGENT-R-002: Implement tartxt.py as a Registered Skill**
   - **Status**: PARTIALLY COMPLETE
   - **Details**: A `TartXTRunner` skill has been implemented in `c4h_agents/skills/tartxt_runner.py`, but it is marked as deprecated in favor of the `CommandLineRunner` skill. We need to determine if we should use the existing implementation or create a new one as specified in the work order.

3. **TASK-AGENT-R-003: Create New Persona for Skill-Based Discovery**
   - **Status**: NOT STARTED
   - **Details**: The `config/personas/discovery_by_skill_v1.yml` file doesn't exist and needs to be created according to the work order requirements.

4. **TASK-AGENT-R-004: Update Default Discovery Task Configuration**
   - **Status**: NOT STARTED
   - **Details**: The `config/system_config.yml` file still references `tartxt_config` for the discovery agent. This needs to be updated to use the new skill-based approach.

## Implementation Tasks

### 1. Verify GenericLLMAgent Refactoring

- **Objective**: Ensure that all tartxt-specific code has been completely removed from `GenericLLMAgent`.
- **Actions**:
  - Review the entire `GenericLLMAgent` class implementation in `c4h_agents/agents/generic.py`
  - Verify that no tartxt-related logic remains beyond the comments
  - If any tartxt code remains, remove it as specified in TASK-AGENT-R-001

### 2. Evaluate TartXTRunner Skill Implementation

- **Objective**: Decide whether to use the existing TartXTRunner or create a new implementation.
- **Actions**:
  - Compare the existing `TartXTRunner` implementation against the requirements in TASK-AGENT-R-002
  - Review the `CommandLineRunner` skill to understand why TartXTRunner is deprecated
  - Make a decision to either:
    - Use the existing TartXTRunner implementation (if it meets requirements)
    - Create a new implementation that isn't deprecated
    - Use CommandLineRunner with tartxt-specific configuration

### 3. Create Discovery-by-Skill Persona

- **Objective**: Create the new persona configuration as specified in TASK-AGENT-R-003.
- **Actions**:
  - Create the file `config/personas/discovery_by_skill_v1.yml`
  - Implement the persona configuration with the exact structure specified in the work order
  - Ensure it uses the appropriate skill (TartXTRunner or CommandLineRunner based on decision from task 2)

### 4. Update System Configuration

- **Objective**: Update the discovery task configuration as specified in TASK-AGENT-R-004.
- **Actions**:
  - Modify `config/system_config.yml`
  - Locate the discovery task within the orchestration.teams section
  - Update the agent_type to `GenericSkillAgent`
  - Change the persona_key to `discovery_by_skill_v1`
  - Remove any tartxt_config sections from the task definition

## Testing Plan

1. **Functional Testing**:
   - Create a test script that exercises the new discovery agent configuration
   - Verify that it correctly uses the skill to run tartxt.py
   - Confirm that the discovered project content is correctly processed

2. **Integration Testing**:
   - Test the discovery agent within the orchestration workflow
   - Ensure it properly passes discovered content to the next agent in the chain
   - Verify that lineage tracking works correctly with the new implementation

## Risks and Considerations

1. **Backwards Compatibility**: The refactoring might affect existing workflows that depend on the discovery agent.
2. **TartXTRunner Deprecation**: We need to carefully consider whether to use a deprecated component or implement an alternative.
3. **Configuration Dependencies**: Other parts of the system might reference the tartxt_config structure and need updates.

## Success Criteria

1. The discovery agent functionality operates through the skill registry mechanism
2. The GenericLLMAgent contains no tartxt-specific code
3. The system configuration uses the new persona and skill approach
4. All tests pass with the new implementation

## Implementation Sequence

1. Verify GenericLLMAgent refactoring (TASK-AGENT-R-001)
2. Evaluate TartXTRunner skill implementation (TASK-AGENT-R-002)
3. Create Discovery-by-Skill persona (TASK-AGENT-R-003)
4. Update System Configuration (TASK-AGENT-R-004)
5. Test the implementation

## Verification and State Restoration

To evaluate success and potentially restore state if needed:

1. Original work order location: `/Users/jim/src/apps/c4h_data/repositories/workorders/WO-trefac.2-02.json`
2. Key files to backup before implementation:
   - `/Users/jim/src/apps/c4h_ai_dev/c4h_agents/agents/generic.py`
   - `/Users/jim/src/apps/c4h_ai_dev/c4h_agents/skills/tartxt_runner.py` 
   - `/Users/jim/src/apps/c4h_ai_dev/config/system_config.yml`
3. If restoration is needed, compare against the original work order requirements and restore from backups.