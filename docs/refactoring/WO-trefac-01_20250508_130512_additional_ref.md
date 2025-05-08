# Refactoring Plan for C4H System Refactoring - Phase 1 (WO-trefac-01)

## Assessment Summary

Based on the analysis of the implementation so far, several key components of the Phase 1 refactoring have been partially implemented but require completion:

1. **TASK-CORE-001**: The ExecutionPlanExecutor class structure is in place but lacks complete implementation of core execution methods.
2. **TASK-CORE-002**: The execution_plan YAML schema has been successfully implemented.
3. **TASK-CORE-003**: The configuration loading and snapshot generation refactoring was not addressed.
4. **TASK-CORE-004**: The Skill Registry implementation is reasonably complete.
5. **TASK-CORE-005**: The Prefect wrappers are partially implemented.

## Remediation Plan

### 1. Complete ExecutionPlanExecutor Implementation
- **Priority**: High
- **Tasks**:
  - Implement `_execute_skill_call()` with proper skill registry integration
  - Implement `_execute_agent_call()` with agent factory pattern
  - Implement `_execute_team_call()` with proper recursion and context handling
  - Implement `_execute_llm_call()` with provider selection
  - Ensure proper error handling and result processing for all execution types
  - Add comprehensive logging for execution tracing

### 2. Refactor Configuration Loading & Snapshot Generation
- **Priority**: Critical
- **Tasks**:
  - Update `materialise_config()` to handle new ArchDocV3 structures
  - Add support for personas explicitly declaring agent_type
  - Implement team configuration with internal execution_plans and agents lists
  - Ensure backward compatibility with existing configurations
  - Add validation for new configuration structures
  - Update tests to cover new configuration scenarios

### 3. Complete Prefect Integration
- **Priority**: Medium
- **Tasks**:
  - Implement missing wrappers for agent/team execution
  - Ensure proper task dependency management
  - Update existing workflow definitions to use new execution model
  - Add proper error handling and retry logic
  - Implement proper status reporting and monitoring

### 4. Ensure End-to-End Testing
- **Priority**: Medium
- **Tasks**:
  - Create test cases for recursive execution plans
  - Test skill registry auto-discovery and instantiation
  - Verify configuration loading with new structures
  - End-to-end test of execution flow with all step types
  - Load testing for complex execution plans

### 5. Documentation and Examples
- **Priority**: Low
- **Tasks**:
  - Document new execution model
  - Create examples of execution plans for common use cases
  - Update existing documentation to reflect new architecture
  - Add migration guide for moving from old to new model

## Implementation Approach

1. **Start with the Configuration Refactoring**:
   - This is the foundation for everything else
   - Focus on backward compatibility while enabling new structures
   - Add comprehensive validation for new configurations

2. **Complete the ExecutionPlanExecutor**:
   - Implement each step type one by one
   - Start with skill_call which has most of the groundwork laid
   - Build up to more complex types like team_call and loop

3. **Integrate with Prefect**:
   - Once execution and configuration are working, wire up Prefect
   - Ensure proper task/flow relationships for parallel execution
   - Implement proper status tracking and reporting

4. **Test and Document**:
   - Create comprehensive test coverage
   - Update documentation as each component is completed
   - Create examples of each execution type

## Timeline Estimate
- Configuration Refactoring: 2-3 days
- ExecutionPlanExecutor Implementation: 3-4 days
- Prefect Integration: 1-2 days
- Testing and Documentation: 2-3 days

**Total Estimate**: 8-12 days to complete all missing/incomplete requirements

## Technical Debt Already Addressed

As part of the initial work, we've successfully reduced some technical debt:
- Implemented a generic `CommandLineRunner` skill to replace the specialized `TartXTRunner`
- Created proper test infrastructure for the new skill
- Added support for command-line tools with various parameter styles
- Enhanced the implementation to support git-based file selection with tartxt
- Added documentation for migration from specialized to generic approach