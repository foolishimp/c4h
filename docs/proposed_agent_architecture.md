# Proposed Type-Based Agent Architecture

This document outlines a mapping from the current persona-based architecture to a more type-based approach using the following agent types:

- `GenericLLMAgent` - For general single-shot LLM interactions
- `GenericOrchestratorAgent` - For coordinating multi-step processes
- `GenericSkillAgent` - For specialized processing tasks
- `GenericFallbackAgent` - For handling failures and fallback scenarios

## Current Architecture

The current architecture uses a persona-centric approach where agent behavior is primarily defined by personas configured in `system_config.yml`. The personas contain LLM settings, prompts, and other configuration that define agent behavior.

The orchestration is managed by teams that define sequences of agents with different personas.

## Proposed Type-Based Architecture

### 1. Agent Type Mapping

| Current Persona/Agent | Proposed Type | Purpose |
|----------------------|---------------|---------|
| discovery_phase | GenericLLMAgent | Project analysis using LLM |
| solution_designer | GenericLLMAgent | Generating code change solutions |
| coder | GenericLLMAgent | Implementing code changes |
| semantic_extract | GenericSkillAgent | Precise extraction of structured data |
| semantic_iterator | GenericOrchestratorAgent | Coordinating extraction process |
| semantic_fast_extractor | GenericSkillAgent | Efficient bulk extraction |
| semantic_slow_extractor | GenericSkillAgent | Sequential parsing for complex content |
| semantic_merge | GenericSkillAgent | Applying diffs to files |
| fallback_coding_phase | GenericFallbackAgent | Conservative fallback implementation |

### 2. Configuration Updates

The `system_config.yml` would be updated to include a `type` field for each agent:

```yaml
# Current configuration style
discovery:
  name: "Discovery Agent"
  prompts:
    system: "You are a project discovery agent..."

# Proposed configuration style 
discovery:
  type: "GenericLLMAgent"  # New type field
  name: "Discovery Agent"
  prompts:
    system: "You are a project discovery agent..."
```

### 3. Team Structure Changes

```yaml
# Example updated team configuration
teams:
  # Discovery team - analyzes project structure
  discovery:
    name: "Discovery Team"
    tasks:
      - name: "discovery_phase"
        agent_type: "GenericLLMAgent"  # Changed from "generic_single_shot"
        persona_key: "discovery"
        description: "Analyze project structure and files"
        requires_approval: false
        max_retries: 2
    routing:
      default: "solution"  # Go to solution team next
```

## Implementation Details

### GenericLLMAgent

This agent performs single-shot LLM interactions based on persona-defined prompts, replacing the current `GenericSingleShotAgent`. It would maintain the same core behavior but with clearer typing.

```python
class GenericLLMAgent(BaseAgent):
    """
    Agent that performs single LLM interactions based on persona configuration.
    Used for generating content, analysis, and other general LLM tasks.
    """
    # Implementation similar to current GenericSingleShotAgent
```

### GenericOrchestratorAgent

This agent would be extended from the current `GenericOrchestratingAgent` but with improved execution plan handling and better support for conditional execution branches.

```python
class GenericOrchestratorAgent(BaseAgent):
    """
    Agent that orchestrates multi-step processes based on an execution plan.
    Used for coordinating complex workflows involving multiple skills or sub-agents.
    """
    # Enhanced implementation based on GenericOrchestratingAgent
```

### GenericSkillAgent

This would be a new agent type focused on specific data processing tasks with minimal LLM interaction:

```python
class GenericSkillAgent(BaseAgent):
    """
    Agent specialized in specific data processing tasks with minimal LLM interaction.
    Used for extraction, transformation, validation, and other specialized processing.
    """
    # Implementation focused on processing capabilities
```

### GenericFallbackAgent

This would be a new agent type specifically designed to handle error cases with more conservative parameters:

```python
class GenericFallbackAgent(BaseAgent):
    """
    Agent designed for handling failure cases with conservative parameters.
    Used as a fallback when primary agents fail, with stricter bounds and safer defaults.
    """
    # Implementation with conservative defaults and specialized error handling
```

## Preserving Current Functionality

To preserve current functionality while transitioning to the type-based architecture:

1. **Configuration Compatibility:** The new agent types would still read persona configurations, ensuring backward compatibility
2. **Factory Method:** Update the agent factory to create the appropriate agent type based on the `agent_type` field in the task configuration
3. **Lineage Tracking:** Maintain the same lineage tracking across all agent types
4. **Skill Integration:** Ensure the new agent types can call skills with proper lineage tracking

## Migration Strategy

1. **Phase 1:** Add type field to agent configurations without changing behavior
2. **Phase 2:** Implement the new agent types while maintaining compatibility 
3. **Phase 3:** Update orchestration to use the new agent types
4. **Phase 4:** Enhance agents with type-specific optimizations

## Benefits of Type-Based Architecture

1. **Clearer Intent:** Agent type clearly communicates its purpose and behavior
2. **Type-Specific Optimization:** Allows optimizing each agent type for its specific purpose
3. **Simplified Configuration:** Less reliance on personas for defining agent behavior
4. **Better Error Handling:** Specialized fallback agent for handling failures
5. **Improved Documentation:** Clearer documentation based on agent types
6. **Enhanced Testing:** More focused testing based on agent type

This approach maintains the flexibility of the current design while providing better structure and more explicit typing of agent behaviors.