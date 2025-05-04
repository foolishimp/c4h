# C4H Pure Type-Based Agent Architecture

## Overview

This document outlines the design for a pure type-based agent architecture for the C4H framework. The type-based architecture is a complete overhaul of the legacy class-based approach, providing a configuration-driven system that adheres to inversion of control principles. This architecture eliminates technical debt and supports complex orchestration patterns through a flexible, extensible design.

## Core Design Principles

1. **Type Over Class Inheritance**: Using type identifiers in configuration rather than class inheritance hierarchies.
2. **Inversion of Control**: The system controls agent instantiation and lifecycle, not the agents themselves.
3. **Immutable Context**: Using immutable context objects with support for snapshots and rollbacks.
4. **Configuration-Driven**: All agent behavior defined in configuration, not code.
5. **Skill Registry**: Dynamic discovery and invocation of skills without direct dependencies.
6. **Template Resolution**: Dynamic resolution of configuration values from context.
7. **JSON Schema Validation**: Strong validation of configuration at load time.

## Architecture Components

### ExecutionContext

The `ExecutionContext` class provides an immutable context with structured access patterns, supporting:

- Dot-notation path access to values
- Immutable updates returning new contexts
- Snapshots and rollbacks
- Sensitive data marking
- Serialization and deserialization

```python
# Example usage
context = ExecutionContext({'user': {'name': 'Alice'}})
updated = context.set('user.age', 30)
snapshot_id = updated.create_snapshot('user_update')
```

### Template Resolution

The `TemplateResolver` provides dynamic resolution of values in configuration:

- String templates: `"Hello, ${user.name:Anonymous}!"`
- Dictionary templates: `{"name": "${user.name:Anonymous}"}`
- List templates: `["${items[0]}", "${items[1]}"]`

```python
# Example usage
template = "Hello, ${user.name:Anonymous}!"
resolved = TemplateResolver.resolve(template, context)
```

### Skill Registry

The `SkillRegistry` provides dynamic discovery and invocation of skills:

- Loading skills from configuration
- Caching skill instances
- Invoking skills with parameters
- Handling different result formats

```python
# Example usage
skill_result = skill_registry.invoke_skill('semantic_extract', text=input_text)
```

### Agent Types

The architecture provides two core agent types:

1. **GenericLLMAgent**: Processes input using an LLM with a persona configuration.
2. **GenericOrchestratorAgent**: Orchestrates execution of skills according to a plan.

```python
# Example agent instantiation
agent = GenericLLMAgent(persona_config, context, skill_registry)
response = agent.process(input=input_text)
```

### Agent Factory

The `AgentFactory` creates agent instances based on configuration:

- Loading system and persona configurations
- Creating skill registry
- Instantiating appropriate agent types
- Registering custom agent types

```python
# Example factory usage
factory = AgentFactory('config/type_system.yml')
agent = factory.create_agent('coder', {'input': input_text})
```

## Configuration Schema

### System Configuration

The system configuration defines the available agents, skills, and orchestration:

```json
{
  "version": "1.0",
  "llm_config": {
    "provider": "anthropic",
    "model": "claude-3-opus-20240229",
    "default_parameters": {
      "temperature": 0.7,
      "max_tokens": 4000
    }
  },
  "agents": {
    "coder": {
      "agent_type": "GenericLLMAgent",
      "persona_key": "coder_v1"
    },
    "orchestrator": {
      "agent_type": "GenericOrchestratorAgent",
      "persona_key": "orchestrator_v1"
    }
  },
  "skills": {
    "semantic_extract": {
      "module": "c4h_agents.skills.semantic_extract",
      "class": "SemanticExtractor",
      "method": "execute",
      "description": "Extract semantic information from text"
    }
  },
  "orchestration": {
    "default_entry_point": "main_flow",
    "error_handling": {
      "fallback": "retry",
      "max_retries": 3
    },
    "lineage_tracking": {
      "enabled": true,
      "storage_path": "lineage"
    },
    "agents": [
      {
        "key": "step_1",
        "agent": "coder",
        "inputs": {
          "input": "${user_input}"
        },
        "outputs": {
          "result": "coder_output"
        },
        "next": {
          "agent": "step_2"
        }
      }
    ]
  }
}
```

### Persona Configuration

The persona configuration defines the behavior of an agent:

```json
{
  "version": "1.0",
  "persona_key": "coder_v1",
  "description": "Software development agent",
  "llm_configuration": {
    "system_prompt": "You are a helpful coding assistant...",
    "user_prompt_template": "Task: ${input}\nProject context: ${project}",
    "prompt_parameters": {
      "temperature": 0.2,
      "max_tokens": 8000
    }
  },
  "skill_bindings": {
    "extract": {
      "skill_key": "semantic_extract",
      "required": true
    }
  },
  "execution_plan": {
    "enabled": true,
    "steps": [
      {
        "name": "analyze_input",
        "type": "skill",
        "skill": "extract",
        "parameters": {
          "text": "${input}",
          "mode": "full"
        },
        "outputs": {
          "result": "analysis"
        }
      },
      {
        "name": "generate_response",
        "type": "skill",
        "skill": "llm_generate",
        "parameters": {
          "prompt": "Analysis: ${analysis}\nTask: ${input}"
        },
        "is_output": true
      }
    ]
  }
}
```

## Migration from Legacy Architecture

The architecture includes a configuration converter tool to migrate from the legacy class-based architecture to the type-based architecture:

1. Converting agent class names to agent types
2. Converting persona configurations
3. Converting skill references
4. Converting orchestration plans

```bash
# Example usage
python tools/convert_config.py config/legacy_system.yml config/type_system.yml --type=system
python tools/convert_config.py config/personas/legacy_coder.yml config/personas/type_coder.yml --type=persona
```

## Execution Flow

1. System loads configuration and creates agent factory
2. Factory instantiates agents based on configuration
3. Agents use skill registry to invoke skills
4. Orchestrator agents execute plans with conditional logic, loops, etc.
5. Results are returned with context updates

## Benefits

- **Reduced Technical Debt**: No inheritance hierarchies, no legacy code paths
- **Increased Flexibility**: Easy to add new agent types and skills
- **Better Testability**: Configuration-driven system is easier to test
- **Improved Maintainability**: Clear separation of concerns
- **Enhanced Extensibility**: Easy to add new features without changing core code

## Implementation

The implementation of the type-based architecture consists of:

1. Context management: execution_context.py, persistence.py, template.py
2. Agent types: type_base_agent.py, type_generic.py
3. Factory: factory.py
4. Skill registry: registry.py
5. Configuration schemas: type_system.json, type_persona.json
6. Configuration converter: convert_config.py
7. Test script: test_type_agent.py

## Conclusion

The pure type-based architecture provides a clean, maintainable, and extensible foundation for the C4H framework. It eliminates technical debt from the legacy architecture while providing a clear migration path for existing configurations.