#!/bin/bash

# Script to create the new configuration structure (v1.3 compatible)
# Run this script from the project root: /Users/jim/src/apps/c4h_ai_dev

# Define the new configuration directory name
CONFIG_DIR="config_teams_0502"
PERSONAS_SUBDIR="personas"
SCHEMAS_SUBDIR="schemas" # Added based on WO-08 output

# --- Create Directories ---
echo "Creating directory structure..."
mkdir -p "${CONFIG_DIR}/${PERSONAS_SUBDIR}"
mkdir -p "${CONFIG_DIR}/${SCHEMAS_SUBDIR}"
echo "Directories created: ${CONFIG_DIR}/, ${CONFIG_DIR}/${PERSONAS_SUBDIR}/, ${CONFIG_DIR}/${SCHEMAS_SUBDIR}/"
echo ""

# --- Create system_config.yml ---
echo "Creating ${CONFIG_DIR}/system_config.yml..."
cat << EOF > "${CONFIG_DIR}/system_config.yml"
# Main System Configuration (v1.3 compatible)

config_locations:
  personas_dir: "${PERSONAS_SUBDIR}" # Relative path to personas dir
  schemas_dir: "${SCHEMAS_SUBDIR}"   # Relative path to schemas dir

llm_config:
  providers:
    # Keep your existing provider definitions here (anthropic, openai, etc.)
    # Example:
    anthropic:
      api_base: "https://api.anthropic.com"
      context_length: 200000
      env_var: "ANTHROPIC_API_KEY"
      default_model: "claude-3-5-sonnet-20241022"
      valid_models:
        - "claude-3-7-sonnet-20250219"
        - "claude-3-5-sonnet-20241022"
        - "claude-3-opus-20240229"
        - "claude-3-sonnet-20240229"
        - "claude-3-haiku-20240307"
      extended_thinking:
        enabled: false
        budget_tokens: 32000
      litellm_params:
        retry: true
        max_retries: 5
        timeout: 30
        backoff:
          initial_delay: 1
          max_delay: 30
          exponential: true
    openai:
      # ... (Your OpenAI config) ...
    gemini:
      # ... (Your Gemini config) ...
    xai:
      # ... (Your XAI config) ...

  default_provider: "anthropic"
  default_model: "claude-3-opus-20240229" # Example default

  # Base structure/schema definition for personas
  # Actual definitions are in the personas/ directory
  personas:
    provider: null
    model: null
    temperature: 0.7
    extended_thinking:
      enabled: true
    prompts:
      system: null
      user: null
    execution_plan:
      enabled: false
      steps: []
    model_params:
      max_tokens: 4096
    # Add other common fields if needed

# Orchestration using the new factory model
orchestration:
  enabled: true
  entry_team: "discovery"
  max_total_teams: 30
  max_recursion_depth: 5
  error_handling:
    retry_teams: true
    max_retries: 2
    log_level: "ERROR"
  teams:
    discovery:
      name: "Discovery Team"
      tasks:
        - name: "discovery_task_1" # Unique instance name
          agent_type: "generic_single_shot"
          persona_key: "discovery_v1" # References personas/discovery_v1.yml
          requires_approval: false
          max_retries: 2
      routing:
        default: "solution"

    solution:
      name: "Solution Design Team"
      tasks:
        - name: "solution_design_task_1" # Unique instance name
          agent_type: "generic_single_shot"
          persona_key: "solution_designer_v1" # References personas/solution_designer_v1.yml
          requires_approval: true
          max_retries: 1
      routing:
        rules:
          - condition:
              task: "solution_design_task_1"
              status: "success"
            next_team: "coder"
        default: "fallback"

    coder:
      name: "Coder Team"
      tasks:
        - name: "coder_task_1" # Unique instance name
          # Assuming coder might need orchestration
          agent_type: "generic_orchestrating"
          persona_key: "coder_v1" # References personas/coder_v1.yml
          requires_approval: true
          max_retries: 1
      routing:
        default: null # End workflow

    fallback:
      name: "Fallback Team"
      tasks:
        - name: "fallback_coder_task_1" # Unique instance name
          agent_type: "generic_orchestrating"
          persona_key: "coder_v1" # Reuse coder persona
          config: # Task-specific override
            temperature: 0 # Make fallback more conservative
      routing:
        default: null

# Runtime configuration
runtime:
  workflow:
    storage:
      enabled: true
      root_dir: "workspaces/workflows"
      format: "yymmdd_hhmm_{workflow_id}"
      retention:
        max_runs: 10
        max_days: 30
      error_handling:
        ignore_storage_errors: true
        log_level: "ERROR"
  lineage: # Note: Lineage config might also belong under llm_config.personas in v1.3 base structure
    enabled: true
    namespace: "c4h_agents"
    separate_input_output: true
    backend:
      type: "file"
      path: "workspaces/lineage"
    error_handling:
      ignore_failures: true
      log_level: "ERROR"
    context:
      include_metrics: true
      include_token_usage: true
      record_timestamps: true
    retry:
      enabled: true
      max_attempts: 3
      initial_delay: 1
      max_delay: 30
      backoff_factor: 2
      retry_on:
        - "overloaded_error"
        - "rate_limit_error"
        - "timeout_error"

# Backup settings
backup:
  enabled: true
  path: "workspaces/backups"

# Logging configuration
logging:
  level: "debug"
  format: "structured"
  agent_level: "debug"
  providers:
    anthropic:
      level: "debug"
    openai:
      level: "debug"
  truncate:
    prefix_length: 2000
    suffix_length: 1000
EOF
echo "Created ${CONFIG_DIR}/system_config.yml"
echo ""

# --- Create Persona Files ---

# Discovery Persona
echo "Creating ${CONFIG_DIR}/${PERSONAS_SUBDIR}/discovery_v1.yml..."
cat << EOF > "${CONFIG_DIR}/${PERSONAS_SUBDIR}/discovery_v1.yml"
provider: "anthropic"
model: "claude-3-5-sonnet-20241022"
temperature: 0
tartxt_config:
  # IMPORTANT: Ensure this path is correct for the execution environment
  # If running via Docker, this might need adjustment or be relative
  script_path: "c4h_agents/skills/tartxt.py" # Relative path assumed to work via PYTHONPATH
  exclusions:
    - '**/node_modules/**'
    - '**/.git/**'
    - '**/__pycache__/**'
    - '**/*.pyc'
    - '**/package-lock.json'
    - '**/dist/**'
    - '**/.DS_Store'
    - '**/README.md'
    - '**/workspaces/**'
    - '**/backup_txt/**'
prompts:
  system: |
    You are a project discovery agent.
    You analyze project structure and files to understand:
    1. Project organization
    2. File relationships
    3. Code dependencies
    4. Available functionality
EOF
echo "Created ${CONFIG_DIR}/${PERSONAS_SUBDIR}/discovery_v1.yml"

# Solution Designer Persona
echo "Creating ${CONFIG_DIR}/${PERSONAS_SUBDIR}/solution_designer_v1.yml..."
cat << EOF > "${CONFIG_DIR}/${PERSONAS_SUBDIR}/solution_designer_v1.yml"
provider: "anthropic"
model: "claude-3-7-sonnet-20250219"
temperature: 1
extended_thinking:
  enabled: true
  budget_tokens: 32000
prompts:
  system: |
    You are a code modification solution designer that returns modifications in a clearly structured text format.
    Return your response in the following format, with no additional explanation:

    ===CHANGE_BEGIN===
    FILE: path/to/file
    TYPE: create|modify|delete
    DESCRIPTION: one line description
    DIFF:
    --- a/existing_file.py
    +++ b/existing_file.py
    @@ -1,3 +1,4 @@
    context line
    -removed line
    +added line
    context line
    ===CHANGE_END===

    PATH REQUIREMENTS:
    Make sure to use the full path for a file as it appears in the manifest.

    FORMAT REQUIREMENTS FOR EACH TYPE:

    1. NEW FILES (type: "create"):
    DIFF:
    --- /dev/null
    +++ b/new_file.py
    @@ -0,0 +1,3 @@
    +new content here

    2. DELETE FILES (type: "delete"):
    DIFF:
    --- a/old_file.py
    +++ /dev/null
    @@ -1,3 +0,0 @@
    -content to remove

    3. MODIFY FILES (type: "modify"):
    DIFF:
    --- a/existing_file.py
    +++ b/existing_file.py
    @@ -1,3 +1,4 @@
    context line
    -removed line
    +added line
    context line

    CONTINUATION HANDLING:
    - For long responses that may be split across multiple completions:
    - Complete each change fully before starting a new one
    - Never split a single DIFF section if possible
    - If a response is cut mid-change, the next continuation should start with the same ===CHANGE_BEGIN=== marker

    RULES:
    - Use /dev/null for create/delete operations
    - Include 3 lines of context for modifications
    - Group related changes into chunks with @@ markers
    - Include imports in first chunk if needed, make sure to include imports for new code
    - Return ONLY the changes in the specified format, no explanations
    - Ensure all paths use forward slashes
    - Keep descriptions short and clear

  solution: |
    Source Code:
    {source_code}

    Intent:
    {intent}

    Return only the content without any markup or explanations.
    IMPORTANT do NOT give any other information, only the asked for content
    IF you provide any other information, you will break the processing pipeline
EOF
echo "Created ${CONFIG_DIR}/${PERSONAS_SUBDIR}/solution_designer_v1.yml"

# Coder Persona
echo "Creating ${CONFIG_DIR}/${PERSONAS_SUBDIR}/coder_v1.yml..."
cat << EOF > "${CONFIG_DIR}/${PERSONAS_SUBDIR}/coder_v1.yml"
provider: "anthropic"
model: "claude-3-opus-20240229"
temperature: 0

# Example execution plan for a coder that uses iterator and asset manager
# This would be used if agent_type is "generic_orchestrating"
execution_plan:
  - name: "setup_iterator"
    skill: "SemanticIterator.initialize" # Assumes skill registry maps this
    input: "{{context.input_data.response}}" # Assuming diffs are here
    output_variable: "change_iterator"
  - name: "apply_changes_loop"
    type: "loop"
    iterate_on: "{{change_iterator}}" # Reference the output variable
    loop_variable: "current_change"   # Variable name for each item in loop
    body:
      - name: "apply_single_change"
        skill: "AssetManager.process_action" # Assumes skill registry maps this
        input: "{{current_change}}"          # Pass the change object
        output_variable: "apply_result_{{loop.index}}" # Optional: store results

prompts: # Prompts might be needed for specific steps or error handling
  system: |
    You are an expert code modification agent. Your task is to safely and precisely apply code changes.
    You receive changes in this exact JSON structure:
    {
      "changes": [
        {
          "file_path": "exact path to file",
          "type": "modify",
          "description": "change description",
          "content": "complete file content" # Or 'diff' depending on input needs
        }
      ]
    }

    Rules:
    1. Always expect input in the above JSON format
    2. If input is a string, parse it as JSON first
    3. Preserve existing functionality unless explicitly told to change it
    4. Maintain code style and formatting
    5. Apply changes exactly as specified
    6. Handle errors gracefully with backups
    7. Validate code after changes
EOF
echo "Created ${CONFIG_DIR}/${PERSONAS_SUBDIR}/coder_v1.yml"

# Add other persona files here based on WO-07 output (semantic_extract, etc.) if needed
# Example for semantic_merge_v1.yml:
echo "Creating ${CONFIG_DIR}/${PERSONAS_SUBDIR}/semantic_merge_v1.yml..."
cat << EOF > "${CONFIG_DIR}/${PERSONAS_SUBDIR}/semantic_merge_v1.yml"
provider: "anthropic"
model: "claude-3-7-sonnet-20250219"
temperature: 1 # Temperature might be 0 for deterministic merge
extended_thinking:
  enabled: true
  budget_tokens: 32000
merge_config: # Specific config for the merge skill/agent
  preserve_formatting: true
  allow_partial: false
  # Add config for fast/slow strategy if needed by the implementation
  # e.g., use_fast_path: true
prompts:
  system: |
    You are a precise code merger that transforms git-style unified diffs...
    # (Rest of the detailed system prompt from WO-07 output)
  merge: |
    You will receive a git-style unified diff representing code changes...
    # (Rest of the detailed merge prompt from WO-07 output)
EOF
echo "Created ${CONFIG_DIR}/${PERSONAS_SUBDIR}/semantic_merge_v1.yml"

# --- Create Schema Files (Optional but good practice) ---
echo "Creating ${CONFIG_DIR}/${SCHEMAS_SUBDIR}/system.json..."
cat << EOF > "${CONFIG_DIR}/${SCHEMAS_SUBDIR}/system.json"
{
  "\$schema": "http://json-schema.org/draft-07/schema#",
  "title": "System Configuration Schema",
  "description": "Schema for validating the system configuration (v1.3 compatible)",
  "type": "object",
  "properties": {
    "config_locations": {
      "type": "object",
      "properties": {
        "personas_dir": { "type": "string", "description": "Relative path to personas directory" },
        "schemas_dir": { "type": "string", "description": "Relative path to schemas directory" }
      },
      "required": ["personas_dir"]
    },
    "llm_config": {
      "type": "object",
      "properties": {
        "providers": {
          "type": "object",
          "additionalProperties": { "$ref": "#/definitions/providerConfig" }
        },
        "default_provider": { "type": "string" },
        "default_model": { "type": "string" },
        "personas": { "$ref": "#/definitions/personaBase" }
      },
      "required": ["providers", "default_provider"]
    },
    "orchestration": { "$ref": "#/definitions/orchestrationConfig" },
    "runtime": { "$ref": "#/definitions/runtimeConfig" },
    "backup": { "$ref": "#/definitions/backupConfig" },
    "logging": { "$ref": "#/definitions/loggingConfig" }
  },
  "required": ["config_locations", "llm_config", "orchestration"],
  "definitions": {
    "providerConfig": {
      "type": "object",
      "properties": {
        "api_base": { "type": "string" },
        "env_var": { "type": "string" },
        "default_model": { "type": "string" },
        "valid_models": { "type": "array", "items": { "type": "string" } },
        "context_length": { "type": ["integer", "null"] },
        "extended_thinking": { "type": ["object", "null"] },
        "litellm_params": { "type": ["object", "null"] },
        "model_params": { "type": ["object", "null"] }
      },
      "required": ["env_var", "default_model"]
    },
    "personaBase": {
       "type": "object",
       "properties": {
         "provider": { "type": ["string", "null"] },
         "model": { "type": ["string", "null"] },
         "temperature": { "type": ["number", "null"] },
         "extended_thinking": { "type": ["object", "null"] },
         "prompts": { "type": ["object", "null"] },
         "execution_plan": { "type": ["object", "null"] },
         "model_params": { "type": ["object", "null"] }
         # Add other common base fields from persona schema
       }
    },
    "orchestrationConfig": {
      "type": "object",
      "properties": {
        "enabled": { "type": "boolean" },
        "entry_team": { "type": "string" },
        "max_total_teams": { "type": "integer" },
        "max_recursion_depth": { "type": "integer" },
        "error_handling": { "type": "object" },
        "teams": {
          "type": "object",
          "additionalProperties": { "$ref": "#/definitions/teamConfig" }
        }
      },
      "required": ["enabled", "entry_team", "teams"]
    },
    "teamConfig": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "max_recursion_depth": { "type": ["integer", "null"] },
        "tasks": {
          "type": "array",
          "items": { "$ref": "#/definitions/taskConfig" }
        },
        "routing": { "$ref": "#/definitions/routingConfig" }
      },
      "required": ["name", "tasks"]
    },
    "taskConfig": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "agent_type": { "type": "string" },
        "persona_key": { "type": "string" },
        "requires_approval": { "type": "boolean" },
        "max_retries": { "type": "integer" },
        "config": { "type": "object" }
      },
      "required": ["name", "agent_type", "persona_key"]
    },
    "routingConfig": {
      "type": "object",
      "properties": {
        "rules": {
          "type": "array",
          "items": { "$ref": "#/definitions/routingRule" }
        },
        "default": { "type": ["string", "null"] }
      }
    },
    "routingRule": {
      "type": "object",
      "properties": {
        "condition": {
          "oneOf": [
            { "type": "object" },
            { "type": "array", "items": { "type": "object" } }
          ]
        },
        "next_team": { "type": ["string", "null"] }
      },
      "required": ["condition", "next_team"]
    },
    "runtimeConfig": {
      "type": "object",
      "properties": {
        "workflow": { "type": "object" },
        "lineage": { "type": "object" }
      }
    },
    "backupConfig": {
      "type": "object",
      "properties": {
        "enabled": { "type": "boolean" },
        "path": { "type": "string" }
      }
    },
    "loggingConfig": {
      "type": "object",
      "properties": {
        "level": { "type": "string", "enum": ["debug", "info", "warning", "error", "critical"] },
        "format": { "type": "string" },
        "agent_level": { "type": "string", "enum": ["debug", "info", "warning", "error", "critical"] },
        "providers": { "type": "object" },
        "truncate": { "type": "object" }
      }
    }
  }
}
EOF
echo "Created ${CONFIG_DIR}/${SCHEMAS_SUBDIR}/system.json"

# Add persona.json and job.json schemas similarly if desired

echo ""
echo "Configuration structure created in ./${CONFIG_DIR}"
echo "Review the generated files, especially provider details and prompts."
echo "Make sure the tartxt script_path in personas/discovery_v1.yml is correct for your environment."
echo "Run the service using: python ./c4h_services/src/bootstrap/prefect_runner.py service --config ${CONFIG_DIR}/system_config.yml"

