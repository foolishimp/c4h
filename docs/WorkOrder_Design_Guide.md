# C4H Agent System: Prompt Design Guidelines

After reviewing the codebase and configuration examples, I'll create a comprehensive guide for designing prompts/work orders for the C4H Agent System. This document will help users structure their prompts effectively.

## Purpose of This Guide

This document provides guidance on how to create effective prompts (work orders) for the C4H Agent System. It covers the structure, key components, and best practices for designing prompts that align with the system's design principles.

## Config Structure Overview

The C4H Agent System uses a hierarchical configuration model with two main sections:

1. **System Configuration** - Defines the agent infrastructure, providers, and global settings
2. **Intent Configuration (Work Order)** - Specifies what you want the system to do

## System Configuration Section

The system configuration establishes the infrastructure and capabilities of the agent system:

```yaml
project:
  path: "/path/to/your/project"  
  workspace_root: "workspaces"

llm_config:
  agents:
    discovery:
      tartxt_config:
        script_path: "/path/to/tartxt.py"
        input_paths:
          - "path/to/folder/to/analyze"
        exclusions:
          - "**/__pycache__/**"
          - "**/.git/**"
    
    solution_designer:
      provider: "anthropic"
      model: "claude-3-7-sonnet-20250219"
      temperature: 1
      extended_thinking:
        enabled: true
        budget_tokens: 32000

    coder:
      provider: "anthropic"
      model: "claude-3-7-sonnet-20250219"
      temperature: 0
      backup_enabled: true

runtime:
  workflow:
    storage:
      enabled: true
      root_dir: "workspaces/workflows"
      format: "yymmdd_hhmm_{workflow_id}"

logging:
  level: "INFO"
  format: "structured"
  agent_level: "DEBUG"
```

### Key System Configuration Components

1. **Project Settings**
   - `path`: Absolute path to the project being processed
   - `workspace_root`: Where workspace and outputs are stored

2. **LLM Configuration**
   - **Discovery Agent**: Analyzes project structure
     - `tartxt_config.input_paths`: Directories to analyze
     - `tartxt_config.exclusions`: Patterns to exclude
   
   - **Solution Designer**: Creates the refactoring plan
     - `provider`: LLM provider (anthropic, openai, gemini)
     - `model`: Specific model to use
     - `temperature`: Creativity level (0-1)
     - `extended_thinking`: For complex reasoning (Claude models)
   
   - **Coder**: Implements the changes
     - Similar settings to Solution Designer
     - `backup_enabled`: Whether to backup files before modifying

3. **Runtime Settings**
   - Storage configuration for workflow artifacts
   - Error handling and retry policies

4. **Logging Configuration**
   - Controls verbosity and format of logs

## Intent Configuration (Work Order) Section

The intent configuration defines what you want the system to do:

```yaml
intent:
  description: |
    A clear, detailed description of what you want to accomplish.
    
    For example:
    "Refactor the codebase to implement a consistent error handling system,
    where all exceptions are logged with contextual information and proper
    hierarchical categorization."
    
    Include:
    1. The goal of the refactoring
    2. Specific requirements and constraints
    3. Implementation details if necessary
    4. Any design principles that should be followed
    
    Be specific about what files or components should be modified.
    Explain the desired outcome in detail.
```

### Best Practices for Intent Description

1. **Be Clear and Specific**
   - State exactly what you want the system to do
   - Specify which files or components should be modified
   - Define the expected outcome

2. **Provide Context**
   - Explain why the change is needed
   - Reference existing patterns or principles to follow
   - Mention any constraints that must be respected

3. **Structure with Numbered Points**
   - Break down complex requirements into numbered sections
   - Prioritize items if appropriate
   - Use hierarchical structure for clarity

4. **Include Implementation Guidelines**
   - Specify technical approaches if needed
   - Reference specific patterns or libraries to use
   - Define any architectural boundaries

5. **Specify Validation Criteria**
   - How will success be measured?
   - Any specific tests or checks that should be performed
   - Expected performance characteristics

## Complete Work Order Template

Here's a complete template combining both system and intent configurations:

```yaml
# Work Order Configuration

# Project settings
project:
  path: "/path/to/your/project"  
  workspace_root: "workspaces"

# Intent description (the actual work order)
intent:
  description: |
    [DETAILED DESCRIPTION OF WHAT YOU WANT TO ACCOMPLISH]
    
    Goal:
    [CLEAR STATEMENT OF END GOAL]
    
    Required Changes:
    1. [FIRST CHANGE]
    2. [SECOND CHANGE]
    3. [THIRD CHANGE]
    
    Implementation Requirements:
    - [REQUIREMENT 1]
    - [REQUIREMENT 2]
    - [REQUIREMENT 3]
    
    Design Principles to Follow:
    - [PRINCIPLE 1]
    - [PRINCIPLE 2]
    
    Files to Modify:
    - [FILE PATH 1]
    - [FILE PATH 2]

# LLM configuration
llm_config:
  agents:
    discovery:
      tartxt_config:
        script_path: "/path/to/tartxt.py"
        input_paths:
          - "path/to/analyze"
        exclusions:
          - "**/__pycache__/**"
          - "**/.git/**"
    
    solution_designer:
      provider: "anthropic"
      model: "claude-3-7-sonnet-20250219" 
      temperature: 1
      extended_thinking:
        enabled: true
        budget_tokens: 32000
    
    coder:
      provider: "anthropic"
      model: "claude-3-7-sonnet-20250219"
      temperature: 0
      backup_enabled: true

# Runtime configuration
runtime:
  workflow:
    storage:
      enabled: true
      root_dir: "workspaces/workflows"
      format: "yymmdd_hhmm_{workflow_id}"
      subdirs:
        - "events"
        - "config"
      error_handling:
        ignore_storage_errors: true
        log_level: "ERROR"

# Logging configuration
logging:
  level: "INFO"
  format: "structured"
  agent_level: "DEBUG"
```

## Work Order Examples

Here are a few example work orders for common tasks:

### Example 1: Adding Error Handling

```yaml
intent:
  description: |
    Add comprehensive error handling to the application with proper logging.
    
    Goal:
    Implement a consistent error handling system across all service functions that captures
    context, provides clear error messages, and ensures proper logging.
    
    Required Changes:
    1. Create a centralized error handling utility
    2. Add try/catch blocks to all service functions
    3. Implement context-aware error logging
    4. Add user-friendly error messages for API responses
    
    Implementation Requirements:
    - Use structured error objects with error codes
    - Include stack traces in dev mode only
    - Log errors with appropriate severity levels
    - Ensure all errors are properly propagated
    
    Files to Modify:
    - src/services/*.js
    - src/api/routes/*.js
    - src/utils/logger.js
```

### Example 2: Refactoring for Performance

```yaml
intent:
  description: |
    Optimize database queries in the data access layer to improve performance.
    
    Goal:
    Reduce API response times by optimizing database queries, implementing caching,
    and reducing unnecessary data fetching.
    
    Required Changes:
    1. Add indexes to frequently queried fields
    2. Implement query batching for related data
    3. Add Redis caching for frequently accessed data
    4. Optimize SELECT queries to fetch only needed fields
    
    Implementation Requirements:
    - Query execution time should not exceed 100ms
    - Cache invalidation must be properly implemented
    - No changes to API contracts or response formats
    - Add performance metrics logging
    
    Design Principles:
    - Minimize database round trips
    - Cache at appropriate layers
    - Prefer eager loading over N+1 queries
    
    Files to Modify:
    - src/data/repositories/*.js
    - src/models/data-access/*.js
    - src/config/database.js
```

### Example 3: Adding a New Feature

```yaml
intent:
  description: |
    Add user authentication with JWT tokens and role-based access control.
    
    Goal:
    Implement a secure authentication system with JWT tokens, user roles,
    and protected routes.
    
    Required Changes:
    1. Create authentication middleware
    2. Implement JWT token generation and validation
    3. Add user roles and permissions system
    4. Protect API routes based on user roles
    
    Implementation Requirements:
    - Use bcrypt for password hashing
    - Implement token refresh mechanism
    - Store user roles in database
    - Add route protection based on roles
    
    Design Principles:
    - Separation of authentication and authorization
    - Clean middleware architecture
    - Security best practices (OWASP)
    
    Files to Create:
    - src/auth/middleware.js
    - src/auth/token-service.js
    - src/auth/role-service.js
    
    Files to Modify:
    - src/api/routes/*.js
    - src/models/user.js
    - src/config/app.js
```

## Workflow Sequence

When the C4H Agent System processes your work order, it follows this sequence:

1. **Discovery Phase**
   - Analyzes project structure and codebase
   - Identifies relevant files and patterns
   - Creates a codebase understanding model

2. **Solution Design Phase**
   - Interprets your intent description
   - Creates a detailed plan for changes
   - Identifies specific modifications needed
   - Designs the solution architecture

3. **Coding Phase**
   - Implements the changes according to the plan
   - Makes precise modifications to the codebase
   - Creates new files if needed
   - Preserves existing functionality

4. **Verification Phase** (if configured)
   - Validates changes against requirements
   - Performs basic testing
   - Ensures consistency of modifications

## Tips for Optimal Results

1. **Be Specific and Detailed**
   - The more specific your intent description, the better the results
   - Include examples where helpful

2. **Reference Design Principles**
   - Mention architectural patterns to follow
   - Reference code style or design principles

3. **Set Clear Boundaries**
   - Specify which parts of the code should change
   - Indicate what should not change

4. **Start Small**
   - Begin with focused, smaller work orders
   - Build confidence in the system before tackling major refactors

5. **Review the Plan**
   - Check the solution design before proceeding to implementation
   - Provide feedback if necessary

By following these guidelines, you'll create effective work orders that leverage the full capabilities of the C4H Agent System while adhering to its design principles.