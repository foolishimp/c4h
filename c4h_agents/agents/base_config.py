"""
Configuration management for agent implementations following design principles.
Path: c4h_agents/agents/base_config.py
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import structlog
from datetime import datetime
from functools import wraps
import time

from c4h_agents.core.project import Project, ProjectPaths
from c4h_agents.config import ConfigNode, create_config_node, locate_config, get_value
from c4h_agents.utils.logging import get_logger
from .types import LogDetail, LLMProvider

logger = get_logger()

def log_operation(operation_name: str):
    """Operation logging decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                self._update_metrics(time.time() - start_time, True)
                return result
            except Exception as e:
                self._update_metrics(time.time() - start_time, False, str(e))
                raise
        return wrapper
    return decorator

class BaseConfig:
    """Configuration management for agent implementations"""
    
    def __init__(self, config: Dict[str, Any] = None, project: Optional[Project] = None):
        """Initialize configuration and project context"""
        self.config = config or {}
        self.config_node = create_config_node(self.config)
        self.project = project
        
        if self.project:
            self.ensure_paths()

        # Set logging detail level from config
        log_level = self.config_node.get_value("logging.agent_level") or "basic"
        self.log_level = LogDetail.from_str(log_level)
        
        # Initialize metrics with project context
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_duration": 0.0,
            "continuation_attempts": 0,
            "last_error": None,
            "start_time": datetime.utcnow().isoformat(),
            "project": self.project.metadata.name if self.project else None
        }
        
        # Check if this agent has an execution plan
        self._has_execution_plan = self._check_for_execution_plan()

    def ensure_paths(self):
        """Ensure project paths exist"""
        # Only create directories if needed
        if hasattr(self.project.paths, 'workspace'):
            self.project.paths.workspace.mkdir(parents=True, exist_ok=True)
        if hasattr(self.project.paths, 'output'):
            self.project.paths.output.mkdir(parents=True, exist_ok=True)

    def lookup(self, path: str) -> Any:
        """
        Lookup a value in the full configuration using a hierarchical path.
        Supports dot notation (e.g., "system.runid") and wildcards.
        """
        return self.config_node.get_value(path)

    def get_agent_node(self) -> ConfigNode:
        """
        Get configuration node for this agent's section.
        Provides relative path access for agent-specific configuration.
        """
        agent_name = self._get_agent_name()
        agent_path = f"llm_config.agents.{agent_name}"
        return self.config_node.get_node(agent_path)

    def _get_runtime_config(self) -> Dict[str, Any]:
        """Get runtime configuration section"""
        runtime_node = self.config_node.get_node("runtime")
        return runtime_node.data or {}

    def _get_lineage_config(self) -> Dict[str, Any]:
        """Get lineage configuration section"""
        runtime_config = self._get_runtime_config()
        lineage_config = self.config_node.get_value("runtime.lineage") or {}
        logger.debug("config.lineage_loaded", 
                    agent=self._get_agent_name(), 
                    enabled=lineage_config.get('enabled', False), 
                    config_keys=list(lineage_config.keys()))
        return lineage_config

    """
    Update the _get_provider_config method to fix the rate limit handling.
    Remove unsupported rate_limit_policy parameter.
    """

    def _get_provider_config(self, provider: LLMProvider) -> Dict[str, Any]:
        """Get provider-specific configuration with correct retry handling"""
        try:
            provider_node = self.config_node.get_node(f"llm_config.providers.{provider.value}")
            provider_config = provider_node.data or {}
            
            # Handle default retry configuration
            litellm_params = provider_config.get("litellm_params", {})
            if not litellm_params:
                litellm_params = {}
                provider_config["litellm_params"] = litellm_params
                
            # Set retry configuration if not already configured
            if "retry" not in litellm_params:
                litellm_params["retry"] = True
                
            if "max_retries" not in litellm_params:
                litellm_params["max_retries"] = 5  # Increased from 3
                
            # Ensure backoff configuration exists
            if "backoff" not in litellm_params:
                litellm_params["backoff"] = {
                    "initial_delay": 2,  # Increased from 1
                    "max_delay": 60,     # Increased from 30
                    "exponential": True
                }
            
            # IMPORTANT: Remove rate_limits or rate_limit_policy if present
            # These are not supported by Anthropic's API
            if "rate_limits" in litellm_params:
                del litellm_params["rate_limits"]
                
            if "rate_limit_policy" in litellm_params:
                del litellm_params["rate_limit_policy"]
            
            # Log all changes
            if self._should_log(LogDetail.DEBUG):
                logger.debug("provider.config_loaded", 
                            provider=str(provider), 
                            retry_config=litellm_params.get("retry"), 
                            max_retries=litellm_params.get("max_retries"),
                            backoff=litellm_params.get("backoff"))
                            
            return provider_config
        except Exception as e:
            logger.error("provider.config_failed", provider=str(provider), error=str(e))
            return {}

    def _get_agent_config(self) -> Dict[str, Any]:
        """
        Get this agent's configuration section.
        Uses hierarchical path lookup for reliable access.
        """
        agent_node = self.get_agent_node()
        return agent_node.data or {}

    def _resolve_model(self, explicit_model: Optional[str], provider_config: Dict[str, Any]) -> str:
        """Resolve model name using hierarchical config lookup"""
        try:
            agent_name = self._get_agent_name()
            
            if explicit_model:
                model = explicit_model
            elif self.lookup(f"llm_config.agents.{agent_name}.model"):
                model = self.lookup(f"llm_config.agents.{agent_name}.model")
            elif "default_model" in provider_config:
                model = provider_config["default_model"]
            elif self.lookup("llm_config.default_model"):
                model = self.lookup("llm_config.default_model")
            else:
                raise ValueError(f"No model specified for provider and no defaults found")
                
            valid_models = provider_config.get("valid_models", [])
            if valid_models and model not in valid_models:
                logger.warning("config.invalid_model", 
                              model=model, 
                              valid_models=valid_models, 
                              using_default=provider_config.get("default_model"))
                model = provider_config.get("default_model")
                if not model:
                    raise ValueError(f"Invalid model {model} and no default available")
                    
            return model
        except Exception as e:
            logger.error("config.model_resolution_failed", 
                        error=str(e), 
                        provider=self.provider.value if hasattr(self, 'provider') else None)
            raise

    def _should_log(self, level: LogDetail) -> bool:
        """Check if current log level includes the specified detail level"""
        log_levels = {
            LogDetail.MINIMAL: 0,
            LogDetail.BASIC: 1, 
            LogDetail.DETAILED: 2,
            LogDetail.DEBUG: 3
        }
        return log_levels[level] <= log_levels[self.log_level]
    
    def _update_metrics(self, duration: float, success: bool, error: Optional[str] = None) -> None:
        """Update operation metrics with timing and success information"""
        self.metrics["total_requests"] += 1
        self.metrics["total_duration"] += duration
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
            self.metrics["last_error"] = error
        if self._should_log(LogDetail.DETAILED):
            logger.info("agent.metrics_updated", metrics=self.metrics, duration=duration, success=success)

    def _get_agent_name(self) -> str:
        """
        Get the agent name for configuration lookup.
        Must be implemented by agent classes.
        """
        raise NotImplementedError("Agent classes must implement _get_agent_name")
        
    def _check_for_execution_plan(self) -> bool:
        """
        Check if this agent has an execution plan in its configuration.
        
        Returns:
            True if an execution plan exists, False otherwise
        """
        try:
            agent_name = self._get_agent_name()
            agent_node = self.config_node.get_node(f"llm_config.agents.{agent_name}")
            
            return "execution_plan" in agent_node.data and isinstance(agent_node.data["execution_plan"], dict)
        except Exception:
            return False
    
    def get_execution_plan(self) -> Optional[Dict[str, Any]]:
        """
        Get the execution plan for this agent if it exists.
        
        Returns:
            Execution plan dictionary or None if not found
        """
        if not self._has_execution_plan:
            return None
            
        try:
            agent_name = self._get_agent_name()
            return self.config_node.get_value(f"llm_config.agents.{agent_name}.execution_plan")
        except Exception as e:
            logger.error("config.get_execution_plan.failed", 
                       agent=agent_name if 'agent_name' in locals() else "unknown",
                       error=str(e))
            return None