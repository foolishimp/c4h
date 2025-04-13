"""
Configuration handling with robust dictionary access and path resolution.
Path: c4h_agents/config.py
"""
import yaml
from typing import Dict, Any, Optional, List, Tuple, Union, Iterator, Pattern
from pathlib import Path
import structlog
from copy import deepcopy
import collections.abc
import json
import fnmatch
import re

logger = structlog.get_logger()

class ConfigNode:
    """
    Node-based configuration access with hierarchical path support.
    Provides relative path queries and wildcard matching.
    """
    def __init__(self, data: Dict[str, Any], base_path: str = ""):
        """
        Initialize config node with data and optional base path.
        
        Args:
            data: Dictionary containing configuration
            base_path: Optional base path for this node (for logging)
        """
        self.data = data
        self.base_path = base_path

    def get_value(self, path: str) -> Any:
        """
        Get value at specified path relative to this node.
        
        Args:
            path: Dot-delimited path string, may include wildcards (*)
            
        Returns:
            Value at the path, or None if not found
        """
        # Handle direct access
        if not path:
            return self.data
            
        # Handle path with wildcards
        if '*' in path:
            matches = list(self._find_wildcard_matches(path))
            if len(matches) == 1:
                return matches[0][1]  # Return the single matched value
            elif len(matches) > 1:
                logger.warning("config.multiple_wildcard_matches", 
                              path=path, 
                              matches=len(matches),
                              returning="first_match")
                return matches[0][1]  # Return first match
            return None
            
        # Standard path access
        path_parts = path.split('.')
        return get_by_path(self.data, path_parts)

    def get_node(self, path: str) -> 'ConfigNode':
        """
        Get configuration node at specified path.
        
        Args:
            path: Dot-delimited path string, may include wildcards (*)
            
        Returns:
            ConfigNode at the path, or empty node if not found
        """
        if not path:
            return self
            
        value = self.get_value(path)
        if isinstance(value, dict):
            full_path = f"{self.base_path}.{path}" if self.base_path else path
            return ConfigNode(value, full_path)
        else:
            logger.warning("config.node_path_not_dict", 
                          path=path, 
                          value_type=type(value).__name__)
            return ConfigNode({}, path)

    def find_all(self, path_pattern: str) -> List[Tuple[str, Any]]:
        """
        Find all values matching a path pattern with wildcards.
        
        Args:
            path_pattern: Dot-delimited path with wildcards
            
        Returns:
            List of (path, value) tuples for all matches
        """
        return list(self._find_wildcard_matches(path_pattern))

    def _find_wildcard_matches(self, path_pattern: str) -> Iterator[Tuple[str, Any]]:
        """
        Iterator for all values matching a wildcard pattern.
        
        Args:
            path_pattern: Dot-delimited path with wildcards
            
        Yields:
            Tuples of (path, value) for each match
        """
        path_parts = path_pattern.split('.')
        
        def _search_recursive(data: Dict[str, Any], current_parts: List[str], 
                             current_path: List[str]) -> Iterator[Tuple[str, Any]]:
            # Base case: no more parts to match
            if not current_parts:
                yield '.'.join(current_path), data
                return
                
            current_part = current_parts[0]
            remaining_parts = current_parts[1:]
            
            # Handle wildcards
            if current_part == '*':
                # Match any key at this level
                if isinstance(data, dict):
                    for key, value in data.items():
                        yield from _search_recursive(value, remaining_parts, current_path + [key])
            elif '*' in current_part:
                # Pattern matching within this level
                pattern = fnmatch.translate(current_part)
                regex = re.compile(pattern)
                if isinstance(data, dict):
                    for key, value in data.items():
                        if regex.match(key):
                            yield from _search_recursive(value, remaining_parts, current_path + [key])
            else:
                # Exact key match
                if isinstance(data, dict) and current_part in data:
                    yield from _search_recursive(data[current_part], remaining_parts, 
                                              current_path + [current_part])
        
        yield from _search_recursive(self.data, path_parts, [])

    def __getitem__(self, key: str) -> Any:
        """
        Dictionary-style access to configuration values.
        
        Args:
            key: Simple key or dot-delimited path
            
        Returns:
            Value at the specified path
        """
        return self.get_value(key)

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in this node.
        
        Args:
            key: Simple key or dot-delimited path
            
        Returns:
            True if the key exists, False otherwise
        """
        return self.get_value(key) is not None

# Original functions enhanced to work with the new approach

def get_by_path(data: Dict[str, Any], path: List[str]) -> Any:
    """
    Access dictionary data using a path list.
    
    Args:
        data: Dictionary to traverse
        path: List of keys forming the path
        
    Returns:
        Value at path or None if not found
    """
    try:
        current = data
        for key in path:
            if isinstance(current, dict): # Check if current level is a dict
                if key not in current:    # Check if key exists in dict
                    return None
                current = current[key]    # Move down one level
            else: # If not a dict, cannot traverse further
                return None
        return current
    except Exception as e:
        logger.error("config.path_access_failed", path=path, error=str(e))
        return None

def get_value(data: Dict[str, Any], path_str: str) -> Any:
    """
    Access dictionary data using a hierarchical path string (e.g. "system.runid").
    Supports both dots (.) and slashes (/) as path separators.
    
    Args:
        data: Dictionary to traverse
        path_str: Delimited key path (using dots or slashes)
        
    Returns:
        Value at the specified path or None if not found.
    """
    # Handle both dot and slash notation for backward compatibility
    if '/' in path_str:
        path_list = path_str.split('/')
    else:
        path_list = path_str.split('.')
        
    return get_by_path(data, path_list)

def locate_keys(data: Dict[str, Any], target_keys: List[str], current_path: List[str] = None) -> Dict[str, Tuple[Any, List[str]]]:
    """
    Locate multiple keys in dictionary using hierarchy tracking.
    
    Args:
        data: Dictionary to search
        target_keys: List of keys to find
        current_path: Path for logging (internal use)
        
    Returns:
        Dict mapping found keys to (value, path) tuples
    """
    try:
        results = {}
        current_path = current_path or []
        if isinstance(data, dict):
            for key in target_keys:
                if key in data:
                    value = data[key]
                    path = current_path + [key]
                    if isinstance(value, str):
                        try:
                            parsed = json.loads(value)
                            value = parsed
                        except json.JSONDecodeError:
                            pass
                    results[key] = (value, path)
                    logger.debug("config.key_located", key=key, path=path, found_type=type(value).__name__)
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    child_results = locate_keys(v, [k for k in target_keys if k not in results], current_path + [k])
                    results.update(child_results)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    child_results = locate_keys(item, [k for k in target_keys if k not in results], current_path + [str(i)])
                    results.update(child_results)
        found_keys = set(results.keys())
        missing_keys = set(target_keys) - found_keys
        if missing_keys:
            logger.debug("config.keys_not_found", keys=list(missing_keys), searched_path=current_path)
        return results
    except Exception as e:
        logger.error("config.locate_keys_failed", target_keys=target_keys, current_path=current_path, error=str(e))
        return {}

def locate_config(config: Dict[str, Any], target_name: str) -> Dict[str, Any]:
    """
    Locate configuration using strict hierarchical path.
    Primary path is always llm_config.agents.[name]
    
    Args:
        config: Configuration dictionary
        target_name: Name of target agent/component
        
    Returns:
        Located config dictionary or empty dict if not found
    """
    try:
        # Use the ConfigNode for more advanced lookup
        config_node = ConfigNode(config)
        standard_path = f"llm_config.agents.{target_name}"
        result = config_node.get_value(standard_path)
        
        if result is not None and isinstance(result, dict):
            logger.debug("config.located_in_hierarchy", 
                        target=target_name, 
                        path=standard_path, 
                        found_keys=list(result.keys()))
            return result
            
        # Try wildcard search as fallback
        wildcard_path = f"*.agents.{target_name}"
        matches = config_node.find_all(wildcard_path)
        if matches:
            result_path, result_value = matches[0]
            logger.debug("config.located_with_wildcard", 
                        target=target_name, 
                        path=result_path, 
                        found_keys=list(result_value.keys()))
            return result_value
            
        logger.warning("config.not_found_in_hierarchy", 
                      target=target_name, 
                      searched_path=standard_path)
        return {}
    except Exception as e:
        logger.error("config.locate_failed", target=target_name, error=str(e))
        return {}

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge dictionaries preserving hierarchical structure.
    
    Rules:
    1. Preserve llm_config.agents hierarchy.
    2. Override values take precedence.
    3. Dictionaries merged recursively.
    4. Lists from override replace lists from base.
    5. None values in override delete keys from base.
    6. Runtime values merged into agent configs.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = deepcopy(base)
    # --- MODIFIED LOGGING ---
    # Log only the top-level keys involved in this specific merge call
    logger.debug("config.deep_merge.called",
                 base_keys=list(base.keys()) if isinstance(base, dict) else type(base).__name__,
                 override_keys=list(override.keys()) if isinstance(override, dict) else type(override).__name__)
    # --- END MODIFIED LOGGING ---
    try:
        for key, value in override.items():
            if value is None:
                result.pop(key, None)
                continue

            # Check if key exists in base and both are dictionaries for recursive merge
            if key in result and isinstance(result[key], collections.abc.Mapping) and isinstance(value, collections.abc.Mapping):
                result[key] = deep_merge(result[key], value)
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                # Override completely if not both dictionaries or if key is new
                result[key] = deepcopy(value)

        # --- MODIFIED LOGGING ---
        # Log only the keys of the final result of this specific call
        logger.debug("config.deep_merge.complete",
                     result_keys=list(result.keys()) if isinstance(result, dict) else type(result).__name__)
        # --- END MODIFIED LOGGING ---
        return result
    except Exception as e:
        logger.error("config.merge.failed", error=str(e), keys_processed=list(override.keys()))
        raise

def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file with comprehensive logging"""
    try:
        logger.info("config.load.starting", path=str(path))
        if not path.exists():
            logger.error("config.load.file_not_found", path=str(path))
            return {}
        with open(path) as f:
            config = yaml.safe_load(f) or {}
        logger.info("config.load.success", path=str(path), keys=list(config.keys()), size=len(str(config)))
        return config
    except yaml.YAMLError as e:
        logger.error("config.load.yaml_error", path=str(path), error=str(e), line=getattr(e, 'line', None), column=getattr(e, 'column', None))
        return {}
    except Exception as e:
        logger.error("config.load.failed", path=str(path), error=str(e), error_type=type(e).__name__)
        return {}

def load_with_app_config(system_path: Path, app_path: Path) -> Dict[str, Any]:
    """Load and merge system config with app config with full logging"""
    try:
        logger.info("config.merge.starting", system_path=str(system_path), app_path=str(app_path))
        system_config = load_config(system_path)
        app_config = load_config(app_path)
        result = deep_merge(system_config, app_config)
        logger.info("config.merge.complete", total_keys=len(result), system_keys=len(system_config), app_keys=len(app_config))
        return result
    except Exception as e:
        logger.error("config.merge.failed", error=str(e), error_type=type(e).__name__)
        return {}

def create_config_node(config: Dict[str, Any]) -> ConfigNode:
    """
    Create a ConfigNode from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ConfigNode for easy hierarchical access
    """
    return ConfigNode(config)