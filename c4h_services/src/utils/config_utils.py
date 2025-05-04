"""
Configuration utilities for schema validation and file handling.
Path: c4h_services/src/utils/config_utils.py
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import jsonschema
from pathlib import Path
import json
import os
import time
from c4h_services.src.utils.logging import get_logger

logger = get_logger()

class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass

def _load_schema(schema_name: str) -> Dict[str, Any]:
    """
    Load a JSON schema from the schemas directory.
    
    Args:
        schema_name: Name of the schema file without extension
        
    Returns:
        Loaded schema as a dictionary
    """
    # Try multiple potential schema locations
    schema_paths = [
        Path(f"config/schemas/{schema_name}.json"),
        Path(os.path.join(os.path.dirname(__file__), f"../../../config/schemas/{schema_name}.json")),
    ]
    
    for path in schema_paths:
        if path.exists():
            with open(path) as f:
                schema = json.load(f)
                logger.info("config.schema.loaded", schema_name=schema_name, path=str(path))
                return schema
    
    logger.warning("config.schema.not_found", schema_name=schema_name)
    return {}

def validate_config_fragment(fragment: Dict[str, Any], schema_name: str, 
                           strict: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate a configuration fragment against a JSON schema.
    
    Args:
        fragment: Configuration fragment to validate
        schema_name: Name of the schema to validate against
        strict: If True, validation errors will be raised as exceptions;
               If False, validation errors will be logged but processing will continue
        
    Returns:
        Tuple of (success, error_message) where success is True if validation passed
        
    Raises:
        ConfigValidationError: If validation fails and strict=True
    """
    start_time = time.time()
    schema = _load_schema(schema_name)
    if not schema:
        logger.warning("config.validation.skipped", schema_name=schema_name)
        return False, "Schema not found"
        
    try:
        jsonschema.validate(instance=fragment, schema=schema)
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info("config.validation.passed", schema_name=schema_name, 
                  fragment_size=len(json.dumps(fragment)), duration_ms=duration_ms)
        return True, None
    except jsonschema.exceptions.ValidationError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_message = f"Configuration validation failed for schema '{schema_name}': {str(e)}"
        
        # Get the path to the validation error for better diagnostics
        path_string = ".".join(str(p) for p in e.path) if e.path else "root"
        logger.error("config.validation.failed", 
                    schema_name=schema_name, 
                    error=str(e),
                    path=path_string,
                    duration_ms=duration_ms)
        
        if strict:
            raise ConfigValidationError(error_message)
        return False, error_message

def validate_config_fragments(fragments: List[Dict[str, Any]], schema_map: Dict[int, str], 
                            strict: bool = False) -> Dict[int, Tuple[bool, Optional[str]]]:
    """
    Validate multiple configuration fragments against their respective schemas.
    
    Args:
        fragments: List of configuration fragments to validate
        schema_map: Mapping of fragment indices to schema names
        strict: If True, first validation error will stop processing and raise an exception
                If False, all fragments will be validated and results returned
                
    Returns:
        Dictionary mapping fragment indices to (success, error_message) tuples
        
    Raises:
        ConfigValidationError: If any validation fails and strict=True
    """
    results = {}
    
    # Log validation start
    logger.info("config.validation.batch_start", 
               fragments_count=len(fragments),
               schemas=list(schema_map.values()))
    
    for idx, schema_name in schema_map.items():
        if idx < 0 or idx >= len(fragments):
            results[idx] = (False, f"Fragment index {idx} out of range")
            continue
            
        fragment = fragments[idx]
        success, error = validate_config_fragment(fragment, schema_name, strict=False)  # Don't raise here
        results[idx] = (success, error)
        
        # If in strict mode and validation failed, stop processing
        if strict and not success:
            raise ConfigValidationError(f"Fragment {idx} ({schema_name}) validation failed: {error}")
    
    # Log overall validation result
    success_count = sum(1 for success, _ in results.values() if success)
    logger.info("config.validation.batch_complete", 
               success_count=success_count,
               total_count=len(schema_map))
    
    return results