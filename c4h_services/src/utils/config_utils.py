"""
Configuration utilities for schema validation and file handling.
Path: c4h_services/src/utils/config_utils.py
"""

from typing import Dict, Any, List, Optional
import jsonschema
from pathlib import Path
import json
import os
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
                return json.load(f)
    
    logger.warning("config.schema.not_found", schema_name=schema_name)
    return {}

def validate_config_fragment(fragment: Dict[str, Any], schema_name: str) -> None:
    """
    Validate a configuration fragment against a JSON schema.
    
    Args:
        fragment: Configuration fragment to validate
        schema_name: Name of the schema to validate against
        
    Raises:
        ConfigValidationError: If validation fails
    """
    schema = _load_schema(schema_name)
    if not schema:
        logger.warning("config.validation.skipped", schema_name=schema_name)
        return
        
    try:
        jsonschema.validate(instance=fragment, schema=schema)
        logger.debug("config.validation.passed", schema_name=schema_name)
    except jsonschema.exceptions.ValidationError as e:
        logger.error("config.validation.failed", 
                    schema_name=schema_name, 
                    error=str(e),
                    path=e.path)
        raise ConfigValidationError(f"Configuration validation failed for schema '{schema_name}': {e}")