"""
JSON Schema validation utility for C4H configuration.
Path: c4h_agents/utils/schema_validation.py
"""

from typing import Dict, Any, List, Union, Optional
import json
import os
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator
import structlog
from copy import deepcopy

logger = structlog.get_logger()

class SchemaValidator:
    """
    Utility class for validating configuration structures against JSON schemas.
    
    Provides methods to validate execution plans, agent configurations, 
    and other structured configuration components.
    """
    
    _schema_cache: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def _load_schema(cls, schema_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a JSON schema from file with caching support.
        
        Args:
            schema_path: Path to the JSON schema file
            
        Returns:
            Loaded schema as a dictionary
        """
        schema_path_str = str(schema_path)
        
        # Return cached schema if available
        if schema_path_str in cls._schema_cache:
            return cls._schema_cache[schema_path_str]
        
        try:
            # Load schema from file
            with open(schema_path, 'r') as f:
                schema = json.load(f)
                
            # Cache and return
            cls._schema_cache[schema_path_str] = schema
            return schema
        except Exception as e:
            logger.error("schema.load_failed", 
                        schema_path=schema_path_str, 
                        error=str(e))
            return {}
    
    @classmethod
    def validate_execution_plan(cls, 
                              execution_plan: Dict[str, Any], 
                              schema_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Validate an execution plan against the execution plan schema.
        
        Args:
            execution_plan: The execution plan to validate
            schema_path: Optional custom schema path (defaults to standard location)
            
        Returns:
            Validated execution plan (original if valid)
            
        Raises:
            ValidationError: If the execution plan is invalid
        """
        # Default schema path if not provided
        if schema_path is None:
            # Find the config directory relative to the current module
            module_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            project_dir = module_dir.parent.parent  # Assuming utils is 2 levels deep
            schema_path = project_dir / "config" / "schemas" / "execution_plan_v1.json"
        
        # Load schema
        schema = cls._load_schema(schema_path)
        if not schema:
            logger.warning("schema.validation_skipped", 
                          reason="schema_not_loaded",
                          schema_path=str(schema_path))
            return execution_plan
        
        # Validate
        try:
            validate(instance=execution_plan, schema=schema)
            logger.debug("schema.validation_passed", 
                        schema_type="execution_plan",
                        steps_count=len(execution_plan.get("steps", [])))
            return execution_plan
        except ValidationError as e:
            logger.error("schema.validation_failed", 
                        schema_type="execution_plan",
                        error=str(e), 
                        path=e.path, 
                        schema_path=e.schema_path)
            raise
    
    @classmethod
    def get_validation_errors(cls, 
                            instance: Dict[str, Any], 
                            schema_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Validate an instance against a schema and return a list of validation errors.
        
        Args:
            instance: The instance to validate
            schema_path: Path to the schema file
            
        Returns:
            List of validation errors (empty if valid)
        """
        schema = cls._load_schema(schema_path)
        if not schema:
            return [{"error": "Failed to load schema", "schema_path": str(schema_path)}]
        
        validator = Draft7Validator(schema)
        errors = []
        
        for error in validator.iter_errors(instance):
            error_info = {
                "message": error.message,
                "path": list(error.path) if error.path else [],
                "schema_path": list(error.schema_path) if error.schema_path else []
            }
            errors.append(error_info)
        
        return errors

    @classmethod
    def validate_and_clean(cls, 
                         instance: Dict[str, Any], 
                         schema_path: Union[str, Path],
                         remove_additional: bool = True) -> Dict[str, Any]:
        """
        Validate an instance against a schema and remove non-conforming properties.
        
        Args:
            instance: The instance to validate and clean
            schema_path: Path to the schema file
            remove_additional: Whether to remove properties not in the schema
            
        Returns:
            Cleaned instance complying with the schema
        """
        schema = cls._load_schema(schema_path)
        if not schema:
            return instance
        
        # Deep copy to avoid modifying the original
        cleaned = deepcopy(instance)
        
        try:
            validate(instance=cleaned, schema=schema)
            return cleaned
        except ValidationError as e:
            logger.warning("schema.cleaning_instance",
                          error=str(e),
                          path=e.path,
                          schema_path=e.schema_path)
            
            # Simple implementation of cleaning by removing offending properties
            if remove_additional:
                if e.validator == 'additionalProperties' and e.path:
                    # Get the parent object
                    parent = cleaned
                    for i in range(len(e.path) - 1):
                        parent = parent[e.path[i]]
                    
                    # Remove the offending property
                    if e.path and isinstance(parent, dict):
                        key = e.path[-1]
                        if key in parent:
                            logger.debug("schema.removing_property", 
                                        key=key, 
                                        path=e.path)
                            del parent[key]
                            
                    # Recursive validation after cleaning
                    return cls.validate_and_clean(cleaned, schema_path, remove_additional)
            
            # Return best effort cleaning
            return cleaned