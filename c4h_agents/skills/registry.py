"""
Skill Registry for C4H Agent System.

This module provides a centralized registry for Skills,
allowing them to be looked up by name and instantiated
at runtime for use by the Execution Plan Executor.
"""

from typing import Dict, Any, Optional, Type, Callable, List, Union, Set
import importlib
import inspect
import structlog
from pathlib import Path
import yaml
import os
import glob
import sys

# Get logger
logger = structlog.get_logger()

class SkillRegistry:
    """
    Registry for C4H Skills that provides lookup and instantiation capabilities.
    This is a singleton to ensure a single registry across the application.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SkillRegistry, cls).__new__(cls)
            cls._instance._skills = {}
            cls._instance._auto_discovery_done = False
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._skills = {}  # name -> skill_config mapping
        self._auto_discovery_done = False
        self._initialized = True
    
    def register_skill(
        self, 
        name: str, 
        module_path: str, 
        class_name: str, 
        method: str = "execute",
        description: str = "",
        default_params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Register a skill with the registry.
        
        Args:
            name: Unique name for the skill
            module_path: Dotted path to the Python module (e.g., 'c4h_agents.skills.semantic_iterator')
            class_name: Name of the skill class in the module
            method: Method to call (defaults to 'execute')
            description: Optional description of the skill
            default_params: Optional default parameters for the skill
            tags: Optional list of tags for categorizing skills
        """
        skill_config = {
            "module": module_path,
            "class": class_name,
            "method": method,
            "description": description,
            "default_params": default_params or {},
            "tags": tags or []
        }
        
        # Check if skill with this name already exists
        if name in self._skills:
            logger.warning("skill.registry.overwriting_existing_skill", 
                          name=name, 
                          old_module=self._skills[name]["module"],
                          new_module=module_path)
        
        self._skills[name] = skill_config
        logger.info("skill.registry.registered_skill", 
                   name=name, 
                   module=module_path, 
                   class_name=class_name)
    
    def get_skill_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get the configuration for a skill by name.
        
        Args:
            name: Name of the skill to look up
            
        Returns:
            Skill configuration dict or None if not found
        """
        # Ensure auto-discovery has been attempted
        if not self._auto_discovery_done:
            self._auto_discover_skills()
            
        return self._skills.get(name)
    
    def instantiate_skill(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Instantiate a skill by name with optional configuration.
        
        Args:
            name: Name of the skill to instantiate
            config: Optional configuration to pass to the skill constructor
            
        Returns:
            Instantiated skill object
            
        Raises:
            ValueError: If skill not found
            ImportError: If module or class cannot be imported
        """
        # Ensure auto-discovery has been attempted
        if not self._auto_discovery_done:
            self._auto_discover_skills()
            
        skill_config = self.get_skill_config(name)
        if not skill_config:
            logger.error("skill.registry.skill_not_found", name=name)
            raise ValueError(f"Skill '{name}' not found in registry")
        
        try:
            # Import the module
            module = importlib.import_module(skill_config["module"])
            
            # Get the class
            skill_class = getattr(module, skill_config["class"])
            
            # Merge configuration
            merged_config = {}
            if skill_config.get("default_params"):
                merged_config.update(skill_config["default_params"])
            if config:
                merged_config.update(config)
            
            # Instantiate the skill
            instance = skill_class(merged_config)
            logger.debug("skill.registry.instantiated_skill", 
                        name=name, 
                        class_name=skill_config["class"])
            return instance
            
        except ImportError as e:
            logger.error("skill.registry.import_error", 
                        name=name, 
                        module=skill_config["module"], 
                        error=str(e))
            raise
        except AttributeError as e:
            logger.error("skill.registry.class_not_found", 
                        name=name, 
                        class_name=skill_config["class"], 
                        error=str(e))
            raise ValueError(f"Class '{skill_config['class']}' not found in module '{skill_config['module']}'")
        except Exception as e:
            logger.error("skill.registry.instantiation_failed", 
                        name=name, 
                        error=str(e))
            raise
    
    def list_skills(self) -> List[str]:
        """Return a list of all registered skill names."""
        # Ensure auto-discovery has been attempted
        if not self._auto_discovery_done:
            self._auto_discover_skills()
            
        return list(self._skills.keys())
        
    def get_all_skill_names(self) -> List[str]:
        """
        Get all registered skill names.
        
        Returns:
            List of all registered skill names
        """
        # Ensure auto-discovery has been attempted
        if not self._auto_discovery_done:
            self._auto_discover_skills()
            
        return list(self._skills.keys())
    
    def get_skill_by_tag(self, tag: str) -> List[str]:
        """
        Get a list of skill names that have the specified tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of skill names with the tag
        """
        # Ensure auto-discovery has been attempted
        if not self._auto_discovery_done:
            self._auto_discover_skills()
            
        return [
            name for name, config in self._skills.items()
            if "tags" in config and tag in config["tags"]
        ]
    
    def load_skills_from_config(self, config: Dict[str, Any]) -> None:
        """
        Load skill definitions from the effective configuration.
        
        Args:
            config: Effective configuration dictionary
        """
        skills_config = config.get("llm_config", {}).get("skills", {})
        if not skills_config:
            logger.warning("skill.registry.no_skills_in_config")
            return
            
        logger.info("skill.registry.loading_from_config", 
                   skills_count=len(skills_config))
                   
        for skill_name, skill_def in skills_config.items():
            if not isinstance(skill_def, dict):
                logger.warning("skill.registry.invalid_skill_definition", 
                              name=skill_name,
                              definition_type=type(skill_def).__name__)
                continue
                
            # Extract skill details
            module_path = skill_def.get("module")
            class_name = skill_def.get("class")
            method = skill_def.get("method", "execute")
            description = skill_def.get("description", "")
            default_params = skill_def.get("default_params", {})
            tags = skill_def.get("tags", [])
            
            if not module_path or not class_name:
                logger.warning("skill.registry.missing_required_fields", 
                              name=skill_name,
                              has_module=bool(module_path),
                              has_class=bool(class_name))
                continue
                
            # Register the skill
            self.register_skill(
                name=skill_name,
                module_path=module_path,
                class_name=class_name,
                method=method,
                description=description,
                default_params=default_params,
                tags=tags
            )
    
    def _auto_discover_skills(self, skills_package: str = "c4h_agents.skills") -> None:
        """
        Auto-discover available skills in the skills package.
        
        This method scans the skills package for potential skill implementations
        and registers them automatically if they meet certain criteria:
        - Have a class with the same name as the module (CamelCase vs snake_case)
        - The class has an "execute" method
        
        Args:
            skills_package: Package to scan for skills
        """
        if self._auto_discovery_done:
            return
            
        try:
            # Import the skills package
            try:
                package = importlib.import_module(skills_package)
                package_path = Path(package.__file__).parent
            except ImportError:
                logger.warning("skill.registry.package_import_failed", 
                              package=skills_package)
                return
                
            logger.info("skill.registry.auto_discovering_skills", 
                       package=skills_package,
                       package_path=str(package_path))
                       
            # Get a list of all Python files in the package
            py_files = list(package_path.glob("*.py"))
            py_files = [f for f in py_files if f.name != "__init__.py"]
            
            discovered_count = 0
            for py_file in py_files:
                module_name = py_file.stem
                
                # Skip certain known non-skill files
                if module_name.startswith("_") or module_name == "registry":
                    continue
                    
                # Convert snake_case to CamelCase for the expected class name
                expected_class_name = "".join(word.title() for word in module_name.split("_"))
                
                # Try to import the module
                full_module_path = f"{skills_package}.{module_name}"
                try:
                    module = importlib.import_module(full_module_path)
                except ImportError as e:
                    logger.warning("skill.registry.module_import_failed", 
                                  module=full_module_path,
                                  error=str(e))
                    continue
                    
                # Look for the expected class
                if hasattr(module, expected_class_name):
                    class_obj = getattr(module, expected_class_name)
                    
                    # Check if it has an execute method
                    if inspect.isclass(class_obj) and hasattr(class_obj, "execute"):
                        # Get method signature to extract doc or description
                        method = getattr(class_obj, "execute")
                        description = method.__doc__ or ""
                        if description:
                            # Extract first line as short description
                            description = description.strip().split("\n")[0].strip()
                            
                        # Register the skill
                        if module_name not in self._skills:
                            self.register_skill(
                                name=module_name,
                                module_path=full_module_path,
                                class_name=expected_class_name,
                                description=description
                            )
                            discovered_count += 1
                        
            logger.info("skill.registry.auto_discovery_complete", 
                       discovered_count=discovered_count,
                       total_skills=len(self._skills))
                       
            # Also do a special check for command_line_runner since it was created as part of this refactoring
            if "command_line_runner" not in self._skills:
                try:
                    module = importlib.import_module("c4h_agents.skills.command_line_runner")
                    if hasattr(module, "CommandLineRunner"):
                        self.register_skill(
                            name="command_line_runner",
                            module_path="c4h_agents.skills.command_line_runner",
                            class_name="CommandLineRunner",
                            description="Generic command line tool and script execution",
                            default_params={
                                "command_configs": {
                                    "tartxt": {
                                        "type": "shell_command",
                                        "command": [sys.executable, "-m", "c4h_agents.skills.tartxt"],
                                        "description": "Project scanning and content extraction"
                                    }
                                }
                            }
                        )
                        logger.info("skill.registry.added_command_line_runner")
                except ImportError:
                    pass
                    
            # Set flag to prevent repeated discovery
            self._auto_discovery_done = True
                
        except Exception as e:
            logger.error("skill.registry.auto_discovery_failed", 
                        error=str(e))
    
    def register_builtin_skills(self) -> None:
        """
        Register built-in skills that should always be available.
        
        This method ensures that critical skills are always registered,
        even if auto-discovery fails or they are not defined in config.
        """
        # Register command_line_runner skill
        self.register_skill(
            name="command_line_runner",
            module_path="c4h_agents.skills.command_line_runner",
            class_name="CommandLineRunner",
            description="Generic command line tool and script execution",
            tags=["utility", "builtin", "execution"],
            default_params={
                "command_configs": {
                    "tartxt": {
                        "type": "shell_command",
                        "command": [sys.executable, "-m", "c4h_agents.skills.tartxt"],
                        "description": "Project scanning and content extraction"
                    }
                }
            }
        )
        
        # Register semantic_iterator skill
        self.register_skill(
            name="semantic_iterator",
            module_path="c4h_agents.skills.semantic_iterator",
            class_name="SemanticIterator",
            description="Iterative semantic processing of content",
            tags=["semantic", "builtin"]
        )
        
        # Register asset_manager skill
        self.register_skill(
            name="asset_manager",
            module_path="c4h_agents.skills.asset_manager",
            class_name="AssetManager",
            description="File and asset management operations",
            tags=["filesystem", "builtin"]
        )
        
        logger.info("skill.registry.registered_builtin_skills", 
                   count=3)
    
    @classmethod
    def get_instance(cls) -> 'SkillRegistry':
        """Get the singleton instance of the registry."""
        return cls()
    
    @classmethod
    def register_default_skills(cls, config: Dict[str, Any]) -> 'SkillRegistry':
        """
        Create and initialize the registry with default skills.
        
        Args:
            config: Effective configuration dictionary
            
        Returns:
            Initialized SkillRegistry instance
        """
        registry = cls.get_instance()
        registry.register_builtin_skills()
        registry.load_skills_from_config(config)
        registry._auto_discover_skills()
        return registry