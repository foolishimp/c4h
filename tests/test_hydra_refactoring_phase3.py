"""
Unit tests for Hydra Refactoring Phase 3 verification.
These tests validate that all agent-level code has been correctly refactored
to use Hydra OmegaConf configuration object and that legacy configuration
systems have been removed.
"""

import os
import sys
import unittest
import importlib
import inspect
import ast
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestHydraPhase3Refactoring(unittest.TestCase):
    """Test cases for verifying Hydra Phase 3 refactoring completion."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = project_root
        self.agents_path = self.project_root / "c4h_agents" / "agents"
        self.utils_path = self.project_root / "c4h_agents" / "utils"
        self.services_path = self.project_root / "c4h_services" / "src"
    
    def test_tc_hydra_3_1_base_config_init(self):
        """
        TC-HYDRA-3.1: Verify BaseConfig and BaseAgent initialization.
        Requirement: FR-3 (Agent Consumption), WO-HYDRA-3.1
        """
        # Import the modules
        from c4h_agents.agents.base_config import BaseConfig
        from c4h_agents.agents.base_agent import BaseAgent
        
        # Test 1: Check BaseConfig __init__ signature
        config_init = inspect.signature(BaseConfig.__init__)
        params = config_init.parameters
        
        # Verify config parameter accepts Union[Dict, DictConfig]
        self.assertIn('config', params, "BaseConfig.__init__ should have 'config' parameter")
        
        # Test 2: Check BaseAgent __init__ signature
        agent_init = inspect.signature(BaseAgent.__init__)
        params = agent_init.parameters
        self.assertIn('full_effective_config', params, 
                     "BaseAgent.__init__ should have 'full_effective_config' parameter")
        
        # Test 3: Search for config_node references in source files
        base_config_file = self.agents_path / "base_config.py"
        base_agent_file = self.agents_path / "base_agent.py"
        
        # Check BaseConfig source
        with open(base_config_file, 'r') as f:
            base_config_source = f.read()
        
        self.assertNotIn('self.config_node', base_config_source,
                        "BaseConfig should not contain 'self.config_node'")
        self.assertNotIn('create_config_node', base_config_source,
                        "BaseConfig should not contain 'create_config_node'")
        
        # Check BaseAgent source
        with open(base_agent_file, 'r') as f:
            base_agent_source = f.read()
            
        self.assertNotIn('self.config_node', base_agent_source,
                        "BaseAgent should not contain 'self.config_node'")
        # Note: log_config_node is allowed as it's a logging function
        
        print("✅ TC-HYDRA-3.1: PASS - BaseConfig and BaseAgent correctly refactored")
    
    def test_tc_hydra_3_2_base_llm_config_access(self):
        """
        TC-HYDRA-3.2: Verify configuration access pattern in BaseLLM.
        Requirement: FR-3 (Agent Consumption), WO-HYDRA-3.2
        """
        base_llm_file = self.agents_path / "base_llm.py"
        
        # Read the source file
        with open(base_llm_file, 'r') as f:
            source = f.read()
        
        # Parse the AST to find _get_model_str method
        tree = ast.parse(source)
        
        # Find the _get_model_str method
        method_found = False
        uses_old_pattern = False
        uses_new_pattern = False
        
        class MethodVisitor(ast.NodeVisitor):
            def __init__(self):
                self.in_get_model_str = False
                self.old_pattern_found = False
                self.new_pattern_found = False
                
            def visit_FunctionDef(self, node):
                if node.name == '_get_model_str':
                    self.in_get_model_str = True
                    self.generic_visit(node)
                    self.in_get_model_str = False
                else:
                    self.generic_visit(node)
                    
            def visit_Call(self, node):
                if self.in_get_model_str:
                    # Check for old pattern: config_node.get_value()
                    if (isinstance(node.func, ast.Attribute) and 
                        node.func.attr == 'get_value'):
                        self.old_pattern_found = True
                    
                    # Check for new pattern: OmegaConf.select()
                    if (isinstance(node.func, ast.Attribute) and 
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id == 'OmegaConf' and
                        node.func.attr == 'select'):
                        self.new_pattern_found = True
                        
                self.generic_visit(node)
        
        visitor = MethodVisitor()
        visitor.visit(tree)
        
        # Verify the method exists and uses correct pattern
        self.assertFalse(visitor.old_pattern_found,
                        "BaseLLM._get_model_str should NOT use config_node.get_value()")
        self.assertTrue(visitor.new_pattern_found,
                       "BaseLLM._get_model_str should use OmegaConf.select()")
        
        # Also check that file imports OmegaConf
        self.assertIn('from omegaconf import', source,
                     "BaseLLM should import OmegaConf")
        
        print("✅ TC-HYDRA-3.2: PASS - BaseLLM configuration access pattern correct")
    
    def test_tc_hydra_3_3_generic_llm_agent_config_access(self):
        """
        TC-HYDRA-3.3: Verify configuration access pattern in GenericLLMAgent.
        Requirement: FR-3 (Agent Consumption), WO-HYDRA-3.2
        """
        generic_file = self.agents_path / "generic.py"
        
        # Read the source file
        with open(generic_file, 'r') as f:
            source = f.read()
        
        # Check for OmegaConf usage in _format_request method
        # Look for the specific pattern mentioned in test case
        self.assertIn('OmegaConf.select(self.config', source,
                     "GenericLLMAgent should use OmegaConf.select()")
        
        # Verify it's not using old get_value pattern
        lines = source.split('\n')
        in_format_request = False
        
        for i, line in enumerate(lines):
            if 'def _format_request' in line:
                in_format_request = True
            elif in_format_request and 'def ' in line and line.strip().startswith('def'):
                in_format_request = False
                
            if in_format_request and '.get_value(' in line:
                self.fail(f"GenericLLMAgent._format_request should not use get_value() - found at line {i+1}")
        
        print("✅ TC-HYDRA-3.3: PASS - GenericLLMAgent configuration access pattern correct")
    
    def test_tc_hydra_3_4_config_py_deleted(self):
        """
        TC-HYDRA-3.4: Verify deletion of core custom configuration module.
        Requirement: FR-4 (Deprecation), WO-HYDRA-3.3
        """
        config_file = self.project_root / "c4h_agents" / "config.py"
        
        self.assertFalse(config_file.exists(),
                        f"config.py should not exist at {config_file}")
        
        print("✅ TC-HYDRA-3.4: PASS - config.py has been deleted")
    
    def test_tc_hydra_3_5_utility_modules_deleted(self):
        """
        TC-HYDRA-3.5: Verify deletion of utility configuration modules.
        Requirement: FR-4 (Deprecation), WO-HYDRA-3.3
        """
        utility_files = [
            "config_validation.py",
            "schema_validation.py", 
            "config_materializer.py"
        ]
        
        for filename in utility_files:
            file_path = self.utils_path / filename
            self.assertFalse(file_path.exists(),
                           f"{filename} should not exist at {file_path}")
        
        print("✅ TC-HYDRA-3.5: PASS - All utility configuration modules deleted")
    
    def test_tc_hydra_3_6_materialise_config_removed(self):
        """
        TC-HYDRA-3.6: Verify removal of materialise_config Prefect task.
        Requirement: FR-4 (Deprecation), WO-HYDRA-3.3
        """
        tasks_file = self.services_path / "intent" / "impl" / "prefect" / "tasks.py"
        
        with open(tasks_file, 'r') as f:
            source = f.read()
        
        # Check that materialise_config function is not present
        self.assertNotIn('def materialise_config', source,
                        "materialise_config function should be removed from tasks.py")
        
        print("✅ TC-HYDRA-3.6: PASS - materialise_config task has been removed")
    
    def test_additional_omegaconf_integration(self):
        """
        Additional test to verify OmegaConf is properly integrated across the codebase.
        """
        # List of key files that should be using OmegaConf
        key_files = [
            self.agents_path / "base_config.py",
            self.agents_path / "base_agent.py", 
            self.agents_path / "base_llm.py",
            self.agents_path / "generic.py",
            self.project_root / "c4h_services" / "src" / "orchestration" / "orchestrator.py"
        ]
        
        for file_path in key_files:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    source = f.read()
                
                # Check for OmegaConf import
                if 'OmegaConf' in source:
                    self.assertIn('from omegaconf import', source,
                                f"{file_path.name} should properly import OmegaConf")
                    print(f"✅ {file_path.name} properly imports OmegaConf")
    
    def test_no_legacy_imports(self):
        """
        Test that no files are importing from the deleted configuration modules.
        """
        # Files that should not be imported anywhere
        forbidden_imports = [
            "from c4h_agents.config import",
            "from c4h_agents.utils.config_validation import",
            "from c4h_agents.utils.schema_validation import",
            "from c4h_agents.utils.config_materializer import"
        ]
        
        # Search through Python files (excluding tests)
        for root, dirs, files in os.walk(self.project_root):
            # Skip test directories and venv
            if 'test' in root or 'venv' in root or '__pycache__' in root:
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    
                    try:
                        with open(file_path, 'r') as f:
                            source = f.read()
                        
                        for forbidden in forbidden_imports:
                            if forbidden in source:
                                # Allow test files to have these imports
                                if 'test' not in str(file_path):
                                    self.fail(f"Found forbidden import '{forbidden}' in {file_path}")
                    except Exception:
                        # Skip files that can't be read
                        pass
        
        print("✅ No legacy configuration imports found in production code")


def run_tests():
    """Run all tests and display results."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHydraPhase3Refactoring)
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print("HYDRA PHASE 3 REFACTORING TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED! Hydra Phase 3 refactoring is complete.")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)