"""
Integration tests for Hydra configuration system.
Verifies that the configuration can be loaded and accessed correctly.
"""

import os
import sys
import unittest
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestHydraIntegration(unittest.TestCase):
    """Integration tests for the Hydra configuration system."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = project_root
        self.conf_path = self.project_root / "conf"
        
        # Clear any existing Hydra instance
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
    
    def tearDown(self):
        """Clean up after tests."""
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
    
    def test_hydra_config_loading(self):
        """Test that Hydra can load the configuration correctly."""
        # Initialize Hydra with the conf directory
        with initialize(version_base=None, config_path=str(self.conf_path)):
            # Compose the configuration
            cfg = compose(config_name="config")
            
            # Verify configuration is loaded
            self.assertIsInstance(cfg, DictConfig, "Configuration should be a DictConfig")
            
            # Check key configuration sections exist
            self.assertIn("llm_config", cfg, "Configuration should contain llm_config")
            self.assertIn("orchestration", cfg, "Configuration should contain orchestration")
            self.assertIn("logging", cfg, "Configuration should contain logging")
            
            # Verify providers are loaded via interpolation
            self.assertIn("providers", cfg.llm_config, "llm_config should contain providers")
            
            # Verify teams are loaded via interpolation
            self.assertIn("teams", cfg.orchestration, "orchestration should contain teams")
            
            print("✅ Hydra configuration loads successfully")
    
    def test_config_interpolation(self):
        """Test that Hydra interpolation works correctly."""
        with initialize(version_base=None, config_path=str(self.conf_path)):
            cfg = compose(config_name="config")
            
            # Check that provider interpolation works
            if "anthropic" in cfg.llm_config.providers:
                anthropic_config = cfg.llm_config.providers.anthropic
                self.assertIsNotNone(anthropic_config, "Anthropic provider config should be loaded")
                self.assertIn("api_key", anthropic_config, "Provider config should contain api_key")
            
            # Check that team interpolation works
            if "discovery" in cfg.orchestration.teams:
                discovery_team = cfg.orchestration.teams.discovery
                self.assertIsNotNone(discovery_team, "Discovery team config should be loaded")
                self.assertIn("tasks", discovery_team, "Team config should contain tasks")
            
            print("✅ Configuration interpolation works correctly")
    
    def test_agent_initialization_with_hydra_config(self):
        """Test that agents can be initialized with Hydra configuration."""
        from c4h_agents.agents.base_config import BaseConfig
        
        # Create a test configuration
        test_config = OmegaConf.create({
            "llm_config": {
                "default_provider": "anthropic",
                "default_model": "claude-3",
                "agents": {
                    "test_agent": {
                        "provider": "anthropic",
                        "model": "claude-3"
                    }
                }
            },
            "logging": {
                "agent_level": "basic"
            }
        })
        
        # Initialize BaseConfig with the configuration
        base_config = BaseConfig(config=test_config)
        
        # Verify configuration is accessible
        self.assertIsInstance(base_config.config, DictConfig)
        self.assertEqual(
            OmegaConf.select(base_config.config, "llm_config.default_provider"),
            "anthropic"
        )
        
        # Test the lookup method
        provider = base_config.lookup("llm_config.default_provider")
        self.assertEqual(provider, "anthropic")
        
        print("✅ Agents can be initialized with Hydra configuration")
    
    def test_omegaconf_select_usage(self):
        """Test that OmegaConf.select is used correctly throughout the codebase."""
        test_config = OmegaConf.create({
            "llm_config": {
                "providers": {
                    "anthropic": {
                        "api_key": "test_key",
                        "models": ["claude-3"]
                    }
                }
            }
        })
        
        # Test safe navigation with OmegaConf.select
        provider_config = OmegaConf.select(test_config, "llm_config.providers.anthropic")
        self.assertIsNotNone(provider_config)
        self.assertEqual(provider_config.api_key, "test_key")
        
        # Test non-existent path returns None
        non_existent = OmegaConf.select(test_config, "llm_config.providers.openai")
        self.assertIsNone(non_existent)
        
        print("✅ OmegaConf.select works correctly for safe navigation")


def run_integration_tests():
    """Run all integration tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHydraIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("HYDRA INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
    else:
        print("\n❌ Some integration tests failed.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)