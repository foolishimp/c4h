"""
Unit tests for the AgentFactory
Path: c4h_services/tests/orchestration/test_factory.py
"""

import unittest
from unittest.mock import patch, MagicMock
from c4h_services.src.orchestration.factory import AgentFactory
from c4h_agents.agents.base_agent import BaseAgent

class TestAgentFactory(unittest.TestCase):
    """Test cases for the AgentFactory class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.effective_config = {
            "llm_config": {
                "agents": {
                    "test_agent": {
                        "provider": "test",
                        "model": "test-model"
                    }
                }
            }
        }
        self.factory = AgentFactory(self.effective_config)
    
    @patch("c4h_services.src.orchestration.factory.importlib.import_module")
    def test_get_agent_class_valid_type(self, mock_import):
        """Test getting agent class with valid agent type"""
        # Setup mock
        mock_module = MagicMock()
        mock_class = MagicMock(spec=BaseAgent)
        mock_module.GenericSingleShotAgent = mock_class
        mock_import.return_value = mock_module
        
        # Test
        result = self.factory._get_agent_class("generic_single_shot")
        
        # Verify
        mock_import.assert_called_once_with("c4h_agents.agents.generic")
        self.assertEqual(result, mock_class)
    
    def test_get_agent_class_invalid_type(self):
        """Test getting agent class with invalid agent type"""
        with self.assertRaises(ValueError) as context:
            self.factory._get_agent_class("invalid_agent_type")
        
        self.assertIn("Unknown agent_type", str(context.exception))
    
    @patch("c4h_services.src.orchestration.factory.importlib.import_module")
    def test_create_agent_success(self, mock_import):
        """Test creating agent successfully"""
        # Setup mock
        mock_module = MagicMock()
        mock_agent_class = MagicMock()
        mock_agent_instance = MagicMock(spec=BaseAgent)
        mock_agent_class.return_value = mock_agent_instance
        mock_module.GenericSingleShotAgent = mock_agent_class
        mock_import.return_value = mock_module
        
        # Test
        task_config = {
            "name": "test_agent",
            "agent_type": "generic_single_shot"
        }
        result = self.factory.create_agent(task_config)
        
        # Verify
        mock_import.assert_called_once_with("c4h_agents.agents.generic")
        mock_agent_class.assert_called_once_with(
            full_effective_config=self.effective_config, 
            unique_name="test_agent"
        )
        self.assertEqual(result, mock_agent_instance)
    
    def test_create_agent_missing_fields(self):
        """Test creating agent with missing required fields"""
        # Test missing agent_type
        with self.assertRaises(ValueError) as context:
            self.factory.create_agent({"name": "test_agent"})
        self.assertIn("Missing required field 'agent_type'", str(context.exception))
        
        # Test missing name
        with self.assertRaises(ValueError) as context:
            self.factory.create_agent({"agent_type": "generic_single_shot"})
        self.assertIn("Missing required field 'name'", str(context.exception))

if __name__ == '__main__':
    unittest.main()