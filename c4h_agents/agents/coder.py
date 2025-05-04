"""
Coder agent stub for backward compatibility.
Path: c4h_agents/agents/coder.py
"""

from typing import Dict, Any
from c4h_agents.agents.generic import GenericLLMAgent

class Coder(GenericLLMAgent):
    """
    Legacy coder agent, now implemented using GenericLLMAgent.
    This class exists for backward compatibility.
    """
    
    def __init__(self, full_effective_config: Dict[str, Any], unique_name: str):
        """Initialize with full_effective_config and unique_name"""
        super().__init__(full_effective_config, unique_name)
        
        # Set persona to coder if not already set
        if not hasattr(self, 'persona_key') or not self.persona_key:
            self.persona_key = "coder"
        
        # Log that we're using the legacy class
        self.logger.info("legacy.coder_agent.initialized", 
                       unique_name=unique_name,
                       using="GenericLLMAgent")