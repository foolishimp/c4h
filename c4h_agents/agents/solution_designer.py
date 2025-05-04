"""
Solution designer agent stub for backward compatibility.
Path: c4h_agents/agents/solution_designer.py
"""

from typing import Dict, Any
from c4h_agents.agents.generic import GenericLLMAgent

class SolutionDesigner(GenericLLMAgent):
    """
    Legacy solution designer agent, now implemented using GenericLLMAgent.
    This class exists for backward compatibility.
    """
    
    def __init__(self, full_effective_config: Dict[str, Any], unique_name: str):
        """Initialize with full_effective_config and unique_name"""
        super().__init__(full_effective_config, unique_name)
        
        # Set persona to solution_designer if not already set
        if not hasattr(self, 'persona_key') or not self.persona_key:
            self.persona_key = "solution_designer"
        
        # Log that we're using the legacy class
        self.logger.info("legacy.solution_designer.initialized", 
                       unique_name=unique_name,
                       using="GenericLLMAgent")