"""
Generic test harness for running agent classes with configuration.
Path: src/testharness.py
"""

from typing import List, Dict, Any, Optional
import structlog
from enum import Enum
import logging.config
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pathlib import Path
import argparse
import yaml
import json
import sys

# Add source directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from c4h_agents.agents.base_agent import BaseAgent
# Import LogDetail enum from c4h_agents.agents.types
from c4h_agents.agents.types import LogDetail

# Update imports to use the new module structure
from c4h_agents.agents.generic import GenericLLMAgent as Coder
from c4h_agents.skills.semantic_iterator import SemanticIterator
from c4h_agents.skills.semantic_merge import SemanticMerge
from c4h_agents.skills.shared.types import ExtractConfig
from c4h_agents.agents.generic import GenericLLMAgent as DiscoveryAgent
from c4h_agents.agents.generic import GenericLLMAgent as SolutionDesigner
from c4h_agents.skills.asset_manager import AssetManager
from c4h_agents.skills.semantic_extract import SemanticExtract
from c4h_agents.config import deep_merge

logger = structlog.get_logger()

class LogMode(str, Enum):
    """Logging modes supported by harness"""
    DEBUG = "debug"     # Maps to LogDetail.DEBUG
    NORMAL = "normal"   # Maps to LogDetail.BASIC

    @property
    def to_log_detail(self) -> LogDetail:
        """Convert harness LogMode to agent LogDetail"""
        return {
            LogMode.DEBUG: LogDetail.DEBUG,
            LogMode.NORMAL: LogDetail.BASIC
        }[self]

def parse_param(param_str: str) -> tuple[str, Any]:
    """Parse a parameter string in format key=value"""
    try:
        key, value = param_str.split('=', 1)
        # Try to interpret as Python literal (for bool, int, etc)
        try:
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # Keep as string if not a valid Python literal
        return key.strip(), value
    except ValueError:
        raise ValueError(f"Invalid parameter format: {param_str}. Use key=value format")

@dataclass
class AgentConfig:
    """Configuration for agent instantiation"""
    agent_type: str
    config_path: Path
    extra_args: Dict[str, Any] = None

class AgentTestHarness:
    """Generic test harness for running agent classes"""
    
    # Registry of supported agent types
    AGENT_TYPES = {
        "coder": Coder,
        "semantic_iterator": SemanticIterator,
        "semantic_merge": SemanticMerge,
        "semantic_extract": SemanticExtract,
        "discovery": DiscoveryAgent,
        "solution_designer": SolutionDesigner,
        "asset_manager": AssetManager
    }

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        # Remove default project root assumption
        self.project_root = None
        
    def setup_logging(self, mode: LogMode) -> None:
        """Configure structured logging based on mode"""
        processors = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        if mode == LogMode.DEBUG:
            processors.extend([
                structlog.dev.ConsoleRenderer(colors=True)
            ])
        else:
            processors.append(structlog.processors.JSONRenderer(indent=2))

        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'DEBUG' if mode == LogMode.DEBUG else 'INFO',
                },
            },
            'root': {
                'handlers': ['console'],
                'level': 'DEBUG' if mode == LogMode.DEBUG else 'INFO',
            }
        })

    def load_configs(self, test_config_path: str) -> Dict[str, Any]:
        """Load and merge configurations"""
        try:
            system_config = self._load_system_config()
            
            # Load test config using yaml
            with open(test_config_path) as f:
                test_config = yaml.safe_load(f)

            # Get project paths from test config, not execution location
            if 'project' in test_config:
                project_config = test_config['project']
                if 'source_path' in project_config:
                    project_config['source_path'] = Path(project_config['source_path']).resolve()
                if 'output_path' in project_config:
                    project_config['output_path'] = Path(project_config['output_path']).resolve()
                if 'default_path' in project_config:
                    project_config['default_path'] = Path(project_config['default_path']).resolve()

            # Merge configs using deep_merge
            config = deep_merge(system_config, test_config)
            
            return config
        except Exception as e:
            logger.error("testharness.config_load_failed", error=str(e))
            raise

    def _load_system_config(self) -> Dict[str, Any]:
        """Load system configuration file"""
        try:
            # Find system config path
            system_config_paths = [
                Path("config/system_config.yml"),
                Path("../config/system_config.yml"),
                Path(__file__).parent.parent / "config" / "system_config.yml"
            ]
            
            for path in system_config_paths:
                if path.exists():
                    logger.info("config.loading", path=str(path))
                    with open(path) as f:
                        return yaml.safe_load(f)
                    
            logger.warning("config.no_system_config_found", 
                        paths=[str(p) for p in system_config_paths])
            return {}

        except Exception as e:
            logger.error("config.load_failed", error=str(e))
            return {}

    def create_agent(self, agent_type: str, config: Dict[str, Any]) -> BaseAgent:
        """Create agent instance based on type"""
        if agent_type not in self.AGENT_TYPES:
            raise ValueError(f"Unsupported agent type: {agent_type}")
                
        agent_class = self.AGENT_TYPES[agent_type]
        # New signature expects full_effective_config and unique_name
        return agent_class(full_effective_config=config, unique_name=agent_type)

    def process_agent(self, config: AgentConfig) -> None:
        """Process agent with configuration"""
        try:
            # Load configuration
            configs = self.load_configs(str(config.config_path))
            logger.debug("testharness.loaded_config", 
                        config_path=str(config.config_path),
                        config_contents=configs)
                
            # Create agent instance
            agent = self.create_agent(config.agent_type, configs)
            logger.debug("testharness.created_agent",
                        agent_type=config.agent_type,
                        agent_class=agent.__class__.__name__)

            # Get any extra parameters passed via command line
            extra_params = config.extra_args or {}
                
            if isinstance(agent, SemanticIterator):
                # Handle iterator case
                extract_config = ExtractConfig(
                    instruction=configs.get('instruction'),
                    format=configs.get('format', 'json')
                )
                
                for key, value in extra_params.items():
                    if hasattr(extract_config, key):
                        setattr(extract_config, key, value)
                        logger.info(f"config.override", key=key, value=value)
                
                agent.configure(
                    content=configs.get('input_data'),
                    config=extract_config
                )
                
                results = []
                for item in agent:
                    results.append(item)
                        
                self._display_output({'results': results})
                    
            else:
                # Generic agent processing
                # Pass input_data directly from configs if available
                if 'input_data' in configs:
                    context = configs['input_data']
                    # Add any extra parameters
                    context.update(extra_params)
                else:
                    # Fallback to basic context structure
                    context = {
                        'content': configs.get('content'),
                        'instruction': configs.get('instruction'),
                        'merge_style': configs.get('merge_style', 'smart'),
                        **extra_params
                    }

                logger.debug("testharness.processing_context", 
                        context_keys=list(context.keys()))
                
                result = agent.process(context)
                
                if not result.success:
                    self._display_output({'error': result.error})
                    return
                        
                self._display_output(result.data)

        except Exception as e:
            logger.error("process_agent.failed", error=str(e))
            raise

    def format_llm_content(self, data: Any) -> str:
        """Format LLM response content for display.
        
        Handles:
        - ModelResponse objects
        - Raw text
        - Dictionary with nested content
        - Escaped multiline strings
        """
        try:
            # Handle ModelResponse objects
            if hasattr(data, 'choices') and data.choices:
                data = data.choices[0].message.content
                
            # Handle dictionary responses
            if isinstance(data, dict):
                # Look for common content fields
                for key in ['content', 'response', 'text', 'result']:
                    if key in data:
                        data = data[key]
                        break
            
            # Convert to string
            content = str(data)
            
            # Handle escaped newlines and indentation
            content = content.replace('\\n', '\n')
            content = content.replace('\\t', '\t')
            content = content.replace('\\"', '"')
            
            # Strip any markdown code block markers
            if content.startswith('```') and content.endswith('```'):
                content = '\n'.join(content.split('\n')[1:-1])
                
            return content
            
        except Exception as e:
            logger.error("display.format_failed", error=str(e))
            return str(data)

    def _display_output(self, output: Dict[str, Any]) -> None:
        """Display formatted output with robust type handling"""
        print("\n=== Results ===\n")
        
        try:
            # For lists of results
            if isinstance(output, dict) and 'results' in output:
                for item in output['results']:
                    # Format each item's content
                    if isinstance(item, dict):
                        for key in ['file_path', 'type', 'description']:
                            if key in item:
                                print(f"{key.replace('_', ' ').title()}: {item[key]}\n")
                        
                        if 'content' in item:
                            print("Content:")
                            print(self.format_llm_content(item['content']))
                        print("-" * 80)
                    else:
                        print(self.format_llm_content(item))
            else:
                # Format top-level output
                print(self.format_llm_content(output))
                        
        except Exception as e:
            logger.error("display.output_failed", error=str(e))
            print(str(output))

def main():
    parser = argparse.ArgumentParser(
        description="Generic agent test harness"
    )
    parser.add_argument(
        "agent_type",
        choices=AgentTestHarness.AGENT_TYPES.keys(),
        help="Type of agent to test"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--log",
        type=LogMode,
        choices=list(LogMode),
        default=LogMode.NORMAL,
        help="Logging mode for test harness"
    )
    parser.add_argument(
        "--param",
        action='append',
        help="Additional parameters in key=value format",
        default=[]
    )
    
    args = parser.parse_args()
    
    try:
        # Parse any additional parameters
        extra_params = {}
        for param_str in args.param:
            try:
                key, value = parse_param(param_str)
                extra_params[key] = value
            except ValueError as e:
                logger.error("param.parse_failed", param=param_str, error=str(e))
                raise SystemExit(1)
        
        logger.info("testharness.starting", 
                   agent_type=args.agent_type,
                   config_file=args.config,
                   log_mode=args.log)

        harness = AgentTestHarness()
        harness.setup_logging(args.log)
        
        try:
            harness.process_agent(AgentConfig(
                agent_type=args.agent_type,
                config_path=Path(args.config),
                extra_args=extra_params
            ))
        except FileNotFoundError as e:
            logger.error("config.not_found", path=args.config)
            print(f"\nError: Configuration file not found: {args.config}")
            raise SystemExit(1)
        except Exception as e:
            logger.error("process_agent.failed", error=str(e))
            raise
            
        logger.info("testharness.completed")
        
    except Exception as e:
        logger.error("harness.failed", error=str(e))
        print(f"\nError: {str(e)}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()