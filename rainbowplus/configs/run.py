import yaml

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
from yaml.parser import ParserError

from rainbowplus.configs.base import LLMConfig


@dataclass
class Configuration:
    """Main configuration class containing all settings."""

    target_llm: LLMConfig
    fitness_llm: LLMConfig
    archive: Dict[str, Any] = field(default_factory=dict)
    sample_prompts: str = ""
    mutator_llm: Optional[LLMConfig] = None
    judge_llm: Optional[LLMConfig] = None

    def __post_init__(self):
        self.mutator_llm = self.mutator_llm or self.target_llm
        self.judge_llm = self.judge_llm or self.target_llm
        # self.sample_prompts = Path(self.sample_prompts)


class ConfigurationLoader:
    @staticmethod
    def load(file_path: str) -> Configuration:
        """Load configuration from YAML file."""
        try:
            with open(file_path, encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
                print(data)
                return ConfigurationLoader._create_config(data)
        except ParserError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except OSError as e:
            raise ValueError(f"Failed to read config file: {e}")

    @staticmethod
    def _create_config(data: Dict[str, Any]) -> Configuration:
        """Create Configuration instance from parsed YAML data."""
        required_llms = ["target_llm", "fitness_llm"]

        try:
            llm_configs = {
                key: ConfigurationLoader._parse_llm_config(data.get(key, {}))
                for key in required_llms
            }

            return Configuration(
                archive=data.get("archive", {}),
                sample_prompts=data.get("sample_prompts", ""),
                mutator_llm=ConfigurationLoader._parse_llm_config(
                    data.get("mutator_llm", {})
                ),
                judge_llm=ConfigurationLoader._parse_llm_config(
                    data.get("judge_llm", {})
                ),
                **llm_configs,
            )
        except KeyError as e:
            raise ValueError(f"Missing required configuration: {e}")

    @staticmethod
    def _parse_llm_config(data: Any) -> LLMConfig:
        """Parse LLM configuration from YAML data."""
        if not isinstance(data, dict):
            return None
        return LLMConfig(**data)
