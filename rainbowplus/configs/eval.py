import yaml

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
from yaml.parser import ParserError

from rainbowplus.configs.base import LLMConfig


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation."""

    log_dir: Path
    algo: str
    method: str
    judge_llm: LLMConfig

    def __post_init__(self):
        self.log_dir = Path(self.log_dir)


class EvalConfigLoader:
    @staticmethod
    def load(file_path: str) -> EvaluationConfig:
        """Load evaluation configuration from YAML file."""
        try:
            with open(file_path, encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
                return EvalConfigLoader._create_config(data)
        except ParserError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except OSError as e:
            raise ValueError(f"Failed to read config file: {e}")

    @staticmethod
    def _create_config(data: Dict[str, Any]) -> EvaluationConfig:
        """Create EvaluationConfig instance from parsed YAML data."""
        required_fields = ["log_dir", "algo", "method", "judge_llm"]
        if missing := [f for f in required_fields if f not in data]:
            raise ValueError(f"Missing required fields: {missing}")

        judge_llm = (
            LLMConfig(**data["judge_llm"])
            if isinstance(data["judge_llm"], dict)
            else None
        )
        if not judge_llm:
            raise ValueError("Invalid judge_llm configuration")

        return EvaluationConfig(
            log_dir=data["log_dir"],
            algo=data["algo"],
            method=data["method"],
            judge_llm=judge_llm,
        )
