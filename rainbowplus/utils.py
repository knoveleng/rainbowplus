import json
import logging
import sys

from typing import TypeVar, List
from datasets import load_dataset
from rainbowplus.switcher import LLMSwitcher
from rainbowplus.configs import ConfigurationLoader

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_txt(file_path: str) -> List[str]:
    """
    Load text file and return non-empty lines.

    Args:
        file_path (str): Path to the text file

    Returns:
        List[str]: List of stripped, non-empty lines
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_json(
    file_path: str,
    field: str,
    num_samples: int = -1,
    shuffle: bool = False,
    seed: int = 0,
) -> List[str]:
    """
    Load JSON dataset with optional sampling and shuffling.

    Args:
        file_path (str): Path to the JSON file
        field (str): Field to extract from the dataset
        num_samples (int, optional): Number of samples to return. Defaults to -1 (all).
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        seed (int, optional): Random seed for shuffling. Defaults to 0.

    Returns:
        List[str]: Extracted and potentially sampled/shuffled data
    """
    data = load_dataset("json", data_files=file_path, split="train")

    if shuffle:
        data = data.shuffle(seed=seed)

    # Determine number of samples
    sample_count = len(data) if num_samples == -1 else min(num_samples, len(data))

    return data[field][:sample_count]


def save_iteration_log(log_dir, iteration, adv_prompts, responses, scores, timestamp):
    """
    Save log of current iteration's results.

    Args:
        log_dir: Directory for saving log files
        iteration: Current iteration number
        adv_prompts: Archive of adversarial prompts
        responses: Archive of model responses
        scores: Archive of prompt scores
        timestamp: Timestamp for log filename
    """
    log_path = log_dir / f"rainbowplus_log_{timestamp}_epoch_{iteration+1}.json"

    with open(log_path, "w") as f:
        json.dump(
            {
                "adv_prompts": {
                    str(key): value for key, value in adv_prompts._archive.items()
                },
                "responses": {
                    str(key): value for key, value in responses._archive.items()
                },
                "scores": {str(key): value for key, value in scores._archive.items()},
            },
            f,
            indent=2,
        )

    logger.info(f"Log saved to {log_path}")


def initialize_language_models(config: ConfigurationLoader):
    """
    Initialize language models from configuration.

    Args:
        config: Configuration object containing model settings

    Returns:
        Dictionary of initialized language models
    """
    # Extract model configurations
    model_configs = [
        config.target_llm,
        config.mutator_llm,
        config.judge_llm,
    ]

    # Create unique language model switchers
    llm_switchers = {}
    seen_model_configs = set()

    for model_config in model_configs:
        # Create a hashable representation of model kwargs
        config_key = tuple(sorted(model_config.model_kwargs.items()))

        # Only create a new LLM switcher if this configuration hasn't been seen before
        if config_key not in seen_model_configs:
            try:
                llm_switcher = LLMSwitcher(model_config)
                model_name = model_config.model_kwargs.get("model", "unnamed_model")
                llm_switchers[model_name] = llm_switcher
                seen_model_configs.add(config_key)
            except ValueError as e:
                logger.error(f"Error initializing model {model_config}: {e}")

    return llm_switchers
