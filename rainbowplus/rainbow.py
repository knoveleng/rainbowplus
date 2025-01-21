import argparse
import random
import json
import time
from pathlib import Path

from rainbowplus.llms.vllm import vLLM
from rainbowplus.scores.bleu import BleuScoreNLTK
from rainbowplus.scores.llama_guard import LlamaGuard
from rainbowplus.utils import load_txt, load_json, LLMSwitcher
from rainbowplus.archive import Archive
from rainbowplus.config import read_config
from rainbowplus.prompts import MUTATOR_PROMPT, JUDGE_PROMPT, TARGET_PROMPT


def parse_arguments():
    """
    Parse command-line arguments for adversarial prompt generation.

    Returns:
        Parsed arguments with configuration for the script
    """
    parser = argparse.ArgumentParser(description="Adversarial Prompt Generation")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=150,
        help="Number of initial seed prompts to process",
    )
    parser.add_argument(
        "--max_iters", type=int, default=1000, help="Maximum number of iteration steps"
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for prompt mutation",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./configs/base.yml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Directory for storing logs"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Number of iterations between log saves",
    )
    return parser.parse_args()


def initialize_language_models(config):
    """
    Initialize language models from configuration.

    Args:
        config: Configuration object containing model settings

    Returns:
        Dictionary of initialized language models
    """
    # Extract model configurations
    model_configs = [config.target_llm, config.mutator_llm, config.judge_llm]

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
                print(f"Error initializing model {model_config}: {e}")

    return llm_switchers


def load_descriptors(config):
    """
    Load descriptors from specified paths.

    Args:
        config: Configuration object with archive paths

    Returns:
        Dictionary of descriptors loaded from text files
    """
    return {
        descriptor: load_txt(path)
        for path, descriptor in zip(
            config.archive["path"], config.archive["descriptor"]
        )
    }


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
    log_path = log_dir / f"rainbow_log_{timestamp}_epoch_{iteration+1}.json"

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

    print(f"Log saved to {log_path}")


def main():
    """
    Main function to execute adversarial prompt generation process.
    Handles prompt mutation, model interactions, and logging.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration and seed prompts
    config = read_config(args.config_file)
    print(config.model_dump_json(indent=2))
    seed_prompts = load_json(
        config.sample_prompts,
        field="question",
        num_samples=args.num_samples,
        shuffle=True,
    )

    # Initialize language models and scoring functions
    llms = initialize_language_models(config)
    fitness_fn = LlamaGuard(config.fitness_llm.model_kwargs)
    similarity_fn = BleuScoreNLTK()

    # Load category descriptors
    descriptors = load_descriptors(config)

    # Initialize archives for adversarial prompts
    adv_prompts = Archive("adv_prompts")
    responses = Archive("responses")
    scores = Archive("scores")

    # Create log directory
    dataset_name = Path(config.sample_prompts).stem
    log_dir = (
        Path(args.log_dir) / config.target_llm.model_kwargs["model"] / dataset_name
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # Main adversarial prompt generation loop
    for i in range(args.max_iters):
        print(f"#####ITERATION: {i}")

        # Select prompt (initial seed or from existing adversarial prompts)
        prompt = (
            seed_prompts[i]
            if i < len(seed_prompts)
            else random.choice(adv_prompts.flatten_values())
        )

        # Sample random descriptors
        descriptor = {key: random.choice(value) for key, value in descriptors.items()}

        # Create unique key for this descriptor set
        key = tuple(descriptor.values())

        # Prepare descriptor string for prompt mutation
        descriptor_str = "- " + "- ".join(
            [f"{key}: {value}\n" for key, value in descriptor.items()]
        )

        # Mutate prompt using mutator LLM
        prompt_ = MUTATOR_PROMPT.format(
            descriptor=descriptor_str.strip(), prompt=prompt
        )
        mutated_prompt = llms[config.mutator_llm.model_kwargs["model"]].generate(
            prompt_, config.mutator_llm.sampling_params
        )

        # Check prompt similarity and process if sufficiently different
        if similarity_fn.score(mutated_prompt, prompt) < args.sim_threshold:
            target_prompt = TARGET_PROMPT.format(prompt=mutated_prompt.strip())
            candidate = llms[config.target_llm.model_kwargs["model"]].generate(
                target_prompt, config.target_llm.sampling_params
            )
            score = fitness_fn.score(
                mutated_prompt, candidate, config.fitness_llm.sampling_params
            )

            print(f"Prompt for Mutator: {prompt_}")
            print("-" * 50)
            print(f"Mutated Prompt: {mutated_prompt}")
            print("-" * 50)
            print(f"Candidate: {candidate}")
            print("-" * 50)
            print("\n\n\n")

            # Update or add new adversarial prompt
            if not adv_prompts.exists(key):
                adv_prompts.add(key, [mutated_prompt])
                responses.add(key, [candidate])
                scores.add(key, [score])
            else:
                # Compare and potentially replace existing prompt
                response = responses.get(key)[0]
                judge_prompt = JUDGE_PROMPT.format(
                    response_1=candidate, response_2=response
                )
                judge_response = llms[config.judge_llm.model_kwargs["model"]].generate(
                    judge_prompt, config.judge_llm.sampling_params
                )

                if "yes" in judge_response.lower():
                    adv_prompts.update(key, [mutated_prompt])
                    responses.update(key, [candidate])
                    scores.update(key, [score])

        # Periodic logging
        if i > 0 and (i + 1) % args.log_interval == 0:
            timestamp = time.strftime(r"%Y%m%d-%H%M%S")
            save_iteration_log(log_dir, i, adv_prompts, responses, scores, timestamp)


if __name__ == "__main__":
    main()
