import sys
import argparse
import random
import json
import time
import logging
from pathlib import Path


from rainbowplus.scores import BleuScoreNLTK, LlamaGuard
from rainbowplus.utils import (
    load_txt,
    load_json,
    initialize_language_models,
    save_iteration_log,
)
from rainbowplus.archive import Archive
from rainbowplus.configs import ConfigurationLoader
from rainbowplus.prompts import MUTATOR_PROMPT, TARGET_PROMPT

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command-line arguments for adversarial prompt generation.

    Returns:
        Parsed arguments with configuration for the script
    """
    parser = argparse.ArgumentParser(description="Adversarial Prompt Generation")
    parser.add_argument(
        "--num_samples", type=int, default=150, help="Number of initial seed prompts"
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
        "--num_mutations",
        type=int,
        default=5,
        help="Number of prompt mutations per iteration",
    )
    parser.add_argument(
        "--fitness_threshold",
        type=float,
        default=0.5,
        help="Minimum fitness score to add prompt to archive",
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
    parser.add_argument(
        "--dataset", type=str, default="./data/do-not-answer.json", help="Dataset name"
    )
    parser.add_argument(
        "--target_llm",
        type=str,
        default="mistralai/Ministral-8B-Instruct-2410",
        help="Path to repository of target LLM",
    )
    return parser.parse_args()


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


def run_rainbowplus(
    args, config, seed_prompts=[], llms=None, fitness_fn=None, similarity_fn=None
):
    """
    Main function to execute adversarial prompt generation process.
    Handles prompt mutation, model interactions, and logging.
    """
    # Load seed prompts
    if not seed_prompts:
        seed_prompts = load_json(
            config.sample_prompts,
            field="question",
            num_samples=args.num_samples,
            shuffle=True,
        )

    # Load category descriptors
    descriptors = load_descriptors(config)

    # Initialize archives for adversarial prompts
    adv_prompts = Archive("adv_prompts")
    responses = Archive("responses")
    scores = Archive("scores")

    # Prepare log directory
    dataset_name = Path(config.sample_prompts).stem
    log_dir = (
        Path(args.log_dir) / config.target_llm.model_kwargs["model"] / dataset_name
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # Main adversarial prompt generation loop
    for i in range(args.max_iters):
        logger.info(f"#####ITERATION: {i}")

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

        # Mutate prompts using mutator LLM
        mutator_model = config.mutator_llm.model_kwargs["model"]
        prompt_ = MUTATOR_PROMPT.format(
            descriptor=descriptor_str.strip(), prompt=prompt
        )
        mutated_prompts = llms[mutator_model].batch_generate(
            [prompt_] * args.num_mutations, config.mutator_llm.sampling_params
        )

        # Filter mutated prompts by similarity
        mutated_prompts = [
            p
            for p in mutated_prompts
            if similarity_fn.score(p, prompt_) < args.sim_threshold
        ]

        if mutated_prompts:
            # Generate responses for mutated prompts
            target_prompts = [
                TARGET_PROMPT.format(prompt=p.strip()) for p in mutated_prompts
            ]
            target_model = config.target_llm.model_kwargs["model"]
            candidates = llms[target_model].batch_generate(
                target_prompts, config.target_llm.sampling_params
            )

            # Score fitness of mutated prompts
            fitness_scores = fitness_fn.batch_score(
                mutated_prompts, candidates, config.fitness_llm.sampling_params
            )

            # Filter prompts based on fitness threshold
            filtered_data = [
                (p, c, s)
                for p, c, s in zip(mutated_prompts, candidates, fitness_scores)
                if s > args.fitness_threshold
            ]

            if filtered_data:
                # Unpack filtered data
                filtered_prompts, filtered_candidates, filtered_scores = zip(
                    *filtered_data
                )

                # Use logger
                logger.info(f"Prompt for Mutator: {prompt_}")
                logger.info(f"Mutated Prompt: {filtered_prompts}")
                logger.info(f"Candidate: {filtered_candidates}")
                logger.info(f"Score: {filtered_scores}")
                logger.info("\n\n\n")

                # Update archives
                if not adv_prompts.exists(key):
                    adv_prompts.add(key, filtered_prompts)
                    responses.add(key, filtered_candidates)
                    scores.add(key, filtered_scores)
                else:
                    adv_prompts.extend(key, filtered_prompts)
                    responses.extend(key, filtered_candidates)
                    scores.extend(key, filtered_scores)

        # Periodic logging
        if i > 0 and (i + 1) % args.log_interval == 0:
            timestamp = time.strftime(r"%Y%m%d-%H%M%S")
            save_iteration_log(log_dir, i, adv_prompts, responses, scores, timestamp)

    # Save final log
    timestamp = time.strftime(r"%Y%m%d-%H%M%S")
    save_iteration_log(log_dir, i, adv_prompts, responses, scores, timestamp)

    # Return final archives
    return adv_prompts, responses, scores


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration and seed prompts
    config = ConfigurationLoader.load(args.config_file)
    print(config)

    # Update configuration based on command-line arguments
    config.target_llm.model_kwargs["model"] = args.target_llm
    config.sample_prompts = args.dataset

    # Initialize language models and scoring functions
    llms = initialize_language_models(config)
    fitness_fn = LlamaGuard(config.fitness_llm.model_kwargs)
    similarity_fn = BleuScoreNLTK()

    # Run the adversarial prompt generation process
    run_rainbowplus(
        args,
        config,
        seed_prompts=[],
        llms=llms,
        fitness_fn=fitness_fn,
        similarity_fn=similarity_fn,
    )
