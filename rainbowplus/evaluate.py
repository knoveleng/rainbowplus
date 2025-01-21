import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from rainbowxplus.archive import Archive
from rainbowxplus.prompts import EVALUATE_PROMPT
from rainbowxplus.configs import EvalConfigLoader
from rainbowxplus.llms import vLLM


class EvaluationProcessor:
    def __init__(
        self,
        config_path: str,
        log_dir: str = None,
        method: str = None,
        algo: str = None,
    ):
        """
        Initialize the evaluation processor.

        :param config_path: Path to the configuration file
        :param log_dir: Override log directory from config
        :param method: Override evaluation method from config
        """
        self.config = EvalConfigLoader.load(config_path)

        # Override config with command-line arguments if provided
        self.log_dir = Path(log_dir or self.config.log_dir)
        self.method = method or self.config.method

        self.sampling_params = self.config.judge_llm.sampling_params
        self.llm = vLLM(self.config.judge_llm.model_kwargs)
        self.algo = algo or self.config.algo

    def _save_eval_log(self, file_name: str, archive_instances: List[Archive]) -> None:
        """Save evaluation logs to a JSON file."""
        log_path = self.log_dir / f"eval_{self.method}_{file_name}.json"

        log_data = {
            archive.name: {str(key): value for key, value in archive._archive.items()}
            for archive in archive_instances
        }

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"Log saved to {log_path}")

    def _process_median_max_method(
        self, adv_prompts: Archive, responses: Archive, scores: Archive
    ) -> Dict[str, Archive]:
        """Process evaluation using median or max method."""
        eval_prompts = Archive("eval_prompts")
        eval_responses = Archive("eval_responses")
        eval_scores = Archive("eval_scores")
        judge_responses = Archive("judge_responses")

        idxs = (
            scores.idx_max_elements()
            if self.method == "max"
            else scores.idx_median_elements()
        )
        for key, idx in idxs.items():
            chosen_prompt = adv_prompts.get(key)[idx]
            chosen_response = responses.get(key)[idx]
            chosen_score = scores.get(key)[idx]

            query = EVALUATE_PROMPT.format(
                prompt=chosen_prompt, response=chosen_response
            )
            res = self.llm.generate(query, self.sampling_params)

            eval_prompts.add(key, chosen_prompt)
            eval_responses.add(key, chosen_response)
            eval_scores.add(key, chosen_score)
            judge_responses.add(key, res)

        return {
            "eval_prompts": eval_prompts,
            "eval_responses": eval_responses,
            "eval_scores": eval_scores,
            "judge_responses": judge_responses,
        }

    def _process_all_method(
        self,
        idx: int,
        log_files: List[str],
        adv_prompts: Archive,
        responses: Archive,
        scores: Archive,
    ) -> Dict[str, Archive]:
        """Process evaluation using all method."""
        eval_prompts = Archive("eval_prompts")
        eval_responses = Archive("eval_responses")
        eval_scores = Archive("eval_scores")
        judge_responses = Archive("judge_responses")

        # Subtract previous iteration's data for rainbowplus algorithm
        if idx > 0 and self.algo == "rainbowplus":
            pre_file = os.path.join(self.log_dir, log_files[idx - 1])
            with open(pre_file, "r") as f:
                pre_log_dict = json.load(f)

            pre_adv_prompts = Archive("pre_adv_prompts")
            pre_adv_prompts.load_from_dict(pre_log_dict["adv_prompts"])
            pre_responses = Archive("pre_responses")
            pre_responses.load_from_dict(pre_log_dict["responses"])
            pre_scores = Archive("pre_scores")
            pre_scores.load_from_dict(pre_log_dict["scores"])

            adv_prompts = adv_prompts.subtract(pre_adv_prompts)
            responses = responses.subtract(pre_responses)
            scores = scores.subtract(pre_scores)

        # Process all prompts
        for key in adv_prompts._archive.keys():
            chosen_prompts = adv_prompts.get(key)
            chosen_responses = responses.get(key)
            chosen_scores = scores.get(key)

            queries = [
                EVALUATE_PROMPT.format(prompt=prompt, response=response)
                for prompt, response in zip(chosen_prompts, chosen_responses)
            ]
            outputs = self.llm.batch_generate(queries, self.sampling_params)

            eval_prompts.add(key, chosen_prompts)
            eval_responses.add(key, chosen_responses)
            eval_scores.add(key, chosen_scores)
            judge_responses.add(key, outputs)

        return {
            "eval_prompts": eval_prompts,
            "eval_responses": eval_responses,
            "eval_scores": eval_scores,
            "judge_responses": judge_responses,
        }

    def process_evaluations(self):
        """Process evaluations across log files."""
        # Get sorted log files starting with the specified method
        log_files = sorted(
            [f for f in os.listdir(self.log_dir) if f.startswith(self.algo)]
        )

        for idx, log_file in enumerate(log_files):
            # Load log file
            path = os.path.join(self.log_dir, log_file)
            with open(path, "r") as f:
                log_dict = json.load(f)

            # Create Archives from log data
            adv_prompts = Archive("adv_prompts")
            adv_prompts.load_from_dict(log_dict["adv_prompts"])
            responses = Archive("responses")
            responses.load_from_dict(log_dict["responses"])
            scores = Archive("scores")
            scores.load_from_dict(log_dict["scores"])

            # Process based on method
            if self.method in ["median", "max"]:
                processed_archives = self._process_median_max_method(
                    adv_prompts, responses, scores
                )
            elif self.method == "all":
                processed_archives = self._process_all_method(
                    idx, log_files, adv_prompts, responses, scores
                )
            else:
                raise ValueError(f"Unsupported method: {self.method}")

            # Save evaluation log
            self._save_eval_log(log_files[idx], list(processed_archives.values()))


def main():
    parser = argparse.ArgumentParser(description="Evaluation Processing Script")
    parser.add_argument(
        "-c", "--config", default="configs/eval.yml", help="Path to configuration file"
    )
    parser.add_argument("-l", "--log_dir", help="Override log directory from config")
    parser.add_argument("-a", "--algo", help="Override algorithm from config")
    parser.add_argument("-m", "--method", help="Override evaluation method from config")

    args = parser.parse_args()

    processor = EvaluationProcessor(
        config_path=args.config,
        log_dir=args.log_dir,
        algo=args.algo,
        method=args.method,
    )
    processor.process_evaluations()


if __name__ == "__main__":
    main()
