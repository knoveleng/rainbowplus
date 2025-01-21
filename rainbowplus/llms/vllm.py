from typing import List
from vllm import LLM, SamplingParams
from rainbowplus.llms.base import BaseLLM


class vLLM(BaseLLM):
    def __init__(self, model_kwargs: dict):
        self.model_kwargs = model_kwargs
        self.llm = LLM(**model_kwargs)

    def get_name(self):
        return self.model_kwargs["model"]

    def generate(self, query: str, sampling_params: dict):
        outputs = self.llm.generate([query], SamplingParams(**sampling_params))
        response = outputs[0].outputs[0].text
        return response

    def batch_generate(self, queries: List[str], sampling_params: dict):
        outputs = self.llm.generate(queries, SamplingParams(**sampling_params))
        return [output.outputs[0].text for output in outputs]
