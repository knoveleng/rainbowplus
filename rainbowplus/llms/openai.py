from openai import OpenAI
from rainbowplus.llms.base import BaseLLM
from pydantic import BaseModel

from typing import List


class LLMviaOpenAI(BaseLLM):
    def __init__(self, config):
        self.config = config
        self.model_kwargs = config.model_kwargs
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def get_name(self):
        return self.model_kwargs["model"]

    def generate(self, query: str, sampling_params: dict):
        response = self.client.chat.completions.create(
            model=self.model_kwargs["model"],
            messages=[
                {"role": "user", "content": query},
            ],
            **sampling_params
        )
        return response.choices[0].message.content

    def batch_generate(self, queries: List[str], sampling_params: dict):
        responses = [self.generate(query, sampling_params) for query in queries]
        return responses

    def generate_format(
        self, query: str, sampling_params: dict, response_format: BaseModel
    ):
        completion = self.client.beta.chat.completions.parse(
            model=self.model_kwargs["model"],
            messages=[
                {"role": "user", "content": query},
            ],
            **sampling_params,
            response_format=response_format
        )
        message = completion.choices[0].message
        return message.parsed
