from setuptools import setup, find_packages

setup(
    name="rainbow-teaming",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "vllm==0.6.3.post1",
        "openai",
        "nltk",
        "pydantic",
        "PyYAML",
        "datasets",
    ],
)
