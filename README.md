# RainbowPlus

## Overview
This repository contains the implementation of the methods described in our research paper **"RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search"**.

## Key Features

- **State-of-the-Art Performance**: Achieves superior results compared to existing methods on HarmBench benchmark
- **Universal Compatibility**: Supports both open-source and proprietary LLMs
- **Computational Efficiency**: Completes an evaluation in just 1.45 hours
- **Flexible Configuration**: Highly customizable for various experimental settings

## Repository Structure

```
├── configs/                  # Configuration files
│   ├── categories/           # Category definitions
│   ├── styles/               # Style definitions
│   ├── base.yml              # Base configuration
│   ├── base-openai.yml       # OpenAI-specific configuration
│   ├── base-sota.yml         # SOTA model configuration
│   └── eval.yml              # Evaluation configuration
│
├── data/                     # Dataset storage
│
├── rainbowplus/              # Core package
│   ├── archive.py            # Archive management
│   ├── configs/              # Configuration utilities
│   ├── evaluate.py           # Current evaluation implementation
│   ├── evaluate_v0.py        # Legacy evaluation implementation
│   ├── get_scores.py         # Metrics extraction utilities
│   ├── llms/                 # LLM integration modules
│   ├── scores/               # Fitness and similarity functions
│   ├── prompts.py            # LLM prompt templates
│   ├── rainbowplus.py        # Main implementation
│   └── utils.py              # Utility functions
│
├── sh/                       # Shell scripts
│   └── run.sh                # All-in-one execution script
│
├── README.md                 # This documentation
└── setup.py                  # Package installation script
```

## Getting Started

### 1. Environment Setup

Create and activate a Python virtual environment, then install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. API Configuration

#### Hugging Face Token (Optional)

Required for accessing certain resources from the Hugging Face Hub (e.g., Llama Guard):

```bash
export HF_AUTH_TOKEN="YOUR_HF_TOKEN"
```

Alternatively:

```bash
huggingface-cli login --token=YOUR_HF_TOKEN
```

#### OpenAI API Key (Optional)

Required when using OpenAI models:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

## Usage

### LLM Configuration

RainbowPlus supports two primary LLM integration methods:

#### 1. vLLM (Open-Source Models)

Example configuration for Qwen-2.5-7B-Instruct:

```yaml
target_llm:
  type_: vllm

  model_kwargs:
    model: Qwen/Qwen2.5-7B-Instruct
    trust_remote_code: True
    max_model_len: 2048
    gpu_memory_utilization: 0.5

  sampling_params:
    temperature: 0.6
    top_p: 0.9
    max_tokens: 1024
```

Additional parameters can be specified according to the [vLLM model documentation](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html) and [sampling parameters documentation](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-parameters).

#### 2. OpenAI API (Proprietary Models)

Example configuration for GPT-4o-mini:

```yaml
target_llm:
  type_: openai

  model_kwargs:
    model: gpt-4o-mini

  sampling_params:
    temperature: 0.6
    top_p: 0.9
    max_tokens: 1024
```

Additional parameters can be specified according to the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat/create).

### Running Experiments

Basic execution with default configuration:

```bash
python -m rainbowplus.rainbowplus --config_file configs/base.yml
```

For customized experiments, you can override specific parameters:

```bash
python -m rainbowplus.rainbowplus \
    --config_file configs/base-sota.yml \
    --num_samples -1 \
    --max_iters 400 \
    --sim_threshold 0.6 \
    --num_mutations 10 \
    --fitness_threshold 0.6 \
    --log_dir logs-sota \
    --dataset ./data/harmbench.json \
    --target_llm "model_identifier" \
    --log_interval 50 \
    --shuffle True
```

#### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `target_llm` | Target LLM identifier |
| `num_samples` | Number of initial seed prompts |
| `max_iters` | Maximum number of iteration steps |
| `sim_threshold` | Similarity threshold for prompt mutation |
| `num_mutations` | Number of prompt mutations per iteration |
| `fitness_threshold` | Minimum fitness score to add prompt to archive |
| `log_dir` | Directory for storing logs |
| `dataset` | Dataset path |
| `shuffle` | Whether to shuffle seed prompts |
| `log_interval` | Number of iterations between log saves |

### Batch Processing Multiple Models

For evaluating multiple models sequentially:

```bash
MODEL_IDS="meta-llama/Llama-2-7b-chat-hf lmsys/vicuna-7b-v1.5 baichuan-inc/Baichuan2-7B-Chat Qwen/Qwen-7B-Chat"

for MODEL in $MODEL_IDS; do
    python -m rainbowplus.rainbowplus \
        --config_file configs/base-sota.yml \
        --num_samples -1 \
        --max_iters 400 \
        --sim_threshold 0.6 \
        --num_mutations 10 \
        --fitness_threshold 0.6 \
        --log_dir logs-sota \
        --dataset ./data/harmbench.json \
        --target_llm $MODEL \
        --log_interval 50 \
        --shuffle True

    # Clean cache between model runs
    rm -r ~/.cache/huggingface/hub/
done
```

## Evaluation

After running experiments, evaluate the results:

```bash
MODEL_IDS="gpt-4.1-nano"  # For multiple models: MODEL_IDS="gpt-4.1-nano gpt-4o-mini"

for MODEL in $MODEL_IDS; do
    # Run evaluation
    python -m rainbowplus.evaluate \
        --config configs/eval.yml \
        --log_dir "./logs-sota/$MODEL/harmbench"

    # Extract metrics
    python rainbowplus/get_scores.py \
        --log_dir "./logs-sota/$MODEL/harmbench" \
        --keyword "global"
done
```

### Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `config` | Path to configuration file |
| `log_dir` | Directory containing experiment logs |
| `keyword` | Keyword for global config file name (default: `global`), you can ignore this param |

### Output Metrics

Results are saved in JSON format:

```json
{
    "General": 0.79,
    "All": 0.8666092943201377
}
```

Where:
- `General`: Metrics calculated using SOTA methods (on original prompts only)
- `All`: Metrics calculated across all generated prompts

## Streamlined Execution

For end-to-end execution, use the provided shell script:

```bash
bash sh/run.sh
```

## Next Features

- [ ] Support OpenAI as fitness function
- [ ] Deploy via FastAPI
- [ ] Support more LLMs

## Citation

```
Coming soon...
```

## License

[License information will be added upon paper publication]