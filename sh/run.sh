#!/bin/bash

# Common experiment parameters
MAX_ITERS=400
NUM_SAMPLES=-1
SIM_THRESHOLD=0.6
NUM_MUTATIONS=10
FITNESS_THRESHOLD=0.6
LOG_INTERVAL=50
DATASET="./data/harmbench.json"
LOG_DIR="./logs-sota"

# Run experiment for a specific model
run_experiment() {
    local model=$1
    local config_file=$2
    
    echo "Running experiment for model: $model"
    python -m rainbowplus.rainbowplus \
        --config_file "$config_file" \
        --num_samples "$NUM_SAMPLES" \
        --max_iters "$MAX_ITERS" \
        --sim_threshold "$SIM_THRESHOLD" \
        --num_mutations "$NUM_MUTATIONS" \
        --fitness_threshold "$FITNESS_THRESHOLD" \
        --log_dir "$LOG_DIR" \
        --dataset "$DATASET" \
        --log_interval "$LOG_INTERVAL" \
        ${model:+--target_llm "$model"}
}

# Run evaluation for a specific model
run_evaluation() {
    local model=$1
    
    echo "Running evaluation for model: $model"
    python -m rainbowplus.evaluate \
        --config configs/eval.yml \
        --log_dir "$LOG_DIR/$model/harmbench"

    echo "Getting scores for model: $model"
    python rainbowplus/get_scores.py \
        --log_dir "$LOG_DIR/$model/harmbench"
}

# Clear HuggingFace cache
clear_hf_cache() {
    echo "Clearing HuggingFace cache"
    rm -rf ~/.cache/huggingface/hub/
}

# Main execution
main() {
    echo "=== Running experiments for open-source models ==="
    # Open-source models
    OPEN_SOURCE_MODELS=(
        "meta-llama/Llama-2-7b-chat-hf"
        "lmsys/vicuna-7b-v1.5"
        "baichuan-inc/Baichuan2-7B-Chat"
        "Qwen/Qwen-7B-Chat"
        "TheBloke/koala-7B-HF"
        "microsoft/Orca-2-7b"
        "mistralai/Mistral-7B-Instruct-v0.2"
        "openchat/openchat-3.5-1210"
        "berkeley-nest/Starling-LM-7B-alpha"
        "HuggingFaceH4/zephyr-7b-beta"
    )
    
    for MODEL in "${OPEN_SOURCE_MODELS[@]}"; do
        run_experiment "$MODEL" "configs/base-opensource.yml"
        clear_hf_cache
    done
    
    for MODEL in "${OPEN_SOURCE_MODELS[@]}"; do
        run_evaluation "$MODEL"
    done
    
    echo "=== Running experiments for closed-source models ==="
    # Closed-source models
    CLOSED_SOURCE_MODELS=(
        "gpt-4.1-nano"
        "gpt-4o-mini"
    )
    
    for MODEL in "${CLOSED_SOURCE_MODELS[@]}"; do
        run_experiment "$MODEL" "configs/base-openai.yml"
    done
    
    for MODEL in "${CLOSED_SOURCE_MODELS[@]}"; do
        run_evaluation "$MODEL"
    done
}

# Execute main function
main