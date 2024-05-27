#!/bin/bash

# Base directory and paths
base_dir="/Users/maritina/Desktop/mt-exercise-5"
model_dir="$base_dir/models/bpe2k"
config_path="$base_dir/configs/transformer_bpe_2k.yaml"
test_data_path="$base_dir/test/test.nl"
reference_data_path="$base_dir/test/test.en"

# Directories for beam search experiments
beam_experiments="$base_dir/beam_experiments"
translations_dir="$beam_experiments/translations"
bleu_scores_dir="$beam_experiments/bleu_scores"

# Create directories if they don't exist
mkdir -p $translations_dir
mkdir -p $bleu_scores_dir

# Parameters
src="nl"
trg="en"
cpu_threads=10
gpu_device=0

# Timer
SECONDS=0

# Model name
model_name="transformer_bpe_2k"

# Optional user input: maximum beam size
beam_max=${1:-10}

# Iterate over each beam size from 1 to beam_max
for (( beam_size=1; beam_size<=${beam_max}; beam_size++ )); do
    # Create a copy of the config with the modified beam size
    sed -r "s/(\s*beam_size\s*:[[:space:]]*).*/\1$beam_size/g" $config_path > $base_dir/configs/${model_name}_beam_${beam_size}.yaml

    echo "########################################"
    echo "Beam size $beam_size"

    # BPE level model translation
    CUDA_VISIBLE_DEVICES=$gpu_device OMP_NUM_THREADS=$cpu_threads python -m joeynmt translate $base_dir/configs/${model_name}_beam_${beam_size}.yaml < $test_data_path > $translations_dir/test.${beam_size}.$trg

    # Compute case-sensitive BLEU on detokenized data
    sacrebleu $reference_data_path < $translations_dir/test.${beam_size}.$trg > $bleu_scores_dir/bleu_${beam_size}.json

    # Clean up the temporary config file
    rm $base_dir/configs/${model_name}_beam_${beam_size}.yaml
done

# Output the time taken
echo "time taken:"
echo "$SECONDS seconds"
