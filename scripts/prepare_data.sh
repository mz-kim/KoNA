#!/usr/bin/env bash
set -euo pipefail

input_path="${1:-dataset/annotations/train.jsonl}"
output_dir="${2:-dataset/train}"
validation_path="${3:-dataset/annotations/validation.jsonl}"

python -m utils.prepare_train_set \
  --dataset_path "$input_path" \
  --num_selective_examples 1000 \
  --num_pure_examples 100 \
  --contain_contrast \
  --num_contrast_examples 100 \
  --is_grpo \
  --output_dir "$output_dir" \
  --seed 807

python -m utils.prepare_train_set \
  --dataset_path "$validation_path" \
  --num_selective_examples 300 \
  --num_contrast_examples 0 \
  --output_dir "$output_dir/validation" \
  --seed 807
