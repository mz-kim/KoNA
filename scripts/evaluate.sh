#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 <gpu_ids> <model> <backend> <test_jsonl> <save_dir> [default|cot|strategy|contrast]" >&2
  exit 1
fi

gpu_ids="$1"
model="$2"
backend="$3"
test_jsonl="$4"
save_dir="$5"
prompt="${6:-default}"
num_gpus=$(awk -F, '{print NF}' <<< "$gpu_ids")
base_url="${MODEL_BASE_URL:-}"
evaluator_model="${EVALUATOR_MODEL:-gpt-5-mini-2025-08-07}"

prompt_args=()
if [[ "$prompt" != "default" ]]; then
  prompt_args=(--prompt "$prompt")
fi

base_url_args=()
if [[ -n "$base_url" ]]; then
  base_url_args=(--model_backend_base_url "$base_url")
fi

CUDA_VISIBLE_DEVICES="$gpu_ids" python -m eval.run_inference \
  --model "$model" \
  --model_engine_backend "$backend" \
  --model_num_gpus "$num_gpus" \
  --data_dir "$test_jsonl" \
  --save_dir "$save_dir" \
  "${base_url_args[@]}" \
  "${prompt_args[@]}"

python -m eval.run_eval \
  --model "$model" \
  --evaluator "$evaluator_model" \
  --evaluator_engine_backend openai \
  --data_dir "$test_jsonl" \
  --save_dir "$save_dir" \
  "${prompt_args[@]}"
