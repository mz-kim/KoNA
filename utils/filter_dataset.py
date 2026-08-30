import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from dataset.dataset_generate_util import FILTERING_DICT, STEP_CONTAINMENT_FILTERING
from mllm import GenerationArgs, UniversalGenParams, VLMInferenceEngine

SPLITS = ["train", "val", "test"]
SOURCES = ["coco", "openimages"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", required=True, choices=[k for k in FILTERING_DICT.keys() if k != "step_containment"], help="Task type to filter.")
    parser.add_argument("--model", required=True, help="Model identifier for VLMInferenceEngine.")
    parser.add_argument("--model_engine_backend", default="vllm-openai", choices=["vllm", "vllm-openai", "openai", "gemini"], help="Backend to use for inference.")
    parser.add_argument("--model_backend_base_url", type=str, default=None, help="Base URL for vllm-openai backend.")
    parser.add_argument("--model_num_gpus", type=int, default=1, help="Tensor parallel size for vllm backend.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vllm backend.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens for filtering generations.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for filtering generations.")
    parser.add_argument("--input_dir", type=str, default="dataset", help="Root directory containing generated datasets.")
    parser.add_argument("--generation_model", choices=["gpt", "gemini"], required=True, help="Generation model type.")
    return parser.parse_args()

def build_backend_kwargs(args: argparse.Namespace) -> Dict:
    if args.model_engine_backend == "vllm":
        return {"tensor_parallel_size": args.model_num_gpus, "gpu_memory_utilization": args.gpu_memory_utilization}
    if args.model_engine_backend == "vllm-openai":
        return {"base_url": args.model_backend_base_url}
    return {}

def load_second_stage_data(input_dir: Path, generation_model: str, task: str, split: str, source: str) -> List[Dict]:
    input_path = input_dir / generation_model / task / source / split / f"second_{task}_{split}_{source}.jsonl"
    if not input_path.exists():
        print(f"[WARN] Skipping missing dataset: {input_path}")
        return []

    records: List[Dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records

def run_inference(engine: VLMInferenceEngine, prompts: List[str], gen_params: UniversalGenParams) -> List[str]:
    gen_args = GenerationArgs(
        engine_input=prompts,
        gen_params=gen_params,
        is_multi_turn_input=False,
        is_batch_input=True,
    )
    outputs = engine.generate(gen_args)
    return [out.output_seqs[0] if (out.output_seqs and out.output_seqs[0]) else "" for out in outputs]

def is_pass(text: str) -> bool:
    if not isinstance(text, str):
        return False
    upper = text.upper()
    if "FINAL EVALUATION:" in upper:
        return "FINAL EVALUATION: PASS" in upper
    return "PASS" in upper and "FAIL" not in upper

def build_step_containment_prompts(records: List[Dict]) -> List[str]:
    prompts = []
    for rec in records:
        prompts.append(
            STEP_CONTAINMENT_FILTERING.format(
                stage1_question=rec.get("question_1", ""),
                stage1_answer=rec.get("answer_1", ""),
                stage2_question=rec.get("question_2", ""),
                stage2_answer=rec.get("answer_2", ""),
            )
        )
    return prompts

def build_task_prompts(task: str, records: List[Dict]) -> List[str]:
    template = FILTERING_DICT[task]
    return [template.format(question=rec.get("question_2", ""), answer=rec.get("answer_2", "")) for rec in records]

def filter_records(
    engine: VLMInferenceEngine,
    task: str,
    records: List[Dict],
    gen_params: UniversalGenParams,
) -> List[Dict]:
    if not records:
        return []

    containment_prompts = build_step_containment_prompts(records)
    stage1_outputs = run_inference(engine, containment_prompts, gen_params)
    
    stage1_records = [
        rec for rec, out in zip(records, stage1_outputs) if is_pass(out)
    ]

    if not stage1_records:
        return []

    task_prompts = build_task_prompts(task, stage1_records)
    stage2_outputs = run_inference(engine, task_prompts, gen_params)
    
    final_records = [
        rec for rec, out in zip(stage1_records, stage2_outputs) if is_pass(out)
    ]

    return final_records

def save_filtered(output_dir: Path, task: str, split: str, source: str, records: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"second_{task}_{split}_{source}_filtered.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(records)} items to {output_path}")

def main():
    args = parse_args()
    backend_kwargs = build_backend_kwargs(args)
    engine = VLMInferenceEngine(args.model, backend=args.model_engine_backend, backend_kwargs=backend_kwargs)

    gen_params = UniversalGenParams(
        n=1,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    input_root = Path(args.input_dir)

    for split in SPLITS:
        for source in SOURCES:
            records = load_second_stage_data(input_root, args.generation_model, args.task_type, split, source)
            if not records:
                continue

            filtered = filter_records(
                engine=engine,
                task=args.task_type,
                records=records,
                gen_params=gen_params,
            )

            save_dir = input_root / args.generation_model / args.task_type / source / split
            save_filtered(save_dir, args.task_type, split, source, filtered)

if __name__ == "__main__":
    main()