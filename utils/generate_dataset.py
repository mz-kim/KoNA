import argparse
import json
import random
import re
from datasets import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dataset.dataset_generate_util import PROMPT_DICT
from mllm import GenerationArgs, UniversalGenParams, VLMInferenceEngine


DEFAULT_IMAGE_ROOTS: Dict[str, Dict[str, Path]] = {
    "coco": {
        "train": Path("dataset/images/coco/train2017"),
        "val": Path("dataset/images/coco/val2017"),
        "test": Path("dataset/images/coco/test2017"),
    },
    "openimages": {
        "train": Path("dataset/images/openimages/train/data"),
        "val": Path("dataset/images/openimages/validation/data"),
        "test": Path("dataset/images/openimages/test/data"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type",required=True,choices=list(PROMPT_DICT.keys()),help="Target task to generate (keys from PROMPT_DICT).",)
    parser.add_argument("--split",default="train",choices=["train", "val", "test"],help="Dataset split to sample images from.",)
    parser.add_argument("--image_source",required=True,choices=["coco", "openimages"],help="Image source to draw from.",)
    parser.add_argument("--model",required=True,help="Model identifier for VLMInferenceEngine.",)
    parser.add_argument("--model_engine_backend",default="vllm-openai",choices=["vllm", "vllm-openai", "openai", "gemini"],help="Backend to use for inference.",)
    parser.add_argument("--model_backend_base_url",type=str,default=None,help="Base URL for vllm-openai backend.",)
    parser.add_argument("--model_num_gpus",type=int,default=1,help="Tensor parallel size for vllm backend.",)
    parser.add_argument("--gpu_memory_utilization",type=float,default=0.9,help="GPU memory utilization for vllm backend.")
    parser.add_argument("--temperature",type=float,default=0.0,help="Generation temperature.")
    parser.add_argument("--max_new_tokens",type=int,default=2048,help="Maximum number of tokens to generate.")
    parser.add_argument("--num_samples", type=int, default=150, help="Number of images to sample for generation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for image sampling.")
    parser.add_argument("--image_dir",type=str,default=None,help="Override directory to read images from (defaults to source/split mapping).",)
    parser.add_argument("--output_dir",type=str,default="dataset/gemini",help="Directory to save generated JSONL files.",)
    parser.add_argument("--image_list_path",type=str,default=None,help="Optional JSON/JSONL manifest to pick image paths or file names from.",)
    return parser.parse_args()


def resolve_image_dir(image_source: str, split: str, override: Optional[str] = None) -> Path:
    if override:
        path = Path(override)
    else:
        source_map = DEFAULT_IMAGE_ROOTS.get(image_source)
        if source_map is None or split not in source_map:
            raise ValueError(f"No default image directory for source={image_source}, split={split}")
        path = source_map[split]
    if not path.exists():
        raise FileNotFoundError(f"Image directory does not exist: {path}")
    return path


def load_image_paths(
    image_source: str,
    split: str,
    seed: int,
    num_samples: int,
    image_dir_override: Optional[str] = None,
    image_list_path: Optional[str] = None,
) -> List[str]:
    rng = random.Random(seed)

    if image_list_path:
        dataset = Dataset.from_json(image_list_path)
        if "image" in dataset.column_names:
            all_paths = [str(p) for p in dataset["image"]]
        elif "file_name" in dataset.column_names:
            base_dir = resolve_image_dir(image_source, split, image_dir_override)
            all_paths = [str(base_dir / fn) for fn in dataset["file_name"]]
        else:
            raise ValueError("image_list_path must contain 'image' or 'file_name' column")
    else:
        base_dir = resolve_image_dir(image_source, split, image_dir_override)
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        all_paths = []
        for pattern in patterns:
            all_paths.extend(base_dir.glob(pattern))
        all_paths = [str(p) for p in sorted(all_paths)]

    if not all_paths:
        raise ValueError("No images found for the given source/split configuration.")

    if num_samples and num_samples < len(all_paths):
        all_paths = rng.sample(all_paths, num_samples)
    return all_paths


def build_backend_kwargs(args: argparse.Namespace) -> Dict:
    if args.model_engine_backend == "vllm":
        return {"tensor_parallel_size": args.model_num_gpus,"gpu_memory_utilization": args.gpu_memory_utilization}
    if args.model_engine_backend == "vllm-openai":
        return {"base_url": args.model_backend_base_url}
    return {}


def qa_from_text(text: str) -> Optional[Tuple[str, str]]:
    if not isinstance(text, str):
        return None
    
    cleaned = text.strip()
    if not cleaned or cleaned.lower().startswith("none"):
        return None

    patterns = [
        r'\{Question:\s*\[([^\]]+)\],\s*Answer:\s*\[([^\]]+)\]\}',
        r'\{Question:\s*([^,]+),\s*Answer:\s*([^}]+)\}',
        r'\{Question:\s*([^}]*?)\s+Answer:\s*([^}]+)\}',
        r'Question\s*[:\-]\s*(.+?)\s*Answer\s*[:\-]\s*(.+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            for trim_char in ("{", "}", "[", "]", '"', "'"):
                question = question.strip(trim_char)
                answer = answer.strip(trim_char) 
            if question and answer:
                return question, answer
                
    return None


def generate_outputs(
    engine: VLMInferenceEngine,
    inputs: Sequence[Sequence],
    gen_params: UniversalGenParams,
) -> List[str]:
    gen_args = GenerationArgs(
        engine_input=list(inputs),
        gen_params=gen_params,
        is_multi_turn_input=False,
        is_batch_input=True,
    )
    model_outputs = engine.generate(gen_args)
    texts = []
    for output in model_outputs:
        seqs = getattr(output, "output_seqs", None) or [""]
        texts.append(seqs[0] if seqs else "")
    return texts


def run_first_stage(
    engine: VLMInferenceEngine,
    image_paths: List[str],
    prompt: str,
    args: argparse.Namespace,
) -> List[Dict]:
    gen_params = UniversalGenParams(
        n=1,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    inputs = [[img, prompt] for img in image_paths]
    raw_outputs = generate_outputs(engine, inputs, gen_params)

    records = []
    for image_path, raw in zip(image_paths, raw_outputs):
        parsed = qa_from_text(raw)
        if parsed is None:
            continue
        question, answer = parsed
        records.append(
            {
                "image": image_path,
                "type": args.task_type,
                "question": question,
                "answer": answer,
                "source": args.image_source,
                "split": args.split,
            }
        )
    return records


def run_second_stage(
    engine: VLMInferenceEngine,
    stage_one_records: List[Dict],
    prompt_template: str,
    args: argparse.Namespace,
) -> List[Dict]:
    if not stage_one_records:
        return []

    gen_params = UniversalGenParams(
        n=1,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    inputs = []
    for rec in stage_one_records:
        prev = f"Question: {rec['question']}\nAnswer: {rec['answer']}"
        prompt = prompt_template.replace("{prev}", prev)
        inputs.append([rec["image"], prompt])

    raw_outputs = generate_outputs(engine, inputs, gen_params)

    records = []
    for rec, raw in zip(stage_one_records, raw_outputs):
        parsed = qa_from_text(raw)
        if parsed is None:
            continue
        question_2, answer_2 = parsed
        records.append(
            {
                "image": rec["image"],
                "type": rec["type"],
                "question_1": rec["question"],
                "answer_1": rec["answer"],
                "question_2": question_2,
                "answer_2": answer_2,
                "source": rec.get("source"),
                "split": rec.get("split"),
            }
        )
    return records


def save_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    if args.task_type not in PROMPT_DICT:
        raise ValueError(f"Unknown task_type '{args.task_type}'. Available: {list(PROMPT_DICT.keys())}")

    backend_kwargs = build_backend_kwargs(args)
    engine = VLMInferenceEngine(args.model, backend=args.model_engine_backend, backend_kwargs=backend_kwargs)

    image_paths = load_image_paths(
        image_source=args.image_source,
        split=args.split,
        seed=args.seed,
        num_samples=args.num_samples,
        image_dir_override=args.image_dir,
        image_list_path=args.image_list_path,
    )

    stage_prompts = PROMPT_DICT[args.task_type]
    if len(stage_prompts) < 2:
        raise ValueError(f"PROMPT_DICT entry for '{args.task_type}' must contain two stages.")
    stage_one_prompt, stage_two_prompt = stage_prompts[0], stage_prompts[1]

    stage_one_records = run_first_stage(engine, image_paths, stage_one_prompt, args)
    stage_two_records = run_second_stage(engine, stage_one_records, stage_two_prompt, args)

    base_name = f"{args.task_type}_{args.split}_{args.image_source}"
    output_dir = Path(args.output_dir) / args.task_type / args.image_source / args.split
    stage_one_path = output_dir / f"first_{base_name}.jsonl"
    stage_two_path = output_dir / f"second_{base_name}.jsonl"

    save_jsonl(stage_one_path, stage_one_records)
    save_jsonl(stage_two_path, stage_two_records)

    print(f"[INFO] Saved first-stage outputs to {stage_one_path}")
    print(f"[INFO] Saved second-stage outputs to {stage_two_path}")


if __name__ == "__main__":
    main()
