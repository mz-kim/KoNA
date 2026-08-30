import argparse
import os
import random
from pathlib import Path

from datasets import Dataset

from eval.compliance_eval_util import STRATEGY_PROMPT
from mllm import GenerationArgs, UniversalGenParams, VLMInferenceEngine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_engine_backend", choices=["vllm", "hf", "vllm-openai", "openai", "gemini"], default="vllm")
    parser.add_argument("--model_backend_base_url", type=str, default=None)
    parser.add_argument("--model_num_gpus", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--prompt", choices=["cot", "strategy", "contrast"], default=None)
    parser.add_argument("--is_batch", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.save_dir is None:
        task = Path(args.data_dir).stem.removesuffix("_test")
        args.save_dir = os.path.join(args.model, "evals", task) if os.path.exists(args.model) else os.path.join("outputs/remote_models", args.model.split("/")[-1], "evals", task)
    return args

def main():
    args = parse_args()

    if args.model_engine_backend == "vllm":
        backend_kwargs = {
            "tensor_parallel_size": args.model_num_gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": 15000,
            "max_num_seqs":64
        }
        args.is_batch = True
    elif args.model_engine_backend == "vllm-openai":
        backend_kwargs = {"base_url": args.model_backend_base_url, "max_model_len": 15000, "max_num_seqs":64}
    else:
        backend_kwargs = {}
    
    model = VLMInferenceEngine(args.model, backend=args.model_engine_backend, backend_kwargs=backend_kwargs)
    dataset = Dataset.from_json(args.data_dir)

    if args.max_num_examples is not None:
        random.seed(0)
        random_ids = sorted(random.sample(range(len(dataset)), min(args.max_num_examples, len(dataset))))
        dataset = dataset.select(random_ids)

    prediction_gen_params = UniversalGenParams(n=1, max_new_tokens=2048, temperature=0)

    instructions_1 = dataset["question_1"]
    images = dataset["image"]

    if args.prompt != "contrast":
        instructions_1 = dataset["question_1"]
        if args.prompt == "cot":
            instructions_1 = [q + " Let’s think step by step." for q in instructions_1]
        if args.prompt == "strategy":
            instructions_1 = [STRATEGY_PROMPT.format(question=inst) for inst in instructions_1]

        inputs_1 = [[image, instruction] for image, instruction in zip(images, instructions_1)]

        gen_args_1 = GenerationArgs(
                engine_input=inputs_1,
                gen_params=prediction_gen_params,
                is_multi_turn_input=False,
                is_batch_input=args.is_batch
        )
        model_outputs_1 = model.generate(gen_args_1)
        model_outputs_1 = [output.output_seqs[0] for output in model_outputs_1]

        if args.prompt == "cot":
            dataset = dataset.add_column(name="cot_output_1", column=model_outputs_1)
        elif args.prompt == "strategy":
            dataset = dataset.add_column(name="strategy_output_1", column=model_outputs_1)
        else:
            dataset = dataset.add_column(name="output_1", column=model_outputs_1)

    instructions_2 = dataset["question_2"]
    if args.prompt == "cot":
        instructions_2 = [q + " Let’s think step by step." for q in instructions_2]
    elif args.prompt == "strategy":
        instructions_2 = [STRATEGY_PROMPT.format(question=inst) for inst in instructions_2]
    elif args.prompt == "contrast":
        instructions_2 = dataset["contrast_question_2"]

    inputs_2 = [[image, instruction] for image, instruction in zip(images, instructions_2)]

    gen_args_2 = GenerationArgs(
            engine_input=inputs_2,
            gen_params=prediction_gen_params,
            is_multi_turn_input=False,
            is_batch_input=args.is_batch
    )
    model_outputs_2 = model.generate(gen_args_2)
    model_outputs_2 = [output.output_seqs[0] for output in model_outputs_2]


    if args.prompt == "cot":
        dataset = dataset.add_column(name="cot_output_2", column=model_outputs_2)
    elif args.prompt == "strategy":
        dataset = dataset.add_column(name="strategy_output_2", column=model_outputs_2)
    elif args.prompt == "contrast":
        dataset = dataset.add_column(name="contrast_output_2", column=model_outputs_2)
    else:
        dataset = dataset.add_column(name="output_2", column=model_outputs_2)

    os.makedirs(args.save_dir, exist_ok=True)
    suffix = args.prompt if args.prompt else "default"
    filename_map = {
        "cot": "inference_cot_outputs.jsonl",
        "strategy": "inference_outputs_with_strategy_prompt.jsonl",
        "contrast": "inference_outputs_contrast_set.jsonl",
        "default": "inference_outputs.jsonl"
    }
    results_save_path = os.path.join(args.save_dir, filename_map.get(suffix, "inference_outputs.jsonl"))
    dataset.to_json(results_save_path, lines=True)

if __name__ == "__main__":
    main()
