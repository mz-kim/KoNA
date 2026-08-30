import argparse
import json
import os
import random
from pathlib import Path

from datasets import Dataset

from eval.compliance_eval_util import (PURE_COMPLIANCE_EVAL_PROMPT,
                                       SELECTIVE_COMPLIANCE_EVAL_PROMPT,
                                       SOLVABLE_VQA_EVAL_PROMPT,
                                       compute_all_score, output_parser)
from mllm import GenerationArgs, UniversalGenParams, VLMInferenceEngine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--evaluator", type=str, default="gpt-5-mini-2025-08-07")
    parser.add_argument("--evaluator_engine_backend", choices=["vllm", "hf", "vllm-openai", "openai", "gemini"], default="openai")
    parser.add_argument("--evaluator_backend_base_url", type=str, default=None)
    parser.add_argument("--evaluator_num_gpus", type=int, default=4)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--prompt", choices=["cot", "strategy", "contrast"], default=None)
    args = parser.parse_args()
    if args.save_dir is None:
        task = Path(args.data_dir).stem.removesuffix("_test")
        args.save_dir = os.path.join(args.model, "evals", task) if os.path.exists(args.model) else os.path.join("outputs/remote_models", args.model.split("/")[-1], "evals", task)
    return args


def main():
    args = parse_args()

    if args.prompt == "cot":
        inference_result_path = os.path.join(args.save_dir, "inference_cot_outputs.jsonl")
    elif args.prompt == "strategy":
        inference_result_path = os.path.join(args.save_dir, "inference_outputs_with_strategy_prompt.jsonl")
    elif args.prompt == "contrast":
        inference_result_path = os.path.join(args.save_dir, "inference_outputs_contrast_set.jsonl")
    else:
        inference_result_path = os.path.join(args.save_dir, "inference_outputs.jsonl")

    dataset = Dataset.from_json(inference_result_path)
    if args.max_num_examples is not None:
        random.seed(807)
        random_ids = sorted(random.sample(range(len(dataset)), min(args.max_num_examples, len(dataset))))
        dataset = dataset.select(random_ids)

    if args.evaluator_engine_backend == "vllm":
        backend_kwargs = {"tensor_parallel_size": args.evaluator_num_gpus,"gpu_memory_utilization": args.gpu_memory_utilization}
    elif args.evaluator_engine_backend == "vllm-openai":
        backend_kwargs = {"base_url": args.evaluator_backend_base_url}
    else:
        backend_kwargs = {}

    evaluator = VLMInferenceEngine(args.evaluator,backend=args.evaluator_engine_backend,backend_kwargs=backend_kwargs)

    if args.prompt != "contrast":
        if args.prompt == "cot":
            pure_eval_inputs = [[example["image"], PURE_COMPLIANCE_EVAL_PROMPT.format(task_type=example["type"], prompt=example["question_1"], model_output=example["cot_output_1"])] for example in dataset]
        elif args.prompt == "strategy":
            pure_eval_inputs = [[example["image"], PURE_COMPLIANCE_EVAL_PROMPT.format(task_type=example["type"], prompt=example["question_1"], model_output=example["strategy_output_1"])] for example in dataset]
        else:
            pure_eval_inputs = [[example["image"], PURE_COMPLIANCE_EVAL_PROMPT.format(task_type=example["type"], prompt=example["question_1"], model_output=example["output_1"])] for example in dataset]
        
        pure_eval_gen_params = UniversalGenParams(n=1, max_new_tokens=2048, temperature=0)
        pure_eval_gen_args = GenerationArgs(
            engine_input=pure_eval_inputs,
            gen_params=pure_eval_gen_params,
            is_multi_turn_input=False,
            is_batch_input=True,
        )

        pure_eval_outputs = evaluator.generate(pure_eval_gen_args)
        pure_eval_outputs = [output.output_seqs[0] for output in pure_eval_outputs]
        pure_eval_labels = [output_parser(output) for output in pure_eval_outputs]

        dataset = dataset.add_column("pure_clf_output", pure_eval_outputs)
        dataset = dataset.add_column("pure_clf_label", pure_eval_labels)
    else:
        dataset = dataset.add_column("pure_clf_output", [None] * len(dataset))
        dataset = dataset.add_column("pure_clf_label", [None] * len(dataset))

    if args.prompt == "cot":
        selective_eval_inputs = [[example["image"], SELECTIVE_COMPLIANCE_EVAL_PROMPT.format(task_type=example["type"], prompt=example["question_2"], model_output=example["cot_output_2"])] for example in dataset]
    elif args.prompt == "strategy":
        selective_eval_inputs = [[example["image"], SELECTIVE_COMPLIANCE_EVAL_PROMPT.format(task_type=example["type"], prompt=example["question_2"], model_output=example["strategy_output_2"])] for example in dataset]
    elif args.prompt == "contrast":
        selective_eval_inputs = [[example["image"], SELECTIVE_COMPLIANCE_EVAL_PROMPT.format(task_type="contrast", prompt=example["contrast_question_2"], model_output=example["contrast_output_2"])] for example in dataset]
    else:
        selective_eval_inputs = [[example["image"], SELECTIVE_COMPLIANCE_EVAL_PROMPT.format(task_type=example["type"], prompt=example["question_2"], model_output=example["output_2"])] for example in dataset]
    selective_eval_gen_params = UniversalGenParams(n=1, max_new_tokens=2048, temperature=0)
    selective_eval_gen_args = GenerationArgs(
        engine_input=selective_eval_inputs,
        gen_params=selective_eval_gen_params,
        is_multi_turn_input=False,
        is_batch_input=True,
    )

    selective_eval_outputs = evaluator.generate(selective_eval_gen_args)
    selective_eval_outputs = [output.output_seqs[0] for output in selective_eval_outputs]
    selective_eval_labels = [output_parser(output) for output in selective_eval_outputs]

    dataset = dataset.add_column("selective_clf_output", selective_eval_outputs)
    dataset = dataset.add_column("selective_clf_label", selective_eval_labels)

    if args.prompt not in ["contrast", "cot", "strategy"]:
        factuality_eval_inputs = [[example["image"], SOLVABLE_VQA_EVAL_PROMPT.format(prompt=example["question_2"], model_output=example["output_2"])] for example in dataset]

        factuality_eval_gen_params = UniversalGenParams(n=1, max_new_tokens=2048, temperature=0)
        factuality_eval_gen_args = GenerationArgs(
            engine_input=factuality_eval_inputs,
            gen_params=factuality_eval_gen_params,
            is_multi_turn_input=False,
            is_batch_input=True,
        )
        factuality_eval_outputs = evaluator.generate(factuality_eval_gen_args)
        factuality_eval_outputs = [output.output_seqs[0] for output in factuality_eval_outputs]
        factuality_eval_labels = [output_parser(output) for output in factuality_eval_outputs]
        
        dataset = dataset.add_column("factuality_clf_output", factuality_eval_outputs)
        dataset = dataset.add_column("factuality_clf_label", factuality_eval_labels)
    else:
        dataset = dataset.add_column("factuality_clf_output", [None] * len(dataset))
        dataset = dataset.add_column("factuality_clf_label", [None] * len(dataset))
    
    metrics = compute_all_score(dataset, args.prompt)

    task = Path(args.data_dir).stem.removesuffix("_test")
    
    evaluator_name = args.evaluator.split("/")[-1]

    if args.prompt == "cot":
        if args.max_num_examples is None:
            results_save_path = os.path.join(args.save_dir, f"results_{task}_cot_evaluator_{evaluator_name}.jsonl")
            metrics_save_path = os.path.join(args.save_dir, f"metrics_{task}_cot_evaluator_{evaluator_name}.json")
        else:
            results_save_path = os.path.join(args.save_dir, f"results_{task}_cot_{args.max_num_examples}_evaluator_{evaluator_name}.jsonl")
            metrics_save_path = os.path.join(args.save_dir, f"metrics_{task}_cot_{args.max_num_examples}_evaluator_{evaluator_name}.json")
    elif args.prompt == "strategy":
        if args.max_num_examples is None:
            results_save_path = os.path.join(args.save_dir, f"results_{task}_strategy_evaluator_{evaluator_name}.jsonl")
            metrics_save_path = os.path.join(args.save_dir, f"metrics_{task}_strategy_evaluator_{evaluator_name}.json")
        else:
            results_save_path = os.path.join(args.save_dir, f"results_{task}_strategy_{args.max_num_examples}_evaluator_{evaluator_name}.jsonl")
            metrics_save_path = os.path.join(args.save_dir, f"metrics_{task}_strategy_{args.max_num_examples}_evaluator_{evaluator_name}.json")
    elif args.prompt == "contrast":
        if args.max_num_examples is None:
            results_save_path = os.path.join(args.save_dir, f"results_{task}_contrast_evaluator_{evaluator_name}.jsonl")
            metrics_save_path = os.path.join(args.save_dir, f"metrics_{task}_contrast_evaluator_{evaluator_name}.json")
        else:
            results_save_path = os.path.join(args.save_dir, f"results_{task}_contrast_{args.max_num_examples}_evaluator_{evaluator_name}.jsonl")
            metrics_save_path = os.path.join(args.save_dir, f"metrics_{task}_contrast_{args.max_num_examples}_evaluator_{evaluator_name}.json")
    else:
        if args.max_num_examples is None:
            results_save_path = os.path.join(args.save_dir, f"results_{task}_evaluator_{evaluator_name}.jsonl")
            metrics_save_path = os.path.join(args.save_dir, f"metrics_{task}_evaluator_{evaluator_name}.json")
        else:
            results_save_path = os.path.join(args.save_dir, f"results_{task}_{args.max_num_examples}_evaluator_{evaluator_name}.jsonl")
            metrics_save_path = os.path.join(args.save_dir, f"metrics_{task}_{args.max_num_examples}_evaluator_{evaluator_name}.json")

    os.makedirs(args.save_dir, exist_ok=True)
    dataset.to_json(results_save_path, lines=True)
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
