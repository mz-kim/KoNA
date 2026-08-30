import json
import argparse
import random
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Mix datasets and optionally sample GRPO set.")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_pure_examples', type=int, default=0, help='Number of single-query examples.')
    parser.add_argument('--num_selective_examples', type=int, default=500, help='Number of compound-query examples.')
    parser.add_argument('--contain_contrast', action='store_true', help='Include fully answerable contrast examples.')
    parser.add_argument('--num_contrast_examples', type=int, default=100)
    parser.add_argument('--is_grpo', action='store_true', help='Create a disjoint GRPO split.')
    parser.add_argument('--grpo_size', type=int, default=100)
    parser.add_argument('--grpo_answerable_examples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=807)
    parser.add_argument('--output_dir', type=str, default='dataset/train')
    return parser.parse_args()

def load_and_group_data(path):
    groups = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            group_key = (item['source'], item['model'], item['type'])
            groups[group_key].append(item)
    return groups

def format_item(item, set_type="selective"):
    new_item = {
        "image": item["image"],
        "type": item["type"],
        "source": item["source"],
        "model": item["model"],
        "set": set_type
    }
    
    if set_type == "pure":
        new_item["question_2"] = item["question_1"]
        new_item["answer_2"] = item["answer_1"]
    elif set_type == "contrast" or set_type == "grpo_contrast":
        new_item["question_2"] = item["contrast_question_2"]
        new_item["answer_2"] = item["contrast_answer_2"]
        if set_type == "grpo_contrast":
            new_item["type"] = "contrast"
    else:
        new_item["question_2"] = item["question_2"]
        new_item["answer_2"] = item["answer_2"]
        
    return new_item

def distribute_samples(total_target, num_groups):
    if total_target <= 0:
        return [0] * num_groups
    base_count = total_target // num_groups
    remainder = total_target % num_groups
    counts = [base_count] * num_groups
    for i in range(remainder):
        counts[i] += 1
    return counts

def save_as_internvl(dataset, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"InternVL format conversion: {output_path}")
    with output_path.open('w', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(dataset)):
            try:
                with Image.open(item['image']) as img:
                    width, height = img.size
            except (OSError, ValueError):
                width, height = 0, 0
            
            internvl_item = {
                "id": idx,
                "image": item['image'],
                "width": width,
                "height": height,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{item['question_2']}"
                    },
                    {
                        "from": "gpt",
                        "value": item['answer_2']
                    }
                ]
            }
            f.write(json.dumps(internvl_item, ensure_ascii=False) + '\n')

    meta_path = output_path.with_suffix('.json')
    metadata = {
        output_path.stem: {
            "root": "",
            "annotation": output_path.as_posix(),
            "data_augment": False,
            "max_dynamic_patch": 6,
            "repeat_time": 1,
            "length": len(dataset),
        }
    }
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f"InternVL metadata saved to: {meta_path}")

def main():
    args = parse_args()
    random.seed(args.seed)
    
    grouped_data = load_and_group_data(args.dataset_path)
    group_keys = list(grouped_data.keys())
    num_groups = len(group_keys)
    
    if num_groups == 0:
        print("No data found.")
        return

    selective_counts = distribute_samples(args.num_selective_examples, num_groups)
    pure_counts = distribute_samples(args.num_pure_examples, num_groups)
    contrast_counts = distribute_samples(args.num_contrast_examples, num_groups)
    grpo_total_counts = distribute_samples(args.grpo_size, num_groups) if args.is_grpo else [0] * num_groups
    
    final_dataset = []
    grpo_dataset = []
    
    type_to_indices = defaultdict(list)
    for idx, key in enumerate(group_keys):
        type_to_indices[key[2]].append(idx)
        
    grpo_std_counts = [0] * num_groups
    grpo_con_counts = [0] * num_groups

    if args.is_grpo:
        sm_to_indices = defaultdict(list)
        for idx, key in enumerate(group_keys):
            sm_to_indices[(key[0], key[1])].append(idx)
        
        total_grpo_con_target = args.grpo_answerable_examples
        if total_grpo_con_target > args.grpo_size:
            raise ValueError('--grpo_answerable_examples cannot exceed --grpo_size')
        
        sources = sorted(list(set(k[0] for k in sm_to_indices.keys())))
        source_quota = distribute_samples(total_grpo_con_target, len(sources))
        source_to_quota = dict(zip(sources, source_quota))

        for src in sources:
            src_total = source_to_quota[src]
            
            current_sm_keys = [k for k in sm_to_indices.keys() if k[0] == src]
            
            for sm_key in current_sm_keys:
                model_name = sm_key[1].lower()
                if 'gpt' in model_name:
                    sm_target = round(src_total * 0.5)
                else:
                    sm_target = src_total - round(src_total * 0.5)
                
                indices = sm_to_indices[sm_key]
                con_distribution = distribute_samples(sm_target, len(indices))
                
                for i, idx in enumerate(indices):
                    allowed_con = min(con_distribution[i], grpo_total_counts[idx])
                    grpo_con_counts[idx] = allowed_con
                    grpo_std_counts[idx] = grpo_total_counts[idx] - allowed_con

    print(f"Sampling starting (Seed: {args.seed}, Groups: {num_groups})")
    
    for i, group_key in enumerate(group_keys):
        items = grouped_data[group_key]
        random.shuffle(items)
        
        target_sel = selective_counts[i]
        target_pure = pure_counts[i]
        target_con = contrast_counts[i] if args.contain_contrast else 0

        target_grpo_std = grpo_std_counts[i]
        target_grpo_con = grpo_con_counts[i]

        current_idx = 0
        
        if target_sel > 0:
            subset = items[current_idx : current_idx + target_sel]
            for item in subset:
                final_dataset.append(format_item(item, set_type="selective"))
            current_idx += len(subset)

        if target_pure > 0:
            subset = items[current_idx : current_idx + target_pure]
            for item in subset:
                final_dataset.append(format_item(item, set_type="pure"))
            current_idx += len(subset)
            
        if target_con > 0:
            subset = items[current_idx : current_idx + target_con]
            for item in subset:
                final_dataset.append(format_item(item, set_type="contrast"))
            current_idx += len(subset)

        if args.is_grpo:
            if target_grpo_std > 0:
                subset = items[current_idx : current_idx + target_grpo_std]
                for item in subset:
                    grpo_dataset.append(format_item(item, set_type="grpo"))
                current_idx += len(subset)

            if target_grpo_con > 0:
                subset = items[current_idx : current_idx + target_grpo_con]
                for item in subset:
                    grpo_dataset.append(format_item(item, set_type="grpo_contrast"))
                current_idx += len(subset)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    
    pure_suffix = f"_pure{args.num_pure_examples}" if args.num_pure_examples > 0 else ""
    sel_suffix = f"_selective{args.num_selective_examples}"
    con_suffix = f"_contrast{args.num_contrast_examples}" if args.contain_contrast else ""
    
    save_filename = f"{base_name}{pure_suffix}{sel_suffix}{con_suffix}.jsonl"
    save_path = os.path.join(args.output_dir, save_filename)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f"Main dataset saved to: {save_path}")

    internvl_save_path = os.path.join(args.output_dir, "internVL", save_filename)
    save_as_internvl(final_dataset, internvl_save_path)

    if args.is_grpo:
        grpo_path = os.path.join(args.output_dir, "grpo.jsonl")
        with open(grpo_path, 'w', encoding='utf-8') as f:
            for entry in grpo_dataset:
                f.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')
        print(f"GRPO dataset saved to: {grpo_path} (Total: {len(grpo_dataset)})")

if __name__ == '__main__':
    main()
