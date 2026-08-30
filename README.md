<div align="center">

<h1>KoNA</h1>
<h3>Knowing What Not to Answer: Selective Non-Compliance in Vision-Language Models</h3>

<p><strong>EMNLP 2026 Main Conference</strong></p>
<p>Minji Kim · Jihyoung Jang · Hyounghun Kim</p>
<p>
  <a href="https://huggingface.co/datasets/mz-kim/KoNA">🤗 Dataset</a> ·
  <a href="https://arxiv.org/abs/2609.04720">📄 Paper</a>
</p>
<p>
  <a href="https://arxiv.org/abs/2609.04720"><img src="https://img.shields.io/badge/arXiv-2609.04720-b31b1b.svg" alt="arXiv:2609.04720"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
</p>

<img src="assets/kona_overview.png" alt="KoNA contrasts single-query and compound-query selective non-compliance" width="100%">

</div>

## Overview

A multimodal request can contain both a valid question and a component that a model cannot or should not answer. KoNA studies **selective non-compliance**: whether a vision-language model can withhold only the inappropriate component while remaining helpful on the valid one.

The benchmark covers False Premise, Visual Inaccessibility, Universal Unknown, Task Feasibility, and Safety. Each instance pairs single and compound queries with an answerable contrast query, making it possible to distinguish targeted non-compliance from indiscriminate refusal.

<p align="center">
  <img src="assets/kona_tasks.png" alt="Examples of KoNA's five selective non-compliance tasks" width="100%">
</p>

## Dataset

The released annotations are stored in `dataset/annotations/`. A packaged version with directly decodable image features and a detailed dataset card is hosted on Hugging Face:

> 🤗 **[mz-kim/KoNA](https://huggingface.co/datasets/mz-kim/KoNA)**

```python
from datasets import load_dataset

dataset = load_dataset("mz-kim/KoNA")
```

| Split | Instances |
| --- | ---: |
| Train | 1,300 |
| Validation | 300 |
| Test | 1,500 |

Each JSONL record contains `image`, `type`, `question_1`, `answer_1`, `question_2`, `answer_2`, `contrast_question_2`, `contrast_answer_2`, `source`, and `model`.

KoNA uses images from [COCO](https://cocodataset.org/#download) and [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html). Source images are not distributed here as dataset files, although the paper-derived overview figures above contain illustrative examples. Download the corresponding source splits and arrange them as follows:

```text
dataset/images/
|-- coco/
|   |-- train2017/
|   |-- val2017/
|   `-- test2017/
`-- openimages/
    |-- train/data/
    |-- validation/data/
    `-- test/data/
```

All annotation paths are repository-relative, so commands should be run from the repository root.

## Setup

The released dataset and evaluation utilities target Python 3.11 and CUDA-capable GPUs.

```bash
conda create -n kona python=3.11
conda activate kona
pip install -r requirements.txt
```

Install a PyTorch/vLLM build compatible with your CUDA driver if the versions in `requirements.txt` do not match your system. The SFT and GRPO environments are not bundled here; follow the installation instructions in the official Qwen2.5-VL or InternVL repository. API-backed generation and evaluation use environment variables; copy `.env.example` only as a template and never commit populated credentials.

```bash
export OPENAI_API_KEY="YOUR_KEY"
export GEMINI_API_KEY="YOUR_KEY"
```

## Dataset construction

Generation and automatic filtering prompts are provided in `dataset/dataset_generate_util.py`. A generation example is:

```bash
python -m utils.generate_dataset \
  --task_type false_premise \
  --split train \
  --image_source coco \
  --model gemini-2.5-flash \
  --model_engine_backend gemini \
  --num_samples 150 \
  --seed 0
```

Filter generated compound-query examples with:

```bash
python -m utils.filter_dataset \
  --task_type false_premise \
  --model gpt-5-mini-2025-08-07 \
  --model_engine_backend openai \
  --generation_model gemini
```

## Training

### Training-data preparation

Create the disjoint SFT and GRPO subsets used in the paper:

```bash
./scripts/prepare_data.sh
```

This produces 1,200 SFT examples (1,000 compound, 100 single, and 100 fully answerable) and 100 GRPO examples (80 compound and 20 fully answerable) under `dataset/train/`. It also creates the InternVL training and validation annotation/metadata files.

### Supervised fine-tuning

The Qwen and InternVL SFT experiments use the model authors' official training implementations without modifying or redistributing their source code:

- Qwen2.5-VL-3B-Instruct: [Qwen2.5-VL official repository](https://github.com/QwenLM/Qwen2.5-VL) with [qwen_sft.yaml](training_configs/qwen_sft.yaml)
- InternVL3-2B-Instruct: [InternVL official repository](https://github.com/OpenGVLab/InternVL) with [internvl_sft.yaml](training_configs/internvl_sft.yaml)

### GRPO training

The GRPO experiments likewise use the unmodified official Qwen2.5-VL and InternVL training code. The upstream trainer implementations are not redistributed; this repository includes only the experiment settings and the KoNA-specific reward function:

- Qwen: [qwen_grpo.yaml](training_configs/qwen_grpo.yaml)
- InternVL: [internvl_grpo.yaml](training_configs/internvl_grpo.yaml)
- Reward: [training_configs/reward.py](training_configs/reward.py), exposed as `llm_judge_reward_function`

The reward function follows the trainer-style `reward_func(completions, **kwargs)` interface and consumes the `image`, `question_2`, and `type` columns produced by `scripts/prepare_data.sh`.

## Evaluation

Run greedy inference followed by GPT-5-mini evaluation:

```bash
./scripts/evaluate.sh \
  0 \
  Qwen/Qwen2.5-VL-3B-Instruct \
  vllm \
  dataset/annotations/test.jsonl \
  outputs/eval/qwen-base
```

Supported inference backends are `vllm`, `hf`, `vllm-openai`, `openai`, and `gemini`. For an OpenAI-compatible vLLM server, set `MODEL_BASE_URL`, for example `http://localhost:8000/v1`.

The evaluation reports query-level non-compliance accuracy, component-level non-compliance accuracy, and factual accuracy. Add `cot`, `strategy`, or `contrast` as the optional sixth argument to evaluate a prompting variant or the answerable contrast set.

## License

The original KoNA code, documentation, configuration files, and annotations in
this repository are released under the [Apache License 2.0](LICENSE). The license does
not cover third-party assets or services. In particular, source images from
COCO and Open Images, external model code and weights, and API services retain
their respective licenses and terms; see
[NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The paper itself is a separate work and is not covered by this repository's
software and annotation license.

## Citation

```bibtex
@misc{kim2026knowinganswerselectivenoncompliance,
  title         = {Knowing What Not to Answer: Selective Non-Compliance in Vision-Language Models},
  author        = {Minji Kim and Jihyoung Jang and Hyounghun Kim},
  year          = {2026},
  eprint        = {2609.04720},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.04720}
}
```

## Acknowledgements

SFT and GRPO use the official [InternVL](https://github.com/OpenGVLab/InternVL) and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) implementations. Their trainer source code is not redistributed; `training_configs/reward.py` is the KoNA-specific reward definition used with those implementations. See `THIRD_PARTY_NOTICES.md` for details.
