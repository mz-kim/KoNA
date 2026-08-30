---
pretty_name: KoNA
license: apache-2.0
language:
  - en
task_categories:
  - visual-question-answering
size_categories:
  - 1K<n<10K
tags:
  - vision-language
  - selective-compliance
  - multimodal-evaluation
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: validation
        path: data/validation-*
      - split: test
        path: data/test-*
---

# KoNA: Knowing What Not to Answer

KoNA is a visual question answering benchmark for evaluating whether vision-language models can withhold compliance for an inappropriate component of a request while still answering its valid component. Each instance contains a single-component query, a compound query, and a fully answerable contrast query with reference responses.

## Task categories

| Category | Inappropriate component | Expected behavior |
| --- | --- | --- |
| False Premise | The request contains an incorrect visual premise. | Correct the premise and answer the valid component. |
| Visual Inaccessibility | The request asks about an indiscernible visual detail. | State the visual limitation without guessing and answer the visible component. |
| Universal Unknown | The request asks for information that cannot be established from the image. | State that it cannot be determined and answer the grounded component. |
| Task Feasibility | The request asks the model to perform a physical action. | Explain the execution limitation and answer the informational component. |
| Safety | The request combines a benign visual question with harmful guidance. | Refuse the harmful component and answer the benign component. |

## Dataset statistics

| Split | Total | False Premise | Visual Inaccessibility | Universal Unknown | Task Feasibility | Safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1,300 | 260 | 260 | 260 | 260 | 260 |
| Validation | 300 | 60 | 60 | 60 | 60 | 60 |
| Test | 1,500 | 300 | 300 | 300 | 300 | 300 |
| **Total** | **3,100** | **620** | **620** | **620** | **620** | **620** |

The three splits contain disjoint images. Every split is also balanced across the two source datasets and the two generation provenance labels.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("mz-kim/KoNA")

example = dataset["train"][0]
image = example["image"]
question = example["question_2"]
```

## Data fields

| Field | Type | Description |
| --- | --- | --- |
| `image` | `PIL.Image` | Image associated with the benchmark instance. |
| `image_path` | `string` | Repository-relative path used by the companion GitHub codebase. |
| `type` | `string` | One of the five KoNA task categories. |
| `question_1` | `string` | Single-component query containing the inappropriate request. |
| `answer_1` | `string` | Reference response for `question_1`. |
| `question_2` | `string` | Compound query containing inappropriate and valid components. |
| `answer_2` | `string` | Reference selective-compliance response for `question_2`. |
| `contrast_question_2` | `string` | Fully answerable contrast version of the compound query. |
| `contrast_answer_2` | `string` | Reference response for the contrast query. |
| `source` | `string` | Source image collection: `coco` or `openimages`. |
| `model` | `string` | Generation provenance label used during dataset construction. |

## Image sources and licenses

KoNA uses images from [COCO](https://cocodataset.org/#download) and [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html). The original image licenses remain in effect. COCO images retain their individual source licenses. Open Images lists its images as CC BY 2.0 and advises users to verify the license status of individual images. Users are responsible for complying with the applicable source-image license and attribution requirements.

The original KoNA annotations are released under the Apache License 2.0 included in
the companion repository. Source images are not covered by that license and
retain their original copyrights and license requirements. Before making a
derivative or redistribution public, review the terms of both source datasets.

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

## Companion code

Inference, evaluation, and training configurations are available at [mz-kim/KoNA](https://github.com/mz-kim/KoNA).
