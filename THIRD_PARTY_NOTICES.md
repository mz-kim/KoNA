# Third-party resources

## External training implementations

The SFT and GRPO experiments were run with the unmodified official implementations from:

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [InternVL](https://github.com/OpenGVLab/InternVL)

Their trainer source code is not copied or redistributed in this repository. The files under `training_configs/` record the KoNA experiment settings, while `training_configs/reward.py` contains the project-specific GRPO reward definition. Users should obtain the model training implementations directly from the upstream repositories and follow their current license and installation requirements.

At the time of this release, the Qwen2.5-VL repository identifies its source
code as Apache-2.0, while the InternVL repository identifies its source code as
MIT-licensed. Model weights may have separate or model-specific terms; users
must review the license attached to the exact checkpoint they use.

## Source images

KoNA annotations refer to images from:

- [COCO](https://cocodataset.org/#download). COCO images retain their individual source licenses; consult the COCO terms of use and the license information associated with each image.
- [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html). Open Images lists its images as CC BY 2.0 and notes that users should verify the license status of individual images.

The KoNA authors do not claim ownership of these source images. Source-dataset
images are not redistributed as standalone dataset files in this repository;
the paper-derived overview figures under `assets/` contain illustrative source
images. The repository's Apache License 2.0 covers the original KoNA code,
documentation, configurations, annotations, and figure composition only; the
original image licenses and attribution requirements remain in effect.

## Paper

The EMNLP 2026 paper is a separate publication and is not licensed under the
repository's Apache License 2.0. Consult the publication venue for the license that
applies to the paper.
