# GaitMax

**Unlocking Motion from Large Vision Models with a Semantic and Kinematic Duality for Gait Recognition**

Zhanbo Huang, Dingqiang Ye, Xiaoming Liu, Yu Kong.

[![Paper](https://img.shields.io/badge/Paper-CVPR%202026-b31b1b.svg)](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Unlocking_Motion_from_Large_Vision_Models_with_a_Semantic_and_CVPR_2026_paper.pdf)
[![Project](https://img.shields.io/badge/Project-Page-1f72b8.svg)](https://actionlab-cv.github.io/GaitMax/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-GCaption-ffce00.svg)](https://huggingface.co/datasets/action-lab/gcaption)

Repository for **GaitMax** (CVPR 2026). See the [paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Unlocking_Motion_from_Large_Vision_Models_with_a_Semantic_and_CVPR_2026_paper.pdf) and [project page](https://actionlab-cv.github.io/GaitMax/).

## GCaption

GCaption is the natural-language attribute resource introduced in GaitMax. Every gait sequence is paired with seven human-interpretable attributes — age, attire, action, related item, location, viewpoint, lighting — and a precomputed OpenCLIP text embedding per attribute, supporting context-aware gait analysis and the paper's Conditional Decorrelation Loss (CDLoss).

🤗 **[action-lab/gcaption](https://huggingface.co/datasets/action-lab/gcaption)** — 103,124 sequences across [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp), [CCPG](https://github.com/BNU-IVC/CCPG), [CCGR](https://github.com/ShinanZou/CCGR), and [SUSTech1K](https://lidargait.github.io).

```python
from huggingface_hub import snapshot_download
snapshot_download("action-lab/gcaption", repo_type="dataset", local_dir="data/gcaption")
```

GCaption ships annotations only (no source frames). Obtain the source datasets from their original providers and join by sequence `id`.

## Gait Light

GaitMax is built on **Gait Light**, a PyTorch + [Lightning](https://lightning.ai) gait-recognition codebase designed for scalable, multi-modal experiments. Relative to the de-facto standard [OpenGait](https://github.com/ShiqiYu/OpenGait), it emphasizes:

- **Unified multi-modal data format.** A single sequence representation carries RGB frames, silhouette masks, poses, and body-part maps together (`SequenceData` / `SequenceBatch`), with modalities toggled per experiment from config. Datasets share one on-disk layout, so they can be pooled and **mixed-trained** without bespoke per-dataset loaders.
- **GPU-optimized losses.** The triplet and other objectives are written as fully vectorized tensor ops — part-wise pairwise distances via batched `bmm`, boolean-mask triplet mining, no per-sample Python loops — so the full loss runs on-GPU.
- **Multi-node DDP out of the box.** Built on Lightning: multi-node / multi-GPU DDP, mixed precision (`16-mixed`), and synchronized BatchNorm are single config flags (`num_nodes`, `devices`, `precision`, `sync_batchnorm`).

## Citation

```bibtex
@InProceedings{Huang_2026_CVPR,
    author    = {Huang, Zhanbo and Ye, Dingqiang and Liu, Xiaoming and Kong, Yu},
    title     = {Unlocking Motion from Large Vision Models with a Semantic and Kinematic Duality for Gait Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {28379-28390}
}
```

Please also cite the source gait datasets (CASIA-B, CCPG, CCGR, SUSTech1K).

## Acknowledgments

We thank the maintainers of CASIA-B, CCPG, CCGR, and SUSTech1K for releasing their datasets to the research community.
