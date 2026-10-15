<div align="center">

# Early Semantic Grounding in Image Editing Models<br/>for Zero-Shot Referring Image Segmentation

**Jingxuan He · Xiyu Wang · Yunke Wang · Mengyu Zheng · Chang Xu***

**NeurIPS 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2605.13122-b31b1b.svg)](https://arxiv.org/abs/2605.13122)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://arxiv.org/abs/2605.13122)

</div>

Instruction-based image editing models must identify where to edit before performing the requested modification.
We reformulate referring expressions as editing instructions and extract segmentation masks from a pretrained image editing model at the very first denoising step.

- 🎯 **Training-free and zero-shot**: no fine-tuning, no extra data, no segmentation model.
- ⚡ **Single step**: one forward pass of the editing model, without generating an image.

## Installation

```bash
conda create -n esg python=3.10 -y
conda activate esg
pip install -r requirements.txt
```

## Models

Download the checkpoints into `models/` (or pass `--model_root` / `--qwen_path`):

| Model | Checkpoints |
|:--|:--|
| Step1X-Edit | [stepfun-ai/Step1X-Edit](https://huggingface.co/stepfun-ai/Step1X-Edit) and [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| FLUX.2-klein 9B | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| Instruction refinement (optional) | [stepfun-ai/Step1X-Edit-v1p2](https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2) |

```bash
hf download stepfun-ai/Step1X-Edit step1x-edit-v1p1-official.safetensors vae.safetensors --local-dir models/stepfun-ai/Step1X-Edit
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen/Qwen2.5-VL-7B-Instruct
hf download black-forest-labs/FLUX.2-klein-9B --local-dir models/black-forest-labs/FLUX.2-klein-9B
```

## Data

We use RefCOCO, RefCOCO+ and RefCOCOg (UMD split) in the LMDB format of
[CRIS](https://github.com/DerrickWang005/CRIS.pytorch/blob/master/tools/prepare_datasets.md). Follow their instructions
and place (or link) the data at `data/RIS`:

```
data/RIS
├── images/train2014
├── lmdb/{refcoco,refcoco+,refcocog_u}/{val,testA,testB,test}.lmdb
└── masks/{refcoco,refcoco+,refcocog_u}/*.png
```

The refined editing instructions used in the paper are in `data/instructions/{dataset}_{split}.json` (one list per
`seg_id`, in the order of the referring expressions). They were obtained by prefixing every expression with "Remove" and
rewriting it with the instruction rewriter of Step1X-Edit v1.2 (see [Instruction refinement](#instruction-refinement)).

## Evaluation

```bash
# Step1X-Edit (feature block l' = 40, 512 px)
python evaluate.py --model step1x --dataset refcoco --split val
# FLUX.2-klein 9B (feature block l' = 20, 1024 px)
python evaluate.py --model flux2 --dataset refcoco --split val
```

Results (oIoU, mIoU, Pr@X) are printed and written to `results/{model}/{dataset}_{split}/metrics.json`, together with
per-expression IoUs. Interrupted runs resume where they stopped.

<details>
<summary><b>Multiple GPUs</b></summary>

Run one process per `--shard_id` with the same `--num_shards`, then merge with `--summarize`:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model flux2 --dataset refcoco --split val --num_shards 2 --shard_id 0
CUDA_VISIBLE_DEVICES=1 python evaluate.py --model flux2 --dataset refcoco --split val --num_shards 2 --shard_id 1
python evaluate.py --model flux2 --dataset refcoco --split val --summarize
```

`scripts/eval_all.sh` does this for all eight splits:

```bash
bash scripts/eval_all.sh step1x 0,1,2,3
bash scripts/eval_all.sh flux2 0,1,2,3
```

</details>

## Demo

```bash
python demo.py --model step1x --image path/to/image.jpg --expression "the man on the left" --output outputs/demo
```

The editing instruction is "Remove {expression}"; add `--refine` to refine it.
The attention, the mask, and an overlay visualization are written to `--output`.

## Instruction refinement

```bash
python refine_instructions.py --dataset refcoco --split val --output_dir data/instructions
```

The rewriter uses stochastic sampling (temperature = 0.7, top-p = 0.9), so newly generated instructions may differ slightly.

## Acknowledgements

This code builds on [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit), [FLUX.2](https://github.com/black-forest-labs/flux2),
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [diffusers](https://github.com/huggingface/diffusers)
and [CRIS](https://github.com/DerrickWang005/CRIS.pytorch). We thank the authors for releasing their code and models.

## License

This project is released under the [MIT License](LICENSE). Code adapted from Step1X-Edit, FLUX, DiffSynth-Studio and
diffusers remains under the Apache License 2.0, as noted in the file headers. The model weights are
subject to their own licenses.

## Citation

If you find this work useful, please cite:

```bibtex
@article{he2026early,
  title={Early Semantic Grounding in Image Editing Models for Zero-Shot Referring Image Segmentation},
  author={He, Jingxuan and Wang, Xiyu and Wang, Yunke and Zheng, Mengyu and Xu, Chang},
  journal={arXiv preprint arXiv:2605.13122},
  year={2026}
}
```
