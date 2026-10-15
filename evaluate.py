"""Zero-shot referring image segmentation on RefCOCO/+/g with an instruction-based image editing model.

Single GPU:
    python evaluate.py --model step1x --dataset refcoco --split val
Multiple GPUs (one process per shard), then merge:
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --model flux2 --dataset refcoco --split val --num_shards 2 --shard_id 0
    CUDA_VISIBLE_DEVICES=1 python evaluate.py --model flux2 --dataset refcoco --split val --num_shards 2 --shard_id 1
    python evaluate.py --model flux2 --dataset refcoco --split val --summarize
"""

import argparse
import gc
import glob
import json
import os
import warnings

import torch
from PIL import Image
from tqdm import tqdm

from esg.data import SPLITS, RISDataset, load_instructions
from esg.segment import IoUMeter, feature_based_localization

warnings.filterwarnings("ignore")

DEFAULT_FEAT_BLOCK = {"step1x": 40, "flux2": 20}


def build_segmenter(args, **kwargs):
    if args.model == "step1x":
        from esg.step1x.segmenter import Step1XSegmenter

        return Step1XSegmenter(
            model_root=args.model_root or "models/stepfun-ai/Step1X-Edit",
            qwen_path=args.qwen_path,
            feat_block=args.feat_block,
            seed=args.seed,
            **kwargs,
        )
    if args.model == "flux2":
        from esg.flux2.segmenter import Flux2Segmenter

        return Flux2Segmenter(
            model_root=args.model_root or "models/black-forest-labs/FLUX.2-klein-9B",
            feat_block=args.feat_block,
            seed=args.seed,
        )
    raise ValueError(args.model)


def summarize(result_dir, split_size=None):
    records = {}
    for path in sorted(glob.glob(os.path.join(result_dir, "shard*.jsonl"))):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                records[(r["seg_id"], r["sent_id"])] = r
    meter = IoUMeter()
    for r in records.values():
        meter.add(r["inter"], r["union"])
    res = meter.summary()
    res["num_images"] = len({k[0] for k in records})
    if split_size is not None and res["num_images"] != split_size:
        print(f"WARNING: only {res['num_images']}/{split_size} images evaluated")
    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(
        f"{result_dir}: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items())
    )
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["step1x", "flux2"], required=True)
    parser.add_argument("--dataset", choices=list(SPLITS), default="refcoco")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data_root", default="data/RIS")
    parser.add_argument("--instruction_dir", default="data/instructions")
    parser.add_argument("--model_root", default=None, help="Step1X-Edit or FLUX.2-klein-9B checkpoint directory")
    parser.add_argument("--qwen_path", default="models/Qwen/Qwen2.5-VL-7B-Instruct", help="Step1X-Edit text encoder")
    parser.add_argument("--feat_block", type=int, default=None, help="block l' of the FSL features")
    parser.add_argument("--thr", type=float, default=0.8, help="threshold tau on the attention prior")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--summarize", action="store_true", help="only merge the shard results and print metrics")
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Step1X-Edit on GPUs with < 48 GB: encode all instructions first, then free the text encoder and run the DiT",
    )
    args = parser.parse_args()
    if args.feat_block is None:
        args.feat_block = DEFAULT_FEAT_BLOCK[args.model]

    result_dir = os.path.join(args.output_dir, args.model, f"{args.dataset}_{args.split}")
    os.makedirs(result_dir, exist_ok=True)
    if args.summarize:
        summarize(result_dir, SPLITS[args.dataset][args.split])
        return

    dataset = RISDataset(args.data_root, args.dataset, args.split)
    instructions = load_instructions(args.instruction_dir, args.dataset, args.split)

    # Resume: skip expressions already evaluated by any earlier run of this split.
    done = set()
    for path in glob.glob(os.path.join(result_dir, "shard*.jsonl")):
        with open(path) as f:
            done |= {(r["seg_id"], r["sent_id"]) for r in map(json.loads, f)}
    out_path = os.path.join(result_dir, f"shard{args.shard_id}of{args.num_shards}.jsonl")

    def pending():
        desc = f"{args.model} {args.dataset} {args.split} [{args.shard_id}/{args.num_shards}]"
        for idx in tqdm(range(args.shard_id, len(dataset), args.num_shards), desc=desc):
            data = dataset[idx]
            seg_id = int(data["seg_id"])
            sents = instructions[str(seg_id)]
            assert len(sents) == len(data["sents"]), f"{seg_id}: instruction count mismatch"
            todo = [(sent_id, sent) for sent_id, sent in enumerate(sents) if (seg_id, sent_id) not in done]
            if todo:
                yield data, seg_id, todo

    texts = None
    if args.low_memory:
        assert args.model == "step1x", "--low_memory is only needed for Step1X-Edit"
        encoder = build_segmenter(args, load_dit=False)
        texts = {}
        for data, seg_id, todo in pending():
            image = Image.fromarray(data["img"])
            for sent_id, sent in todo:
                text = encoder.encode_text(image, sent)
                texts[(seg_id, sent_id)] = {"txt": text["txt"].cpu(), "token_range": text["token_range"]}
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        segmenter = build_segmenter(args, load_text_encoder=False)
    else:
        segmenter = build_segmenter(args)

    with open(out_path, "a") as f:
        for data, seg_id, todo in pending():
            gt = data["mask"] > 0
            image = Image.fromarray(data["img"])
            for sent_id, sent in todo:
                if texts is None:
                    prior, feat = segmenter(image, sent)
                else:
                    prior, feat = segmenter.segment(image, texts.pop((seg_id, sent_id)))
                pred = feature_based_localization(prior, feat, size=gt.shape, thr=args.thr).cpu().numpy()
                inter, union = int((pred & gt).sum()), int((pred | gt).sum())
                rec = {"seg_id": seg_id, "sent_id": sent_id, "inter": inter, "union": union, "iou": inter / union}
                f.write(json.dumps(rec) + "\n")
            f.flush()

    if args.num_shards == 1:
        summarize(result_dir, SPLITS[args.dataset][args.split])


if __name__ == "__main__":
    main()
