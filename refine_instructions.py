"""Turn referring expressions into refined editing instructions: "Remove <expression>" -> Step1X-Edit v1.2 rewriter.

The refined instructions used in the paper are released in data/instructions. Since the rewriter samples
(temperature 0.7, top-p 0.9), re-running this script gives slightly different instructions.

    python refine_instructions.py --dataset refcoco --split val --output_dir data/instructions_new
"""

import argparse
import json
import os
import warnings

import torch
from PIL import Image
from tqdm import tqdm

from esg.data import SPLITS, RISDataset
from esg.refine import InstructionRefiner

warnings.filterwarnings("ignore")


def merge(output_dir, dataset, split, num_shards):
    name = f"{dataset}_{split}"
    merged = {}
    for shard_id in range(num_shards):
        with open(os.path.join(output_dir, f"{name}.shard{shard_id}of{num_shards}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                merged[str(r["seg_id"])] = r["sents"]
    assert len(merged) == SPLITS[dataset][split], f"{len(merged)} images refined"
    with open(os.path.join(output_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(SPLITS), default="refcoco")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data_root", default="data/RIS")
    parser.add_argument("--model_root", default="models/stepfun-ai/Step1X-Edit-v1p2")
    parser.add_argument("--prefix", default="Remove")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data/instructions_new")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--merge", action="store_true", help="only merge the shard outputs into one json file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.merge:
        merge(args.output_dir, args.dataset, args.split, args.num_shards)
        return

    torch.manual_seed(args.seed)
    dataset = RISDataset(args.data_root, args.dataset, args.split)
    refiner = InstructionRefiner(args.model_root)

    out_path = os.path.join(
        args.output_dir, f"{args.dataset}_{args.split}.shard{args.shard_id}of{args.num_shards}.jsonl"
    )
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["seg_id"] for line in f}

    with open(out_path, "a", encoding="utf-8") as f:
        for idx in tqdm(range(args.shard_id, len(dataset), args.num_shards)):
            data = dataset[idx]
            seg_id = int(data["seg_id"])
            if seg_id in done:
                continue
            image = Image.fromarray(data["img"])
            sents = []
            for sent in data["sents"]:
                sent = f"{args.prefix} {sent}"
                try:
                    sent = refiner(image, sent)
                except Exception as e:  # keep the unrefined instruction, as in the paper
                    print(e)
                sents.append(sent)
            f.write(json.dumps({"seg_id": seg_id, "sents": sents}, ensure_ascii=False) + "\n")
            f.flush()

    if args.num_shards == 1:
        merge(args.output_dir, args.dataset, args.split, 1)


if __name__ == "__main__":
    main()
