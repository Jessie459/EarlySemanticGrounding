"""Segment the object referred to by a natural-language expression in a single image.

python demo.py --model step1x --image examples/zebras.jpg --expression "zebra on left looking at you" --output outputs/demo

Other examples (from RefCOCO val): examples/broccoli.jpg "brocoli at top", examples/buses.jpg "bus on far right".
"""

import argparse
import gc
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from esg.segment import feature_based_localization

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["step1x", "flux2"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expression", required=True, help='referring expression, e.g. "the man on the left"')
    parser.add_argument("--refine", action="store_true", help="refine the instruction with Step1X-Edit v1.2")
    parser.add_argument("--refiner_root", default="models/stepfun-ai/Step1X-Edit-v1p2")
    parser.add_argument("--model_root", default=None)
    parser.add_argument("--qwen_path", default="models/Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--feat_block", type=int, default=None)
    parser.add_argument("--thr", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/demo")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    instruction = f"Remove {args.expression}"
    if args.refine:
        from esg.refine import InstructionRefiner

        refiner = InstructionRefiner(args.refiner_root)
        torch.manual_seed(args.seed)
        instruction = refiner(image, instruction)
        del refiner
        torch.cuda.empty_cache()
    print(f"Instruction: {instruction}")

    from evaluate import DEFAULT_FEAT_BLOCK, build_segmenter

    if args.feat_block is None:
        args.feat_block = DEFAULT_FEAT_BLOCK[args.model]
    if args.model == "step1x":
        # Load the text encoder and the DiT one after the other to save GPU memory.
        encoder = build_segmenter(args, load_dit=False)
        text = encoder.encode_text(image, instruction)
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        prior, feat = build_segmenter(args, load_text_encoder=False).segment(image, text)
    else:
        prior, feat = build_segmenter(args)(image, instruction)
    size = (image.height, image.width)
    mask = feature_based_localization(prior, feat, size=size, thr=args.thr).cpu().numpy()

    os.makedirs(args.output, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(args.output, "mask.png"))
    prior = F.interpolate(prior[None, None].float(), size=size, mode="bilinear")[0, 0].cpu().numpy()
    Image.fromarray((prior * 255).astype(np.uint8)).save(os.path.join(args.output, "attention.png"))
    overlay = np.array(image).astype(np.float32)
    overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])
    Image.fromarray(overlay.astype(np.uint8)).save(os.path.join(args.output, "overlay.png"))
    print(f"Saved mask.png, attention.png and overlay.png to {args.output}")


if __name__ == "__main__":
    main()
