import numpy as np
import torch
import torch.nn.functional as F


def minmax_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + eps)


def masked_average_pooling(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """feat: [C, h, w], mask: [h, w] -> [C]"""
    mask = mask.to(feat.dtype)
    return torch.einsum("chw,hw->c", feat, mask / (mask.sum() + eps))


def feature_based_localization(prior: torch.Tensor, feat: torch.Tensor, size: tuple[int, int], thr: float = 0.8):
    """Feature-based Semantic Localization (FSL).

    The attention prior is thresholded into a coarse foreground region, foreground/background prototypes are obtained by
    masked average pooling over the l2-normalized features, and every pixel is assigned to the prototype with the higher
    cosine similarity.

    Args:
        prior: refined attention map in [0, 1], shape [h, w].
        feat: image-token features, shape [C, h, w].
        size: (H, W) of the output mask.
        thr: threshold tau on the attention prior.

    Returns:
        Boolean foreground mask of shape [H, W].
    """
    coarse = prior > thr
    feat = F.normalize(feat.float(), dim=0)
    prototypes = torch.stack([masked_average_pooling(feat, ~coarse), masked_average_pooling(feat, coarse)])  # [2, C]
    scores = torch.einsum("kc,chw->khw", prototypes, feat)
    # Bilinear upsampling is linear, so upsampling the scores equals scoring the upsampled features.
    scores = F.interpolate(scores[None], size=size, mode="bilinear")[0]
    return scores.argmax(dim=0).bool()


class IoUMeter:
    def __init__(self):
        self.inter, self.union, self.ious = 0, 0, []

    def add(self, inter: int, union: int) -> float:
        iou = inter / union if union > 0 else 1.0
        self.inter += inter
        self.union += union
        self.ious.append(iou)
        return iou

    def summary(self) -> dict:
        ious = np.asarray(self.ious)
        res = {"oIoU": 100.0 * self.inter / max(self.union, 1), "mIoU": 100.0 * ious.mean()}
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
            res[f"Pr@{int(thr * 100)}"] = 100.0 * (ious > thr).mean()
        res["num"] = len(ious)
        return res
