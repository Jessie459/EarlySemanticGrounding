import json
import os

import cv2
import lmdb
import numpy as np
import pyarrow as pa

SPLITS = {
    "refcoco": {"val": 3811, "testA": 1975, "testB": 1810},
    "refcoco+": {"val": 3805, "testA": 1975, "testB": 1798},
    "refcocog_u": {"val": 2573, "test": 5023},
}


class RISDataset:
    """RefCOCO/+/g in the LMDB format of CRIS (https://github.com/DerrickWang005/CRIS.pytorch).

    Expects `{data_root}/lmdb/{dataset}/{split}.lmdb` and `{data_root}/masks/{dataset}/{seg_id}.png`.
    """

    def __init__(self, data_root, dataset, split):
        self.lmdb_dir = os.path.join(data_root, "lmdb", dataset, f"{split}.lmdb")
        self.mask_dir = os.path.join(data_root, "masks", dataset)
        self.length = SPLITS[dataset][split]
        self.env = None

    def _init_db(self):
        self.env = lmdb.open(
            self.lmdb_dir,
            subdir=os.path.isdir(self.lmdb_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            length = pa.deserialize(txn.get(b"__len__"))
            self.keys = pa.deserialize(txn.get(b"__keys__"))
        assert length == self.length, "precomputed length is incorrect"

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Open LMDB lazily so that the dataset can be forked: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        with self.env.begin(write=False) as txn:
            loaded = pa.deserialize(txn.get(self.keys[index]))

        img = cv2.imdecode(np.frombuffer(loaded["img"], np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg_id = loaded["seg_id"]
        mask = cv2.imread(os.path.join(self.mask_dir, f"{seg_id}.png"), flags=cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0

        return {"img": img, "seg_id": seg_id, "sents": loaded["sents"], "mask": mask}


def load_instructions(instruction_dir, dataset, split):
    """Editing instructions ("Remove <expression>", refined by Step1X-Edit v1.2), keyed by str(seg_id)."""
    with open(os.path.join(instruction_dir, f"{dataset}_{split}.json"), "r", encoding="utf-8") as f:
        return json.load(f)
