import random

import numpy as np
from torch.utils.data import Dataset

ALLOWED_REGION_NAMES = {"promoter", "cds", "utr", "exon", "intron", "nig", "dig"}


def normalize_region_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in ALLOWED_REGION_NAMES:
        raise ValueError(
            f"Unsupported region name '{name}'. "
            f"Expected subset of {sorted(ALLOWED_REGION_NAMES)}."
        )
    return normalized


def _normalize_region_label_map(raw) -> dict[str, int]:
    if raw is None:
        return {}
    if isinstance(raw, (list, tuple)):
        return {normalize_region_name(str(name)): int(idx) for idx, name in enumerate(raw)}
    if isinstance(raw, dict):
        return {normalize_region_name(str(k)): int(v) for k, v in raw.items()}
    raise TypeError(f"region_label_map must be a dict or list, got {type(raw)}")


def _build_region_ids(record, seq_len: int) -> np.ndarray:
    """
    Build dense region_ids array for multi-region ratio loss.
    Requires region_ids (dense ids) or region_segments (RLE).
    """
    if "region_ids" in record:
        arr = np.array(record["region_ids"], dtype=np.int64)
        if arr.shape[0] < seq_len:
            raise ValueError("region_ids shorter than sequence length.")
        return arr[:seq_len]
    if "region_segments" in record:
        region_ids = np.full(seq_len, -1, dtype=np.int64)
        for s, e, rid in record["region_segments"]:
            s_clamped = max(0, min(int(s), seq_len))
            e_clamped = max(0, min(int(e), seq_len))
            if e_clamped > s_clamped:
                region_ids[s_clamped:e_clamped] = int(rid)
        if (region_ids < 0).any():
            raise ValueError("region_segments do not cover the full sequence.")
        return region_ids
    raise ValueError("record missing region_ids or region_segments.")


class RegionAwareDNADataset(Dataset):
    """
    RefSeq dataset variant that also returns per-base region labels.

    Requires region_ids or region_segments for multi-region labels.
    """

    def __init__(self, tokenizer, dataset, data_max_length):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.data_max_length = data_max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]
        sequence = record["sequence"]
        seq_len = len(sequence)
        region_ids = _build_region_ids(record, seq_len)

        if seq_len > self.data_max_length:
            start_idx = random.randint(0, seq_len - self.data_max_length)
            end_idx = start_idx + self.data_max_length
            sequence = sequence[start_idx:end_idx]
            region_ids = region_ids[start_idx:end_idx]

        inputs = {"input_ids": self.tokenizer(sequence), "region_ids": region_ids}
        return inputs
