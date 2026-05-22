from __future__ import annotations

"""Downstream dataset helpers for DNALongBench tasks.

This module intentionally keeps release utilities in one place:
- CMP-Seq helpers for contact map prediction and embedding caching.
- DNALongBench sequence loaders for eQTL and ETGP binary classification.
- TISP sequence-regression loaders.

The public DNALongBench dataset layout is:
`andyjzhao/dnalongbench/{tisp,cmp,eqtl,etgp}`.
"""

from bisect import bisect_left
from collections import defaultdict
import csv
import gzip
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence

import numpy as np
import torch
from datasets import Dataset as HFDataset, DatasetDict
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    Subset,
    get_worker_info,
)
from torch.utils.data import Dataset as TorchDataset

from src.genezip.embedding import coverage_aware_pooling, coverage_aware_pooling_num_den
from src.downstream.tokenize import (
    extract_token_stream_from_sequence,
    iter_token_stream_windows_from_sequence,
    build_tokenizer,
    setup_extractor,
)
from src.downstream.utils import build_model


def _decode_uint8_sequence(seq_value: Any) -> str:
    arr = np.asarray(seq_value, dtype=np.uint8)
    lut = np.full(256, ord("N"), dtype=np.uint8)
    lut[0] = ord("A")
    lut[1] = ord("C")
    lut[2] = ord("G")
    lut[3] = ord("T")
    lut[4] = ord("N")
    return lut[arr].tobytes().decode("ascii")


def _resolve_checkpoint_dir(checkpoint_path: str) -> str:
    checkpoint_path = os.path.abspath(checkpoint_path)
    return checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)


def resolve_cmp_split(dataset: DatasetDict, split: str) -> HFDataset:
    split = str(split or "train").lower()
    if split in dataset:
        return dataset[split]
    alt = {"val": "validation", "validate": "validation", "valid": "validation"}
    if split in alt and alt[split] in dataset:
        return dataset[alt[split]]
    raise KeyError(f"Split '{split}' not found in CMPDataset; available: {list(dataset.keys())}")


@dataclass
class FeatureRecord:
    sample_id: str
    spans: torch.Tensor
    tokens: Optional[torch.Tensor]
    embedding: Optional[torch.Tensor]
    h_bins: Optional[torch.Tensor]
    meta: Dict[str, Any]


class EmbeddingProvider(Protocol):
    def get(self, sample_id: str) -> FeatureRecord:
        ...

    def get_many(self, sample_ids: Sequence[str]) -> Sequence[FeatureRecord]:
        ...

    def validate(self) -> None:
        ...


class CMPEmbeddingProviderBase:
    def __init__(self, dataset: HFDataset) -> None:
        self.dataset = dataset
        self._id_to_idx: Optional[Dict[str, int]] = None
        self._build_index()

    def _build_index(self) -> None:
        if "sample_id" not in self.dataset.column_names:
            raise ValueError("CMPDataset missing sample_id column.")
        ids = self.dataset["sample_id"]
        self._id_to_idx = {str(sample_id): idx for idx, sample_id in enumerate(ids)}

    def _row(self, sample_id: str) -> Dict[str, Any]:
        if self._id_to_idx is None:
            self._build_index()
        idx = self._id_to_idx.get(str(sample_id))
        if idx is None:
            raise KeyError(f"sample_id not found in CMPDataset: {sample_id}")
        return self.dataset[idx]

    def get_many(self, sample_ids: Sequence[str]) -> Sequence[FeatureRecord]:
        return [self.get(sample_id) for sample_id in sample_ids]

    def validate(self) -> None:
        if "sample_id" not in self.dataset.column_names:
            raise ValueError("CMPDataset missing sample_id column.")


def _accumulate_hbins(
    window_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],
    num_bins: int,
    bin_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    num = None
    den = None
    last_dtype = None
    for emb, spans in window_iter:
        if emb.numel() == 0:
            continue
        last_dtype = emb.dtype
        num_chunk, den_chunk = coverage_aware_pooling_num_den(
            emb,
            spans,
            bin_size,
            num_bins=num_bins,
        )
        num = num_chunk if num is None else num + num_chunk
        den = den_chunk if den is None else den + den_chunk
    if num is None or den is None:
        raise ValueError("No tokens produced for h_bins pooling.")
    pooled = num / (den[:, None] + eps)
    if last_dtype is not None:
        pooled = pooled.to(last_dtype)
    return pooled


class SequenceProvider(CMPEmbeddingProviderBase):
    def __init__(
        self,
        dataset: HFDataset,
        sequence_key: str,
        sequence_format: str,
        reference_id: Optional[str],
        tokenizer_name: str,
        embedding_layer: str,
        embedding_stream: str,
        trunk_len: int,
        routing_step: Optional[int],
        encode_batch_size: int,
        validate_spans: bool,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
        strict: bool,
        disable_routing: bool = False,
        cache_dir: Optional[str] = None,
        cache_tokens: bool = True,
        cache_embeddings: bool = False,
        num_bins: Optional[int] = None,
        bin_size: Optional[int] = None,
        cache_hbins: bool = False,
    ) -> None:
        super().__init__(dataset)
        self.sequence_key = str(sequence_key or "sequence")
        self.sequence_format = str(sequence_format or "string").lower()
        if self.sequence_format not in ("string", "uint8"):
            raise ValueError("sequence_format must be 'string' or 'uint8'")
        self.reference_id = reference_id
        self.tokenizer_name = tokenizer_name
        self.embedding_layer = embedding_layer
        self.embedding_stream = embedding_stream
        self.trunk_len = int(trunk_len)
        self.routing_step = routing_step
        self.encode_batch_size = int(encode_batch_size)
        self.validate_spans = bool(validate_spans)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype
        self.strict = strict
        self.disable_routing = bool(disable_routing)
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else None
        self.cache_tokens = bool(cache_tokens)
        self.cache_embeddings = bool(cache_embeddings)
        self.num_bins = int(num_bins) if num_bins is not None else None
        self.bin_size = int(bin_size) if bin_size is not None else None
        self.cache_hbins = bool(cache_hbins)

        self.token_dim: Optional[int] = None
        self._model = None
        self._tokenizer = None
        self._extractor = None
        self._cache: Optional[Dict[str, FeatureRecord]] = {} if self.cache_embeddings else None

    def _sequence_from_row(self, row: Dict[str, Any]) -> str:
        if self.sequence_key not in row:
            raise ValueError(f"CMP-Seq dataset missing '{self.sequence_key}' column.")
        seq_value = row[self.sequence_key]
        if self.sequence_format == "string":
            return str(seq_value)
        return _decode_uint8_sequence(seq_value)

    def _cache_path(self, sample_id: str) -> Optional[str]:
        if not self.cache_dir or not self.cache_tokens:
            return None
        safe_id = str(sample_id)
        subdir = os.path.join(self.cache_dir, safe_id[:2])
        return os.path.join(subdir, f"{safe_id}.pt")

    def cache_path(self, sample_id: str) -> Optional[str]:
        return self._cache_path(sample_id)

    def has_cache(self, sample_id: str) -> bool:
        cache_path = self._cache_path(sample_id)
        return bool(cache_path and os.path.isfile(cache_path))

    def _hbins_path(self, sample_id: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        safe_id = str(sample_id)
        subdir = os.path.join(self.cache_dir, safe_id[:2])
        return os.path.join(subdir, f"{safe_id}.hbins.pt")

    def hbins_cache_path(self, sample_id: str) -> Optional[str]:
        return self._hbins_path(sample_id)

    def has_hbins_cache(self, sample_id: str) -> bool:
        cache_path = self._hbins_path(sample_id)
        return bool(cache_path and os.path.isfile(cache_path))

    def _load_disk_cache(self, path: str, sample_id: str) -> FeatureRecord:
        payload = torch.load(path, map_location="cpu")
        return FeatureRecord(
            sample_id=str(sample_id),
            spans=payload["spans"],
            tokens=payload.get("tokens"),
            embedding=payload.get("embedding"),
            h_bins=payload.get("h_bins"),
            meta=payload.get("meta", {}),
        )

    def _load_hbins_cache(self, path: str, sample_id: str) -> FeatureRecord:
        payload = torch.load(path, map_location="cpu")
        h_bins = payload.get("h_bins")
        if h_bins is None:
            raise ValueError(f"h_bins cache missing data for sample {sample_id}")
        return FeatureRecord(
            sample_id=str(sample_id),
            spans=payload.get("spans", torch.empty((0, 2), dtype=torch.int64)),
            tokens=None,
            embedding=None,
            h_bins=h_bins,
            meta=payload.get("meta", {}),
        )

    def _save_disk_cache(self, path: str, feat: FeatureRecord) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "spans": feat.spans.cpu(),
                "tokens": None if feat.tokens is None else feat.tokens.cpu(),
                "embedding": None if feat.embedding is None else feat.embedding.cpu(),
                "meta": feat.meta,
            },
            path,
        )

    def _save_hbins_cache(
        self,
        path: str,
        h_bins: torch.Tensor,
        spans: Optional[torch.Tensor],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        meta_payload = dict(meta or {})
        meta_payload.setdefault("feature_kind", "h_bins")
        if self.num_bins is not None:
            meta_payload.setdefault("num_bins", int(self.num_bins))
        if self.bin_size is not None:
            meta_payload.setdefault("bin_size", int(self.bin_size))
        payload: Dict[str, Any] = {
            "h_bins": h_bins.cpu(),
            "meta": meta_payload,
        }
        if spans is not None:
            payload["spans"] = spans.cpu()
        torch.save(payload, path)

    def _compute_hbins(self, feat: FeatureRecord) -> torch.Tensor:
        if self.num_bins is None or self.bin_size is None:
            raise ValueError("num_bins/bin_size must be set to compute h_bins.")
        tokens = feat.embedding if feat.embedding is not None else feat.tokens
        if tokens is None:
            raise ValueError("FeatureRecord missing embedding/tokens for h_bins.")
        spans = feat.spans
        pooled, _ = coverage_aware_pooling(
            tokens,
            spans,
            self.bin_size,
            num_bins=self.num_bins,
            return_den=True,
        )
        return pooled

    def _iter_hbins_windows(self, seq: str) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        if self._model is None or self._tokenizer is None or self._extractor is None:
            raise RuntimeError("Tokenizer model not initialized.")
        for emb, starts, ends in iter_token_stream_windows_from_sequence(
            seq,
            self.trunk_len,
            self._tokenizer,
            self._model,
            self.device,
            self._extractor,
            self.embedding_layer,
            routing_step=self.routing_step,
            encode_batch_size=self.encode_batch_size,
            validate_spans_flag=self.validate_spans,
        ):
            if emb.numel() == 0:
                continue
            if self.token_dim is None:
                self.token_dim = int(emb.shape[1])
            spans = torch.from_numpy(np.stack([starts, ends], axis=1)).to(torch.int64)
            yield emb, spans

    def _compute_hbins_stream(self, seq: str) -> torch.Tensor:
        if self.num_bins is None or self.bin_size is None:
            raise ValueError("num_bins/bin_size must be set to compute h_bins.")
        return _accumulate_hbins(
            self._iter_hbins_windows(seq),
            self.num_bins,
            self.bin_size,
        )

    def get_hbins(self, sample_id: str) -> FeatureRecord:
        sample_id = str(sample_id)
        cache_path = self._hbins_path(sample_id)
        if cache_path and os.path.isfile(cache_path):
            return self._load_hbins_cache(cache_path, sample_id)
        row = self._row(sample_id)
        if self.reference_id is not None and "reference_id" in row:
            if str(row["reference_id"]) != str(self.reference_id):
                raise ValueError("reference_id mismatch between CMP-Seq dataset and config.")
        self._ensure_runtime()
        seq = self._sequence_from_row(row)
        h_bins = self._compute_hbins_stream(seq)
        meta = {
            "feature_kind": "orca_embedding",
            "embedding_layer": self.embedding_layer,
            "embedding_stream": self.embedding_stream,
            "tokenizer": self.tokenizer_name,
            "reference_id": row.get("reference_id"),
            "sequence_format": self.sequence_format,
            "schema_version": "v1",
        }
        if cache_path and self.cache_hbins:
            self._save_hbins_cache(cache_path, h_bins, None, meta=meta)
        return FeatureRecord(
            sample_id=sample_id,
            spans=torch.empty((0, 2), dtype=torch.int64),
            tokens=None,
            embedding=None,
            h_bins=h_bins,
            meta=meta,
        )

    def get(self, sample_id: str) -> FeatureRecord:
        sample_id = str(sample_id)
        if self._cache is not None and sample_id in self._cache:
            return self._cache[sample_id]
        cache_path = self._cache_path(sample_id)
        if cache_path and os.path.isfile(cache_path):
            feat = self._load_disk_cache(cache_path, sample_id)
            if self._cache is not None:
                self._cache[sample_id] = feat
            return feat
        row = self._row(sample_id)
        if self.reference_id is not None and "reference_id" in row:
            if str(row["reference_id"]) != str(self.reference_id):
                raise ValueError("reference_id mismatch between CMP-Seq dataset and config.")
        self._ensure_runtime()
        seq = self._sequence_from_row(row)
        tokens, start_tok, end_tok, stats = extract_token_stream_from_sequence(
            seq,
            self.trunk_len,
            self._tokenizer,
            self._model,
            self.device,
            self._extractor,
            self.embedding_layer,
            routing_step=self.routing_step,
            encode_batch_size=self.encode_batch_size,
            validate_spans_flag=self.validate_spans,
        )
        if tokens.numel() == 0:
            raise ValueError(f"No tokens produced for sample {sample_id}")
        if self.token_dim is None:
            self.token_dim = int(tokens.shape[1])

        spans = torch.stack([torch.from_numpy(start_tok), torch.from_numpy(end_tok)], dim=1).to(torch.int64)
        meta = {
            "feature_kind": "orca_embedding",
            "embedding_layer": self.embedding_layer,
            "embedding_stream": self.embedding_stream,
            "tokenizer": self.tokenizer_name,
            "reference_id": row.get("reference_id"),
            "sequence_format": self.sequence_format,
            "schema_version": "v1",
        }
        feat = FeatureRecord(
            sample_id=sample_id,
            spans=spans,
            tokens=None,
            embedding=tokens,
            h_bins=None,
            meta={**meta, **(stats or {})},
        )
        if self._cache is not None:
            self._cache[sample_id] = feat
        if cache_path:
            self._save_disk_cache(cache_path, feat)
        if self.cache_hbins:
            hbins_path = self._hbins_path(sample_id)
            if hbins_path and not os.path.isfile(hbins_path):
                h_bins = self._compute_hbins(feat)
                self._save_hbins_cache(hbins_path, h_bins, feat.spans, meta=meta)
        return feat

    def _ensure_runtime(self) -> None:
        if self._model is not None:
            return
        checkpoint_dir = _resolve_checkpoint_dir(self.checkpoint_path)
        self._model, prev_cfg = build_model(
            checkpoint_dir,
            self.checkpoint_path,
            self.device,
            self.dtype,
            self.strict,
            disable_routing=self.disable_routing,
        )
        self._tokenizer = build_tokenizer(self.tokenizer_name)
        model_cfg = prev_cfg.model_cfg
        self._extractor = setup_extractor(
            self._model,
            self.embedding_layer,
            self.embedding_stream,
            model_cfg=model_cfg,
        )

    def close(self) -> None:
        if self._extractor is not None:
            try:
                self._extractor.close()
            except Exception:
                pass
            self._extractor = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_model"] = None
        state["_tokenizer"] = None
        state["_extractor"] = None
        return state


class CMPTrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        provider: EmbeddingProvider,
        label_key: str = "label_ut",
        mask_key: Optional[str] = None,
        target_index: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.provider = provider
        self.label_key = label_key
        self.mask_key = mask_key
        self.target_index = int(target_index) if target_index is not None else None

    def _select_target(self, value: Any) -> torch.Tensor:
        target = torch.as_tensor(value, dtype=torch.float32)
        if self.target_index is None or target.dim() == 1:
            return target
        if target.dim() != 2:
            raise ValueError(f"CMP target must be 1D or 2D, got shape={tuple(target.shape)}")
        return target[:, self.target_index]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset[idx]
        sample_id = str(row["sample_id"])
        feat = self.provider.get(sample_id)
        target = self._select_target(row[self.label_key])
        out = {"sample_id": sample_id, "target": target, "feat": feat}
        if self.mask_key and self.mask_key in row:
            out["mask_ut"] = self._select_target(row[self.mask_key]).bool()
        return out


class CMPHbinsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        provider: SequenceProvider,
        label_key: str = "label_ut",
        mask_key: Optional[str] = None,
        target_index: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.provider = provider
        self.label_key = label_key
        self.mask_key = mask_key
        self.target_index = int(target_index) if target_index is not None else None

    def _select_target(self, value: Any) -> torch.Tensor:
        target = torch.as_tensor(value, dtype=torch.float32)
        if self.target_index is None or target.dim() == 1:
            return target
        if target.dim() != 2:
            raise ValueError(f"CMP target must be 1D or 2D, got shape={tuple(target.shape)}")
        return target[:, self.target_index]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset[idx]
        sample_id = str(row["sample_id"])
        feat = self.provider.get_hbins(sample_id)
        if feat.h_bins is None:
            raise ValueError(f"h_bins missing for sample {sample_id}")
        target = self._select_target(row[self.label_key])
        out = {"sample_id": sample_id, "target": target, "h_bins": feat.h_bins}
        if self.mask_key and self.mask_key in row:
            out["mask_ut"] = self._select_target(row[self.mask_key]).bool()
        return out


def collate_cmp_features(
    batch: Sequence[Dict[str, Any]],
    pad_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    feats = [item["feat"] for item in batch]
    token_list: list[torch.Tensor] = []
    start_list: list[torch.Tensor] = []
    end_list: list[torch.Tensor] = []
    for feat in feats:
        tokens = feat.embedding if feat.embedding is not None else feat.tokens
        if tokens is None:
            raise ValueError("FeatureRecord missing embedding/tokens.")
        if tokens.dim() != 2:
            raise ValueError("FeatureRecord tokens must be a 2D tensor (T, D).")
        spans = feat.spans
        if spans.dim() != 2 or spans.shape[1] != 2:
            raise ValueError("FeatureRecord spans must have shape (T, 2).")
        token_list.append(tokens)
        start_list.append(spans[:, 0])
        end_list.append(spans[:, 1])

    lengths = [t.shape[0] for t in token_list]
    if not lengths:
        raise ValueError("Empty batch")
    max_len = max(lengths)
    token_dim = token_list[0].shape[1]
    batch_size = len(token_list)

    tokens_pad = token_list[0].new_full((batch_size, max_len, token_dim), pad_value)
    start_pad = start_list[0].new_zeros((batch_size, max_len))
    end_pad = end_list[0].new_zeros((batch_size, max_len))
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for idx, length in enumerate(lengths):
        tokens_pad[idx, :length] = token_list[idx]
        start_pad[idx, :length] = start_list[idx]
        end_pad[idx, :length] = end_list[idx]
        mask[idx, :length] = True

    target = torch.stack([item["target"] for item in batch], dim=0)
    payload: Dict[str, Any] = {
        "tokens": tokens_pad,
        "start_bp": start_pad,
        "end_bp": end_pad,
        "mask": mask,
        "target": target,
        "sample_id": [item.get("sample_id") for item in batch],
    }
    if "mask_ut" in batch[0]:
        payload["mask_ut"] = torch.stack([item["mask_ut"] for item in batch], dim=0)
    return payload


def collate_cmp_hbins(
    batch: Sequence[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    h_bins = torch.stack([item["h_bins"] for item in batch], dim=0)
    target = torch.stack([item["target"] for item in batch], dim=0)
    payload: Dict[str, Any] = {
        "h_bins": h_bins,
        "target": target,
        "sample_id": [item.get("sample_id") for item in batch],
    }
    if "mask_ut" in batch[0]:
        payload["mask_ut"] = torch.stack([item["mask_ut"] for item in batch], dim=0)
    return payload


# DNALongBench sequence-classification helpers.
_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _rc_dna(seq: str) -> str:
    return "".join({"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}[ch.upper()] for ch in seq[::-1])


def parse_config(config_file: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    with open(config_file, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            key, raw_value, raw_type = parts[0], parts[1], parts[2]
            value: Any = raw_value
            if raw_type == "int":
                value = int(raw_value)
            elif raw_type == "float":
                value = float(raw_value)
            elif raw_type == "bool":
                lowered = raw_value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    value = True
                elif lowered in {"false", "0", "no"}:
                    value = False
                else:
                    value = bool(raw_value)
            elif raw_type == "list":
                items = [item for item in raw_value.split(",") if item != ""]
                parsed: list[Any] = []
                for item in items:
                    item = item.strip()
                    if item == "":
                        continue
                    if item.isnumeric():
                        parsed.append(int(item))
                        continue
                    try:
                        parsed.append(float(item))
                        continue
                    except Exception:
                        parsed.append(str(item))
                value = parsed
            else:
                value = str(raw_value)
            config[key] = value
    return config


class FastaStringExtractor:
    def __init__(self, fasta_file: str) -> None:
        try:
            import pyfaidx  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pyfaidx is required for DNALongBench loaders. Add it via uv sync.") from exc

        self._fasta_file = str(fasta_file)
        self._fasta = pyfaidx.Fasta(self._fasta_file, rebuild=True, as_raw=True, sequence_always_upper=True)
        self._chrom_sizes: dict[str, int] = {name: len(seq) for name, seq in self._fasta.items()}

    def _resolve_chrom(self, chrom: str) -> str:
        if chrom in self._chrom_sizes:
            return chrom
        if chrom.startswith("chr"):
            alt = chrom.removeprefix("chr")
        else:
            alt = f"chr{chrom}"
        if alt in self._chrom_sizes:
            return alt
        raise KeyError(f"chrom {chrom!r} not found in fasta={self._fasta_file}")

    def extract(self, chrom: str, start: int, end: int) -> str:
        chrom = self._resolve_chrom(chrom)
        chrom_size = int(self._chrom_sizes[chrom])
        if end <= start:
            return ""

        pad_upstream = "N" * max(0, -int(start))
        pad_downstream = "N" * max(0, int(end) - chrom_size)

        trimmed_start = max(0, int(start))
        trimmed_end = min(chrom_size, int(end))
        if trimmed_end <= trimmed_start:
            return pad_upstream + pad_downstream

        # pyfaidx expects 1-based inclusive coordinates.
        seq = str(self._fasta.get_seq(chrom, trimmed_start + 1, trimmed_end))
        return f"{pad_upstream}{seq}{pad_downstream}"

    def close(self) -> None:
        self._fasta.close()


class BedIntervalIndex:
    def __init__(self, bed_gz_path: str) -> None:
        self._path = str(bed_gz_path)
        intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        with gzip.open(self._path, "rt", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except Exception:
                    continue
                if end <= start:
                    continue
                intervals[chrom].append((start, end))

        self._intervals: dict[str, list[tuple[int, int]]] = {}
        self._starts: dict[str, list[int]] = {}
        for chrom, items in intervals.items():
            items.sort(key=lambda x: (x[0], x[1]))
            merged: list[tuple[int, int]] = []
            for start, end in items:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            self._intervals[chrom] = merged
            self._starts[chrom] = [start for start, _ in merged]

    def _resolve_chrom(self, chrom: str) -> str | None:
        if chrom in self._intervals:
            return chrom
        if chrom.startswith("chr"):
            alt = chrom.removeprefix("chr")
        else:
            alt = f"chr{chrom}"
        if alt in self._intervals:
            return alt
        return None

    def iter_overlaps(self, chrom: str, start: int, end: int) -> Iterable[tuple[int, int]]:
        chrom_key = self._resolve_chrom(chrom)
        if chrom_key is None or end <= start:
            return
        intervals = self._intervals[chrom_key]
        starts = self._starts[chrom_key]
        idx = max(0, bisect_left(starts, int(start)) - 1)
        while idx < len(intervals):
            s, e = intervals[idx]
            if s >= end:
                break
            if e > start:
                yield max(start, s), min(end, e)
            idx += 1


def _mask_sequence(seq: str, overlaps: Iterable[tuple[int, int]], sequence_start: int) -> str:
    if not seq:
        return seq
    buf = bytearray(seq.encode("ascii"))
    n = len(buf)
    for o_start, o_end in overlaps:
        rel_start = max(0, int(o_start) - int(sequence_start))
        rel_end = min(n, int(o_end) - int(sequence_start))
        if rel_end <= rel_start:
            continue
        buf[rel_start:rel_end] = b"N" * (rel_end - rel_start)
    return buf.decode("ascii")


def _parse_tsv_dicts(path: str) -> Iterable[dict[str, str]]:
    with open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            yield {k: (v if v is not None else "") for k, v in row.items()}


def parse_eqtl_sequences(
    *,
    root_path: str,
    eqtl_file: str,
    blacklist_bed_gz: str,
    fasta: FastaStringExtractor,
    subset: str,
    seq_len_cutoff: int,
    tss_flank_upstream: int,
    tss_flank_downstream: int,
    region_flank_upstream: int,
    region_flank_downstream: int,
    max_records: int | None = None,
) -> list[tuple[str, str, str]]:
    if subset not in {"train", "valid", "test"}:
        raise ValueError(f"subset must be one of train/valid/test, got {subset!r}")

    records: list[tuple[str, str, str]] = []
    blacklist = BedIntervalIndex(os.path.join(root_path, blacklist_bed_gz))
    for row in _parse_tsv_dicts(os.path.join(root_path, eqtl_file)):
        if row.get("subset") != subset:
            continue
        if row.get("gene_chrom") != row.get("region_chrom"):
            continue
        gene_strand = row.get("gene_strand")
        try:
            gene_start = int(row["gene_start"])
            gene_end = int(row["gene_end"])
            region_start = int(row["region_start"])
            region_end = int(row["region_end"])
        except Exception:
            continue

        if gene_strand == "+":
            tss_anchor = gene_start
        elif gene_strand == "-":
            tss_anchor = gene_end - 1
        else:
            continue

        tss_start = tss_anchor - int(tss_flank_upstream)
        tss_end = (tss_anchor + 1) + int(tss_flank_downstream)

        variant_start = region_start
        variant_end = region_end
        region_start = region_start - int(region_flank_upstream)
        region_end = region_end + int(region_flank_downstream)

        distance = max(0, max(tss_start, region_start) - min(tss_end, region_end))
        if distance > int(seq_len_cutoff):
            continue

        sequence_start = min(tss_start, region_start)
        sequence_end = max(tss_end, region_end)
        chrom = str(row["region_chrom"])
        region_seq = fasta.extract(chrom, sequence_start, sequence_end)

        allele1 = str(row.get("allele1", ""))
        allele2 = str(row.get("allele2", ""))
        variant_rel_start = variant_start - sequence_start
        variant_rel_end = variant_end - sequence_start
        if allele1 and region_seq[variant_rel_start:variant_rel_end] != allele1:
            # Reference mismatch (rare) - skip instead of crashing.
            continue

        if distance > 0:
            if tss_start > region_end:
                query_start, query_end = region_end, tss_start
            else:
                query_start, query_end = tss_end, region_start
            overlaps = blacklist.iter_overlaps(chrom, int(query_start), int(query_end))
            region_seq = _mask_sequence(region_seq, overlaps, sequence_start)

        region_seq_var = f"{region_seq[:variant_rel_start]}{allele2}{region_seq[variant_rel_end:]}"

        if gene_start > region_end:
            region_seq = _rc_dna(region_seq)
            region_seq_var = _rc_dna(region_seq_var)

        if len(region_seq) < int(seq_len_cutoff):
            region_seq = region_seq + ("N" * (int(seq_len_cutoff) - len(region_seq)))
        else:
            region_seq = region_seq[: int(seq_len_cutoff)]

        if len(region_seq_var) < int(seq_len_cutoff):
            region_seq_var = region_seq_var + ("N" * (int(seq_len_cutoff) - len(region_seq_var)))
        else:
            region_seq_var = region_seq_var[: int(seq_len_cutoff)]

        records.append((region_seq, region_seq_var, str(row.get("target", ""))))
        if max_records is not None and len(records) >= int(max_records):
            break
    return records


def parse_etgp_sequences(
    *,
    root_path: str,
    epi_file: str,
    blacklist_bed_gz: str,
    fasta: FastaStringExtractor,
    subset: str,
    seq_len_cutoff: int,
    tss_flank_upstream: int,
    tss_flank_downstream: int,
    region_flank_upstream: int,
    region_flank_downstream: int,
    max_records: int | None = None,
) -> list[tuple[str, str]]:
    if subset not in {"train", "valid", "test"}:
        raise ValueError(f"subset must be one of train/valid/test, got {subset!r}")

    records: list[tuple[str, str]] = []
    blacklist = BedIntervalIndex(os.path.join(root_path, blacklist_bed_gz))
    for row in _parse_tsv_dicts(os.path.join(root_path, epi_file)):
        if row.get("subset") != subset:
            continue
        if row.get("gene_chrom") != row.get("region_chrom"):
            continue
        gene_strand = row.get("gene_strand")
        try:
            gene_start = int(row["gene_start"])
            gene_end = int(row["gene_end"])
            region_start = int(row["region_start"])
            region_end = int(row["region_end"])
        except Exception:
            continue

        if gene_strand == "+":
            tss_anchor = gene_start
        elif gene_strand == "-":
            tss_anchor = gene_end - 1
        else:
            continue

        tss_start = tss_anchor - int(tss_flank_upstream)
        tss_end = (tss_anchor + 1) + int(tss_flank_downstream)

        region_start = region_start - int(region_flank_upstream)
        region_end = region_end + int(region_flank_downstream)

        distance = max(0, max(tss_start, region_start) - min(tss_end, region_end))
        if distance > int(seq_len_cutoff):
            continue

        sequence_start = min(tss_start, region_start)
        sequence_end = max(tss_end, region_end)
        chrom = str(row["region_chrom"])
        region_seq = fasta.extract(chrom, sequence_start, sequence_end)

        if distance > 0:
            if tss_start > region_end:
                query_start, query_end = region_end, tss_start
            else:
                query_start, query_end = tss_end, region_start
            overlaps = blacklist.iter_overlaps(chrom, int(query_start), int(query_end))
            region_seq = _mask_sequence(region_seq, overlaps, sequence_start)

        if gene_start > region_end:
            region_seq = _rc_dna(region_seq)

        if len(region_seq) < int(seq_len_cutoff):
            region_seq = region_seq + ("N" * (int(seq_len_cutoff) - len(region_seq)))
        else:
            region_seq = region_seq[: int(seq_len_cutoff)]

        records.append((region_seq, str(row.get("target", ""))))
        if max_records is not None and len(records) >= int(max_records):
            break
    return records


class EQTLseqDataset(TorchDataset):
    def __init__(
        self,
        *,
        tokenizer: Any,
        root_path: str,
        config_file: str,
        subset: str,
        data_max_length: int,
        conjoin_test: bool,
        max_records: int | None = None,
    ) -> None:
        super().__init__()
        self.config = parse_config(config_file)
        self.subset = str(subset)
        self.tokenizer = tokenizer
        self.conjoin_test = bool(conjoin_test)
        self.data_max_length = int(data_max_length)

        genome_fa = self.config.get("genome_fa")
        if not genome_fa:
            raise ValueError("Missing genome_fa in config.")
        fasta = FastaStringExtractor(os.path.join(root_path, str(genome_fa)))

        self.dataset = parse_eqtl_sequences(
            root_path=root_path,
            eqtl_file=str(self.config["eQTL_file"]),
            blacklist_bed_gz=str(self.config["eQTL_tabix_file"]),
            fasta=fasta,
            subset=self.subset,
            seq_len_cutoff=int(self.config.get("seq_len_cutoff", 450000)),
            tss_flank_upstream=int(self.config["tss_flank_upstream"]),
            tss_flank_downstream=int(self.config["tss_flank_downstream"]),
            region_flank_upstream=int(self.config["region_flank_upstream"]),
            region_flank_downstream=int(self.config["region_flank_downstream"]),
            max_records=max_records,
        )
        fasta.close()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        _, variant_seq, target = self.dataset[index]
        target2int = {"positive": 1, "negative": 0}
        label = target2int.get(str(target).lower(), 0)

        tokens_fwd = self.tokenizer(variant_seq[: self.data_max_length])
        if self.conjoin_test and self.subset != "train":
            tokens_rc = self.tokenizer(_reverse_complement(variant_seq)[: self.data_max_length])
            input_ids = torch.stack((torch.LongTensor(tokens_fwd), torch.LongTensor(tokens_rc)), dim=1)
        else:
            input_ids = torch.LongTensor(tokens_fwd)
        return {"input_ids": input_ids, "labels": torch.tensor(label, dtype=torch.long)}


class ETGPseqDataset(TorchDataset):
    def __init__(
        self,
        *,
        tokenizer: Any,
        root_path: str,
        config_file: str,
        subset: str,
        data_max_length: int,
        conjoin_test: bool,
        max_records: int | None = None,
    ) -> None:
        super().__init__()
        self.config = parse_config(config_file)
        self.subset = str(subset)
        self.tokenizer = tokenizer
        self.conjoin_test = bool(conjoin_test)
        self.data_max_length = int(data_max_length)

        genome_fa = self.config.get("genome_fa")
        if not genome_fa:
            raise ValueError("Missing genome_fa in config.")
        fasta = FastaStringExtractor(os.path.join(root_path, str(genome_fa)))

        self.dataset = parse_etgp_sequences(
            root_path=root_path,
            epi_file=str(self.config["EPI_file"]),
            blacklist_bed_gz=str(self.config["enhancer_tabix_file"]),
            fasta=fasta,
            subset=self.subset,
            seq_len_cutoff=int(self.config.get("seq_len_cutoff", 450000)),
            tss_flank_upstream=int(self.config["tss_flank_upstream"]),
            tss_flank_downstream=int(self.config["tss_flank_downstream"]),
            region_flank_upstream=int(self.config["region_flank_upstream"]),
            region_flank_downstream=int(self.config["region_flank_downstream"]),
            max_records=max_records,
        )
        fasta.close()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        seq, target = self.dataset[index]
        target2int = {"positive": 1, "negative": 0}
        label = target2int.get(str(target).lower(), 0)

        tokens_fwd = self.tokenizer(seq[: self.data_max_length])
        if self.conjoin_test and self.subset != "train":
            tokens_rc = self.tokenizer(_reverse_complement(seq)[: self.data_max_length])
            input_ids = torch.stack((torch.LongTensor(tokens_fwd), torch.LongTensor(tokens_rc)), dim=1)
        else:
            input_ids = torch.LongTensor(tokens_fwd)
        return {"input_ids": input_ids, "labels": torch.tensor(label, dtype=torch.long)}


def load_data(
    *,
    root: str,
    task_name: str,
    cell_type: str,
    sequence_length: int,
    tokenizer: Any,
    conjoin_test: bool = False,
    max_train_records: int | None = None,
    max_valid_records: int | None = None,
    max_test_records: int | None = None,
) -> tuple[TorchDataset, TorchDataset, TorchDataset]:
    root_path = os.path.abspath(os.path.expanduser(str(root)))
    task_name = str(task_name)
    cell_type = str(cell_type)
    data_max_length = int(sequence_length)

    if task_name == "eqtl_prediction":
        task_root = os.path.join(root_path, "eqtl")
        cfg_file = os.path.join(task_root, "config", f"gtex_hg38.{cell_type}.config")
        train = EQTLseqDataset(
            tokenizer=tokenizer,
            root_path=task_root,
            config_file=cfg_file,
            subset="train",
            data_max_length=data_max_length,
            conjoin_test=conjoin_test,
            max_records=max_train_records,
        )
        valid = EQTLseqDataset(
            tokenizer=tokenizer,
            root_path=task_root,
            config_file=cfg_file,
            subset="valid",
            data_max_length=data_max_length,
            conjoin_test=conjoin_test,
            max_records=max_valid_records,
        )
        test = EQTLseqDataset(
            tokenizer=tokenizer,
            root_path=task_root,
            config_file=cfg_file,
            subset="test",
            data_max_length=data_max_length,
            conjoin_test=conjoin_test,
            max_records=max_test_records,
        )
        return train, valid, test

    if task_name == "enhancer_target_gene_prediction":
        task_root = os.path.join(root_path, "etgp")
        cfg_file = os.path.join(task_root, "config", f"{cell_type}.config")
        train = ETGPseqDataset(
            tokenizer=tokenizer,
            root_path=task_root,
            config_file=cfg_file,
            subset="train",
            data_max_length=data_max_length,
            conjoin_test=conjoin_test,
            max_records=max_train_records,
        )
        valid = ETGPseqDataset(
            tokenizer=tokenizer,
            root_path=task_root,
            config_file=cfg_file,
            subset="valid",
            data_max_length=data_max_length,
            conjoin_test=conjoin_test,
            max_records=max_valid_records,
        )
        test = ETGPseqDataset(
            tokenizer=tokenizer,
            root_path=task_root,
            config_file=cfg_file,
            subset="test",
            data_max_length=data_max_length,
            conjoin_test=conjoin_test,
            max_records=max_test_records,
        )
        return train, valid, test

    raise ValueError(
        f"Unsupported task_name={task_name!r}. Supported: eqtl_prediction, enhancer_target_gene_prediction."
    )


# TISP sequence regression helpers.
TISP_SEQUENCE_LENGTH = 100_000
TISP_WINDOW_STEP = 50_000
TISP_VALID_HOLDOUT = ("chr10",)
TISP_TEST_HOLDOUT = ("chr8", "chr9")
TISP_FASTA = "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
TISP_FASTA_GZ = f"{TISP_FASTA}.gz"
TISP_TRACK_FILES = (
    "agg.plus.bw.bedgraph.bw",
    "agg.encodecage.plus.v2.bedgraph.bw",
    "agg.encoderampage.plus.v2.bedgraph.bw",
    "agg.plus.grocap.bedgraph.sorted.merged.bw",
    "agg.plus.allprocap.bedgraph.sorted.merged.bw",
    "agg.minus.allprocap.bedgraph.sorted.merged.bw",
    "agg.minus.grocap.bedgraph.sorted.merged.bw",
    "agg.encoderampage.minus.v2.bedgraph.bw",
    "agg.encodecage.minus.v2.bedgraph.bw",
    "agg.minus.bw.bedgraph.bw",
)
TISP_TRACK_NAMES = (
    "cage_plus",
    "encodecage_plus",
    "encoderampage_plus",
    "grocap_plus",
    "procap_plus",
    "procap_minus",
    "grocap_minus",
    "encoderampage_minus",
    "encodecage_minus",
    "cage_minus",
)
TISP_BLACKLIST_FILES = (
    "blacklists/fantom.blacklist8.plus.bed.gz",
    "blacklists/fantom.blacklist8.minus.bed.gz",
)

_BASE_TO_ONE_HOT = np.zeros((256, 4), dtype=np.float32)
_BASE_TO_ONE_HOT[ord("A")] = (1.0, 0.0, 0.0, 0.0)
_BASE_TO_ONE_HOT[ord("C")] = (0.0, 1.0, 0.0, 0.0)
_BASE_TO_ONE_HOT[ord("G")] = (0.0, 0.0, 1.0, 0.0)
_BASE_TO_ONE_HOT[ord("T")] = (0.0, 0.0, 0.0, 1.0)
_BASE_TO_ONE_HOT[ord("a")] = _BASE_TO_ONE_HOT[ord("A")]
_BASE_TO_ONE_HOT[ord("c")] = _BASE_TO_ONE_HOT[ord("C")]
_BASE_TO_ONE_HOT[ord("g")] = _BASE_TO_ONE_HOT[ord("G")]
_BASE_TO_ONE_HOT[ord("t")] = _BASE_TO_ONE_HOT[ord("T")]
_BASE_TO_ONE_HOT[ord("N")] = (0.25, 0.25, 0.25, 0.25)
_BASE_TO_ONE_HOT[ord("n")] = _BASE_TO_ONE_HOT[ord("N")]
_RC_TRANSLATION = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def _reverse_complement_tisp_target(target_cage: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(target_cage[::-1, ::-1])


def resolve_tisp_task_root(root: str) -> str:
    root = os.path.abspath(os.path.expanduser(str(root)))
    if os.path.isdir(os.path.join(root, "seqs")) and os.path.isdir(os.path.join(root, "targets")):
        return root
    subdir = os.path.join(root, "tisp")
    if os.path.isdir(os.path.join(subdir, "seqs")) and os.path.isdir(os.path.join(subdir, "targets")):
        return subdir
    raise FileNotFoundError(f"Expected DNALongBench TISP data at {subdir}.")


def _resolve_chrom_name(chrom_lens: dict[str, int], chrom: str) -> str:
    if chrom in chrom_lens:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in chrom_lens:
        return chrom[3:]
    prefixed = f"chr{chrom}"
    if prefixed in chrom_lens:
        return prefixed
    raise KeyError(f"Chromosome {chrom!r} not found.")


def _resolve_chrom_name_or_none(chrom_lens: dict[str, int], chrom: str) -> str | None:
    try:
        return _resolve_chrom_name(chrom_lens, chrom)
    except KeyError:
        return None


class FastaGenome:
    def __init__(self, input_path: str) -> None:
        from pyfaidx import Fasta

        self.input_path = input_path
        self.fasta = Fasta(input_path, as_raw=False, sequence_always_upper=True)
        self.chrom_lens = {str(name): len(record) for name, record in self.fasta.items()}

    def get_chr_lens(self) -> dict[str, int]:
        return dict(self.chrom_lens)

    def get_sequence_from_coords(self, chrom: str, start: int, end: int, strand: str = "+") -> str:
        resolved = _resolve_chrom_name(self.chrom_lens, chrom)
        start = int(start)
        end = int(end)
        if start < 0 or end < start:
            raise ValueError(f"Invalid FASTA coordinates: {chrom}:{start}-{end}")
        chrom_len = int(self.chrom_lens[resolved])
        clipped_start = max(0, start)
        clipped_end = min(end, chrom_len)
        left_pad = "N" * max(0, -start)
        right_pad = "N" * max(0, end - chrom_len)
        seq = left_pad + str(self.fasta[resolved][clipped_start:clipped_end].seq).upper() + right_pad
        if strand == "-":
            seq = seq.translate(_RC_TRANSLATION)[::-1]
        return seq

    def get_encoding_from_coords(self, chrom: str, start: int, end: int, strand: str = "+") -> np.ndarray:
        seq = self.get_sequence_from_coords(chrom, start, end, strand)
        seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        return np.ascontiguousarray(_BASE_TO_ONE_HOT[seq_bytes])


class GenomicSignalFeatures:
    def __init__(
        self,
        input_paths,
        features,
        shape,
        blacklists=None,
        blacklists_indices=None,
        replacement_indices=None,
        replacement_scaling_factors=None,
    ):
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors
        self.n_features = len(features)
        self.feature_index_dict = {feat: idx for idx, feat in enumerate(features)}
        self.shape = (len(input_paths), *shape)
        self._chrom_sizes = None

    def _initialize(self) -> None:
        if not self.initialized:
            import pyBigWig
            import tabix

            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist) for blacklist in self.blacklists]
            self._chrom_sizes = self.data[0].chroms()
            self.initialized = True

    def get_chrom_sizes(self) -> dict[str, int]:
        self._initialize()
        return dict(self._chrom_sizes)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        self._initialize()

        resolved_chrom = _resolve_chrom_name(self._chrom_sizes, chrom)

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        chrom_end = int(self._chrom_sizes[resolved_chrom])
        start = int(start)
        end = min(int(end), chrom_end)
        wigmat = np.zeros((len(self.data), end - start), dtype=np.float32)
        for idx in feature_indices:
            try:
                wigmat[idx, :] = self.data[idx].values(resolved_chrom, start, end, numpy=True)
            except Exception:
                print(resolved_chrom, start, end, self.input_paths[idx], flush=True)
                raise

        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                        for _, s, e in blacklist.query(resolved_chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s) - start, 0) : int(e) - start] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(resolved_chrom, start, end):
                            wigmat[:, np.fmax(int(s) - start, 0) : int(e) - start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(
                    self.blacklists,
                    self.blacklists_indices,
                    self.replacement_indices,
                    self.replacement_scaling_factors,
                ):
                    for _, s, e in blacklist.query(resolved_chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0) : int(e) - start] = (
                            wigmat[replacement_indices, np.fmax(int(s) - start, 0) : int(e) - start]
                            * replacement_scaling_factor
                        )

        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat


class TISPChromosomeSlidingWindowDataset(Dataset):
    def __init__(
        self,
        genome,
        target: GenomicSignalFeatures,
        chrom: str,
        chrom_len: int,
        *,
        window_size: int = TISP_SEQUENCE_LENGTH,
        step_size: int = TISP_WINDOW_STEP,
    ) -> None:
        if chrom_len < window_size:
            raise ValueError(
                f"chromosome length {chrom_len} is shorter than TISP window_size={window_size}"
            )
        self.genome = genome
        self.target = target
        self.chrom = chrom
        self.chrom_len = int(chrom_len)
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.window_starts = list(range(0, self.chrom_len - self.window_size + 1, self.step_size))
        last_start = self.chrom_len - self.window_size
        if not self.window_starts or self.window_starts[-1] != last_start:
            self.window_starts.append(last_start)
        self.num_windows = len(self.window_starts)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int):
        start = int(self.window_starts[int(idx)])
        end = start + self.window_size
        input_seq = self.genome.get_encoding_from_coords(self.chrom, start, end)
        target_cage = self.target.get_feature_data(self.chrom, start, end)
        return input_seq, target_cage


class TISPRandomTrainDataset(IterableDataset):
    def __init__(
        self,
        genome,
        target: GenomicSignalFeatures,
        chrom_lens: dict[str, int],
        train_chroms: list[str],
        *,
        num_samples: int,
        sequence_length: int = TISP_SEQUENCE_LENGTH,
        seed: int = 3,
        random_strand: bool = False,
    ) -> None:
        super().__init__()
        self.genome = genome
        self.target = target
        self.chrom_lens = chrom_lens
        self.sequence_length = int(sequence_length)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.random_strand = bool(random_strand)

        chroms = []
        weights = []
        for chrom in train_chroms:
            chrom_len = int(chrom_lens[chrom])
            interval_len = chrom_len - 2 * self.sequence_length
            if interval_len <= 0:
                continue
            chroms.append(chrom)
            weights.append(interval_len)

        if not chroms:
            raise ValueError("No eligible training chromosomes were found for TISP.")

        self.chroms = chroms
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        self.window_radius = self.sequence_length // 2

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            worker_samples = self.num_samples
        else:
            worker_id = worker.id
            worker_samples = self.num_samples // worker.num_workers
            if worker_id < (self.num_samples % worker.num_workers):
                worker_samples += 1

        rng = np.random.default_rng(self.seed + worker_id)
        produced = 0
        while produced < worker_samples:
            chrom = rng.choice(self.chroms, p=self.weights)
            chrom_len = int(self.chrom_lens[chrom])
            position = int(rng.integers(self.sequence_length, chrom_len - self.sequence_length))
            start = position - self.window_radius
            end = position + self.window_radius
            strand = "+"
            if self.random_strand:
                strand = "+" if int(rng.integers(0, 2)) == 0 else "-"

            input_seq = self.genome.get_encoding_from_coords(chrom, start, end, strand)
            if input_seq.shape[0] == 0:
                continue
            if np.mean(input_seq == 0.25) > 0.30:
                continue

            target_cage = self.target.get_feature_data(chrom, start, end)
            if strand == "-":
                target_cage = _reverse_complement_tisp_target(target_cage)
            produced += 1
            yield input_seq, target_cage


def _make_signal_target(task_root: str) -> GenomicSignalFeatures:
    target_root = os.path.join(task_root, "targets")
    input_paths = [os.path.join(target_root, name) for name in TISP_TRACK_FILES]
    blacklists = [os.path.join(target_root, name) for name in TISP_BLACKLIST_FILES]
    return GenomicSignalFeatures(
        input_paths,
        TISP_TRACK_NAMES,
        (TISP_SEQUENCE_LENGTH,),
        blacklists,
        [0, 9],
        [1, 8],
        [0.61357, 0.61357],
    )


def _resolve_tisp_fasta_path(task_root: str) -> str:
    seq_root = os.path.join(task_root, "seqs")
    fasta_path = os.path.join(seq_root, TISP_FASTA)
    if os.path.exists(fasta_path):
        return fasta_path
    fasta_gz_path = os.path.join(seq_root, TISP_FASTA_GZ)
    if os.path.exists(fasta_gz_path):
        raise FileNotFoundError(
            "TISP requires an uncompressed FASTA for pyfaidx. "
            f"Found gzip FASTA at {fasta_gz_path} but missing {fasta_path}."
        )
    raise FileNotFoundError(f"Missing TISP FASTA at {fasta_path}.")


def _maybe_limit_dataset(dataset, limit: int):
    limit = int(limit)
    if limit <= 0:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))


def _default_train_sample_count(chrom_lens: dict[str, int], train_chroms: list[str]) -> int:
    total = 0
    for chrom in train_chroms:
        chrom_len = int(chrom_lens[chrom])
        if chrom_len < TISP_SEQUENCE_LENGTH:
            continue
        total += (chrom_len - TISP_SEQUENCE_LENGTH) // TISP_WINDOW_STEP + 1
    return total


def build_tisp_loaders(
    *,
    root: str,
    batch_size: int,
    sequence_length: int,
    num_train_samples: int = 0,
    num_valid_samples: int = 0,
    num_test_samples: int = 0,
    random_strand: bool = False,
) -> tuple[Any, DataLoader, DataLoader]:
    if int(sequence_length) != TISP_SEQUENCE_LENGTH:
        raise ValueError("TISP requires sequence_length=100000 to match the benchmark setup.")

    task_root = resolve_tisp_task_root(root)
    fasta_path = _resolve_tisp_fasta_path(task_root)
    target = _make_signal_target(task_root)

    genome = FastaGenome(fasta_path)
    chrom_lens = genome.get_chr_lens()
    target_chrom_lens = target.get_chrom_sizes()
    valid_chrom = _resolve_chrom_name(chrom_lens, TISP_VALID_HOLDOUT[0])
    test_holdout = [_resolve_chrom_name(chrom_lens, chrom) for chrom in TISP_TEST_HOLDOUT]

    missing_holdout = [
        chrom
        for chrom in (valid_chrom, *test_holdout)
        if _resolve_chrom_name_or_none(target_chrom_lens, chrom) is None
    ]
    if missing_holdout:
        raise ValueError(
            "TISP target tracks are missing required holdout chromosomes: "
            + ", ".join(sorted(missing_holdout))
        )

    holdout_chroms = {valid_chrom, *test_holdout}
    train_chroms = [
        chrom
        for chrom in chrom_lens
        if chrom not in holdout_chroms and _resolve_chrom_name_or_none(target_chrom_lens, chrom) is not None
    ]
    train_samples = int(num_train_samples)
    if train_samples <= 0:
        train_samples = _default_train_sample_count(chrom_lens, train_chroms)

    train_dataset = TISPRandomTrainDataset(
        genome,
        target,
        chrom_lens,
        train_chroms,
        num_samples=train_samples,
        sequence_length=TISP_SEQUENCE_LENGTH,
        seed=3,
        random_strand=bool(random_strand),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)

    valid_dataset = TISPChromosomeSlidingWindowDataset(
        genome,
        target,
        valid_chrom,
        int(chrom_lens[valid_chrom]),
    )
    valid_loader = DataLoader(
        _maybe_limit_dataset(valid_dataset, num_valid_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_datasets = [
        TISPChromosomeSlidingWindowDataset(
            genome,
            target,
            resolved_chrom,
            int(chrom_lens[resolved_chrom]),
        )
        for resolved_chrom in test_holdout
    ]
    test_loader = DataLoader(
        _maybe_limit_dataset(ConcatDataset(test_datasets), num_test_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, valid_loader, test_loader
