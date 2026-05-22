from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from src.genezip.tokenizer import FastSingleNucleotideTokenizer, SingleNucleotideTokenizer, SixMerTokenizer
from src.utils.tqdm_progress import TqdmProgress


class EmbeddingExtractor:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.outputs: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self.supports_boundaries = True

    def register(self, name: str, module: torch.nn.Module, tuple_output: bool = False) -> None:
        def _hook(_module, _inputs, output):
            if tuple_output:
                output = output[0] if isinstance(output, (tuple, list)) else output
            self.outputs[name] = output

        self._handles.append(module.register_forward_hook(_hook))

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def get_stage_modules(model: torch.nn.Module) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
    stage0 = getattr(model, "backbone", None)
    stage1 = None
    if stage0 is not None and not getattr(stage0, "is_innermost", False):
        stage1 = getattr(stage0, "main_network", None)
    return stage0, stage1


def setup_extractor(
    model: torch.nn.Module,
    embedding_layer: str,
    embedding_stream: str,
    model_cfg: Optional[Mapping[str, Any]] = None,
) -> EmbeddingExtractor:
    extractor = EmbeddingExtractor(model)
    stage0, stage1 = get_stage_modules(model)
    need_io = embedding_layer == "io"
    need_dc0 = embedding_layer in ("dc0", "dc0+dc1")
    need_dc1 = embedding_layer in ("dc1", "dc0+dc1")

    if need_io:
        if stage0 is None or not hasattr(stage0, "decoder"):
            raise ValueError("Model does not expose stage0 decoder for IO embeddings.")
        extractor.register("io", stage0.decoder)
        extractor.supports_boundaries = False

    if need_dc0 or need_dc1:
        if stage1 is None or not hasattr(stage1, "encoder"):
            if embedding_layer == "dc0+dc1":
                raise ValueError("Single-stage models do not support dc0+dc1 embeddings.")
            backbone = getattr(model, "backbone", None)
            if backbone is None:
                raise ValueError("Model does not expose backbone for single-stage embeddings.")
            key = "dc0" if need_dc0 else "dc1"
            extractor.register(key, backbone, tuple_output=True)
            extractor.supports_boundaries = False
            return extractor
        if need_dc0:
            if embedding_stream == "encoder":
                extractor.register("dc0", stage1.encoder)
            else:
                extractor.register("dc0", stage1.decoder)
        if need_dc1:
            if embedding_stream == "encoder":
                extractor.register("dc1", stage1.chunk_layer, tuple_output=True)
            else:
                extractor.register("dc1", stage1.main_network, tuple_output=True)

    return extractor


def run_model_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    routing_step: Optional[int],
):
    if "routing_step" in getattr(model.forward, "__code__").co_varnames:
        return model(input_ids, mask=mask, routing_step=routing_step)
    return model(input_ids, mask=mask)


def boundary_mask_to_spans(boundary_mask: torch.Tensor, total_len: int) -> Tuple[np.ndarray, np.ndarray]:
    mask = boundary_mask
    if mask.dim() == 2:
        mask = mask[0]
    mask = mask.to(torch.bool).cpu()
    idx = mask.nonzero(as_tuple=False).flatten().numpy()
    if idx.size == 0:
        idx = np.array([0], dtype=np.int64)
    elif idx[0] != 0:
        idx = np.concatenate(([0], idx))
    idx = np.unique(idx)
    ends = np.concatenate((idx[1:], [total_len]))
    return idx.astype(np.int64), ends.astype(np.int64)


def align_embeddings_with_spans(
    embeddings: torch.Tensor, starts: np.ndarray, ends: np.ndarray
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    n = min(embeddings.shape[0], starts.shape[0])
    if n == 0:
        return embeddings[:0], starts[:0], ends[:0]
    if embeddings.shape[0] != starts.shape[0]:
        embeddings = embeddings[:n]
        starts = starts[:n]
        ends = ends[:n]
    return embeddings, starts, ends


def validate_spans(name: str, starts: np.ndarray, ends: np.ndarray, total_len: int) -> None:
    if starts.size == 0:
        return
    if starts[0] < 0 or ends[-1] > total_len:
        raise ValueError(f"{name} spans out of bounds: [{starts[0]}, {ends[-1]}] vs {total_len}")
    if not np.all(starts[:-1] <= starts[1:]):
        raise ValueError(f"{name} starts not non-decreasing.")
    if not np.all(ends[:-1] <= ends[1:]):
        raise ValueError(f"{name} ends not non-decreasing.")
    if np.any(ends <= starts):
        raise ValueError(f"{name} has non-positive span length.")


def encode_sequence(tokenizer, seq: str) -> np.ndarray:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode([seq], add_bos=False, add_eos=False)[0]
        input_ids = encoded["input_ids"]
    else:
        input_ids = tokenizer(seq)
    if isinstance(input_ids, dict):
        input_ids = input_ids["input_ids"]
    return np.asarray(input_ids, dtype=np.int64)


def build_tokenizer(name: str):
    name = name.lower()
    if name == "fast":
        return FastSingleNucleotideTokenizer()
    if name == "singlenucleotide":
        return SingleNucleotideTokenizer(RC_augmentation=False)
    if name == "sixmer":
        return SixMerTokenizer()
    raise ValueError(f"Unknown tokenizer: {name}")


def iter_token_stream_windows_from_sequence(
    sequence: str,
    trunk_len: int,
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    extractor: EmbeddingExtractor,
    embedding_layer: str,
    routing_step: Optional[int] = None,
    encode_batch_size: int = 1,
    validate_spans_flag: bool = False,
    progress: Optional[TqdmProgress] = None,
    window_task: Optional[int] = None,
) -> Iterable[Tuple[torch.Tensor, np.ndarray, np.ndarray]]:
    seq_len = len(sequence)
    embedding_layer = str(embedding_layer).lower()
    if embedding_layer not in ("dc0", "dc1", "io"):
        raise ValueError("token extraction supports embedding_layer in {dc0, dc1, io}")
    supports_boundaries = extractor.supports_boundaries
    window_batch_size = int(encode_batch_size or 1)
    if window_batch_size < 1:
        window_batch_size = 1
    if progress is not None and window_task is not None:
        num_windows = int(math.ceil(seq_len / float(trunk_len)))
        progress.update(window_task, total=num_windows, completed=0, description="windows")

    def select_mask(mask: torch.Tensor, index: int) -> torch.Tensor:
        if mask.dim() == 1:
            return mask
        return mask[index]

    def iter_window_batches():
        batch: List[Tuple[int, int]] = []
        batch_len = None
        for start in range(0, seq_len, trunk_len):
            end = min(start + trunk_len, seq_len)
            length = end - start
            if batch_len is None:
                batch_len = length
            if length != batch_len or len(batch) >= window_batch_size:
                if batch:
                    yield batch
                batch = []
                batch_len = length
            batch.append((start, end))
        if batch:
            yield batch

    def process_batch(batch_windows: List[Tuple[int, int]]):
        if not batch_windows:
            return
        input_ids_list = []
        for start, end in batch_windows:
            chunk = sequence[start:end]
            input_ids_list.append(encode_sequence(tokenizer, chunk))

        lengths = {len(ids) for ids in input_ids_list}
        if len(lengths) != 1:
            for start, end in batch_windows:
                yield from process_batch([(start, end)])
            return

        input_ids = torch.tensor(np.stack(input_ids_list), dtype=torch.long, device=device)
        mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)

        extractor.clear()
        with torch.inference_mode():
            outputs = run_model_forward(model, input_ids, mask, routing_step)

        if embedding_layer == "dc0":
            emb_batch = extractor.outputs.get("dc0")
            if emb_batch is None:
                raise RuntimeError("DC0 embeddings not captured.")
        elif embedding_layer == "dc1":
            emb_batch = extractor.outputs.get("dc1")
            if emb_batch is None:
                raise RuntimeError("DC1 embeddings not captured.")
        else:
            emb_batch = extractor.outputs.get("io")
            if emb_batch is None:
                raise RuntimeError("IO embeddings not captured.")

        if emb_batch.dim() == 2:
            emb_batch = emb_batch.unsqueeze(0)
        emb_batch = emb_batch.detach().cpu()

        if supports_boundaries:
            bpred_outputs = getattr(outputs, "bpred_output", None)
            if bpred_outputs is None:
                raise RuntimeError("Model output missing bpred_output for boundary extraction.")

            if len(bpred_outputs) < 1:
                raise RuntimeError("Model output missing DC0 boundary predictions.")
            if embedding_layer == "dc1" and len(bpred_outputs) < 2:
                raise RuntimeError("Model output missing DC1 boundary predictions.")

            dc0_mask_batch = bpred_outputs[0].boundary_mask.detach()
            if dc0_mask_batch.is_cuda:
                dc0_mask_batch = dc0_mask_batch.cpu()
            dc1_mask_batch = None
            if embedding_layer == "dc1":
                dc1_mask_batch = bpred_outputs[1].boundary_mask.detach()
                if dc1_mask_batch.is_cuda:
                    dc1_mask_batch = dc1_mask_batch.cpu()

            for batch_idx, (start, end) in enumerate(batch_windows):
                dc0_mask = select_mask(dc0_mask_batch, batch_idx)
                dc0_starts, dc0_ends = boundary_mask_to_spans(dc0_mask, end - start)

                if embedding_layer == "dc0":
                    emb = emb_batch[batch_idx]
                    starts = dc0_starts + start
                    ends = dc0_ends + start
                else:
                    dc1_mask = select_mask(dc1_mask_batch, batch_idx)
                    dc1_starts_idx, dc1_ends_idx = boundary_mask_to_spans(
                        dc1_mask, int(len(dc0_starts))
                    )
                    dc0_starts_global = dc0_starts + start
                    total_end_bp = start + (end - start)
                    dc0_bounds = np.concatenate([dc0_starts_global, [total_end_bp]])
                    starts = dc0_bounds[dc1_starts_idx]
                    ends = dc0_bounds[dc1_ends_idx]
                    emb = emb_batch[batch_idx]

                if validate_spans_flag:
                    validate_spans(embedding_layer, starts, ends, start + (end - start))

                emb, starts, ends = align_embeddings_with_spans(emb, starts, ends)
                if emb.numel() == 0:
                    continue
                yield emb, starts, ends
        else:
            for batch_idx, (start, end) in enumerate(batch_windows):
                emb = emb_batch[batch_idx]
                span_len = end - start
                starts = np.arange(span_len, dtype=np.int64) + start
                ends = starts + 1
                if validate_spans_flag:
                    validate_spans(embedding_layer, starts, ends, start + (end - start))

                emb, starts, ends = align_embeddings_with_spans(emb, starts, ends)
                if emb.numel() == 0:
                    continue
                yield emb, starts, ends

        if progress is not None and window_task is not None:
            progress.advance(window_task, advance=len(batch_windows))

    for batch_windows in iter_window_batches():
        yield from process_batch(batch_windows)


def extract_token_stream_from_sequence(
    sequence: str,
    trunk_len: int,
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    extractor: EmbeddingExtractor,
    embedding_layer: str,
    routing_step: Optional[int] = None,
    encode_batch_size: int = 1,
    validate_spans_flag: bool = False,
    progress: Optional[TqdmProgress] = None,
    window_task: Optional[int] = None,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Dict[str, float]]:
    tokens: List[torch.Tensor] = []
    starts_all: List[np.ndarray] = []
    ends_all: List[np.ndarray] = []
    stats: Dict[str, float] = {}
    token_lengths: List[np.ndarray] = []

    for emb, starts, ends in iter_token_stream_windows_from_sequence(
        sequence,
        trunk_len,
        tokenizer,
        model,
        device,
        extractor,
        embedding_layer,
        routing_step=routing_step,
        encode_batch_size=encode_batch_size,
        validate_spans_flag=validate_spans_flag,
        progress=progress,
        window_task=window_task,
    ):
        tokens.append(emb)
        starts_all.append(starts)
        ends_all.append(ends)
        token_lengths.append((ends - starts).astype(np.int64))

    if not tokens:
        return (
            torch.empty((0, 0), dtype=torch.float32),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            {},
        )

    tokens_cat = torch.cat(tokens, dim=0)
    start_bp = np.concatenate(starts_all, axis=0)
    end_bp = np.concatenate(ends_all, axis=0)

    if token_lengths:
        lengths = np.concatenate(token_lengths, axis=0)
        stats["token_count"] = float(tokens_cat.shape[0])
        stats["avg_bpt"] = float(lengths.mean())
        stats["span_min"] = float(lengths.min())
        stats["span_med"] = float(np.median(lengths))
        stats["span_max"] = float(lengths.max())

    return tokens_cat, start_bp, end_bp, stats
