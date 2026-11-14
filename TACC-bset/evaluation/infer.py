"""Debug-oriented inference script that completes after logging diagnostics.

Compared with ``evaluation/infer.py``, this variant prints rich inference-time
diagnostics and completes normally after the debug logs are emitted.
"""

import json
import logging
import math
import os
import time
from collections import Counter
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    FeatureSchema,
    NUM_HOUR_BUCKETS,
    NUM_SEQ_TIME_STATS,
    NUM_TIME_BUCKETS,
    NUM_WEEKDAY_BUCKETS,
    PCVRParquetDataset,
)
from model import ModelInput, PCVRHyFormer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


DEBUG_STOP_AFTER_SAVE = False
DEBUG_SAMPLE_LIMIT = 50
DEBUG_TOP_ZERO_FEATURES = 5
DEBUG_PROGRESS_EVERY = 100
DEBUG_THRESHOLD_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9]
DEBUG_TIME_BUCKET_TOP_K = 8
DEBUG_TIME_BUCKET_DIST_TOP_K = 16
DEBUG_SUCCESS_FULL_BUCKET_PRINT_LIMIT = 64
DEBUG_PROB_BINS = [
    (0.0, 0.001),
    (0.001, 0.005),
    (0.005, 0.01),
    (0.01, 0.05),
    (0.05, 0.1),
    (0.1, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 0.9),
    (0.9, 0.95),
    (0.95, 0.99),
    (0.99, 1.0),
]


_FALLBACK_MODEL_CFG = {
    "d_model": 64,
    "emb_dim": 64,
    "num_queries": 1,
    "num_hyformer_blocks": 2,
    "num_heads": 4,
    "seq_encoder_type": "transformer",
    "hidden_mult": 4,
    "dropout_rate": 0.01,
    "seq_top_k": 50,
    "seq_causal": False,
    "action_num": 1,
    "num_time_buckets": NUM_TIME_BUCKETS,
    "num_weekday_buckets": 0,
    "num_hour_buckets": 0,
    "seq_time_stats_dim": 0,
    "rank_mixer_mode": "full",
    "use_rope": False,
    "rope_base": 10000.0,
    "emb_skip_threshold": 0,
    "seq_id_threshold": 10000,
    "ns_tokenizer_type": "rankmixer",
    "user_ns_tokens": 0,
    "item_ns_tokens": 0,
    "dense_tokenizer_type": "single",
    "query_generator_type": "baseline",
    "target_attn_type": "none",
    "ui_shortcut_type": "none",
    "ui_shortcut_alpha_init": 1.0,
    "first_decoded_shortcut_type": "none",
    "first_decoded_detach": False,
    "first_decoded_alpha_init": 0.01,
    "target_attn_alpha_init": 0.1,
}

_FALLBACK_SEQ_MAX_LENS = "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16
_DEFAULT_INFER_BATCH_SIZE_CAP = 64
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(","):
        k, v = pair.split(":")
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    train_config_path = os.path.join(model_dir, "train_config.json")
    if os.path.exists(train_config_path):
        with open(train_config_path, "r") as f:
            cfg = json.load(f)
        logging.info("Loaded train_config from %s", train_config_path)
        return cfg
    logging.warning(
        "train_config.json not found in %s, falling back to hardcoded defaults. "
        "Shape mismatch may occur if training used non-default hyperparameters.",
        model_dir,
    )
    return {}


def resolve_infer_batch_size(train_config: Dict[str, Any]) -> int:
    train_batch_size = int(train_config.get("batch_size", _FALLBACK_BATCH_SIZE))
    batch_size_cap = int(
        os.environ.get("INFER_BATCH_SIZE_CAP", _DEFAULT_INFER_BATCH_SIZE_CAP)
    )
    if batch_size_cap > 0 and train_batch_size > batch_size_cap:
        logging.info(
            "Capping inference batch_size from %s to %s to reduce CUDA memory pressure",
            train_batch_size,
            batch_size_cap,
        )
        return batch_size_cap
    return train_batch_size


def get_infer_autocast_context(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        amp_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        logging.info("Using CUDA autocast for inference with dtype=%s", amp_dtype)
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from ``train_config``; missing keys fall
    back to ``_FALLBACK_MODEL_CFG``.

    Special handling for ``num_time_buckets``: it is not exposed on the CLI
    as an independent hyperparameter; the bucket count is determined by
    ``NUM_TIME_BUCKETS`` (default boundaries). Resolution order:

      1) ``train_config`` contains ``num_time_buckets`` directly (legacy ckpt)
         -> use that value;
      2) ``train_config`` contains ``use_time_buckets`` (new-style training)
         -> use ``NUM_TIME_BUCKETS`` if enabled, else 0;
      3) neither is present -> fall back to ``_FALLBACK_MODEL_CFG[...]``.
    """
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == "num_time_buckets":
            if "num_time_buckets" in train_config:
                cfg[key] = train_config["num_time_buckets"]
            elif "use_time_buckets" in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config["use_time_buckets"] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
            continue
        if key == "num_weekday_buckets":
            if "num_weekday_buckets" in train_config:
                cfg[key] = train_config["num_weekday_buckets"]
            elif "use_calendar_time_features" in train_config:
                cfg[key] = (
                    NUM_WEEKDAY_BUCKETS if train_config["use_calendar_time_features"] else 0
                )
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
            continue
        if key == "num_hour_buckets":
            if "num_hour_buckets" in train_config:
                cfg[key] = train_config["num_hour_buckets"]
            elif "use_calendar_time_features" in train_config:
                cfg[key] = NUM_HOUR_BUCKETS if train_config["use_calendar_time_features"] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
            continue
        if key == "seq_time_stats_dim":
            if "seq_time_stats_dim" in train_config:
                cfg[key] = train_config["seq_time_stats_dim"]
            elif "use_seq_time_stats" in train_config:
                cfg[key] = NUM_SEQ_TIME_STATS if train_config["use_seq_time_stats"] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    "train_config missing both 'seq_time_stats_dim' and "
                    "'use_seq_time_stats', using fallback = %s",
                    cfg[key],
                )
            continue
        cfg[key] = train_config.get(key, _FALLBACK_MODEL_CFG[key])
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = "cpu",
) -> PCVRHyFormer:
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info("Loading NS groups from %s", ns_groups_json)
        with open(ns_groups_json, "r") as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        user_ns_groups = [
            [user_fid_to_idx[f] for f in fids]
            for fids in ns_groups_cfg["user_ns_groups"].values()
        ]
        item_ns_groups = [
            [item_fid_to_idx[f] for f in fids]
            for fids in ns_groups_cfg["item_ns_groups"].values()
        ]
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes
    )
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes
    )

    logging.info("Building PCVRHyFormer with cfg: %s", model_cfg)
    return PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        user_dense_entries=dataset.user_dense_schema.entries,
        item_dense_entries=dataset.item_dense_schema.entries,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    ).to(device)


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Unsupported checkpoint payload type: {type(state_dict)!r}. "
            "Expected a state_dict or a dict containing 'state_dict'."
        )

# Checkpoints saved from ``torch.compile(model)`` carry an ``_orig_mod.``
    # prefix on every parameter name. Strip it here so inference can rebuild
    # the uncompiled model and still load weights strictly.
    orig_mod_prefix = '_orig_mod.'
    if state_dict and all(
        isinstance(k, str) and k.startswith(orig_mod_prefix)
        for k in state_dict.keys()
    ):
        logging.info("Detected torch.compile checkpoint; stripping '_orig_mod.' prefix")
        state_dict = {
            k[len(orig_mod_prefix):]: v
            for k, v in state_dict.items()
        }

    model_state = model.state_dict()
    if 'ui_shortcut_alpha' in model_state and 'ui_shortcut_alpha' not in state_dict:
        logging.warning(
            "Checkpoint is missing ui_shortcut_alpha; using initialized value "
            "from model construction."
        )
        state_dict = dict(state_dict)
        state_dict['ui_shortcut_alpha'] = model_state['ui_shortcut_alpha']

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the model "
            "constructed by build_model does NOT match the checkpoint."
        )
        raise exc


def get_ckpt_path() -> Optional[str]:
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(batch: Dict[str, Any], device: str) -> ModelInput:
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch["_seq_domains"]
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    seq_time_stats: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f"{domain}_len"]
        bsz, _, seq_len = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f"{domain}_time_bucket",
            torch.zeros(bsz, seq_len, dtype=torch.long, device=device),
        )
        seq_time_stats[domain] = device_batch.get(
            f"{domain}_time_stats",
            torch.zeros(bsz, 0, dtype=torch.float32, device=device),
        )

    return ModelInput(
        user_int_feats=device_batch["user_int_feats"],
        item_int_feats=device_batch["item_int_feats"],
        user_dense_feats=device_batch["user_dense_feats"],
        item_dense_feats=device_batch["item_dense_feats"],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        seq_time_stats=seq_time_stats,
        timestamp_weekday=device_batch["timestamp_weekday"],
        timestamp_hour=device_batch["timestamp_hour"],
    )


def _to_float(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _summary_stats(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"min": 0.0, "p05": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    q = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "min": _to_float(np.min(arr)),
        "p05": _to_float(q[0]),
        "p25": _to_float(q[1]),
        "p50": _to_float(q[2]),
        "p75": _to_float(q[3]),
        "p95": _to_float(q[4]),
        "max": _to_float(np.max(arr)),
        "mean": _to_float(np.mean(arr)),
        "std": _to_float(np.std(arr)),
    }


def _shape_stats(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"skewness": 0.0, "kurtosis": 0.0, "cv": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    centered = arr - mean
    if std < 1e-12:
        return {"skewness": 0.0, "kurtosis": 0.0, "cv": 0.0}
    skewness = float(np.mean((centered / std) ** 3))
    kurtosis = float(np.mean((centered / std) ** 4) - 3.0)
    cv = float(std / mean) if abs(mean) > 1e-12 else 0.0
    return {"skewness": skewness, "kurtosis": kurtosis, "cv": cv}


def _log_shape(prefix: str, stats: Dict[str, float]) -> None:
    skew = stats["skewness"]
    kurt = stats["kurtosis"]
    cv = stats["cv"]
    skew_desc = "symmetric" if abs(skew) < 0.5 else ("right-skewed" if skew > 0 else "left-skewed")
    kurt_desc = "normal-tailed" if abs(kurt) < 1.0 else ("heavy-tailed" if kurt > 0 else "light-tailed")
    logging.info(
        "%s_shape: skewness=%.4f(%s), kurtosis=%.4f(%s), cv=%.4f",
        prefix,
        skew,
        skew_desc,
        kurt,
        kurt_desc,
        cv,
    )


def _seq_bin_label(length: int) -> str:
    if length <= 0:
        return "<=0"
    if length <= 1:
        return "1-1"
    if length <= 5:
        return "2-5"
    if length <= 10:
        return "6-10"
    if length <= 25:
        return "11-25"
    if length <= 50:
        return "26-50"
    if length <= 100:
        return "51-100"
    if length <= 200:
        return "101-200"
    return "201-512"


def _coverage_bin(value: float) -> str:
    if value < 0.1:
        return "0.00-0.10"
    if value < 0.2:
        return "0.10-0.20"
    if value < 0.3:
        return "0.20-0.30"
    if value < 0.5:
        return "0.30-0.50"
    if value < 0.7:
        return "0.50-0.70"
    if value < 0.9:
        return "0.70-0.90"
    if value < 0.95:
        return "0.90-0.95"
    if value < 0.99:
        return "0.95-0.99"
    return "0.99-1.00"


def _top_counts(arr: np.ndarray, top_k: int) -> List[Tuple[int, int]]:
    if arr.size == 0:
        return []
    counter = Counter(int(x) for x in arr.tolist() if int(x) > 0)
    return counter.most_common(top_k)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0
    return _to_float(np.corrcoef(x, y)[0, 1])


def _format_top_counts(pairs: List[Tuple[int, int]]) -> str:
    if not pairs:
        return "none"
    return " | ".join(f"{k}:{v}" for k, v in pairs)


def _time_bucket_bin(bucket_id: int) -> str:
    if bucket_id <= 0:
        return "pad_or_missing"
    if bucket_id <= 8:
        return "very_recent"
    if bucket_id <= 24:
        return "recent"
    if bucket_id <= 40:
        return "mid"
    if bucket_id <= 56:
        return "old"
    return "very_old"


def _missing_pattern_key(seq_lens: Dict[str, int]) -> str:
    parts = []
    for domain in sorted(seq_lens.keys()):
        parts.append(f"{domain}:{'1' if seq_lens[domain] > 0 else '0'}")
    return "|".join(parts)


def _truncation_rate(seq_lens: np.ndarray, max_len: int) -> float:
    if seq_lens.size == 0:
        return 0.0
    return _to_float(np.mean(seq_lens >= max_len))


def _bucket_prob_summary(bucket_ids: np.ndarray, probs: np.ndarray) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = {}
    for bid, prob in zip(bucket_ids.tolist(), probs.tolist()):
        grouped.setdefault(_time_bucket_bin(int(bid)), []).append(float(prob))
    return {
        key: {
            "mean_prob": _to_float(np.mean(vals)),
            "count": int(len(vals)),
        }
        for key, vals in grouped.items()
    }


def _bucket_distribution_summary(bucket_ids: np.ndarray, top_k: int) -> Dict[str, Any]:
    arr = np.asarray(bucket_ids, dtype=np.int64)
    total = int(arr.size)
    if total == 0:
        return {
            "total": 0,
            "nonzero_total": 0,
            "zero_count": 0,
            "zero_rate": 0.0,
            "unique_nonzero_buckets": 0,
            "top_buckets": [],
        }
    zero_count = int(np.sum(arr <= 0))
    nonzero = arr[arr > 0]
    top_pairs = _top_counts(nonzero, top_k)
    return {
        "total": total,
        "nonzero_total": int(nonzero.size),
        "zero_count": zero_count,
        "zero_rate": _to_float(zero_count / max(total, 1)),
        "unique_nonzero_buckets": int(len(np.unique(nonzero))) if nonzero.size > 0 else 0,
        "top_buckets": [
            {
                "bucket": int(bucket),
                "count": int(count),
                "rate": _to_float(count / max(total, 1)),
                "nonzero_rate": _to_float(count / max(int(nonzero.size), 1)),
            }
            for bucket, count in top_pairs
        ],
    }


def _bucket_full_distribution(bucket_ids: np.ndarray) -> Dict[str, Dict[str, float]]:
    arr = np.asarray(bucket_ids, dtype=np.int64)
    total = int(arr.size)
    if total == 0:
        return {}
    counter = Counter(int(x) for x in arr.tolist())
    return {
        str(bucket): {
            "count": int(count),
            "rate": _to_float(count / max(total, 1)),
        }
        for bucket, count in sorted(counter.items())
    }


def _bucket_quantiles(bucket_ids: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(bucket_ids, dtype=np.int64)
    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        return {
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "min": _to_float(np.min(nonzero)),
        "p25": _to_float(np.percentile(nonzero, 25)),
        "p50": _to_float(np.percentile(nonzero, 50)),
        "p75": _to_float(np.percentile(nonzero, 75)),
        "p90": _to_float(np.percentile(nonzero, 90)),
        "p95": _to_float(np.percentile(nonzero, 95)),
        "p99": _to_float(np.percentile(nonzero, 99)),
        "max": _to_float(np.max(nonzero)),
        "mean": _to_float(np.mean(nonzero)),
    }


def _bucket_coverage_summary(bucket_ids: np.ndarray, max_bucket_id: int) -> Dict[str, Any]:
    arr = np.asarray(bucket_ids, dtype=np.int64)
    nonzero = arr[arr > 0]
    used = sorted({int(x) for x in nonzero.tolist()})
    if max_bucket_id <= 0:
        return {
            "configured_bucket_count": 0,
            "used_bucket_count": int(len(used)),
            "bucket_coverage_rate": 0.0,
            "missing_bucket_count": 0,
            "missing_buckets_head": [],
            "used_buckets_head": used[:DEBUG_SUCCESS_FULL_BUCKET_PRINT_LIMIT],
        }
    full = set(range(1, max_bucket_id + 1))
    used_set = set(used)
    missing = sorted(full - used_set)
    return {
        "configured_bucket_count": int(max_bucket_id),
        "used_bucket_count": int(len(used)),
        "bucket_coverage_rate": _to_float(len(used) / max(max_bucket_id, 1)),
        "missing_bucket_count": int(len(missing)),
        "missing_buckets_head": missing[:DEBUG_SUCCESS_FULL_BUCKET_PRINT_LIMIT],
        "used_buckets_head": used[:DEBUG_SUCCESS_FULL_BUCKET_PRINT_LIMIT],
    }


def _bucket_cross_summary(
    bucket_ids: np.ndarray,
    group_labels: List[str],
    probs: np.ndarray,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for bucket_id, label, prob in zip(bucket_ids.tolist(), group_labels, probs.tolist()):
        bucket_key = _time_bucket_bin(int(bucket_id))
        grouped.setdefault(bucket_key, {}).setdefault(label, []).append(float(prob))
    return {
        bucket_key: {
            label: {
                "mean_prob": _to_float(np.mean(vals)),
                "count": int(len(vals)),
            }
            for label, vals in label_map.items()
        }
        for bucket_key, label_map in grouped.items()
    }
def _log_json(prefix: str, payload: Any) -> None:
    logging.info("%s: %s", prefix, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _log_samples(prefix: str, samples: Dict[str, List[Dict[str, Any]]]) -> None:
    for bucket, rows in samples.items():
        for row in rows:
            payload = {"bucket": bucket, **row}
            logging.info("%s: %s", prefix, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _append_samples(bucket: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> None:
    needed = DEBUG_SAMPLE_LIMIT - len(bucket)
    if needed > 0:
        bucket.extend(candidates[:needed])


def _write_status_files(result_dir: str, status: str, message: str) -> None:
    logging.info("status_result: %s", json.dumps({"status": status, "message": message}, ensure_ascii=False, sort_keys=True))
    logging.info("status_meta: %s", json.dumps({"status": status, "message": message}, ensure_ascii=False, sort_keys=True))


def main() -> None:
    model_dir = os.environ.get("MODEL_OUTPUT_PATH")
    data_dir = os.environ.get("EVAL_DATA_PATH")
    result_dir = os.environ.get("EVAL_RESULT_PATH")
    if not model_dir or not data_dir or not result_dir:
        raise ValueError("MODEL_OUTPUT_PATH, EVAL_DATA_PATH, and EVAL_RESULT_PATH must all be set.")

    os.makedirs(result_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    schema_path = os.path.join(model_dir, "schema.json")
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, "schema.json")
    logging.info("Using schema: %s", schema_path)

    train_config = load_train_config(model_dir)
    sml_str = train_config.get("seq_max_lens", _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info("seq_max_lens: %s", seq_max_lens)

    batch_size = resolve_infer_batch_size(train_config)
    num_workers = int(train_config.get("num_workers", _FALLBACK_NUM_WORKERS))
    debug_use_labels = os.environ.get("INFER_DEBUG_USE_LABELS", "0") == "1"
    logging.info(
        "Inference debug analysis enabled: use_labels=%s, sample_limit=%s, ablation=%s, ablation_samples=%s",
        debug_use_labels,
        DEBUG_SAMPLE_LIMIT,
        False,
        0,
    )

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    total_test_samples = test_dataset.num_rows
    logging.info(
        "Total test samples: %s, user_dense_dim=%s",
        total_test_samples,
        test_dataset.user_dense_schema.total_dim,
    )

    model_cfg = resolve_model_cfg(train_config)
    ns_groups_json = train_config.get("ns_groups_json", None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}.")
    logging.info("Loading checkpoint from %s", ckpt_path)
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
    )

    all_probs: List[float] = []
    all_logits: List[float] = []
    all_emb_norms: List[float] = []
    all_user_ids: List[str] = []
    weekday_values: List[int] = []
    hour_values: List[int] = []
    coverage_values: List[float] = []
    seq_len_rows: Dict[str, List[int]] = {d: [] for d in test_dataset.seq_domains}
    seq_time_bucket_rows: Dict[str, List[int]] = {d: [] for d in test_dataset.seq_domains}
    seq_nonzero_token_rows: Dict[str, List[int]] = {d: [] for d in test_dataset.seq_domains}
    seq_effective_cov_rows: Dict[str, List[float]] = {d: [] for d in test_dataset.seq_domains}
    missing_pattern_counts: Counter[str] = Counter()
    missing_pattern_prob_sums: Dict[str, float] = {}
    user_int_active: List[int] = []
    item_int_active: List[int] = []
    user_dense_active: List[int] = []
    user_int_zero_counts = np.zeros(test_dataset.user_int_schema.total_dim, dtype=np.int64)
    item_int_zero_counts = np.zeros(test_dataset.item_int_schema.total_dim, dtype=np.int64)
    user_dense_nonzero_counts = np.zeros(test_dataset.user_dense_schema.total_dim, dtype=np.int64)

    sample_buckets: Dict[str, List[Dict[str, Any]]] = {
        "high_conf_positive": [],
        "high_conf_negative": [],
        "uncertain_near_0_5": [],
        "low_feature_coverage": [],
    }

    start_time = time.time()
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get("user_id", [])

            with get_infer_autocast_context(device):
                logits, embedding = model.predict(model_input)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)

            logits_np = logits.cpu().numpy()
            probs_np = probs.cpu().numpy()
            emb_norm_np = embedding.norm(dim=-1).cpu().numpy()
            user_int_np = batch["user_int_feats"].numpy()
            item_int_np = batch["item_int_feats"].numpy()
            user_dense_np = batch["user_dense_feats"].numpy()
            weekday_np = batch["timestamp_weekday"].numpy()
            hour_np = batch["timestamp_hour"].numpy()

            batch_seq_lens: Dict[str, np.ndarray] = {}
            for domain in batch["_seq_domains"]:
                lengths_np = batch[f"{domain}_len"].numpy()
                batch_seq_lens[domain] = lengths_np
                seq_len_rows[domain].extend(lengths_np.tolist())

                seq_np = batch[domain].numpy()
                seq_nonzero_token_rows[domain].extend(
                    np.count_nonzero(seq_np, axis=(1, 2)).tolist()
                )
                denom = max(1, seq_np.shape[1] * seq_np.shape[2])
                seq_effective_cov_rows[domain].extend(
                    (np.count_nonzero(seq_np, axis=(1, 2)).astype(np.float64) / denom).tolist()
                )
                time_bucket_np = batch[f"{domain}_time_bucket"].numpy()
                valid_bucket_mask = time_bucket_np > 0
                if valid_bucket_mask.any(axis=1).size:
                    recent_bucket = np.where(valid_bucket_mask, time_bucket_np, np.iinfo(np.int64).max).min(axis=1)
                    recent_bucket = np.where(valid_bucket_mask.any(axis=1), recent_bucket, 0)
                else:
                    recent_bucket = np.zeros(lengths_np.shape[0], dtype=np.int64)
                seq_time_bucket_rows[domain].extend(recent_bucket.tolist())

            user_int_active_np = np.count_nonzero(user_int_np, axis=1)
            item_int_active_np = np.count_nonzero(item_int_np, axis=1)
            user_dense_active_np = np.count_nonzero(np.abs(user_dense_np) > 1e-12, axis=1)
            coverage_np = (
                user_int_active_np
                + item_int_active_np
                + user_dense_active_np
            ).astype(np.float64)
            coverage_np /= max(
                1,
                user_int_np.shape[1] + item_int_np.shape[1] + user_dense_np.shape[1],
            )

            user_int_zero_counts += np.count_nonzero(user_int_np == 0, axis=0)
            item_int_zero_counts += np.count_nonzero(item_int_np == 0, axis=0)
            if user_dense_np.shape[1] > 0:
                user_dense_nonzero_counts += np.count_nonzero(np.abs(user_dense_np) > 1e-12, axis=0)

            all_probs.extend(probs_np.tolist())
            all_logits.extend(logits_np.tolist())
            all_emb_norms.extend(emb_norm_np.tolist())
            all_user_ids.extend(user_ids)
            weekday_values.extend(weekday_np.tolist())
            hour_values.extend(hour_np.tolist())
            coverage_values.extend(coverage_np.tolist())
            user_int_active.extend(user_int_active_np.tolist())
            item_int_active.extend(item_int_active_np.tolist())
            user_dense_active.extend(user_dense_active_np.tolist())

            pos_candidates: List[Dict[str, Any]] = []
            neg_candidates: List[Dict[str, Any]] = []
            uncertain_candidates: List[Dict[str, Any]] = []
            coverage_candidates: List[Dict[str, Any]] = []
            for i, uid in enumerate(user_ids):
                row = {
                    "user_id": uid,
                    "prob": float(probs_np[i]),
                    "logit": float(logits_np[i]),
                    "coverage": float(coverage_np[i]),
                    "weekday": int(weekday_np[i]),
                    "hour": int(hour_np[i]),
                    "seq_lens": {d: int(batch_seq_lens[d][i]) for d in batch["_seq_domains"]},
                    "seq_nonzero_tokens": {d: int(seq_nonzero_token_rows[d][-len(user_ids) + i]) for d in batch["_seq_domains"]},
                    "seq_recent_bucket": {d: int(seq_time_bucket_rows[d][-len(user_ids) + i]) for d in batch["_seq_domains"]},
                }
                missing_key = _missing_pattern_key(row["seq_lens"])
                missing_pattern_counts[missing_key] += 1
                missing_pattern_prob_sums[missing_key] = missing_pattern_prob_sums.get(missing_key, 0.0) + float(probs_np[i])
                if probs_np[i] >= 0.95:
                    pos_candidates.append(row)
                if probs_np[i] <= 0.01:
                    neg_candidates.append(row)
                if abs(probs_np[i] - 0.5) <= 0.02:
                    uncertain_candidates.append(row)
                if coverage_np[i] <= 0.2:
                    coverage_candidates.append(row)

            _append_samples(sample_buckets["high_conf_positive"], pos_candidates)
            _append_samples(sample_buckets["high_conf_negative"], neg_candidates)
            _append_samples(sample_buckets["uncertain_near_0_5"], uncertain_candidates)
            _append_samples(sample_buckets["low_feature_coverage"], coverage_candidates)

            if (batch_idx + 1) % DEBUG_PROGRESS_EVERY == 0:
                logging.info("Processed %s samples", (batch_idx + 1) * batch_size)

    elapsed = time.time() - start_time
    logging.info("Inference complete: %s predictions", len(all_probs))

    predictions = {"predictions": dict(zip(all_user_ids, all_probs))}
    output_path = os.path.join(result_dir, "predictions.json")
    with open(output_path, "w") as f:
        json.dump(predictions, f)
    logging.info("Saved %s predictions to %s", len(all_probs), output_path)

    probs_arr = np.asarray(all_probs, dtype=np.float64)
    logits_arr = np.asarray(all_logits, dtype=np.float64)
    emb_norm_arr = np.asarray(all_emb_norms, dtype=np.float64)
    coverage_arr = np.asarray(coverage_values, dtype=np.float64)
    confidence_arr = np.maximum(probs_arr, 1.0 - probs_arr)
    entropy_arr = -(
        probs_arr * np.log(np.clip(probs_arr, 1e-12, 1.0))
        + (1.0 - probs_arr) * np.log(np.clip(1.0 - probs_arr, 1e-12, 1.0))
    )

    prob_stats = _summary_stats(probs_arr)
    logit_stats = _summary_stats(logits_arr)
    confidence_stats = _summary_stats(confidence_arr)
    entropy_stats = _summary_stats(entropy_arr)
    emb_norm_stats = _summary_stats(emb_norm_arr)

    logging.info(
        "Inference debug summary: prob_mean=%s, p05=%s, p50=%s, p95=%s, nonfinite=%s, diagnostics=logged",
        prob_stats["mean"],
        prob_stats["p05"],
        prob_stats["p50"],
        prob_stats["p95"],
        int(np.size(probs_arr) - np.count_nonzero(np.isfinite(probs_arr))),
    )
    logging.info("elapsed=%.1fs, throughput=%d samples/s", elapsed, int(len(all_probs) / max(elapsed, 1e-12)))
    logging.info(
        "logit: min=%.4f, p05=%.4f, p50=%.4f, p95=%.4f, max=%.4f, mean=%.4f, std=%.4f",
        logit_stats["min"],
        logit_stats["p05"],
        logit_stats["p50"],
        logit_stats["p95"],
        logit_stats["max"],
        logit_stats["mean"],
        logit_stats["std"],
    )

    logging.info("prob_bins:")
    prob_bin_summary: Dict[str, int] = {}
    for lo, hi in DEBUG_PROB_BINS:
        if hi < 1.0:
            mask = (probs_arr >= lo) & (probs_arr < hi)
            label = f"[{lo},{hi})"
        else:
            mask = (probs_arr >= lo) & (probs_arr <= hi)
            label = f"[{lo},{hi}]"
        count = int(mask.sum())
        prob_bin_summary[label] = count
        logging.info("%s: %d (%.4f%%)", label, count, 100.0 * count / max(len(probs_arr), 1))

    logging.info(
        "confidence: p25=%.4f, p50=%.4f, p75=%.4f, p95=%.4f, p99=%.4f",
        confidence_stats["p25"],
        confidence_stats["p50"],
        confidence_stats["p75"],
        confidence_stats["p95"],
        _to_float(np.quantile(confidence_arr, 0.99)) if confidence_arr.size else 0.0,
    )
    for level in DEBUG_THRESHOLD_LEVELS:
        count = int((probs_arr >= level).sum())
        logging.info(
            "threshold>=%s: %d samples (%.4f%%)",
            level,
            count,
            100.0 * count / max(len(probs_arr), 1),
        )

    logging.info(
        "entropy: p25=%.4f, p50=%.4f, p75=%.4f (max=ln2≈0.693)",
        entropy_stats["p25"],
        entropy_stats["p50"],
        entropy_stats["p75"],
    )
    logging.info(
        "calendar: invalid_ts_rate=%.4f%%",
        100.0 * sum(1 for x in weekday_values if x == 0) / max(len(weekday_values), 1),
    )

    weekday_hist = Counter(int(x) for x in weekday_values if int(x) > 0)
    hour_hist = Counter(int(x) for x in hour_values if int(x) > 0)
    logging.info("weekday_hist: %s", ", ".join(f"{k}={v}" for k, v in sorted(weekday_hist.items())))
    logging.info("hour_hist: %s", ", ".join(f"{k}={v}" for k, v in sorted(hour_hist.items())))

    top_user_zero = np.argsort(-user_int_zero_counts)[:DEBUG_TOP_ZERO_FEATURES]
    for idx in top_user_zero.tolist():
        logging.info(
            "feature user_int:%s zero for %.2f%% rows",
            idx,
            100.0 * user_int_zero_counts[idx] / max(len(probs_arr), 1),
        )
    top_item_zero = np.argsort(-item_int_zero_counts)[:DEBUG_TOP_ZERO_FEATURES]
    for idx in top_item_zero.tolist():
        logging.info(
            "feature item_int:%s zero for %.2f%% rows",
            idx,
            100.0 * item_int_zero_counts[idx] / max(len(probs_arr), 1),
        )

    def _active_ratio(zero_counts: np.ndarray) -> int:
        active_ratio = 1.0 - zero_counts / max(len(probs_arr), 1)
        return int(np.count_nonzero(active_ratio > 0.01))

    logging.info(
        "feature_group[user_int]: %d/%d features with >1%% active rate",
        _active_ratio(user_int_zero_counts),
        len(user_int_zero_counts),
    )
    logging.info(
        "feature_group[item_int]: %d/%d features with >1%% active rate",
        _active_ratio(item_int_zero_counts),
        len(item_int_zero_counts),
    )
    logging.info(
        "feature_group[user_dense]: %d/%d features with >1%% active rate",
        int(
            np.count_nonzero(
                user_dense_nonzero_counts / max(len(probs_arr), 1) > 0.01
            )
        ) if test_dataset.user_dense_schema.total_dim > 0 else 0,
        test_dataset.user_dense_schema.total_dim,
    )

    user_int_active_arr = np.asarray(user_int_active, dtype=np.float64)
    item_int_active_arr = np.asarray(item_int_active, dtype=np.float64)
    user_dense_active_arr = np.asarray(user_dense_active, dtype=np.float64)
    u_stats = _summary_stats(user_int_active_arr)
    i_stats = _summary_stats(item_int_active_arr)
    d_stats = _summary_stats(user_dense_active_arr)
    logging.info(
        "user_int_active: min=%.1f, p25=%.1f, p50=%.1f, p75=%.1f, max=%.1f",
        u_stats["min"], u_stats["p25"], u_stats["p50"], u_stats["p75"], u_stats["max"],
    )
    logging.info(
        "item_int_active: min=%.1f, p25=%.1f, p50=%.1f, p75=%.1f, max=%.1f",
        i_stats["min"], i_stats["p25"], i_stats["p50"], i_stats["p75"], i_stats["max"],
    )
    logging.info(
        "user_dense_active: min=%.1f, p25=%.1f, p50=%.1f, p75=%.1f, max=%.1f",
        d_stats["min"], d_stats["p25"], d_stats["p50"], d_stats["p75"], d_stats["max"],
    )

    _log_shape("prob", _shape_stats(probs_arr))
    _log_shape("logit", _shape_stats(logits_arr))
    iqr = float(np.quantile(logits_arr, 0.75) - np.quantile(logits_arr, 0.25)) if logits_arr.size else 0.0
    rng = float(np.max(logits_arr) - np.min(logits_arr)) if logits_arr.size else 0.0
    centered = logits_arr - np.mean(logits_arr) if logits_arr.size else logits_arr
    std = float(np.std(logits_arr)) if logits_arr.size else 0.0
    kurtosis = float(np.mean((centered / std) ** 4)) if logits_arr.size and std > 1e-12 else 0.0
    skewness = float(np.mean((centered / std) ** 3)) if logits_arr.size and std > 1e-12 else 0.0
    bimodality = ((skewness ** 2) + 1.0) / (kurtosis + 3.0) if kurtosis > -3.0 else 0.0
    logging.info(
        "logit_shape: range=%.4f, iqr=%.4f, bimodality_coeff=%.4f (>0.555 suggests bimodal)",
        rng,
        iqr,
        bimodality,
    )

    for x in [-5, -3, -1, 0, 1, 3]:
        s = 1.0 / (1.0 + math.exp(-x))
        logging.info("sigmoid_sensitivity@logit=%+d: deriv=%.6f", x, s * (1.0 - s))

    logging.info(
        "emb_norm: min=%.4f, p05=%.4f, p50=%.4f, p95=%.4f, std=%.4f",
        emb_norm_stats["min"],
        emb_norm_stats["p05"],
        emb_norm_stats["p50"],
        emb_norm_stats["p95"],
        emb_norm_stats["std"],
    )

    pred_by_coverage: Dict[str, List[float]] = {}
    for cov, prob in zip(coverage_arr.tolist(), probs_arr.tolist()):
        pred_by_coverage.setdefault(_coverage_bin(float(cov)), []).append(float(prob))
    coverage_parts = []
    for label, vals in pred_by_coverage.items():
        coverage_parts.append(f"{label}:mean_p={np.mean(vals):.4f}({len(vals)})")
    logging.info("pred_by_coverage: %s", " | ".join(coverage_parts))

    missing_pattern_summary = []
    for key, count in missing_pattern_counts.most_common(DEBUG_TIME_BUCKET_TOP_K):
        mean_prob = missing_pattern_prob_sums.get(key, 0.0) / max(count, 1)
        missing_pattern_summary.append(f"{key}:mean_p={mean_prob:.4f}({count})")
    logging.info("pred_by_missing_pattern: %s", " | ".join(missing_pattern_summary) if missing_pattern_summary else "none")

    pred_by_weekday_parts = []
    pred_by_hour_parts = []
    time_bucket_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    seq_truncation_summary: Dict[str, Dict[str, float]] = {}
    seq_density_summary: Dict[str, Dict[str, float]] = {}
    seq_corr_summary: Dict[str, Dict[str, float]] = {}
    for bucket, values, probs in [
        ("weekday", weekday_values, probs_arr.tolist()),
        ("hour", hour_values, probs_arr.tolist()),
    ]:
        groups: Dict[int, List[float]] = {}
        for key, prob in zip(values, probs):
            if int(key) <= 0:
                continue
            groups.setdefault(int(key), []).append(float(prob))
        parts = [f"{'d' if bucket == 'weekday' else 'h'}{k}:{np.mean(v):.4f}({len(v)})" for k, v in sorted(groups.items())]
        if bucket == "weekday":
            pred_by_weekday_parts = parts
        else:
            pred_by_hour_parts = parts
    logging.info("pred_by_weekday: %s", " | ".join(pred_by_weekday_parts))
    logging.info("pred_by_hour: %s", " | ".join(pred_by_hour_parts))

    seq_len_dist_summary: Dict[str, Dict[str, int]] = {}
    seq_impact_summary: Dict[str, Dict[str, float]] = {}
    pred_by_seq_len_summary: Dict[str, Dict[str, float]] = {}
    time_bucket_dist_summary: Dict[str, Dict[str, Any]] = {}
    time_bucket_full_dist_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    time_bucket_quantile_summary: Dict[str, Dict[str, float]] = {}
    time_bucket_coverage_summary: Dict[str, Dict[str, Any]] = {}
    pred_by_recent_bucket_and_seq_len_summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    pred_by_recent_bucket_and_missing_pattern_summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for domain in test_dataset.seq_domains:
        lens = np.asarray(seq_len_rows[domain], dtype=np.int64)
        seq_len_dist_summary[domain] = {str(k): int(v) for k, v in sorted(Counter(lens.tolist()).items())}
        logging.info("seq_%s_len_dist: %s", domain, seq_len_dist_summary[domain])

        empty_mask = lens <= 0
        nonempty_mask = lens > 0
        empty_mean = float(np.mean(probs_arr[empty_mask])) if empty_mask.any() else 0.0
        nonempty_mean = float(np.mean(probs_arr[nonempty_mask])) if nonempty_mask.any() else 0.0
        seq_impact_summary[domain] = {
            "empty_mean": empty_mean,
            "empty_count": int(empty_mask.sum()),
            "nonempty_mean": nonempty_mean,
            "nonempty_count": int(nonempty_mask.sum()),
            "delta": nonempty_mean - empty_mean,
        }
        logging.info(
            "seq_%s_impact: empty=%.4f(%d) vs nonempty=%.4f(%d), delta=%+.4f",
            domain,
            empty_mean,
            int(empty_mask.sum()),
            nonempty_mean,
            int(nonempty_mask.sum()),
            nonempty_mean - empty_mean,
        )

        grouped: Dict[str, List[float]] = {}
        bucket_ids = np.asarray(seq_time_bucket_rows[domain], dtype=np.int64)
        nonzero_tokens = np.asarray(seq_nonzero_token_rows[domain], dtype=np.float64)
        effective_cov = np.asarray(seq_effective_cov_rows[domain], dtype=np.float64)
        truncation_rate = _truncation_rate(lens, int(seq_max_lens.get(domain, 0)))
        at_cap_count = int(np.sum(lens >= int(seq_max_lens.get(domain, 0)))) if int(seq_max_lens.get(domain, 0)) > 0 else 0
        seq_truncation_summary[domain] = {
            "max_len": int(seq_max_lens.get(domain, 0)),
            "at_cap_count": at_cap_count,
            "at_cap_rate": truncation_rate,
        }
        seq_density_summary[domain] = {
            "nonzero_token_stats": _summary_stats(nonzero_tokens),
            "effective_coverage_stats": _summary_stats(effective_cov),
        }
        seq_corr_summary[domain] = {
            "len_prob_corr": _safe_corr(lens.astype(np.float64), probs_arr),
            "recent_bucket_prob_corr": _safe_corr(bucket_ids.astype(np.float64), probs_arr),
            "effective_coverage_prob_corr": _safe_corr(effective_cov, probs_arr),
        }
        time_bucket_summary[domain] = _bucket_prob_summary(bucket_ids, probs_arr)
        time_bucket_dist_summary[domain] = _bucket_distribution_summary(
            bucket_ids,
            DEBUG_TIME_BUCKET_DIST_TOP_K,
        )
        time_bucket_full_dist_summary[domain] = _bucket_full_distribution(bucket_ids)
        time_bucket_quantile_summary[domain] = _bucket_quantiles(bucket_ids)
        time_bucket_coverage_summary[domain] = _bucket_coverage_summary(
            bucket_ids,
            int(model_cfg.get("num_time_buckets", 0)) - 1,
        )
        seq_len_labels = [_seq_bin_label(int(length)) for length in lens.tolist()]
        missing_pattern_labels = [
            "nonempty" if int(length) > 0 else "empty"
            for length in lens.tolist()
        ]
        pred_by_recent_bucket_and_seq_len_summary[domain] = _bucket_cross_summary(
            bucket_ids,
            seq_len_labels,
            probs_arr,
        )
        pred_by_recent_bucket_and_missing_pattern_summary[domain] = _bucket_cross_summary(
            bucket_ids,
            missing_pattern_labels,
            probs_arr,
        )
        logging.info(
            "seq_%s_time_bucket_dist: zero_rate=%.4f unique_nonzero=%d top=%s",
            domain,
            time_bucket_dist_summary[domain]["zero_rate"],
            time_bucket_dist_summary[domain]["unique_nonzero_buckets"],
            " | ".join(
                f"b{row['bucket']}:{row['count']}({row['rate']:.4f})"
                for row in time_bucket_dist_summary[domain]["top_buckets"]
            ) or "none",
        )
        logging.info(
            "seq_%s_time_bucket_quantiles: min=%.1f p25=%.1f p50=%.1f p75=%.1f p90=%.1f p95=%.1f p99=%.1f max=%.1f mean=%.2f",
            domain,
            time_bucket_quantile_summary[domain]["min"],
            time_bucket_quantile_summary[domain]["p25"],
            time_bucket_quantile_summary[domain]["p50"],
            time_bucket_quantile_summary[domain]["p75"],
            time_bucket_quantile_summary[domain]["p90"],
            time_bucket_quantile_summary[domain]["p95"],
            time_bucket_quantile_summary[domain]["p99"],
            time_bucket_quantile_summary[domain]["max"],
            time_bucket_quantile_summary[domain]["mean"],
        )
        logging.info(
            "seq_%s_time_bucket_coverage: used=%d/%d rate=%.4f missing=%d head_missing=%s",
            domain,
            time_bucket_coverage_summary[domain]["used_bucket_count"],
            time_bucket_coverage_summary[domain]["configured_bucket_count"],
            time_bucket_coverage_summary[domain]["bucket_coverage_rate"],
            time_bucket_coverage_summary[domain]["missing_bucket_count"],
            ",".join(str(x) for x in time_bucket_coverage_summary[domain]["missing_buckets_head"]) or "none",
        )
        logging.info(
            "seq_%s_time_bucket_full_dist: %s",
            domain,
            " | ".join(
                f"b{bucket}:{stats['count']}({stats['rate']:.4f})"
                for bucket, stats in list(time_bucket_full_dist_summary[domain].items())[:DEBUG_SUCCESS_FULL_BUCKET_PRINT_LIMIT]
            ) or "none",
        )
        logging.info(
            "seq_%s_truncation: at_cap=%d/%d rate=%.4f max_len=%d",
            domain,
            at_cap_count,
            len(lens),
            truncation_rate,
            int(seq_max_lens.get(domain, 0)),
        )
        logging.info(
            "seq_%s_density: nonzero_tokens_p50=%.1f effective_cov_p50=%.4f",
            domain,
            seq_density_summary[domain]["nonzero_token_stats"]["p50"],
            seq_density_summary[domain]["effective_coverage_stats"]["p50"],
        )
        logging.info(
            "seq_%s_corr: len_prob=%.4f recent_bucket_prob=%.4f effective_cov_prob=%.4f",
            domain,
            seq_corr_summary[domain]["len_prob_corr"],
            seq_corr_summary[domain]["recent_bucket_prob_corr"],
            seq_corr_summary[domain]["effective_coverage_prob_corr"],
        )
        logging.info(
            "seq_%s_recent_bucket_top: %s",
            domain,
            _format_top_counts(_top_counts(bucket_ids, DEBUG_TIME_BUCKET_TOP_K)),
        )
        logging.info(
            "pred_by_recent_bucket[%s]: %s",
            domain,
            " | ".join(
                f"{k}:mean_p={v['mean_prob']:.4f}({v['count']})"
                for k, v in time_bucket_summary[domain].items()
            ),
        )
        for length, prob in zip(lens.tolist(), probs_arr.tolist()):
            grouped.setdefault(_seq_bin_label(int(length)), []).append(float(prob))
        pred_by_seq_len_summary[domain] = {
            label: float(np.mean(vals)) for label, vals in grouped.items()
        }
        parts = [f"{label}:{np.mean(vals):.4f}({len(vals)})" for label, vals in grouped.items()]
        logging.info("pred_by_seq_len[%s]: %s", domain, " | ".join(parts))

    for bucket_name, rows in sample_buckets.items():
        logging.info("sample_bucket[%s]: %d", bucket_name, len(rows))

    if not debug_use_labels:
        logging.info(
            "[DIAG] Ground-truth labels are not available in competition inference; "
            "false_positive/false_negative buckets are emitted only when "
            "INFER_DEBUG_USE_LABELS=1 on labeled data."
        )

    summary_payload = {
        "prob_stats": prob_stats,
        "logit_stats": logit_stats,
        "confidence_stats": confidence_stats,
        "entropy_stats": entropy_stats,
        "embedding_norm_stats": emb_norm_stats,
        "prob_bins": prob_bin_summary,
        "seq_len_distributions": seq_len_dist_summary,
        "seq_impact": seq_impact_summary,
        "pred_by_seq_len": pred_by_seq_len_summary,
        "pred_by_missing_pattern": {
            k: {
                "count": int(v),
                "mean_prob": _to_float(missing_pattern_prob_sums.get(k, 0.0) / max(v, 1)),
            }
            for k, v in missing_pattern_counts.items()
        },
        "seq_truncation": seq_truncation_summary,
        "seq_density": seq_density_summary,
        "seq_correlations": seq_corr_summary,
        "pred_by_recent_bucket": time_bucket_summary,
        "time_bucket_distribution": time_bucket_dist_summary,
        "sample_bucket_sizes": {k: len(v) for k, v in sample_buckets.items()},
        "elapsed_seconds": elapsed,
        "throughput": len(probs_arr) / max(elapsed, 1e-12),
    }

    _log_json("inference_debug_summary", summary_payload)
    _log_samples("inference_debug_sample", sample_buckets)

    message = (
        "Debug inference complete — all diagnostics were emitted via logs. "
        "Set INFER_DEBUG=0 for a real inference run."
    )
    if DEBUG_STOP_AFTER_SAVE:
        raise RuntimeError(message)
    _write_status_files(result_dir, "success", message)


if __name__ == "__main__":
    result_dir_env = os.environ.get("EVAL_RESULT_PATH", ".")
    try:
        main()
    except Exception as exc:
        logging.error("Inference failed: %s", exc)
        try:
            _write_status_files(result_dir_env, "error", str(exc))
        finally:
            raise
