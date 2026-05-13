import argparse
import os
from typing import Any, Dict, Iterable

from .dataset_config import DATASET_DEFAULTS

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to load config.yaml. Please install pyyaml.") from exc


_TOP_LEVEL_KEYS = ("shared", "generator", "evaluator")
_PATH_LIKE_KEYS = {"dataset_path", "save_dir", "generated_dir"}
_SHARED_REQUIRED_KEYS = (
    "model_path",
    "model_type",
    "with_tag",
    "vision_token_id",
    "patch_size",
    "merge_size",
)
_GENERATOR_REQUIRED_KEYS = (
    "max_new_tokens",
    "overwrite",
    "disable_monkey_patch",
)
_EVALUATOR_REQUIRED_KEYS = (
    "global_save_fig",
    "overwrite",
    "outlier_ratio",
    "dominance_ratio"
)


def _normalize_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return os.path.normpath(os.path.expandvars(os.path.expanduser(value)))


def _ensure_mapping(name: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Section '{name}' must be a mapping, got {type(value).__name__}.")
    return value


def _require_keys(section_name: str, cfg: Dict[str, Any], required_keys: Iterable[str]) -> None:
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in section '{section_name}': {missing}")


def _apply_dataset_defaults(dataset: str, generator: Dict[str, Any], evaluator: Dict[str, Any]) -> None:
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Available: {list(DATASET_DEFAULTS.keys())}")
    dcfg = DATASET_DEFAULTS[dataset]

    generator.setdefault("dataset_path", dcfg["dataset_path"])
    generator.setdefault("save_dir", dcfg["output_root"])
    generator.setdefault("replace_path", dcfg["replace_path"])

    evaluator.setdefault("generated_dir", dcfg["output_root"])
    evaluator.setdefault("replace_path", dcfg["replace_path"])


def _normalize_return_aggregate(evaluator: Dict[str, Any]) -> None:
    if "return_aggregate" not in evaluator:
        raise KeyError("Missing required key in section 'evaluator': ['return_aggregate']")

    
def _normalize_our_method_options(evaluator: Dict[str, Any]) -> None:
    raw_topk = evaluator.get("topk_spike_patches", 3)
    try:
        topk = int(raw_topk)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"evaluator.topk_spike_patches must be an integer, got: {raw_topk}") from exc
    if topk <= 0:
        raise ValueError(f"evaluator.topk_spike_patches must be > 0, got: {topk}")
    evaluator["topk_spike_patches"] = topk

    raw_share_thr = evaluator.get("share_thr", 0.3)
    try:
        share_thr = float(raw_share_thr)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"evaluator.share_thr must be a float, got: {raw_share_thr}") from exc
    if not 0.0 < share_thr <= 1.0:
        raise ValueError(f"evaluator.share_thr must be in (0, 1], got: {share_thr}")
    evaluator["share_thr"] = share_thr

    sink_filter_aliases = {
        "all_tokens": "all_tokens",
        "all": "all_tokens",
        "full": "all_tokens",
        "random": "random",
        "random_topk": "random",
        "random_tokens": "random",
        "anomaly_related_topk": "anomaly_related_topk",
        "related_topk": "anomaly_related_topk",
        "anomaly_related": "anomaly_related_topk",
        "related": "anomaly_related_topk",
        "anomaly_unrelated_topk": "anomaly_unrelated_topk",
        "unrelated_topk": "anomaly_unrelated_topk",
        "anomaly_unrelated": "anomaly_unrelated_topk",
        "unrelated": "anomaly_unrelated_topk",
        "pos_content": "pos_content",
        "content": "pos_content",
        "meaningful": "pos_content",
        "pos_meaningful": "pos_content",
        "pos_function": "pos_function",
        "function": "pos_function",
        "functional": "pos_function",
        "pos_irrelevant": "pos_function",
        "irrelevant": "pos_function",
    }
    raw_sink_filter_mode = evaluator.get("sink_head_token_filter_mode", "pos_function")
    normalized_sink_filter_mode = sink_filter_aliases.get(str(raw_sink_filter_mode).strip().lower())
    if normalized_sink_filter_mode is None:
        raise ValueError(
            f"Unsupported evaluator.sink_head_token_filter_mode: {raw_sink_filter_mode}. "
            "Use one of {all_tokens, random, anomaly_related_topk, anomaly_unrelated_topk, pos_content, pos_function}."
        )
    evaluator["sink_head_token_filter_mode"] = normalized_sink_filter_mode

    raw_sink_topk = evaluator.get("sink_head_token_topk", 8)
    try:
        sink_head_token_topk = int(raw_sink_topk)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"evaluator.sink_head_token_topk must be an integer, got: {raw_sink_topk}") from exc
    if sink_head_token_topk <= 0:
        raise ValueError(f"evaluator.sink_head_token_topk must be > 0, got: {sink_head_token_topk}")
    evaluator["sink_head_token_topk"] = sink_head_token_topk


def _normalize_openrouter_api_key(shared: Dict[str, Any]) -> None:
    raw_key = shared.get("OPENROUTER_API_KEY", "")
    shared["OPENROUTER_API_KEY"] = "" if raw_key is None else str(raw_key)


def load_config(config_path: str, dataset: str) -> Dict[str, Dict[str, Any]]:
    resolved_config_path = _normalize_path(config_path)
    if not os.path.isfile(resolved_config_path):
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    with open(resolved_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise TypeError("Top-level YAML content must be a mapping.")

    for top_key in _TOP_LEVEL_KEYS:
        if top_key not in cfg:
            raise KeyError(f"Missing top-level section '{top_key}' in config file.")

    shared = _ensure_mapping("shared", cfg["shared"])
    generator = _ensure_mapping("generator", cfg["generator"])
    evaluator = _ensure_mapping("evaluator", cfg["evaluator"])

    _require_keys("shared", shared, _SHARED_REQUIRED_KEYS)
    _require_keys("generator", generator, _GENERATOR_REQUIRED_KEYS)
    _require_keys("evaluator", evaluator, _EVALUATOR_REQUIRED_KEYS)
    _normalize_openrouter_api_key(shared)
    _normalize_return_aggregate(evaluator)
    _normalize_our_method_options(evaluator)
    _apply_dataset_defaults(dataset, generator, evaluator)

    for section in (shared, generator, evaluator):
        for key, value in list(section.items()):
            if key in _PATH_LIKE_KEYS:
                section[key] = _normalize_path(value)

    return {"shared": shared, "generator": generator, "evaluator": evaluator}


def build_stage_namespace(config_path: str, stage: str, dataset: str) -> argparse.Namespace:
    stage_key = stage.lower()
    if stage_key not in {"generator", "evaluator"}:
        raise ValueError(f"Unsupported stage: {stage}")

    cfg = load_config(config_path, dataset=dataset)
    merged = dict(cfg["shared"])
    merged.update(cfg[stage_key])
    merged["dataset"] = dataset
    merged["config_path"] = _normalize_path(config_path)
    return argparse.Namespace(**merged)
