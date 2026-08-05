from __future__ import annotations

import pandas as pd


VALIDATION_NAMES = {"val", "valid", "validation"}


def _enabled(config: dict, key: str, default=False):
    return bool(config.get(key, config.get("validation", {}).get(key, config.get("training", {}).get(key, default))))


def resolve_train_validation_test_splits(frame: pd.DataFrame, config: dict):
    """Keep test out of model selection unless the user explicitly opts in."""
    splits = frame["split"].astype(str).str.casefold()
    train = frame[splits == "train"].copy()
    validation = frame[splits.isin(VALIDATION_NAMES)].copy()
    test = frame[splits == "test"].copy()
    use_test = _enabled(config, "use_test_as_validation")
    selected = "validation"
    if validation.empty and not test.empty and use_test:
        validation, selected = test.copy(), "test_opt_in"
    if train.empty:
        raise ValueError("No train samples found; training is blocked")
    if validation.empty:
        suffix = "; set use_test_as_validation: true only for a temporary fallback" if not test.empty else ""
        raise ValueError("No validation split found; training is blocked" + suffix)
    return train, validation, test, {
        "validation_source": selected,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "test_used_for_validation": selected == "test_opt_in",
    }


def quick_subset(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Deterministic, per-dataset quick slice; test is never included here."""
    quick = config.get("quick", {})
    train_limit = int(quick.get("train_samples_per_dataset", 32))
    val_limit = int(quick.get("validation_samples_per_dataset", 8))
    split = frame["split"].astype(str).str.casefold()
    allowed = frame[split.isin({"train", *VALIDATION_NAMES})].copy()
    limits = {"train": train_limit, "val": val_limit, "valid": val_limit, "validation": val_limit}
    pieces = []
    for (split_name, _), group in allowed.sort_values("sample_id").groupby([split[allowed.index], "dataset_name"], sort=True):
        if group.empty:
            continue
        pieces.append(group.head(limits[split_name]))
    return pd.concat(pieces, ignore_index=True) if pieces else allowed.iloc[0:0].copy()
