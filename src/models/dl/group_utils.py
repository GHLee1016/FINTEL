"""Feature-group helpers for GroupNN.

GroupNN reuses the existing extended DL arrays, whose feature axis is ordered
by ``dataset_DL/feature_columns_{market}.txt``. This module maps those feature
names to economically meaningful groups so the model can slice ``X[:, :, idx]``
without creating duplicated group-specific ``.npy`` files.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from ...preprocess.features import CORE, EXTENDED_ADD, MOMENTUM_ADD


GROUP_ORDER = ("core", "momentum", "macro", "spillover")


def _indices_for(feature_cols: List[str], names: Iterable[str]) -> List[int]:
    name_set = set(names)
    return [i for i, col in enumerate(feature_cols) if col in name_set]


def build_feature_groups(feature_cols: List[str]) -> Dict[str, List[int]]:
    """Return GroupNN feature indices for an extended feature column list.

    Parameters
    ----------
    feature_cols:
        Feature names in the exact order used by the DL input tensor.

    Returns
    -------
    dict
        Mapping with keys ``core``, ``momentum``, ``macro``, and ``spillover``.

    Raises
    ------
    ValueError
        If a group is empty, a feature is assigned twice, or an input feature is
        not covered by the four GroupNN groups.
    """
    feature_cols = list(feature_cols)
    groups: Dict[str, List[int]] = {
        "core": _indices_for(feature_cols, CORE),
        "momentum": _indices_for(feature_cols, MOMENTUM_ADD),
        "macro": _indices_for(feature_cols, EXTENDED_ADD),
        "spillover": [
            i for i, col in enumerate(feature_cols) if col.startswith("spillover_")
        ],
    }

    for name in GROUP_ORDER:
        if not groups[name]:
            raise ValueError(
                f"GroupNN requires non-empty {name!r} group; "
                "load data with tier='extended'."
            )

    assigned = [idx for indices in groups.values() for idx in indices]
    duplicates = sorted({idx for idx in assigned if assigned.count(idx) > 1})
    if duplicates:
        dup_names = [feature_cols[i] for i in duplicates]
        raise ValueError(f"features assigned to multiple groups: {dup_names}")

    missing = sorted(set(range(len(feature_cols))) - set(assigned))
    if missing:
        missing_names = [feature_cols[i] for i in missing]
        raise ValueError(f"features not assigned to any GroupNN group: {missing_names}")

    return {name: groups[name] for name in GROUP_ORDER}
