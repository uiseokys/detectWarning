from __future__ import annotations

import os


def _enable_optional_mean_max_pooling() -> None:
    mode = str(os.environ.get("DETECTWARNING_TEMPORAL_POOLING") or "").strip().lower()
    if mode not in {"mean_max", "mean+max", "mean-max", "max_mean", "max+mean"}:
        return
    try:
        from mean_max_pooling import enable_mean_max_pooling

        enable_mean_max_pooling()
    except Exception as exc:
        print(f"[train] failed to enable mean+max temporal pooling: {exc}")


_enable_optional_mean_max_pooling()
