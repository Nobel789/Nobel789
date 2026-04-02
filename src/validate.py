from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import pandas as pd


@dataclass
class QualityMetrics:
    rows_before: int
    rows_after: int
    duplicates_before: int
    duplicates_after: int
    missing_before: int
    missing_after: int
    columns_before: int
    columns_after: int
    dropped_pii_columns: list[str]
    outlier_clip_counts: Dict[str, int]

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_quality_metrics(
    before: pd.DataFrame,
    after: pd.DataFrame,
    dropped_pii_columns: list[str],
    outlier_clip_counts: Dict[str, int],
) -> QualityMetrics:
    return QualityMetrics(
        rows_before=len(before),
        rows_after=len(after),
        duplicates_before=int(before.duplicated().sum()),
        duplicates_after=int(after.duplicated().sum()),
        missing_before=int(before.isna().sum().sum()),
        missing_after=int(after.isna().sum().sum()),
        columns_before=before.shape[1],
        columns_after=after.shape[1],
        dropped_pii_columns=dropped_pii_columns,
        outlier_clip_counts=outlier_clip_counts,
    )
