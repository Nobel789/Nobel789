from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PII_CANDIDATES = {
    "patient_name",
    "name",
    "email",
    "emailid",
    "phone",
    "address",
    "ssn",
}

GENDER_MAP = {
    "m": "Male",
    "male": "Male",
    "f": "Female",
    "female": "Female",
    "other": "Other",
    "non-binary": "Other",
    "nonbinary": "Other",
}

DIAGNOSIS_MAP = {
    "htn": "Hypertension",
    "high blood pressure": "Hypertension",
    "dm": "Diabetes",
    "t2dm": "Diabetes",
}


@dataclass
class CleanResult:
    df: pd.DataFrame
    dropped_pii_columns: List[str]
    outlier_clip_counts: Dict[str, int]


def _normalize_colname(col: str) -> str:
    return col.strip().lower().replace(" ", "_")


def _drop_pii(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    normalized = {_normalize_colname(c): c for c in df.columns}
    drop = [original for norm, original in normalized.items() if norm in PII_CANDIDATES]
    return df.drop(columns=drop, errors="ignore"), drop


def _normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = {_normalize_colname(c): c for c in out.columns}

    if "gender" in cols:
        c = cols["gender"]
        out[c] = (
            out[c]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(GENDER_MAP)
            .fillna(out[c])
        )

    for key in ("diagnosis", "diagnosis_code", "dx"):
        if key in cols:
            c = cols[key]
            out[c] = (
                out[c]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(DIAGNOSIS_MAP)
                .fillna(out[c])
            )
    return out


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(out[c].median())
        else:
            mode = out[c].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "Unknown"
            out[c] = out[c].fillna(fill)
    return out


def _clip_outliers_iqr(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    counts: Dict[str, int] = {}
    numeric = out.select_dtypes(include=[np.number]).columns

    for c in numeric:
        q1, q3 = out[c].quantile(0.25), out[c].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            counts[c] = 0
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = ((out[c] < lo) | (out[c] > hi)).sum()
        out[c] = out[c].clip(lo, hi)
        counts[c] = int(before)
    return out, counts


def clean_healthcare_dataframe(df: pd.DataFrame) -> CleanResult:
    out, dropped = _drop_pii(df)
    out = _normalize_categories(out)
    out = out.drop_duplicates().reset_index(drop=True)
    out = _impute_missing(out)
    out, outlier_counts = _clip_outliers_iqr(out)
    return CleanResult(df=out, dropped_pii_columns=dropped, outlier_clip_counts=outlier_counts)
