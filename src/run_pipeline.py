from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from clean import clean_healthcare_dataframe
from validate import compute_quality_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Healthcare data cleaning pipeline")
    p.add_argument("--input", required=True, help="Path to raw input CSV")
    p.add_argument("--output", required=True, help="Path to cleaned output CSV")
    p.add_argument("--report", required=True, help="Path to quality report JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    report_path = Path(args.report)

    df_raw = pd.read_csv(in_path)
    result = clean_healthcare_dataframe(df_raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    result.df.to_csv(out_path, index=False)

    metrics = compute_quality_metrics(
        before=df_raw,
        after=result.df,
        dropped_pii_columns=result.dropped_pii_columns,
        outlier_clip_counts=result.outlier_clip_counts,
    )
    report_path.write_text(json.dumps(metrics.to_dict(), indent=2))

    print(f"Saved cleaned data: {out_path}")
    print(f"Saved quality report: {report_path}")


if __name__ == "__main__":
    main()
