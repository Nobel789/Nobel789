# healthcare-data-cleaning-pipeline

Production-friendly healthcare data cleaning pipeline for de-identification, quality validation, and ML-ready transformation.

## What this improves
This version addresses common portfolio gaps:
- Moves logic from notebooks to reusable Python modules (`src/`)
- Adds runnable CLI entrypoint (`src/run_pipeline.py`)
- Adds measurable before/after quality report (`reports/quality_report.json`)
- Provides proper dependency file (`requirements.txt`)

## Project structure
```text
.
├── data/
│   ├── raw/               # input csv files
│   └── processed/         # cleaned output csv files
├── reports/
│   └── quality_report.json
├── src/
│   ├── clean.py
│   ├── validate.py
│   └── run_pipeline.py
├── requirements.txt
└── README.md
```

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/run_pipeline.py \
  --input data/raw/healthcare.csv \
  --output data/processed/healthcare_clean.csv \
  --report reports/quality_report.json
```

## What the pipeline does
1. Removes PII columns when present (`patient_name`, `email`, etc.)
2. Standardizes common categorical variants (gender, diagnosis labels)
3. Handles missing values (median for numeric, mode for categorical)
4. Removes duplicate rows
5. Clips numeric outliers using IQR bounds
6. Generates before/after quality metrics

## Example quality metrics reported
- row counts before/after
- duplicate counts removed
- missing values before/after
- columns dropped as PII
- outlier clipping counts by numeric column

## Notes
- This repo contains a generic, reusable cleaning baseline.
- Adapt mappings/rules to your healthcare schema and governance requirements.
