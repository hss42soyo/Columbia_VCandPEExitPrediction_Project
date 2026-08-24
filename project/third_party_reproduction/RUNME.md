# RUNME: Chapter 9.2 Licensed Reproduction Package

## 1. Confirm the paired data package exists

Expected data package:

`D:\dev\ECR Capital Management\Book2\data\eamples\Chapter 9\VC & PE Exit Prediction\third_party_reproduction_20260521`

The package should contain `data\`, `manifests\`, `DATA_LICENSE_NOTES.md`, and
`PACKAGE_CONTENTS.md`.

## 2. Install dependencies

```powershell
python -m pip install -r .\requirements.txt
```

## 3. Validate package integrity

```powershell
python .\scripts\build_delivery_package.py
```

The script is safe to rerun. It copies named source roots only, skips files that
already match by hash, and records every transaction in the copy ledger.

## 4. Run the notebook

```powershell
jupyter notebook .\notebooks\paper7_vc_exit_reproduction.ipynb
```

Run all cells. The notebook checks the copy ledger, reads schema catalogs, and
loads the frozen headline result tables used by the chapter narrative.

## Acceptance Criteria

- `manifests\copy_ledger.csv` contains only `ok` statuses.
- `schemas\crunchbase_schema.json`, `schemas\preqin_schema.json`, and
  `schemas\commoncrawl_schema.json` load successfully.
- The frozen `final_regime_comparison.csv` table displays the accepted baseline,
  current-regime anchor, Common Crawl panel, and policy-refit rows.
