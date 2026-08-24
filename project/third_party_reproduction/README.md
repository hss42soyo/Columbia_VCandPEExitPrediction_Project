# Chapter 9.2 Licensed Reproduction Package

This folder contains the minimal code surface for a private third-party
reproduction package of the VC/PE exit-prediction work. It is not a public data
release. The paired data package is staged outside `repo/` at:

`D:\dev\ECR Capital Management\Book2\data\eamples\Chapter 9\VC & PE Exit Prediction\third_party_reproduction_20260521`

## Contents

- `scripts/build_delivery_package.py` builds manifests, schema catalogs, and the
  private data package by copying named canonical roots.
- `notebooks/paper7_vc_exit_reproduction.ipynb` validates the package and
  reproduces the frozen headline tables.
- `requirements.txt` lists the Python dependencies used by the notebook and
  package builder.

This notebook is the private licensed reproduction surface. The public reader
notebook lives one level up at
`../notebooks/L3_9_2_leakage_controlled_private_markets.ipynb` and runs the
sample workflow without the paired private data package.

## Safety Rules

- Do not delete source files.
- Do not move source files.
- Do not use mirror or purge-style synchronization.
- Licensed rows belong only in the paired data package, not in this repo folder.

## Build

From this folder:

```powershell
python .\scripts\build_delivery_package.py
```

The build script is copy-only. It writes a source manifest, copy ledger, hash
manifest, schema catalogs, and a package validation report.

## Validate

Open `notebooks/paper7_vc_exit_reproduction.ipynb` and run all cells. The
notebook validates the copy ledger and loads the frozen result tables that anchor
the chapter narrative.
