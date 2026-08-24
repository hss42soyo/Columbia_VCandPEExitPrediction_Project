# Leakage-Controlled Prediction in Private Markets (Python)

> [!warning] Retired hard-liquidity implementation
> This Python package is the older hard-liquidity teaching implementation. It remains useful as a runnable companion and provenance record, but it is not the current Paper 7 evidence path. The current 9.2 manuscript evidence is controlled by the 2026-05-14 Żbikowski & Antosiuk (2021) forensic reproduction, traceability manifest, and diagnostic screening extensions.

Python is the canonical Chapter 9 public implementation.

## What This Chapter Covers

The current Chapter 9 empirical draft models a narrower venture headline target:

- `hard_timely_liquidity_by_8q`
- hard routes only: `ipo`, `mna`, `sponsor_sale`
- stage 1: binary hazard
- stage 2: conditional route model among realized hard exits

The public package keeps buyout as a provisional extension and honest blocker
case study, not as chapter-headline evidence.

## Modes

- `sample`: default public mode; runs from `repo/packs/scenario/pe-vc-hazard/`
- `live`: optional proprietary rerender path; requires local licensed inputs
  outside `repo/`

Live mode resolves proprietary inputs from:

- `--live-data-dir`, or
- `BOOK2_PE_VC_HAZARD_LIVE_DATA_DIR`, or
- a local `paths.local.yml` file when present

## Files

- `VC-PE-Exit-Prediction.ipynb`: notebook-facing teaching surface
- `../notebooks/L3_9_2_leakage_controlled_private_markets.ipynb`: canonical
  reader notebook for the public/sample workflow
- `vc_pe_exit_prediction.py`: canonical helper implementation
- `run_vc_pe_exit_prediction.py`: runnable entrypoint
- `path_helpers.py`: sample/live path resolution helpers
- `assumptions.md`: current analytical-contract notes
- `rendered-sample/`: bundled sample figures
- `rendered-live/`: static publication figures only

## Typical Commands

```powershell
python run_vc_pe_exit_prediction.py --data-mode sample
python run_vc_pe_exit_prediction.py --data-mode live
```

`rendered-live/` is intentionally a static-publication surface. Do not commit
machine-readable live outputs derived from licensed vendor data.

This chapter package is an empirical draft / current-state packaging, not a
claim that the Chapter 9 evidence gate has passed.

The code directory retains the original `VC & PE Exit Prediction` folder name for backward compatibility with existing repository paths; the manuscript title is now `Leakage-Controlled Prediction in Private Markets`.
