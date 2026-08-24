# Leakage-Controlled Prediction in Private Markets

> [!warning] Retired package surface
> This folder documents the older hard-liquidity public-example package. It is preserved as a runnable/provenance surface, not as the current Chapter 9.2 / Paper 7 manuscript claim. The current manuscript claim is the 2026-05-14 Żbikowski & Antosiuk (2021) forensic reproduction and diagnostic diligence-screen extension.

This Chapter 9 public example packages the current state of the book's VC/PE
exit-prediction work as an honest empirical draft.

- `python/` is the canonical implementation.
- `notebooks/L3_9_2_leakage_controlled_private_markets.ipynb` is the reader
  notebook for the public/sample workflow.
- `matlab/` and `r/` are wrapper companions over the Python logic.
- `sample` is the default public path and runs from the bundled scenario pack.
- `live` rerenders the chapter-facing figures from proprietary local inputs
  outside `repo/`.
- `third_party_reproduction/notebooks/paper7_vc_exit_reproduction.ipynb` is the
  private licensed reproduction notebook for the accepted manuscript evidence.

The current empirical doctrine is narrower than the older broad "any exit"
story:

- venture headline target: `hard_timely_liquidity_by_8q`
- hard routes only: `ipo`, `mna`, `sponsor_sale`
- two-stage design: binary hazard plus conditional route model
- buyout remains a provisional appendix-level extension

This folder is publication packaging for the current state of the project. It
is not a claim that the Chapter 9 research is finished or evidence-ready.

Use the reader notebook for a clean public rerun. Use the licensed reproduction
notebook only with the paired private data package outside `repo/`.

The code directory retains the original `VC & PE Exit Prediction` folder name for backward compatibility with existing repository paths; the manuscript title is now `Leakage-Controlled Prediction in Private Markets`.
