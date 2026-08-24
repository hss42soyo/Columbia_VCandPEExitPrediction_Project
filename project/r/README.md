# Leakage-Controlled Prediction in Private Markets (R Companion)

> [!warning] Retired hard-liquidity companion
> This R companion wraps the older hard-liquidity public-example package. It is not the current Paper 7 reproduction evidence path and should not be cited as the current Chapter 9.2 manuscript result.

This R folder is a wrapper companion over the canonical Python Chapter 9
implementation.

- `sample` is the default public path.
- `live` requires proprietary data outside `repo/`.
- `rendered-live/` contains static publication figures only.

Use this surface when you want to rerender the Chapter 9 figure package from a
Quarto notebook without maintaining a separate native R analytics stack.

Render:

```powershell
quarto render VC-PE-Exit-Prediction.qmd
```

Set `data_mode` near the top of the notebook to `sample` or `live`.

This is an empirical draft / current-state packaging, not a claim that the
research is evidence-ready.

The code directory retains the original `VC & PE Exit Prediction` folder name for backward compatibility with existing repository paths; the manuscript title is now `Leakage-Controlled Prediction in Private Markets`.
