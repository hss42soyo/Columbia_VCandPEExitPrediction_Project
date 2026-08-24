# Leakage-Controlled Prediction in Private Markets (MATLAB Companion)

> [!warning] Retired hard-liquidity companion
> This MATLAB companion wraps the older hard-liquidity public-example package. It is not the current Paper 7 reproduction evidence path and should not be cited as the current Chapter 9.2 manuscript result.

This MATLAB folder is a wrapper companion over the canonical Python Chapter 9
implementation.

- `sample` is the default public path.
- `live` requires proprietary data outside `repo/`.
- `rendered-live/` contains static publication figures only.

Use this surface when you want to run the Chapter 9 figure package from MATLAB
without maintaining a separate native analytics stack.

Run:

```matlab
VC_PE_Exit_Prediction
```

Edit the `data_mode` variable near the top of the script to switch between
`sample` and `live`.

This is an empirical draft / current-state packaging, not a claim that the
research is evidence-ready.

The code directory retains the original `VC & PE Exit Prediction` folder name for backward compatibility with existing repository paths; the manuscript title is now `Leakage-Controlled Prediction in Private Markets`.
