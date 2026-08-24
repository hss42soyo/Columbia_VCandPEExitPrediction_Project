# Chapter 9 Assumptions

This note maps the current Chapter 9 notation to the Python implementation in
`vc_pe_exit_prediction.py`.

## Headline Target

The live headline target is:

- `hard_timely_liquidity_by_8q`

It is a point-in-time binary target for whether a company realizes a direct
high- or medium-confidence liquidity event within the next eight quarters.

Included headline routes:

- `ipo`
- `mna`
- `sponsor_sale`

Excluded from the headline target:

- `soft_failure_sensitivity`

Code mapping:

- target construction: `add_redesigned_targets`
- target definition artifacts: `build_target_definition_main`,
  `build_target_definition_sensitivity`
- route-confidence audit: `build_label_confidence_audit`

## Target Exploration Layer

The Chapter 9 build now carries a controlled target-exploration layer on top of
the locked live headline target.

Important rules:

- `venture_growth` keeps `hard_timely_liquidity_by_8q` as the locked baseline
- `buyout_pe` candidates are compared on objective definition, not on a fresh
  broad feature sweep
- the bounded buyout-only empirical path now finalizes policy choice with a
  validation-only constrained selector and a pooled stage-2 route-support gate
- the final doctrine bundle is `rendered-live-final-doctrine-20260404/`
- the canonical buyout-only bounded milestone remains
  `rendered-live-buyout-policy-20260403/`, but the final doctrine pass
  downgrades buyout from promoted to `provisional` because target/policy
  stability was insufficient across nearby bounded robustness settings
- soft-failure evidence stays out of headline candidate labels
- unsupported candidates, such as dated partial realizations without usable
  source fields, remain definition-only and are reported as such

Code mapping:

- target registry: `build_target_registry`
- candidate panel construction: `build_target_candidate_panel`
- candidate evaluation: `evaluate_target_candidate`
- recommendation table: `build_target_recommendation_table`

## Time and PIT Rule

- Quarter `t` is the outcome quarter.
- Company features are observable by the end of quarter `t-1`.
- The forward underwriting horizon is `T = 8` quarters.

Operational implications:

- Crunchbase snapshot-only status fields are not used directly as headline
  predictors
- low-confidence closed / realized proxies stay in sensitivity-only outputs
- live mode uses quarter fixed effects instead of building macro state from
  realized exits in the labeled panel

Code mapping:

- quarter indexing: `quarter_idx_from_dates`, `quarter_idx_from_label`
- panel construction: `build_company_quarter_panel`
- realized 8-quarter horizon labels: `add_realized_exit_within_horizon`

## Two-Stage Design

### Stage 1

Stage 1 is a binary hazard for:

- `hard_timely_liquidity_by_8q`

The implementation is a discrete-time logistic model estimated separately by
empirical universe.

Code mapping:

- feature selection for stage 1: `stage1_feature_columns`
- binary matrix preparation: `prepare_binary_model_matrix`
- fitting: `fit_binary_hazard`, `fit_stage1_models_by_universe`
- scoring: `score_stage1_panel`, `score_stage1_holdout_panel`

### Stage 2

Stage 2 is conditional on realized hard exits only.

Route handling:

- pooled strategic route: `pooled_strategic = ipo + mna`
- sponsor route: `sponsor_sale`
- fallback to pooled routes when IPO support is too thin for honest live
  evaluation

Code mapping:

- route mapping: `map_stage2_route_label`
- route-class selection: `select_stage2_route_classes`
- conditional route probabilities: `build_stage2_probability_tables`,
  `predict_stage2_route_probs`

## Universe Separation

The live build separates at least two universes:

- `venture_growth`
- `buyout_pe`

Universe labels are attached before estimation and evaluation. Sector-stage
heterogeneity is then reported within each universe.

Code mapping:

- universe map: `build_company_universe_map`
- universe attachment: `attach_universe_labels`
- support audits: `build_universe_support`, `build_sector_stage_metrics`

## Feature Blocks

The governed feature blocks remain:

- `macro_time`
- `company_core`
- `sector_stage`
- `financing_trajectory`
- `sponsor_fund`
- `lp_demand`
- `patent_core`
- `patent_quality`
- `network_team`
- `interaction_bundle`

Important interpretation rule:

- feature search is diagnostic, not the main methodology
- live claims are made off the redesigned target and evaluation tables, not off
  arbitrary feature-combination wins
- live-mode CLI reruns skip feature search by default unless
  `--run-feature-search` is supplied explicitly

Code mapping:

- registry: `build_feature_registry`
- feature search: `run_feature_combo_search`
- feature coverage: `build_feature_coverage_by_block`

## Patent Handling

The patent layer is conservative and PIT-safe:

- applications become externally visible at `appldate + 18 months`
- active live predictors are `patent_apps_visible_l4q`,
  `patent_stock_visible`, and `patent_grants_l4q`
- forward citations are not used as a time-varying predictor
- filing-date backward citations are not used as a filing-time predictor
- aggregate patent effects are treated as sector-conditional diagnostics rather
  than a universal live claim

Code mapping:

- patent crosswalk confidence: `build_patent_crosswalk_confidence`
- sector-conditional activation: `add_sector_conditional_patent_features`
- patent challenger comparison: `build_patent_sector_model_comparison`

## Valuation and Decision Layer

The valuation layer remains reduced-form:

- pathwise NPV uses ownership, route-conditioned proceeds, discounting, and a
  terminal-value rule

The redesigned decision layer reports two active policy views:

1. probability-threshold screening on `hard_timely_liquidity_by_8q`
2. simplified economic screen on conservative predicted proceeds

The old CE-only rule remains appendix-grade diagnostic material and is not the
only policy reported.

Code mapping:

- future state paths: `build_future_states`
- route-conditioned incidence: `build_stage2_cumulative_incidence`
- valuation: `simulate_npv`
- redesigned policy backtests: `build_redesigned_policy_backtests`

## Evaluation and Promotion Gate

Calibration remains the headline criterion.

The live build reports:

- Brier
- integrated Brier
- ROC-AUC
- PR-AUC
- calibration slope / intercept when estimable
- top-decile lift
- policy acceptance and hit rate

On the canonical bounded buyout-only run, the selected buyout target is:

- `any_direct_realization_by_16q`

and the selected policy is chosen on validation only subject to the acceptance
band, then confirmed once on the locked test split.

In the final textbook-resolution bundle, that buyout result is retained only as
`provisional` and appendix-level because nearby bounded settings do not keep the
same buyout target/policy stable.

Those are emitted for:

- `full_test`
- `high_confidence_subset`
- `stress_slice`
- `venture_growth`
- `buyout_pe`

The redesign promotion gate is controlled by:

- `target_name`
- `enough_route_support`
- `enough_policy_activation`
- `acceptable_label_confidence`
- `acceptable_calibration_full`
- `acceptable_calibration_high_confidence`
- `chapter_evidence_ready`

Code mapping:

- evaluation tables: `build_evaluation_metrics_main`,
  `build_evaluation_metrics_by_universe`
- promotion gate: `build_promotion_gate_v2`
- explanation note: `write_promotion_gate_v2_explanation`

## Sample Mode

The bundled public sample pack lives under:

- `repo/packs/scenario/pe-vc-hazard/`

It preserves:

- the two-stage workflow surface
- the same headline target name
- pooled-route logic
- policy backtests
- deterministic outputs under fixed seeds

It does not preserve:

- real vendor identifiers
- real route-confidence mix
- real patent timelines
- real sponsor or LP histories

