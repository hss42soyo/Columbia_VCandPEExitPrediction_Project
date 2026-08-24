# Appendix - Internal Project History, Failures, Drift, and Continuation Requirements

> [!warning] Superseded internal history
> This file is preserved as an internal audit trail for the retired hard-liquidity prototype. It is not the current Chapter 9.2 / Paper 7 claim. The current 2026-05-14 position is the Żbikowski & Antosiuk (2021) forensic reproduction plus current-regime, censoring, and early-stage diligence-screen diagnostics. Do not use the "Current Frozen State" below as the current manuscript status.

Internal and removable. This appendix is not written for final publication. It
exists so the current Chapter 9 freeze can be reopened later without losing the
actual project history.

## Bottom Line

We did not achieve the original goal.

The original goal was an economically actionable VC/PE exit-prediction study
with good decision surfaces on direct realization outcomes. We did not get
there.

What we did achieve is narrower:

- we mapped the data contract honestly
- we identified which objectives were actually supportable
- we proved that proxy milestone models can work better than direct realization
  models on the current staged data
- we redesigned the public chapter around a more defensible venture
  hazard/liquidity target
- we documented why buyout remains provisional
- we proved that the Paper 7 branch, as attempted on the staged extract, is not
  a true reproduction and still does not produce a good confusion matrix

That is useful, but it is not success against the original economic objective.

## Current Frozen State

The current public Chapter 9 package is frozen around the hazard/liquidity line,
not around the earlier proxy-objective line and not around the later Paper 7
replication line.

Current chapter doctrine:

- venture main doctrine: `hard_timely_liquidity_by_8q`
- included headline routes: `ipo`, `mna`, `sponsor_sale`
- stage 1: binary hazard for hard timely liquidity
- stage 2: conditional route model among realized hard exits only
- buyout extension target: `any_direct_realization_by_16q`
- buyout status: `provisional`

Current chapter gate:

- route support: pass
- policy activation: pass
- label confidence: pass
- full-test calibration: fail
- high-confidence calibration: fail
- `chapter_evidence_ready = false`

Frozen current-state metrics from the canonical hazard/liquidity bundle:

- full-test mean absolute calibration gap: `0.1875`
- high-confidence mean absolute calibration gap: `0.1973`
- conditional stage-2 route accuracy: `0.3189`
- active screening policy acceptance: `23373` names with hit rate `0.0288`
- active economic policy acceptance: `11007` names with hit rate `0.0341`

Interpretation:

- the current chapter package is honest and runnable
- the doctrine and claim boundary are much cleaner than earlier versions
- the chapter is still a current-state empirical draft, not a promoted result

## Original Goal

The original goal was not "predict a statistically convenient proxy."

The original goal was closer to:

- can we predict direct VC/PE realizations well enough to support an investment
  decision?
- can we get strong confusion matrices or otherwise decision-grade operating
  performance on economically meaningful outcomes?
- can we replicate or extend the academic papers that appear to solve adjacent
  problems?

That goal had two sub-goals:

1. paper-aligned empirical credibility
2. practical actionability

Those two sub-goals were repeatedly in tension with the data we actually had.

## Project Timeline

### 1. Early feasibility and literature-alignment pass

We started by auditing what papers we wanted to emulate and what fields the
staged data could actually support.

This phase was useful and should not be forgotten. It established:

- not all papers were solving the same problem
- venture milestone papers were much easier to support than direct
  realization/actionability papers
- we matched some paper predictor families reasonably well
- the strongest missing predictors were relational and exit-specific:
  investor identity, syndicate structure, PE exit date, PE exit route, PE exit
  value

Main lesson:

- objective choice had to come before model choice

Main failure:

- we still kept drifting toward whatever target the staged data supported best,
  rather than holding the economically preferred target fixed and refusing to
  overclaim

### 2. Third-time and early venture-proxy work

The early serious modeling path kept converging on venture follow-on and
progression signals because those labels and predictors were the cleanest in the
staged venture data.

This branch produced real signal, but it was economically misaligned.

What worked:

- venture financing events were numerous and well dated
- company/round/team linkage was much stronger than PE exit linkage
- some engineered venture features had real signal

What failed:

- the strongest measurable target was venture follow-on, not investor
  realization
- the work felt unsatisfying because it did not answer the economic question we
  actually cared about

Main lesson:

- a statistically feasible proxy is not automatically a good headline objective

### 3. Fourth-time objective and predictor reset

The fourth-time line was the first major attempt to impose discipline on the
problem.

Core changes:

- explicit direct-vs-proxy objective families
- predictor registry with statuses like available, engineerable, cross-source,
  or structurally blocked
- full-superset screening before subset search
- true exhaustive subset search over shortlisted predictor pools rather than
  fake "full search" over impossible combinations

What worked:

- the objective mismatch became explicit
- the predictor gap became explicit
- the best proxy combinations were identified honestly
- the direct objectives were shown to be searchable but weak

What failed:

- direct realization still did not become strong
- the practical value remained mostly in proxy outcomes

Main lesson:

- feature-engineering depth helps the venture proxy side
- it does not solve missing direct exit labels or missing investor-linkage
  tables

### 4. Fifth-time funnel and model pass

The fifth-time line extended the proxy family, froze a model-worthy objective
funnel, and built actual models only for objectives that survived the evidence
gate.

Winners from that pass:

- `venture_next_priced_round_12m`
- `venture_reach_series_a_24m`
- `venture_reach_series_bplus_24m`
- `venture_up_round_within_12m`

Screening-only direct objectives:

- `venture_direct_liquidity_24m`
- `venture_direct_liquidity_36m`

Discarded after modeling:

- `venture_next_priced_round_6m`
- `venture_reach_series_a_12m`

What worked:

- we stopped pretending all objectives deserved model-building
- the proxy winners were real enough to package
- the direct-liquidity line was honestly demoted
- the data-acquisition roadmap became clearer

What failed:

- the direct economically aligned objectives still did not produce good
  confusion matrices
- the proxy winners still were not the same as solving the actionability
  problem

Main lesson:

- proxy models can be publishable in a narrow sense
- they do not solve the direct realization problem

### 5. Hazard/liquidity redesign and public chapter freeze

The current public Chapter 9 package is the result of a later redesign that
abandoned the old broad "any exit" framing and recentered the chapter on a
narrower hard-liquidity doctrine.

What worked:

- the target is more defensible
- the route set is cleaner
- the venture and buyout claim surfaces are explicitly separated
- the public sample/live package is now honest and runnable
- the note and figures match the current doctrine

What failed:

- the chapter still does not pass its own evidence gate
- buyout remains provisional
- the policy usefulness is still too weak for a strong chapter-headline claim

Main lesson:

- this redesign improved honesty and educational value
- it did not convert the empirical draft into a finished result

### 6. Paper 7 reset

After losing confidence in the earlier branches, we reset privately around the
one audited paper that looked most reproducible on paper: Paper 7
("A machine learning, bias-free approach for predicting business success using
Crunchbase data").

The plan was:

1. reproduce the paper exactly
2. confirm the result
3. then add Preqin as an extension

This failed for two reasons:

- the staged Crunchbase extract did not actually match the paper's historical
  universe
- our first implementation diverged materially from the paper on top of that

The first baseline was invalid:

- labeled sample only `429`
- test set only `86`
- class balance wildly wrong
- target and negative policy materially different from the paper

That branch should be remembered mostly as a warning.

### 7. Paper 7 updated-cohort and corrected-updated-cohort attempts

We then stopped pretending strict reproduction was possible and rebuilt the
study as a Paper 7-style updated-cohort study on the staged `2015+` extract.

First updated cohort:

- `2015-2022`
- paper-style open-ended label
- far better grounded than the 429-row attempt
- still weak

Corrected updated cohort:

- `2015-2019`
- `Series B` only
- all non-positives as negatives
- first-degree education features
- bucketed organization-size features
- CV tuning and locked-threshold selection

Best corrected result:

- best locked-threshold test model: `svm`
- precision: `0.2163`
- recall: `0.8507`
- F1: `0.3449`
- ROC-AUC: `0.6579`
- confusion matrix: `TN=352`, `FP=413`, `FN=20`, `TP=114`

That result is more defensible than the earlier Paper 7 attempts, but it still
fails the operational test because `FP >> TP`.

Main lesson:

- we corrected the main self-inflicted mistakes
- the remaining gap is mostly structural

## What Actually Worked

These are the real successes and should be preserved:

1. We eventually became much more honest about objectives.
   We stopped pretending every measurable label was the right economic target.

2. We built explicit blocker registries instead of hand-waving.
   Missing exit fields and missing investor-linkage tables were documented as
   structural blockers, not just feature-engineering TODOs.

3. We proved some proxy venture models can work at a moderate level.
   The fifth-time winners are real proxy progression models, even if they are
   not actionability models.

4. We created a clean public sample/live packaging layer for Chapter 9.
   The repo now preserves the current doctrine and claim boundary without
   shipping licensed rows.

5. We disproved our own over-optimistic branches.
   The invalid 429-row Paper 7 branch and the stronger-than-warranted early
   proxy interpretations were both corrected and frozen as non-successes.

6. We now know what data would actually matter.
   The next step is clearer than it was before: data first, not more tuning.

## What Failed

These are the failures that matter:

1. We never got a strong direct-actionability model.
   Direct liquidity and buyout realization remained weak.

2. We never got a good chapter-headline confusion matrix on the economically
   preferred outcomes.

3. The current hazard/liquidity chapter still fails the evidence gate.

4. The buyout extension never became stable enough to promote.

5. We did not reproduce Paper 7.
   Not in the strict sense, and not at the paper's performance level.

6. We spent too much time asking model families and thresholds to rescue
   problems that were really label and data problems.

## Goal Drift

The project drifted several times. This needs to be stated plainly because
future continuation should not repeat the same drift without noticing it.

### Drift 1: direct actionability to venture proxy progression

Why it happened:

- the venture labels and features were much stronger
- the PE/actionability fields were missing

What we got:

- workable proxy milestone models

What we lost:

- alignment with the original economic question

### Drift 2: proxy progression to hazard/liquidity packaging

Why it happened:

- the chapter still needed a coherent public story
- the broader proxy work was not a good final manuscript shape

What we got:

- a cleaner chapter doctrine
- stronger claim boundary

What we lost:

- continuity with some of the earlier venture-proxy work

### Drift 3: strict paper reproduction to updated-cohort paper-style study

Why it happened:

- the staged extract starts in `2015`
- strict reproduction was impossible

What we got:

- a more defensible test of what the staged extract can actually support

What we lost:

- the ability to call the result a reproduction

## Compromises We Accepted

These compromises were rational, but they should still be remembered as
compromises rather than successes:

1. We accepted proxy objectives when direct realization objectives were blocked
   or weak.

2. We accepted venture as the main empirical surface and buyout as provisional.

3. We accepted a current-state empirical draft instead of an evidence-ready
   chapter result.

4. We accepted that the Paper 7 line could only become a paper-style updated
   cohort study, not a true replication.

5. We accepted public educational packaging as a way to crystallize the work
   even though the main empirical claims remain limited.

## Blockers

### Data blockers

- no direct-dated PE exit fields such as `exit_date`, `exit_type`, `exit_value`
- no complete round-level investor participation table for the staged venture
  data
- no investor identity / reputation / syndicate-network surface
- no pre-2015 Crunchbase history in the staged Paper 7 extract
- missing `category_list` and `category_groups_list` in the staged Paper 7
  extract
- missing homepage/workflow inputs used in the original Paper 7 environment

### Label blockers

- direct realization labels remain sparse or structurally unavailable
- buyout target stability is too weak across nearby bounded settings
- Paper 7-style success labels are dominated by milestone-style positives in the
  staged extract rather than a richer realized-success surface

### Feature blockers

- sponsor/fund-state and LP-demand features remain coverage-constrained
- investor-quality features are blocked by missing relational tables
- some education and industry features remain weaker than their paper analogues

### Evaluation blockers

- current hazard/liquidity calibration is still not good enough for promotion
- current direct classifiers still have poor decision surfaces
- corrected Paper 7-style results still show `FP >> TP`

### Process blockers

- too much historical time was spent on estimator churn before hardening the
  data-contract diagnosis
- the project repeatedly slid toward what was easiest to predict rather than
  what was most economically meaningful

## What New Data Would Materially Change The Project

These items are the highest-value additions. They are listed in priority order.

1. Direct-dated PE realization fields
- `exit_date`
- `exit_type`
- `exit_value` or realized multiple

2. Round-level investor participation and investor identity
- who invested in which round
- returning-investor flags
- syndicate composition
- investor reputation / network features

3. Stronger sponsor/fund/company joins
- fund-level state
- sponsor history
- LP-demand-related features where coverage is real

4. Richer Crunchbase history and category fields
- pre-2015 history if Paper 7 or similar replications remain a goal
- `category_list`
- `category_groups_list`
- missing page/homepage workflow inputs if still needed

5. Better cross-source entity linkage
- higher-confidence Crunchbase-Preqin company matching
- normalized sector and geography taxonomies across sources

## What Model Work Is Still Worth Doing

Only after the relevant data improve:

1. Reopen direct VC/PE realization modeling.
- direct labels first
- then direct objective definition
- then model testing

2. Reopen buyout as a real empirical branch.
- only after direct-dated buyout realization fields and stronger joins exist

3. Reopen paper replication work.
- only after the source extract actually matches the target paper's universe and
  features

4. Consider bounded-follow-up redesigns on the current extract if no better data
   arrive.
- this is the only current-data model change that still looks materially
  different from what we have already tested
- it would be a new study, not a rescue of the current open-ended classifier

## What Is Not Worth Doing Again Without Better Data

These should be treated as stop-rule items:

1. More threshold churn on the same weak classifiers.
2. More hyperparameter churn on the same weak direct labels.
3. More exhaustive subset/model-family searches on structurally blocked targets.
4. Pretending proxy milestone success solves direct actionability.
5. Pretending the corrected Paper 7 branch is "almost a reproduction."

## Recommended Restart Order

If the project resumes, the order should be:

1. Rebuild the data inventory and blocker register first.
   Confirm exactly which new fields were added and which old blockers remain.

2. Re-evaluate objective families before fitting any model.
   Lock the economically preferred target first.

3. Rebuild labels and predictor registries.
   Use the same classification discipline:
   available, engineerable, cross-source derivable, or blocked.

4. Only then rerun objective-specific screening and feature construction.

5. Only then choose model families.

6. Stop immediately if direct labels are still missing or still too sparse.
   Do not rerun the same modeling loop hoping the answer changes.

## Artifact Map

Future continuation should start by reading these files in roughly this order:

### Current chapter freeze

- `repo/docs/implementation_briefs/pe_vc_hazard.md`
- `repo/examples/Chapter 9/VC & PE Exit Prediction/README.md`
- `repo/examples/Chapter 9/VC & PE Exit Prediction/python/assumptions.md`
- `Private/Cruchbase+Preqin/old/python/rendered-live-canonical-20260403/chapter_summary.md`
- `Private/Cruchbase+Preqin/old/python/rendered-live-final-doctrine-20260404/final_target_doctrine.md`
- `Private/Cruchbase+Preqin/old/python/rendered-live-final-doctrine-20260404/final_claim_boundary.md`
- `Private/Cruchbase+Preqin/old/python/rendered-live-final-doctrine-20260404/final_empirical_status_table.md`

### Objective and predictor drift

- `Private/Cruchbase+Preqin/old/fourth-time/objective_discussion.md`
- `Private/Cruchbase+Preqin/old/fourth-time/predictor_alignment_discussion.md`
- `Private/Cruchbase+Preqin/old/fifth-time/rendered-fifth-time-data-audit/data_acquisition_roadmap.md`
- `Private/Cruchbase+Preqin/old/fifth-time/rendered-fifth-time-models/summary.md`
- `Private/Cruchbase+Preqin/old/fifth-time/rendered-fifth-time-models/tables/final_model_status.csv`

### Paper 7 line

- `Private/Cruchbase+Preqin/Paper 7/paper7_execution_plan.md`
- `Private/Cruchbase+Preqin/Paper 7/docs/paper_vs_implementation.md`
- `Private/Cruchbase+Preqin/Paper 7/docs/paper7_corrected_updated_cohort_spec.md`
- `Private/Cruchbase+Preqin/Paper 7/outputs/corrected-updated-cohort-paper7/updated_cohort_assessment.md`
- `Private/Cruchbase+Preqin/Paper 7/failed-again-7.md`

## Final Recommendation

If new data arrive, resume the project from the data-contract layer, not from
the modeling layer.

If no new data arrive, do not reopen the same direct-actionability or Paper 7
classifier loops. The only remaining current-data line that still looks
meaningfully different is a bounded-follow-up redesign, and even that should be
treated as a new study rather than a continuation of a failed reproduction.
