# (c) 2027, Michael Robbins
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import zipfile
from itertools import combinations
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from pandas.api.types import CategoricalDtype

from path_helpers import LIVE_DATA_ENV_VAR, resolve_sample_pack_dir


ROUTES = ["no_exit", "ipo", "mna", "sponsor_sale", "writeoff"]
EXIT_ROUTES = ROUTES[1:]
MAIN_DIRECT_ROUTES = ["ipo", "mna", "sponsor_sale"]
HARD_TIMELY_LIQUIDITY_TARGET = "hard_timely_liquidity_by_8q"
UNIVERSE_ORDER = ["venture_growth", "buyout_pe"]
COMPANY_FEATURES = [
    "age_q",
    "time_since_last_round_q",
    "log_last_round_usd",
    "patent_flow_l4q",
    "sponsor_score",
]
PATENT_FEATURE_COLUMNS = [
    "patent_apps_visible_l4q",
    "patent_stock_visible",
    "patent_grants_l4q",
]
SECTOR_BUCKET_ORDER = [
    "technology",
    "industrial_deeptech",
    "health_lifesciences",
    "financial_services",
    "consumer_retail",
    "generic_services",
    "unknown_other",
]
SECTOR_BUCKET_BASE = "unknown_other"
SECTOR_DUMMY_BUCKETS = [bucket for bucket in SECTOR_BUCKET_ORDER if bucket != SECTOR_BUCKET_BASE]
STAGE_BUCKET_ORDER = [
    "early",
    "venture_growth",
    "buyout_late",
    "other_unknown",
]
STAGE_BUCKET_BASE = "other_unknown"
STAGE_DUMMY_BUCKETS = [bucket for bucket in STAGE_BUCKET_ORDER if bucket != STAGE_BUCKET_BASE]
PATENT_PLAUSIBLE_BUCKETS = {"technology", "industrial_deeptech", "health_lifesciences"}
BASELINE_FEATURE_GROUPS = ["macro_time", "company_core", "sector_stage", "financing_trajectory"]
OPTIONAL_FEATURE_GROUPS = ["sponsor_fund", "patent_core"]
PLACEHOLDER_FEATURE_GROUPS = ["patent_quality", "lp_demand", "network_team", "interaction_bundle"]
CANONICAL_TARGET_CALIBRATION_METRIC = "mean_abs_calibration_gap"
TARGET_BASE_FEATURE_BACKBONE = "macro_time|company_core|sector_stage|financing_trajectory"
TARGET_SPONSOR_FUND_FEATURE_BACKBONE = f"{TARGET_BASE_FEATURE_BACKBONE}|sponsor_fund"
BUYOUT_SPONSOR_FUND_FEATURES = [
    "buyout_fund_launch_count_l4q",
    "buyout_fund_final_close_usd_l4q",
    "buyout_fund_interim_close_usd_l4q",
    "buyout_fund_dpi_median_lagged",
    "buyout_fund_rvpi_median_lagged",
    "buyout_fund_multiple_median_lagged",
    "buyout_fund_net_cashflow_l4q",
    "buyout_lp_next12m_allocation_usd_lagged",
    "buyout_sponsor_raise_10y_lagged",
    "buyout_sponsor_coinvest_share_lagged",
    "buyout_returning_lp_pct_lagged",
    "buyout_fund_months_to_final_close_lagged",
]
BUYOUT_POLICY_PROBABILITY_THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
BUYOUT_POLICY_TOP_QUANTILES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
DEFAULT_CONFIG = {
    "data_mode": "sample",
    "pack_label": "scenario",
    "analysis_start_quarter": "2010Q1",
    "analysis_end_quarter": "2023Q4",
    "train_end_quarter": "2018Q4",
    "validation_end_quarter": "2021Q4",
    "test_end_quarter": "2023Q4",
    "holdout_horizon_quarters": 8,
    "n_simulations": 1000,
    "ownership": 0.12,
    "purchase_price_fraction_of_v0": 0.90,
    "tv_fraction_of_v0": 0.00,
    "gamma_risk_aversion": 2.0,
    "random_seed": 42,
    "max_train_rows": 900000,
    "target_panel_rows": 1500000,
    "entry_year_floor": 2010,
    "min_entry_year": None,
    "min_train_exits": 100,
    "min_test_exits": 50,
    "min_train_route_support": 5,
    "min_train_years": 3,
    "min_validation_quarters": 4,
    "min_test_quarters": 4,
    "company_chunk_size": 1000,
    "use_quarter_fixed_effects": None,
    "freeze_regime_shift": -1.50,
    "freeze_exit_logit_shift": -0.85,
    "freeze_kappa_multiplier": 0.80,
    "freeze_multiple_multiplier": 0.85,
    "decision_eval_paths": 48,
    "fixed_probability_thresholds": [0.01, 0.02, 0.03],
    "primary_confusion_threshold": 0.02,
    "certainty_equivalent_threshold": 0.0,
    "stress_slice_start_quarter": "2020Q1",
    "stress_slice_end_quarter": "2020Q4",
    "stylized_bucket_size": 24,
    "stress_visibility_prob_gap": 0.0025,
    "stress_visibility_npv_gap": 0.0100,
    "optimizer_maxiter": 500,
    "ridge_penalty": 1e-3,
    "patent_min_confidence": "medium",
    "feature_search_max_train_rows": 30000,
    "feature_search_decision_eval_paths": 16,
    "feature_search_validation_max_rows": 120000,
    "feature_search_test_max_rows": 160000,
    "sector_support_min_rows": 5000,
    "sector_support_min_exits": 25,
    "stage_support_min_rows": 2500,
    "stage_support_min_exits": 20,
    "stage1_min_train_events_per_universe": 25,
    "stage2_min_route_support": 5,
    "stage2_min_bucket_support": 20,
    "policy_validation_quantile": 0.90,
    "target_exploration_probability_threshold": 0.03,
    "promotion_gate_calibration_gap_max": 0.05,
    "promotion_gate_high_conf_gap_max": 0.08,
    "promotion_gate_min_policy_acceptance": 0.005,
    "promotion_gate_min_label_confidence_share": 0.85,
    "target_selection_min_direct_dated_share": 0.25,
    "target_selection_max_policy_acceptance": 0.50,
    "skip_feature_search": False,
    "buyout_only": False,
    "buyout_min_validation_positives": 25,
    "buyout_min_test_positives": 25,
    "buyout_max_inferred_transition_share": 0.75,
    "buyout_realization_min_gap_quarters": 4,
    "buyout_policy_acceptance_min": 0.005,
    "buyout_policy_acceptance_max": 0.50,
    "buyout_policy_acceptance_max_fallback": 0.60,
}
FALLBACK_MULTIPLE_PARAMS = {
    "ipo": {"mu": math.log(1.60), "sigma": 0.35, "kappa": 0.85},
    "mna": {"mu": math.log(1.25), "sigma": 0.30, "kappa": 0.75},
    "sponsor_sale": {"mu": math.log(1.10), "sigma": 0.28, "kappa": 0.70},
    "writeoff": {"mu": math.log(0.08), "sigma": 0.20, "kappa": 0.05},
}
DEFERRED_FEATURE_BLOCKS = [
    {
        "feature_family": "core_placeholder",
        "feature_name": "patent_flow_l4q",
        "status": "pit_adapter_active_in_live_mode",
        "used_in_model": "yes",
        "current_construction": "visible_apps_plus_grants_l4q",
        "dependency_note": "Actual mode now uses a PIT-safe patent adapter with 18-month application visibility; sample mode keeps a synthetic teaching proxy.",
    },
    {
        "feature_family": "core_placeholder",
        "feature_name": "sponsor_score",
        "status": "baseline_active",
        "used_in_model": "yes",
        "current_construction": "preqin_indicator_plus_investor_term",
        "dependency_note": "The richer sponsor/fund-state layer is blocked by missing firm_id and fund_id joins in the local staged Preqin deal extracts.",
    },
    {
        "feature_family": "deferred_enrichment",
        "feature_name": "fund_state_score",
        "status": "deferred_placeholder",
        "used_in_model": "no",
        "current_construction": "not_built",
        "dependency_note": "Requires Preqin fund-level link fields and PIT-safe reporting lags.",
    },
    {
        "feature_family": "deferred_enrichment",
        "feature_name": "lp_demand_score",
        "status": "deferred_placeholder",
        "used_in_model": "no",
        "current_construction": "not_built",
        "dependency_note": "Requires LP-to-fund commitment coverage not present in the current staged local files.",
    },
    {
        "feature_family": "deferred_enrichment",
        "feature_name": "service_provider_signal",
        "status": "deferred_placeholder",
        "used_in_model": "no",
        "current_construction": "not_built",
        "dependency_note": "Requires advisory/service-provider fields with consistent PIT timestamps.",
    },
    {
        "feature_family": "deferred_enrichment",
        "feature_name": "patent_citation_flow_l4q",
        "status": "deferred_placeholder",
        "used_in_model": "no",
        "current_construction": "not_built",
        "dependency_note": "Requires a patent-event adapter with PIT-safe filing and citation lags.",
    },
    {
        "feature_family": "deferred_enrichment",
        "feature_name": "company_network_signal",
        "status": "deferred_placeholder",
        "used_in_model": "no",
        "current_construction": "not_built",
        "dependency_note": "Requires company-team-network enrichment beyond the current Chapter 9 live build.",
    },
]


def quarter_idx_from_dates(series: pd.Series) -> pd.Series:
    return series.dt.year * 4 + (series.dt.quarter - 1)


def quarter_idx_from_label(label: str) -> int:
    match = re.fullmatch(r"(\d{4})Q([1-4])", label)
    if not match:
        raise ValueError(f"Invalid quarter label: {label}")
    year = int(match.group(1))
    quarter = int(match.group(2))
    return year * 4 + (quarter - 1)


def quarter_label_from_idx(idx: int) -> str:
    year = idx // 4
    quarter = idx % 4 + 1
    return f"{year}Q{quarter}"


def quarter_end_from_idx(idx: pd.Series | np.ndarray | int) -> pd.Series:
    values = pd.Series(np.atleast_1d(idx), copy=False)
    years = (values // 4).astype(int)
    quarters = (values % 4 + 1).astype(int)
    periods = pd.PeriodIndex.from_fields(year=years, quarter=quarters, freq="Q")
    result = pd.Series(periods.to_timestamp(how="end"))
    if np.isscalar(idx):
        return result.iloc[[0]]
    return result


def buyout_fund_mask(fund_details: pd.DataFrame) -> pd.Series:
    text = (
        fund_details.get("fund_type", pd.Series(index=fund_details.index, dtype=object)).fillna("").astype(str)
        + " "
        + fund_details.get("fund_focus", pd.Series(index=fund_details.index, dtype=object)).fillna("").astype(str)
    ).str.lower()
    mask = text.str.contains("buyout|private equity|pe buyout", regex=True)
    if int(mask.sum()) == 0:
        mask = pd.Series(np.ones(len(fund_details), dtype=bool), index=fund_details.index)
    return mask


def aggregate_quarter_events(
    frame: pd.DataFrame,
    date_col: str,
    value_col: str | None = None,
    agg_name: str = "value",
) -> pd.DataFrame:
    working = frame.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.loc[working[date_col].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["quarter_idx", agg_name])
    working["quarter_idx"] = quarter_idx_from_dates(working[date_col])
    if value_col is None:
        grouped = working.groupby("quarter_idx", as_index=False).size().rename(columns={"size": agg_name})
    else:
        grouped = working.groupby("quarter_idx", as_index=False).agg(
            **{agg_name: (value_col, "sum")}
        )
    return grouped.sort_values("quarter_idx").reset_index(drop=True)


def build_buyout_sponsor_fund_market_panel(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fund_details = sources.get("preqin_fund_details", pd.DataFrame()).copy()
    if fund_details.empty:
        return pd.DataFrame(columns=["quarter_idx", *BUYOUT_SPONSOR_FUND_FEATURES])
    buyout_funds = fund_details.loc[buyout_fund_mask(fund_details)].copy()
    if buyout_funds.empty:
        return pd.DataFrame(columns=["quarter_idx", *BUYOUT_SPONSOR_FUND_FEATURES])
    buyout_fund_ids = set(buyout_funds["fund_id"].dropna().astype(str))
    buyout_firm_ids = set(buyout_funds["firm_id"].dropna().astype(str))
    for column in ["final_size_usd", "latest_interim_close_size_usd"]:
        buyout_funds[column] = pd.to_numeric(buyout_funds[column], errors="coerce")
    launches = aggregate_quarter_events(
        buyout_funds,
        "fundraising_launch_date",
        value_col=None,
        agg_name="fund_launch_count",
    )
    final_closes = aggregate_quarter_events(
        buyout_funds,
        "final_close_date",
        value_col="final_size_usd",
        agg_name="final_close_usd",
    )
    interim_closes = aggregate_quarter_events(
        buyout_funds,
        "latest_interim_close_date",
        value_col="latest_interim_close_size_usd",
        agg_name="interim_close_usd",
    )
    performance = sources.get("preqin_fund_performance", pd.DataFrame()).copy()
    if not performance.empty:
        performance = performance.loc[performance["fund_id"].astype(str).isin(buyout_fund_ids)].copy()
        performance["date_reported"] = pd.to_datetime(performance["date_reported"], errors="coerce")
        for column in ["distr_dpi_pcent", "value_rvpi_pcent", "multiple"]:
            performance[column] = pd.to_numeric(performance[column], errors="coerce")
        performance = performance.loc[performance["date_reported"].notna()].copy()
        if not performance.empty:
            performance["quarter_idx"] = quarter_idx_from_dates(performance["date_reported"])
            perf_quarter = performance.groupby("quarter_idx", as_index=False).agg(
                dpi_median=("distr_dpi_pcent", "median"),
                rvpi_median=("value_rvpi_pcent", "median"),
                multiple_median=("multiple", "median"),
            )
        else:
            perf_quarter = pd.DataFrame(columns=["quarter_idx", "dpi_median", "rvpi_median", "multiple_median"])
    else:
        perf_quarter = pd.DataFrame(columns=["quarter_idx", "dpi_median", "rvpi_median", "multiple_median"])
    cashflow = sources.get("preqin_cashflow", pd.DataFrame()).copy()
    if not cashflow.empty:
        cashflow = cashflow.loc[cashflow["fund_id"].astype(str).isin(buyout_fund_ids)].copy()
        cashflow["net_cashflow"] = pd.to_numeric(cashflow["net_cashflow"], errors="coerce")
        cashflow = aggregate_quarter_events(
            cashflow,
            "transaction_date",
            value_col="net_cashflow",
            agg_name="net_cashflow",
        )
    else:
        cashflow = pd.DataFrame(columns=["quarter_idx", "net_cashflow"])
    manager_details = sources.get("preqin_manager_details", pd.DataFrame()).copy()
    if not manager_details.empty:
        manager_details = manager_details.loc[manager_details["firm_id"].astype(str).isin(buyout_firm_ids)].copy()
        manager_details["lastupdated"] = pd.to_datetime(manager_details["lastupdated"], errors="coerce")
        manager_details["totalfundsraised10yearsmn"] = pd.to_numeric(
            manager_details["totalfundsraised10yearsmn"],
            errors="coerce",
        )
        manager_details = manager_details.loc[manager_details["lastupdated"].notna()].copy()
        if not manager_details.empty:
            manager_details["quarter_idx"] = quarter_idx_from_dates(manager_details["lastupdated"])
            manager_details["coinvest_flag"] = manager_details["investorcoinvestmentrights"].astype(str).str.lower().str.contains(
                "yes|true|offer|right",
                regex=True,
            ).astype(float)
            manager_quarter = manager_details.groupby("quarter_idx", as_index=False).agg(
                sponsor_raise_10y_median=("totalfundsraised10yearsmn", "median"),
                sponsor_coinvest_share=("coinvest_flag", "mean"),
            )
        else:
            manager_quarter = pd.DataFrame(
                columns=["quarter_idx", "sponsor_raise_10y_median", "sponsor_coinvest_share"]
            )
    else:
        manager_quarter = pd.DataFrame(columns=["quarter_idx", "sponsor_raise_10y_median", "sponsor_coinvest_share"])
    fund_terms = sources.get("preqin_fund_terms", pd.DataFrame()).copy()
    if not fund_terms.empty:
        fund_terms = fund_terms.loc[fund_terms["fund_id"].astype(str).isin(buyout_fund_ids)].copy()
        fund_terms["returninginvestorspcent"] = pd.to_numeric(fund_terms["returninginvestorspcent"], errors="coerce")
        terms_dates = buyout_funds[
            ["fund_id", "fundraising_launch_date", "latest_interim_close_date", "final_close_date"]
        ].copy()
        fund_terms = fund_terms.merge(terms_dates, on="fund_id", how="left")
        fund_terms["observable_date"] = pd.to_datetime(fund_terms["final_close_date"], errors="coerce")
        fund_terms.loc[fund_terms["observable_date"].isna(), "observable_date"] = pd.to_datetime(
            fund_terms["latest_interim_close_date"],
            errors="coerce",
        )
        fund_terms.loc[fund_terms["observable_date"].isna(), "observable_date"] = pd.to_datetime(
            fund_terms["fundraising_launch_date"],
            errors="coerce",
        )
        fund_terms["months_to_final_close"] = (
            (
                pd.to_datetime(fund_terms["final_close_date"], errors="coerce")
                - pd.to_datetime(fund_terms["fundraising_launch_date"], errors="coerce")
            ).dt.days
            / 30.4375
        )
        fund_terms = fund_terms.loc[fund_terms["observable_date"].notna()].copy()
        if not fund_terms.empty:
            fund_terms["quarter_idx"] = quarter_idx_from_dates(fund_terms["observable_date"])
            terms_quarter = fund_terms.groupby("quarter_idx", as_index=False).agg(
                returning_lp_pct_median=("returninginvestorspcent", "median"),
                fund_months_to_final_close_median=("months_to_final_close", "median"),
            )
        else:
            terms_quarter = pd.DataFrame(
                columns=["quarter_idx", "returning_lp_pct_median", "fund_months_to_final_close_median"]
            )
    else:
        terms_quarter = pd.DataFrame(columns=["quarter_idx", "returning_lp_pct_median", "fund_months_to_final_close_median"])
    investor_details = sources.get("preqin_investor_details", pd.DataFrame()).copy()
    if not investor_details.empty:
        investor_details["next_12_months_quarter"] = pd.to_datetime(investor_details["next_12_months_quarter"], errors="coerce")
        investor_details["next12monthsallocationmax_pe_usd"] = pd.to_numeric(
            investor_details["next12monthsallocationmax_pe_usd"],
            errors="coerce",
        )
        investor_details = investor_details.loc[investor_details["next_12_months_quarter"].notna()].copy()
        if not investor_details.empty:
            investor_details["quarter_idx"] = quarter_idx_from_dates(investor_details["next_12_months_quarter"])
            lp_quarter = investor_details.groupby("quarter_idx", as_index=False).agg(
                lp_next12m_allocation_usd=("next12monthsallocationmax_pe_usd", "median")
            )
        else:
            lp_quarter = pd.DataFrame(columns=["quarter_idx", "lp_next12m_allocation_usd"])
    else:
        lp_quarter = pd.DataFrame(columns=["quarter_idx", "lp_next12m_allocation_usd"])
    quarter_sources = []
    for frame in [launches, final_closes, interim_closes, perf_quarter, cashflow, manager_quarter, terms_quarter, lp_quarter]:
        if not frame.empty:
            quarter_sources.extend(frame["quarter_idx"].dropna().astype(int).tolist())
    if not quarter_sources:
        return pd.DataFrame(columns=["quarter_idx", *BUYOUT_SPONSOR_FUND_FEATURES])
    quarter_idx = np.arange(min(quarter_sources), max(quarter_sources) + 1, dtype=int)
    panel = pd.DataFrame({"quarter_idx": quarter_idx})
    for frame in [launches, final_closes, interim_closes, cashflow]:
        if not frame.empty:
            panel = panel.merge(frame, on="quarter_idx", how="left")
    for column in ["fund_launch_count", "final_close_usd", "interim_close_usd", "net_cashflow"]:
        panel[column] = pd.to_numeric(panel.get(column), errors="coerce").fillna(0.0)
    for frame in [perf_quarter, manager_quarter, terms_quarter, lp_quarter]:
        if not frame.empty:
            panel = panel.merge(frame, on="quarter_idx", how="left")
    for column in [
        "dpi_median",
        "rvpi_median",
        "multiple_median",
        "lp_next12m_allocation_usd",
        "sponsor_raise_10y_median",
        "sponsor_coinvest_share",
        "returning_lp_pct_median",
        "fund_months_to_final_close_median",
    ]:
        panel[column] = pd.to_numeric(panel.get(column), errors="coerce")
        panel[column] = panel[column].ffill()
    panel["buyout_fund_launch_count_l4q"] = panel["fund_launch_count"].rolling(4, min_periods=1).sum()
    panel["buyout_fund_final_close_usd_l4q"] = panel["final_close_usd"].rolling(4, min_periods=1).sum()
    panel["buyout_fund_interim_close_usd_l4q"] = panel["interim_close_usd"].rolling(4, min_periods=1).sum()
    panel["buyout_fund_dpi_median_lagged"] = panel["dpi_median"]
    panel["buyout_fund_rvpi_median_lagged"] = panel["rvpi_median"]
    panel["buyout_fund_multiple_median_lagged"] = panel["multiple_median"]
    panel["buyout_fund_net_cashflow_l4q"] = panel["net_cashflow"].rolling(4, min_periods=1).sum()
    panel["buyout_lp_next12m_allocation_usd_lagged"] = panel["lp_next12m_allocation_usd"]
    panel["buyout_sponsor_raise_10y_lagged"] = panel["sponsor_raise_10y_median"]
    panel["buyout_sponsor_coinvest_share_lagged"] = panel["sponsor_coinvest_share"]
    panel["buyout_returning_lp_pct_lagged"] = panel["returning_lp_pct_median"]
    panel["buyout_fund_months_to_final_close_lagged"] = panel["fund_months_to_final_close_median"]
    output = panel[["quarter_idx", *BUYOUT_SPONSOR_FUND_FEATURES]].copy()
    for column in BUYOUT_SPONSOR_FUND_FEATURES:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0).astype(np.float32)
    return output


def attach_buyout_sponsor_fund_market_features(
    panel: pd.DataFrame,
    buyout_market_panel: pd.DataFrame,
) -> pd.DataFrame:
    enriched = panel.copy()
    for column in BUYOUT_SPONSOR_FUND_FEATURES:
        if column not in enriched.columns:
            enriched[column] = 0.0
    if buyout_market_panel.empty:
        return enriched
    base = enriched.reset_index(drop=False).rename(columns={"index": "_row_id"})
    base["lookup_idx"] = pd.to_numeric(base["quarter_idx"], errors="coerce").fillna(0).astype(np.int64) - 1
    market = buyout_market_panel.copy().sort_values("quarter_idx").reset_index(drop=True)
    market["quarter_idx"] = pd.to_numeric(market["quarter_idx"], errors="coerce").fillna(0).astype(np.int64)
    market = market.rename(
        columns={column: f"{column}_market" for column in BUYOUT_SPONSOR_FUND_FEATURES if column in market.columns}
    )
    merged = pd.merge_asof(
        base.sort_values("lookup_idx"),
        market.rename(columns={"quarter_idx": "market_quarter_idx"}),
        left_on="lookup_idx",
        right_on="market_quarter_idx",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("_row_id").reset_index(drop=True)
    buyout_mask = merged["universe"].astype(str).eq("buyout_pe")
    for column in BUYOUT_SPONSOR_FUND_FEATURES:
        values = pd.to_numeric(
            merged.get(f"{column}_market", pd.Series(index=merged.index, dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        merged[column] = np.where(buyout_mask, values, 0.0).astype(np.float32)
    drop_columns = ["_row_id", "lookup_idx", "market_quarter_idx"] + [f"{column}_market" for column in BUYOUT_SPONSOR_FUND_FEATURES]
    return merged.drop(columns=drop_columns, errors="ignore")


def normalize_name(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(
        r"\b(inc|llc|ltd|corp|corporation|company|co|holdings|group|plc|sa|ag|lp|partners|capital)\b",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_location_key(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


COUNTRY_CODE_ALIASES = {
    "us": "US",
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "canada": "CA",
    "ca": "CA",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "england": "GB",
    "germany": "DE",
    "de": "DE",
    "france": "FR",
    "fr": "FR",
    "singapore": "SG",
    "sg": "SG",
    "australia": "AU",
    "au": "AU",
    "netherlands": "NL",
    "nl": "NL",
    "sweden": "SE",
    "se": "SE",
    "switzerland": "CH",
    "ch": "CH",
    "ireland": "IE",
    "ie": "IE",
    "israel": "IL",
    "il": "IL",
    "japan": "JP",
    "jp": "JP",
    "south korea": "KR",
    "korea": "KR",
    "kr": "KR",
    "china": "CN",
    "cn": "CN",
    "hong kong": "HK",
    "hk": "HK",
    "india": "IN",
    "in": "IN",
}


def normalize_country_code(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in COUNTRY_CODE_ALIASES:
        return COUNTRY_CODE_ALIASES[lowered]
    if len(text) == 2 and text.isalpha():
        return text.upper()
    if len(text) == 3 and text.isalpha():
        return text[:2].upper()
    return COUNTRY_CODE_ALIASES.get(lowered)


def normalize_domain(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if "://" not in text:
        text = "http://" + text
    try:
        netloc = urlparse(text).netloc.lower()
    except ValueError:
        return None
    netloc = netloc.split("@")[-1].split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or None


def sample_sector_token(company_name: object) -> str:
    text = str(company_name).strip() if pd.notna(company_name) else ""
    parts = text.split()
    return parts[1] if len(parts) >= 2 else "Unknown"


def map_sector_bucket(raw_sector_text: object, company_name: object | None = None) -> str:
    parts = []
    if pd.notna(raw_sector_text):
        parts.append(str(raw_sector_text))
    if company_name is not None and pd.notna(company_name):
        parts.append(str(company_name))
    text = " ".join(parts).strip().lower()
    if not text:
        return "unknown_other"
    health_terms = (
        "bio",
        "biotech",
        "pharma",
        "therap",
        "med",
        "health",
        "clinical",
        "diagnostic",
        "genom",
        "life science",
    )
    industrial_terms = (
        "semiconductor",
        "chip",
        "robot",
        "battery",
        "energy",
        "grid",
        "forge",
        "works",
        "industrial",
        "manufactur",
        "hardware",
        "aerospace",
        "materials",
        "climate",
        "deep tech",
        "deep-tech",
    )
    financial_terms = (
        "fintech",
        "insur",
        "insurance",
        "lending",
        "payments",
        "payment",
        "bank",
        "wealth",
        "treasury",
        "risk",
        "credit",
    )
    consumer_terms = (
        "consumer",
        "retail",
        "commerce",
        "ecommerce",
        "e-commerce",
        "marketplace",
        "food",
        "travel",
        "fashion",
        "restaurant",
        "delivery",
        "gaming",
        "media",
    )
    technology_terms = (
        "cloud",
        "data",
        "logic",
        "system",
        "software",
        "saas",
        "ai",
        "artificial intelligence",
        "analytics",
        "cyber",
        "developer",
        "infrastructure",
        "platform",
        "automation",
    )
    service_terms = (
        "service",
        "consult",
        "staffing",
        "outsourc",
        "logistics",
        "agency",
        "real estate",
        "property",
        "support",
        "operations",
    )
    if any(term in text for term in health_terms):
        return "health_lifesciences"
    if any(term in text for term in industrial_terms):
        return "industrial_deeptech"
    if any(term in text for term in financial_terms):
        return "financial_services"
    if any(term in text for term in consumer_terms):
        return "consumer_retail"
    if any(term in text for term in technology_terms):
        return "technology"
    if any(term in text for term in service_terms):
        return "generic_services"
    return "unknown_other"


def map_stage_bucket(raw_stage_label: object) -> str:
    text = str(raw_stage_label).strip().lower() if pd.notna(raw_stage_label) else ""
    if not text:
        return "other_unknown"
    early_terms = ("seed", "angel", "grant")
    venture_terms = ("series a", "series b", "series c", "series d", "growth", "venture capital", "venture debt", "pre-ipo")
    buyout_terms = (
        "buyout",
        "public to private",
        "recapital",
        "restructuring",
        "turnaround",
        "special situations",
        "corporate investment",
        "add-on",
        "pipe",
        "merger",
        "growth capital",
    )
    if any(term in text for term in early_terms):
        return "early"
    if any(term in text for term in buyout_terms):
        return "buyout_late"
    if any(term in text for term in venture_terms):
        return "venture_growth"
    return "other_unknown"


def sector_dummy_columns() -> list[str]:
    return [f"sector_bucket_{bucket}" for bucket in SECTOR_DUMMY_BUCKETS]


def stage_dummy_columns() -> list[str]:
    return [f"stage_bucket_{bucket}" for bucket in STAGE_DUMMY_BUCKETS]


def add_bucket_feature_columns(panel: pd.DataFrame) -> pd.DataFrame:
    enriched = panel.copy()
    if "sector_bucket" not in enriched.columns:
        enriched["sector_bucket"] = "unknown_other"
    enriched["sector_bucket"] = (
        enriched["sector_bucket"]
        .where(enriched["sector_bucket"].isin(SECTOR_BUCKET_ORDER), "unknown_other")
        .fillna("unknown_other")
        .astype(str)
    )
    if "raw_stage_label" not in enriched.columns:
        stage_source = enriched["stage_or_type"] if "stage_or_type" in enriched.columns else pd.Series(np.nan, index=enriched.index)
        enriched["raw_stage_label"] = stage_source
    enriched["stage_bucket"] = enriched["raw_stage_label"].map(map_stage_bucket).astype(str)
    enriched["sector_patent_plausible"] = enriched["sector_bucket"].isin(PATENT_PLAUSIBLE_BUCKETS).astype(np.int8)
    for bucket in SECTOR_DUMMY_BUCKETS:
        enriched[f"sector_bucket_{bucket}"] = enriched["sector_bucket"].eq(bucket).astype(np.int8)
    for bucket in STAGE_DUMMY_BUCKETS:
        enriched[f"stage_bucket_{bucket}"] = enriched["stage_bucket"].eq(bucket).astype(np.int8)
    return enriched


def add_interaction_candidate_columns(panel: pd.DataFrame) -> pd.DataFrame:
    enriched = panel.copy()
    enriched["interaction_sector_patent_apps_plausible"] = (
        pd.to_numeric(enriched.get("patent_apps_visible_l4q"), errors="coerce").fillna(0.0)
        * pd.to_numeric(enriched.get("sector_patent_plausible"), errors="coerce").fillna(0.0)
    ).astype(np.float32)
    enriched["interaction_sector_patent_grants_plausible"] = (
        pd.to_numeric(enriched.get("patent_grants_l4q"), errors="coerce").fillna(0.0)
        * pd.to_numeric(enriched.get("sector_patent_plausible"), errors="coerce").fillna(0.0)
    ).astype(np.float32)
    enriched["interaction_sector_patent_stock_plausible"] = (
        pd.to_numeric(enriched.get("patent_stock_visible"), errors="coerce").fillna(0.0)
        * pd.to_numeric(enriched.get("sector_patent_plausible"), errors="coerce").fillna(0.0)
    ).astype(np.float32)
    enriched["interaction_stage_patent_apps_growth"] = (
        pd.to_numeric(enriched.get("patent_apps_visible_l4q"), errors="coerce").fillna(0.0)
        * pd.to_numeric(enriched.get("stage_bucket_venture_growth"), errors="coerce").fillna(0.0)
    ).astype(np.float32)
    enriched["interaction_stage_patent_stock_growth"] = (
        pd.to_numeric(enriched.get("patent_stock_visible"), errors="coerce").fillna(0.0)
        * pd.to_numeric(enriched.get("stage_bucket_venture_growth"), errors="coerce").fillna(0.0)
    ).astype(np.float32)
    enriched["interaction_sector_sponsor_financial"] = (
        pd.to_numeric(enriched.get("sponsor_score"), errors="coerce").fillna(0.0)
        * pd.to_numeric(enriched.get("sector_bucket_financial_services"), errors="coerce").fillna(0.0)
    ).astype(np.float32)
    enriched["interaction_sector_sponsor_generic_services"] = (
        pd.to_numeric(enriched.get("sponsor_score"), errors="coerce").fillna(0.0)
        * pd.to_numeric(enriched.get("sector_bucket_generic_services"), errors="coerce").fillna(0.0)
    ).astype(np.float32)
    enriched["interaction_stage_financing_round_buyout"] = (
        pd.to_numeric(enriched.get("log_last_round_usd"), errors="coerce").fillna(0.0)
        * pd.to_numeric(enriched.get("stage_bucket_buyout_late"), errors="coerce").fillna(0.0)
    ).astype(np.float32)
    enriched["interaction_stage_financing_tslr_growth"] = (
        pd.to_numeric(enriched.get("time_since_last_round_q"), errors="coerce").fillna(0.0)
        * pd.to_numeric(enriched.get("stage_bucket_venture_growth"), errors="coerce").fillna(0.0)
    ).astype(np.float32)
    enriched["interaction_macro_sponsor"] = (
        pd.to_numeric(enriched.get("market_regime"), errors="coerce").fillna(0.0)
        * pd.to_numeric(enriched.get("sponsor_score"), errors="coerce").fillna(0.0)
    ).astype(np.float32)
    return enriched


def read_path_aliases(
    paths_file: str | Path | None = None,
    live_data_dir: str | Path | None = None,
) -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[4]
    candidates: list[Path] = []
    if paths_file:
        candidates.append(Path(paths_file))
    candidates.append(repo_root / "local" / "paths.local.yml")
    candidates.append(repo_root / "local" / "paths.example.yml")
    selected = None
    for candidate in candidates:
        if candidate.exists():
            selected = candidate
            break
    aliases: dict[str, Path] = {}
    if selected is not None:
        current_section = None
        for raw_line in selected.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" ") and line.endswith(":"):
                current_section = line[:-1].strip()
                continue
            if current_section != "path_aliases":
                continue
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            aliases[key.strip()] = Path(value.strip())
    env_live_root = os.environ.get(LIVE_DATA_ENV_VAR, "").strip()
    licensed_root = None
    if live_data_dir:
        licensed_root = Path(live_data_dir).resolve()
    elif env_live_root:
        licensed_root = Path(env_live_root).resolve()
    aliases.setdefault("licensed_data_root", licensed_root or Path("D:/data"))
    aliases.setdefault(
        "wrds_patent_data_csv",
        aliases["licensed_data_root"] / "WRDS" / "WRDS - Patents" / "wrdsapps.patents.csv",
    )
    return aliases


def find_single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched {pattern} under {directory}")
    return matches[0]


def find_all_files(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern))


def read_zipped_csv_header(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        entry_name = archive.namelist()[0]
        with archive.open(entry_name) as handle:
            return pd.read_csv(handle, nrows=0, low_memory=False).columns.tolist()


def read_zipped_csv(zip_path: Path, desired_columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        entry_name = archive.namelist()[0]
        with archive.open(entry_name) as handle:
            header = pd.read_csv(handle, nrows=0, low_memory=False).columns.tolist()
        usecols = [column for column in desired_columns if column in header]
        with archive.open(entry_name) as handle:
            frame = pd.read_csv(handle, usecols=usecols, low_memory=False)
    missing = [column for column in desired_columns if column not in frame.columns]
    for column in missing:
        frame[column] = np.nan
    return frame[desired_columns]


def canonicalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        for column in frame.columns
    }
    return frame.rename(columns=renamed)


def read_and_union_zipped_csvs(zip_paths: list[Path], desired_columns: list[str]) -> pd.DataFrame:
    frames = []
    for zip_path in zip_paths:
        frame = canonicalize_column_names(read_zipped_csv(zip_path, read_zipped_csv_header(zip_path)))
        keep_columns = [column for column in desired_columns if column in frame.columns]
        trimmed = frame[keep_columns].copy() if keep_columns else pd.DataFrame(index=frame.index)
        for column in desired_columns:
            if column not in trimmed.columns:
                trimmed[column] = np.nan
        frames.append(trimmed[desired_columns])
    if not frames:
        return pd.DataFrame(columns=desired_columns)
    combined = pd.concat(frames, ignore_index=True)
    return combined[desired_columns]


def safe_git_hash(start_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(start_path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def load_actual_inputs(config: dict) -> dict[str, pd.DataFrame]:
    aliases = read_path_aliases(config.get("paths_file"), config.get("live_data_dir"))
    licensed_root = aliases["licensed_data_root"]
    preqin_root = licensed_root / "WRDS" / "Preqin"
    crunchbase_root = licensed_root / "CrunchBase"
    source_file_rows: list[dict[str, object]] = []

    def register_source(table_name: str, source_path: Path, source_kind: str = "zipped_csv") -> Path:
        if source_kind == "zipped_csv":
            header = read_zipped_csv_header(source_path)
        else:
            header = pd.read_csv(source_path, nrows=0).columns.tolist()
        source_file_rows.append(
            {
                "table_name": table_name,
                "source_path": str(source_path),
                "source_kind": source_kind,
                "raw_field_count": int(len(header)),
                "raw_fields": "|".join(header),
            }
        )
        return source_path

    vc_zip = register_source(
        "preqin_vc",
        find_single_file(preqin_root / "Venture Capital Deals", "*.csv.zip"),
    )
    vc = read_zipped_csv(
        vc_zip,
        [
            "portfolio_company_id",
            "deal_date",
            "ventureid",
            "stage",
            "deal_status",
            "investment_status",
            "deal_financing_size_usd",
            "total_known_funding_usd",
            "portfolio_company_name",
            "portfolio_company_website",
            "portfolio_company_country",
            "portfolio_company_region",
            "year_established",
        ],
    )
    buyout_zip = register_source(
        "preqin_buyout",
        find_single_file(preqin_root / "Buyout Deals", "*.csv.zip"),
    )
    buyout = read_zipped_csv(
        buyout_zip,
        [
            "portfolio_company_id",
            "deal_date",
            "buyout_id",
            "fund_id",
            "firm_id",
            "investment_type",
            "currency",
            "deal_status",
            "investment_status",
            "deal_size_usd",
            "dealsizeequity_usd",
            "enterprisevalue",
            "debtsize_usd",
            "managementinvests",
            "individualinvestors",
            "otherunspecifiedinvestors",
            "investmentstake",
            "acquired_share_pcent",
            "portfolio_company_name",
            "portfolio_company_website",
            "portfolio_company_state",
            "portfolio_company_country",
            "portfolio_company_region",
            "deal_description",
            "pressreleaseurl",
            "firm_about",
            "firm_othernames",
            "industry_classification",
            "primary_industry",
            "sub_industries",
            "industry_verticals",
            "industry_subverticals",
            "year_established",
        ],
    )
    vc["deal_date"] = pd.to_datetime(vc["deal_date"], errors="coerce")
    buyout["deal_date"] = pd.to_datetime(buyout["deal_date"], errors="coerce")

    register_source("cb_companies", crunchbase_root / "companies.csv", source_kind="csv")
    cb_companies = pd.read_csv(
        crunchbase_root / "companies.csv",
        usecols=[
            "company_uuid",
            "name",
            "description",
            "website",
            "founded_on",
            "operating_status",
            "last_funding_at",
            "num_funding_rounds",
            "hq_city",
            "hq_country",
            "collected_at",
        ],
    )
    cb_companies["founded_on"] = pd.to_datetime(cb_companies["founded_on"], errors="coerce")
    cb_companies["last_funding_at"] = pd.to_datetime(cb_companies["last_funding_at"], errors="coerce")
    cb_companies["collected_at"] = pd.to_datetime(cb_companies["collected_at"], errors="coerce")

    register_source("cb_rounds", crunchbase_root / "funding_rounds.csv", source_kind="csv")
    cb_rounds = pd.read_csv(
        crunchbase_root / "funding_rounds.csv",
        usecols=[
            "company_uuid",
            "announced_on",
            "investment_type",
            "money_raised_usd",
            "num_investors",
        ],
    )
    cb_rounds["announced_on"] = pd.to_datetime(cb_rounds["announced_on"], errors="coerce")
    register_source("cb_acquisitions", crunchbase_root / "acquisitions.csv", source_kind="csv")
    cb_acq = pd.read_csv(
        crunchbase_root / "acquisitions.csv",
        usecols=["acquiree_uuid", "announced_on", "price_usd", "acquisition_type"],
    )
    cb_acq["announced_on"] = pd.to_datetime(cb_acq["announced_on"], errors="coerce")
    register_source("cb_ipos", crunchbase_root / "ipos.csv", source_kind="csv")
    cb_ipos = pd.read_csv(
        crunchbase_root / "ipos.csv",
        usecols=["company_uuid", "went_public_on", "money_raised_usd", "stock_exchange"],
    )
    cb_ipos["went_public_on"] = pd.to_datetime(cb_ipos["went_public_on"], errors="coerce")
    fund_details_zip = register_source(
        "preqin_fund_details",
        find_single_file(preqin_root / "Fund Details", "*.csv.zip"),
    )
    fund_details = read_zipped_csv(
        fund_details_zip,
        [
            "firm_id",
            "fund_id",
            "fund_name",
            "firm_name",
            "vintage",
            "fund_type",
            "fundraising_launch_date",
            "target_size_usd",
            "final_size_usd",
            "latest_interim_close_size_usd",
            "latest_interim_close_date",
            "fund_status",
            "final_close_date",
            "fund_focus",
            "industry",
            "region",
            "fund_number_overall",
            "fund_number_series",
            "fund_structure",
            "geographic_scope",
            "placement_agents",
            "law_firm",
            "administrator",
            "auditor",
        ],
    )
    fund_details["fundraising_launch_date"] = pd.to_datetime(fund_details["fundraising_launch_date"], errors="coerce")
    fund_details["latest_interim_close_date"] = pd.to_datetime(fund_details["latest_interim_close_date"], errors="coerce")
    fund_details["final_close_date"] = pd.to_datetime(fund_details["final_close_date"], errors="coerce")
    fund_perf_zip = register_source(
        "preqin_fund_performance",
        find_single_file(preqin_root / "Fund Performance", "*.csv.zip"),
    )
    fund_performance = read_zipped_csv(
        fund_perf_zip,
        [
            "fund_id",
            "date_reported",
            "called_pcent",
            "distr_dpi_pcent",
            "value_rvpi_pcent",
            "multiple",
            "net_irr_pcent",
            "benchmark_id",
        ],
    )
    fund_performance["date_reported"] = pd.to_datetime(fund_performance["date_reported"], errors="coerce")
    fund_terms_zip = register_source(
        "preqin_fund_terms",
        find_single_file(preqin_root / "Fund Terms", "*.csv.zip"),
    )
    fund_terms = read_zipped_csv(
        fund_terms_zip,
        [
            "fund_id",
            "numberinvestorsmin",
            "numberinvestorsmax",
            "returninginvestorspcent",
            "investmentperiodyears",
            "annualmgmtfeeduringperiodpcent",
            "carriedinterestpercent",
            "hurdleratepercent",
        ],
    )
    manager_zip = register_source(
        "preqin_manager_details",
        find_single_file(preqin_root / "Manager Details", "*.csv.zip"),
    )
    manager_details = read_zipped_csv(
        manager_zip,
        [
            "firm_id",
            "firmname",
            "lastupdated",
            "firmtype",
            "status",
            "mainfirmstrategy",
            "totalfundsraised10yearsmn",
            "investorcoinvestmentrights",
            "totalnumberofportfoliocompanies",
            "currentnumberofportfoliocompanie",
            "geofocus",
            "countryfocus",
            "industryfocus",
            "about",
        ],
    )
    manager_details["lastupdated"] = pd.to_datetime(manager_details["lastupdated"], errors="coerce")
    investor_zips = find_all_files(preqin_root / "Investor Details", "*.csv.zip")
    for idx, investor_zip in enumerate(investor_zips, start=1):
        register_source(f"preqin_investor_details_{idx}", investor_zip)
    investor_details = read_and_union_zipped_csvs(
        investor_zips,
        [
            "firm_id",
            "firm_name",
            "firm_type",
            "currently_investing_pe",
            "funds_under_management_usd",
            "current_pe_allocation_usd",
            "target_pe_allocation_usd",
            "coinvest_with_gp",
            "first_close_investor",
            "separate_accounts",
            "next12monthsallocationmin_pe_usd",
            "next12monthsallocationmax_pe_usd",
            "next_12_months_quarter",
        ],
    )
    investor_details["next_12_months_quarter"] = pd.to_datetime(investor_details["next_12_months_quarter"], errors="coerce")
    cashflow_zip = register_source(
        "preqin_cashflow",
        find_single_file(preqin_root / "Cash Flow", "*.csv.zip"),
    )
    cashflow = read_zipped_csv(
        cashflow_zip,
        [
            "fund_id",
            "transaction_date",
            "firm_id",
            "transaction_type",
            "transaction_amount",
            "net_cashflow",
        ],
    )
    cashflow["transaction_date"] = pd.to_datetime(cashflow["transaction_date"], errors="coerce")
    return {
        "preqin_vc": vc,
        "preqin_buyout": buyout,
        "cb_companies": cb_companies,
        "cb_rounds": cb_rounds,
        "cb_acquisitions": cb_acq,
        "cb_ipos": cb_ipos,
        "preqin_fund_details": fund_details,
        "preqin_fund_performance": fund_performance,
        "preqin_fund_terms": fund_terms,
        "preqin_manager_details": manager_details,
        "preqin_investor_details": investor_details,
        "preqin_cashflow": cashflow,
        "source_file_inventory": pd.DataFrame(source_file_rows),
    }


def sample_pack_root(config: dict) -> Path:
    return resolve_sample_pack_dir(
        Path(__file__).resolve().parent,
        config.get("sample_pack_dir"),
    )


def load_sample_inputs(config: dict) -> dict[str, pd.DataFrame]:
    pack_root = sample_pack_root(config)
    company = pd.read_csv(pack_root / "dim_private_company.csv")
    company["founded_date"] = pd.to_datetime(company["founded_date"], errors="coerce")
    rounds = pd.read_csv(pack_root / "fact_private_round.csv")
    rounds["round_date"] = pd.to_datetime(rounds["round_date"], errors="coerce")
    investors = pd.read_csv(pack_root / "fact_private_investor_participation.csv")
    exits = pd.read_csv(pack_root / "fact_private_exit.csv")
    exits["exit_date"] = pd.to_datetime(exits["exit_date"], errors="coerce")
    funds = pd.read_csv(pack_root / "dim_fund.csv")
    regimes = pd.read_csv(pack_root / "event_market_regime.csv")
    return {
        "company": company,
        "rounds": rounds,
        "investors": investors,
        "exits": exits,
        "funds": funds,
        "regimes": regimes,
    }


def first_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def build_preqin_company_master(vc: pd.DataFrame, buyout: pd.DataFrame) -> pd.DataFrame:
    vc_company = vc[
        [
            "portfolio_company_id",
            "portfolio_company_name",
            "portfolio_company_website",
            "portfolio_company_country",
            "portfolio_company_region",
            "year_established",
        ]
    ].copy()
    buyout_company = buyout[
        [
            "portfolio_company_id",
            "portfolio_company_name",
            "portfolio_company_website",
            "portfolio_company_country",
            "portfolio_company_region",
            "year_established",
        ]
    ].copy()
    combined = pd.concat([vc_company, buyout_company], ignore_index=True)
    combined["normalized_name"] = combined["portfolio_company_name"].map(normalize_name)
    combined["normalized_domain"] = combined["portfolio_company_website"].map(normalize_domain)
    company_master = (
        combined.sort_values("portfolio_company_id")
        .groupby("portfolio_company_id", as_index=False)
        .agg(
            {
                "portfolio_company_name": first_non_null,
                "portfolio_company_website": first_non_null,
                "portfolio_company_country": first_non_null,
                "portfolio_company_region": first_non_null,
                "year_established": first_non_null,
                "normalized_name": first_non_null,
                "normalized_domain": first_non_null,
            }
        )
    )
    return company_master


def build_crosswalk(preqin_company_master: pd.DataFrame, cb_companies: pd.DataFrame) -> pd.DataFrame:
    preqin = preqin_company_master.copy()
    cb = cb_companies.copy()
    preqin["normalized_country"] = preqin["portfolio_company_country"].str.lower()
    cb["normalized_name"] = cb["name"].map(normalize_name)
    cb["normalized_domain"] = cb["website"].map(normalize_domain)
    cb["normalized_country"] = cb["hq_country"].str.lower()

    preqin_domain_counts = (
        preqin.dropna(subset=["normalized_domain"])
        .groupby("normalized_domain")["portfolio_company_id"]
        .nunique()
    )
    cb_domain_counts = (
        cb.dropna(subset=["normalized_domain"]).groupby("normalized_domain")["company_uuid"].nunique()
    )
    unique_domains = set(preqin_domain_counts[preqin_domain_counts == 1].index) & set(
        cb_domain_counts[cb_domain_counts == 1].index
    )
    domain_matches = preqin[preqin["normalized_domain"].isin(unique_domains)][
        ["portfolio_company_id", "normalized_domain"]
    ].merge(
        cb[cb["normalized_domain"].isin(unique_domains)][["company_uuid", "normalized_domain"]],
        on="normalized_domain",
        how="inner",
    )
    domain_matches["match_method"] = "domain"
    domain_matches["match_confidence"] = "high"

    matched_preqin = set(domain_matches["portfolio_company_id"])
    matched_cb = set(domain_matches["company_uuid"])

    preqin_name_pool = preqin[
        (~preqin["portfolio_company_id"].isin(matched_preqin)) & preqin["normalized_name"].notna()
    ].copy()
    cb_name_pool = cb[(~cb["company_uuid"].isin(matched_cb)) & cb["normalized_name"].notna()].copy()
    preqin_name_counts = preqin_name_pool.groupby("normalized_name")["portfolio_company_id"].nunique()
    cb_name_counts = cb_name_pool.groupby("normalized_name")["company_uuid"].nunique()
    unique_names = set(preqin_name_counts[preqin_name_counts == 1].index) & set(
        cb_name_counts[cb_name_counts == 1].index
    )
    preqin_name_pool = preqin_name_pool[preqin_name_pool["normalized_name"].isin(unique_names)]
    cb_name_pool = cb_name_pool[cb_name_pool["normalized_name"].isin(unique_names)]
    name_matches = preqin_name_pool[
        ["portfolio_company_id", "normalized_name", "normalized_country"]
    ].merge(
        cb_name_pool[["company_uuid", "normalized_name", "normalized_country"]],
        on="normalized_name",
        how="inner",
        suffixes=("_preqin", "_cb"),
    )
    country_ok = (
        name_matches["normalized_country_preqin"].isna()
        | name_matches["normalized_country_cb"].isna()
        | (name_matches["normalized_country_preqin"] == name_matches["normalized_country_cb"])
    )
    name_matches = name_matches[country_ok].copy()
    name_matches["match_method"] = "name"
    name_matches["match_confidence"] = "medium"
    return pd.concat(
        [
            domain_matches[
                ["portfolio_company_id", "company_uuid", "match_method", "match_confidence"]
            ],
            name_matches[
                ["portfolio_company_id", "company_uuid", "match_method", "match_confidence"]
            ],
        ],
        ignore_index=True,
    ).drop_duplicates(["portfolio_company_id", "company_uuid"])


def build_company_master(
    preqin_master: pd.DataFrame,
    cb_companies: pd.DataFrame,
    cb_rounds: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    cb_with_rounds = set(cb_rounds["company_uuid"].dropna().unique())
    matched_cb = set(crosswalk["company_uuid"].dropna().unique())

    preqin = preqin_master.merge(crosswalk, on="portfolio_company_id", how="left").merge(
        cb_companies.add_prefix("cb_"),
        left_on="company_uuid",
        right_on="cb_company_uuid",
        how="left",
    )
    preqin["company_id"] = "preqin_" + preqin["portfolio_company_id"].astype(str)
    preqin["company_name"] = preqin["cb_name"].fillna(preqin["portfolio_company_name"])
    preqin["preqin_company_name"] = preqin["portfolio_company_name"]
    preqin["cb_company_name"] = preqin["cb_name"]
    preqin["website"] = preqin["cb_website"].fillna(preqin["portfolio_company_website"])
    preqin["country"] = preqin["cb_hq_country"].fillna(preqin["portfolio_company_country"])
    preqin["city"] = preqin["cb_hq_city"]
    preqin["region"] = preqin["portfolio_company_region"]
    preqin["founded_date"] = preqin["cb_founded_on"]
    missing_founded = preqin["founded_date"].isna() & preqin["year_established"].notna()
    preqin.loc[missing_founded, "founded_date"] = pd.to_datetime(
        preqin.loc[missing_founded, "year_established"].astype(int).astype(str) + "-01-01",
        errors="coerce",
    )
    preqin["company_source"] = "preqin"
    preqin["raw_sector_text"] = preqin["cb_description"].fillna(preqin["company_name"])
    preqin["sector_bucket"] = [
        map_sector_bucket(raw_text, company_name)
        for raw_text, company_name in zip(preqin["raw_sector_text"], preqin["company_name"], strict=True)
    ]

    cb_only = cb_companies[
        cb_companies["company_uuid"].isin(cb_with_rounds - matched_cb)
    ].copy()
    cb_only["company_id"] = "crunchbase_" + cb_only["company_uuid"].astype(str)
    cb_only["company_name"] = cb_only["name"]
    cb_only["preqin_company_name"] = np.nan
    cb_only["cb_company_name"] = cb_only["name"]
    cb_only["country"] = cb_only["hq_country"]
    cb_only["city"] = cb_only["hq_city"]
    cb_only["region"] = np.nan
    cb_only["founded_date"] = cb_only["founded_on"]
    cb_only["company_source"] = "crunchbase"
    cb_only["portfolio_company_id"] = np.nan
    cb_only["match_method"] = np.nan
    cb_only["match_confidence"] = np.nan
    cb_only["cb_company_uuid"] = cb_only["company_uuid"]
    cb_only["cb_operating_status"] = cb_only["operating_status"]
    cb_only["cb_last_funding_at"] = cb_only["last_funding_at"]
    cb_only["cb_num_funding_rounds"] = cb_only["num_funding_rounds"]
    cb_only["cb_collected_at"] = cb_only["collected_at"]
    cb_only["company_uuid"] = cb_only["company_uuid"]
    cb_only["website"] = cb_only["website"]
    cb_only["raw_sector_text"] = cb_only["description"].fillna(cb_only["company_name"])
    cb_only["sector_bucket"] = [
        map_sector_bucket(raw_text, company_name)
        for raw_text, company_name in zip(cb_only["raw_sector_text"], cb_only["company_name"], strict=True)
    ]

    preqin_master_columns = [
        "company_id",
        "company_name",
        "preqin_company_name",
        "cb_company_name",
        "website",
        "country",
        "city",
        "region",
        "founded_date",
        "company_source",
        "portfolio_company_id",
        "match_method",
        "match_confidence",
        "company_uuid",
        "cb_company_uuid",
        "cb_operating_status",
        "cb_last_funding_at",
        "cb_num_funding_rounds",
        "cb_collected_at",
        "raw_sector_text",
        "sector_bucket",
    ]
    company_master = pd.concat(
        [
            preqin[preqin_master_columns],
            cb_only[preqin_master_columns],
        ],
        ignore_index=True,
    )
    company_master["normalized_name"] = company_master["company_name"].map(normalize_name)
    company_master["normalized_domain"] = company_master["website"].map(normalize_domain)
    company_master["normalized_country_code"] = company_master["country"].map(normalize_country_code)
    company_master["normalized_city"] = company_master["city"].map(normalize_location_key)
    return company_master.drop_duplicates("company_id").reset_index(drop=True)


def build_round_events(
    company_master: pd.DataFrame,
    vc: pd.DataFrame,
    buyout: pd.DataFrame,
    cb_rounds: pd.DataFrame,
) -> pd.DataFrame:
    preqin_id_map = (
        company_master.dropna(subset=["portfolio_company_id"])[
            ["portfolio_company_id", "company_id"]
        ]
        .drop_duplicates()
    )
    cb_id_map = (
        company_master.dropna(subset=["cb_company_uuid"])[["cb_company_uuid", "company_id"]]
        .drop_duplicates()
        .rename(columns={"cb_company_uuid": "company_uuid"})
    )

    vc_events = vc.merge(preqin_id_map, on="portfolio_company_id", how="left")
    vc_events = vc_events.rename(columns={"deal_date": "event_date"})
    vc_events["round_amount_usd"] = vc_events["deal_financing_size_usd"].fillna(
        vc_events["total_known_funding_usd"]
    )
    vc_events["num_investors"] = np.nan
    vc_events["source"] = "preqin_vc"
    vc_events["stage_or_type"] = vc_events["stage"]

    buyout_events = buyout.merge(preqin_id_map, on="portfolio_company_id", how="left")
    buyout_events = buyout_events.rename(columns={"deal_date": "event_date"})
    buyout_events["round_amount_usd"] = buyout_events["dealsizeequity_usd"].fillna(
        buyout_events["deal_size_usd"]
    )
    buyout_events["num_investors"] = np.nan
    buyout_events["source"] = "preqin_buyout"
    buyout_events["stage_or_type"] = buyout_events["investment_type"]

    cb_events = cb_rounds.merge(cb_id_map, on="company_uuid", how="left")
    cb_events = cb_events.rename(columns={"announced_on": "event_date"})
    cb_events["round_amount_usd"] = cb_events["money_raised_usd"]
    cb_events["source"] = "crunchbase_round"
    cb_events["stage_or_type"] = cb_events["investment_type"]

    combined = pd.concat(
        [
            vc_events[["company_id", "event_date", "round_amount_usd", "num_investors", "source", "stage_or_type"]],
            buyout_events[
                ["company_id", "event_date", "round_amount_usd", "num_investors", "source", "stage_or_type"]
            ],
            cb_events[["company_id", "event_date", "round_amount_usd", "num_investors", "source", "stage_or_type"]],
        ],
        ignore_index=True,
    )
    combined = combined.dropna(subset=["company_id", "event_date"]).copy()
    combined["quarter_idx"] = quarter_idx_from_dates(combined["event_date"])
    combined = (
        combined.sort_values(["company_id", "event_date"])
        .groupby(["company_id", "quarter_idx"], as_index=False)
        .agg(
            round_date=("event_date", "max"),
            round_amount_usd=("round_amount_usd", "max"),
            num_investors=("num_investors", "max"),
            event_sources=("source", lambda values: "|".join(sorted(set(values)))),
            stage_or_type=("stage_or_type", first_non_null),
        )
    )
    return combined


def classify_buyout_route(investment_type: object, deal_description: object) -> tuple[str | None, str]:
    investment_text = str(investment_type).strip().lower() if pd.notna(investment_type) else ""
    description_text = str(deal_description).strip().lower() if pd.notna(deal_description) else ""
    merger_terms = ("merger", "acquisition", "acquired", "strategic", "trade sale")
    sponsor_terms = (
        "buyout",
        "secondary buyout",
        "recapitalisation",
        "recapitalization",
        "growth capital",
        "public to private",
        "add-on",
        "corporate investment",
        "pipe",
        "sponsor",
        "private equity",
    )
    if investment_text in {"merger"} or any(term in description_text for term in merger_terms):
        return "mna", "medium"
    if any(term in investment_text for term in sponsor_terms):
        return "sponsor_sale", "medium"
    if any(term in description_text for term in sponsor_terms):
        return "sponsor_sale", "medium"
    return None, "unresolved"


def build_direct_exit_candidates(
    company_master: pd.DataFrame,
    round_events: pd.DataFrame,
    vc: pd.DataFrame,
    buyout: pd.DataFrame,
    cb_acq: pd.DataFrame,
    cb_ipos: pd.DataFrame,
) -> pd.DataFrame:
    first_round = round_events.groupby("company_id", as_index=False).agg(
        entry_date=("round_date", "min"),
        first_round_quarter=("quarter_idx", "min"),
    )
    cb_id_map = (
        company_master.dropna(subset=["cb_company_uuid"])[["cb_company_uuid", "company_id"]]
        .drop_duplicates()
        .rename(columns={"cb_company_uuid": "company_uuid"})
    )
    preqin_id_map = (
        company_master.dropna(subset=["portfolio_company_id"])[["portfolio_company_id", "company_id"]]
        .drop_duplicates()
    )

    ipo_events = cb_ipos.merge(cb_id_map, on="company_uuid", how="left").rename(
        columns={"went_public_on": "event_date", "money_raised_usd": "event_value_usd"}
    )
    ipo_events["route_label"] = "ipo"
    ipo_events["confidence_tier"] = "high"
    ipo_events["route_source"] = "crunchbase_ipo"

    acquisition_events = cb_acq.merge(
        cb_id_map, left_on="acquiree_uuid", right_on="company_uuid", how="left"
    ).rename(columns={"announced_on": "event_date", "price_usd": "event_value_usd"})
    acquisition_events["route_label"] = "mna"
    acquisition_events["confidence_tier"] = "high"
    acquisition_events["route_source"] = "crunchbase_acquisition"

    buyout_events = buyout.merge(preqin_id_map, on="portfolio_company_id", how="left").merge(
        first_round[["company_id", "entry_date"]], on="company_id", how="left"
    )
    buyout_events["route_and_confidence"] = buyout_events.apply(
        lambda row: classify_buyout_route(row["investment_type"], row["deal_description"]), axis=1
    )
    buyout_events["route_label"] = buyout_events["route_and_confidence"].map(lambda value: value[0])
    buyout_events["confidence_tier"] = buyout_events["route_and_confidence"].map(lambda value: value[1])
    buyout_events = buyout_events[
        buyout_events["route_label"].notna()
        & buyout_events["company_id"].notna()
        & buyout_events["deal_date"].notna()
        & buyout_events["entry_date"].notna()
        & (buyout_events["deal_date"] > buyout_events["entry_date"])
    ].copy()
    buyout_events["event_date"] = buyout_events["deal_date"]
    buyout_events["event_value_usd"] = buyout_events["deal_size_usd"].fillna(
        buyout_events["enterprisevalue"]
    )
    buyout_events["route_source"] = "preqin_buyout_transition"

    direct_candidates = pd.concat(
        [
            ipo_events[["company_id", "event_date", "event_value_usd", "route_label", "confidence_tier", "route_source"]],
            acquisition_events[
                ["company_id", "event_date", "event_value_usd", "route_label", "confidence_tier", "route_source"]
            ],
            buyout_events[
                ["company_id", "event_date", "event_value_usd", "route_label", "confidence_tier", "route_source"]
            ],
        ],
        ignore_index=True,
    )
    direct_candidates = direct_candidates.dropna(subset=["company_id", "event_date"]).copy()
    direct_candidates["quarter_idx"] = quarter_idx_from_dates(direct_candidates["event_date"])
    direct_candidates["priority"] = direct_candidates["confidence_tier"].map(
        {"high": 0, "medium": 1, "low": 2}
    )
    return direct_candidates


def build_sensitivity_exit_candidates(
    company_master: pd.DataFrame,
    round_events: pd.DataFrame,
    vc: pd.DataFrame,
    direct_candidates: pd.DataFrame,
) -> pd.DataFrame:
    company_with_direct = set(direct_candidates["company_id"].unique())
    cb_writeoff = company_master[
        company_master["cb_operating_status"].fillna("").str.lower().eq("closed")
        & company_master["company_id"].isin(set(company_master["company_id"]) - company_with_direct)
    ][["company_id", "cb_last_funding_at"]].merge(
        round_events.groupby("company_id", as_index=False).agg(last_round_date=("round_date", "max")),
        on="company_id",
        how="left",
    )
    cb_writeoff["event_date"] = cb_writeoff["cb_last_funding_at"].fillna(cb_writeoff["last_round_date"]) + pd.DateOffset(
        months=24
    )
    cb_writeoff["event_value_usd"] = 0.0
    cb_writeoff["route_label"] = "soft_failure_sensitivity"
    cb_writeoff["confidence_tier"] = "low"
    cb_writeoff["route_source"] = "crunchbase_closed_proxy"
    cb_writeoff["priority"] = 3

    vc_company_state = (
        vc.dropna(subset=["deal_date"])
        .sort_values(["portfolio_company_id", "deal_date"])
        .groupby("portfolio_company_id", as_index=False)
        .agg(last_vc_date=("deal_date", "max"), last_vc_status=("investment_status", "last"))
    )
    preqin_writeoff = company_master[["company_id", "portfolio_company_id"]].merge(
        vc_company_state,
        on="portfolio_company_id",
        how="left",
    )
    preqin_writeoff = preqin_writeoff[
        preqin_writeoff["company_id"].isin(set(company_master["company_id"]) - company_with_direct)
        & preqin_writeoff["last_vc_status"].fillna("").str.lower().eq("realized")
    ].copy()
    preqin_writeoff["event_date"] = preqin_writeoff["last_vc_date"] + pd.DateOffset(months=24)
    preqin_writeoff["event_value_usd"] = 0.0
    preqin_writeoff["route_label"] = "soft_failure_sensitivity"
    preqin_writeoff["confidence_tier"] = "low"
    preqin_writeoff["route_source"] = "preqin_realized_proxy"
    preqin_writeoff["priority"] = 3

    sensitivity_candidates = pd.concat(
        [
            cb_writeoff[
                [
                    "company_id",
                    "event_date",
                    "event_value_usd",
                    "route_label",
                    "confidence_tier",
                    "route_source",
                    "priority",
                ]
            ],
            preqin_writeoff[
                [
                    "company_id",
                    "event_date",
                    "event_value_usd",
                    "route_label",
                    "confidence_tier",
                    "route_source",
                    "priority",
                ]
            ],
        ],
        ignore_index=True,
    )
    sensitivity_candidates = sensitivity_candidates.dropna(subset=["company_id", "event_date"]).copy()
    sensitivity_candidates["quarter_idx"] = quarter_idx_from_dates(pd.to_datetime(sensitivity_candidates["event_date"]))
    return sensitivity_candidates


def choose_first_exit(exit_candidates: pd.DataFrame, analysis_end_idx: int) -> pd.DataFrame:
    candidates = exit_candidates.copy()
    candidates = candidates[candidates["quarter_idx"] <= analysis_end_idx].copy()
    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    candidates["confidence_rank"] = candidates["confidence_tier"].map(confidence_rank).fillna(3)
    chosen = (
        candidates.sort_values(["company_id", "quarter_idx", "priority", "confidence_rank"])
        .groupby("company_id", as_index=False)
        .first()
    )
    return chosen[
        [
            "company_id",
            "event_date",
            "quarter_idx",
            "route_label",
            "confidence_tier",
            "route_source",
            "event_value_usd",
        ]
    ].rename(columns={"quarter_idx": "exit_quarter_idx", "event_date": "exit_date"})


def build_route_audit(exit_candidates: pd.DataFrame, chosen_exits: pd.DataFrame) -> pd.DataFrame:
    raw = (
        exit_candidates.groupby(["route_label", "confidence_tier", "route_source"], as_index=False)
        .size()
        .rename(columns={"size": "candidate_count"})
    )
    chosen = (
        chosen_exits.groupby(["route_label", "confidence_tier", "route_source"], as_index=False)
        .size()
        .rename(columns={"size": "chosen_exit_count"})
    )
    audit = raw.merge(
        chosen,
        on=["route_label", "confidence_tier", "route_source"],
        how="outer",
    ).fillna(0)
    return audit.sort_values(["route_label", "confidence_tier", "route_source"]).reset_index(drop=True)


def build_route_confidence_summary(
    direct_candidates: pd.DataFrame,
    chosen_main: pd.DataFrame,
    sensitivity_candidates: pd.DataFrame,
    chosen_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for mapping_scope, candidates, chosen in [
        ("main", direct_candidates, chosen_main),
        (
            "sensitivity",
            sensitivity_candidates,
            chosen_sensitivity[chosen_sensitivity["route_label"] == "soft_failure_sensitivity"].copy(),
        ),
    ]:
        raw = (
            candidates.groupby(["confidence_tier", "route_source"], as_index=False)
            .size()
            .rename(columns={"size": "candidate_count"})
        )
        selected = (
            chosen.groupby(["confidence_tier", "route_source"], as_index=False)
            .size()
            .rename(columns={"size": "chosen_exit_count"})
        )
        merged = raw.merge(selected, on=["confidence_tier", "route_source"], how="outer").fillna(0)
        merged["mapping_scope"] = mapping_scope
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)[
        ["mapping_scope", "confidence_tier", "route_source", "candidate_count", "chosen_exit_count"]
    ].sort_values(["mapping_scope", "confidence_tier", "route_source"]).reset_index(drop=True)


def build_route_mapping_comparison(
    chosen_main: pd.DataFrame,
    chosen_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for mapping_scope, frame in [("main", chosen_main), ("sensitivity", chosen_sensitivity)]:
        counts = frame["route_label"].value_counts().to_dict()
        for route_label, chosen_count in sorted(counts.items()):
            rows.append(
                {
                    "mapping_scope": mapping_scope,
                    "route_label": route_label,
                    "chosen_exit_count": int(chosen_count),
                }
            )
    return pd.DataFrame(rows).sort_values(["mapping_scope", "route_label"]).reset_index(drop=True)


def build_target_definition_main() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_name": HARD_TIMELY_LIQUIDITY_TARGET,
                "included_route_label": "ipo",
                "required_confidence_tier": "high_or_medium",
                "included_in_headline_target": 1,
                "note": "Direct IPO liquidity route only.",
            },
            {
                "target_name": HARD_TIMELY_LIQUIDITY_TARGET,
                "included_route_label": "mna",
                "required_confidence_tier": "high_or_medium",
                "included_in_headline_target": 1,
                "note": "Direct acquisition or strategic sale route only.",
            },
            {
                "target_name": HARD_TIMELY_LIQUIDITY_TARGET,
                "included_route_label": "sponsor_sale",
                "required_confidence_tier": "high_or_medium",
                "included_in_headline_target": 1,
                "note": "Direct sponsor-to-sponsor or sponsor-led sale route only.",
            },
            {
                "target_name": HARD_TIMELY_LIQUIDITY_TARGET,
                "included_route_label": "soft_failure_sensitivity",
                "required_confidence_tier": "low",
                "included_in_headline_target": 0,
                "note": "Soft-failure proxies are excluded from the headline target and kept only as sensitivity evidence.",
            },
        ]
    )


def build_target_definition_sensitivity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_name": "soft_failure_sensitivity",
                "route_label": "soft_failure_sensitivity",
                "included_in_headline_target": 0,
                "reporting_role": "sensitivity_only",
                "note": "Tracks low-confidence failure-side proxies such as Crunchbase closed-status and Preqin realized proxies.",
            }
        ]
    )


def build_label_confidence_audit(
    chosen_main: pd.DataFrame,
    chosen_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for target_scope, frame in [
        ("main_hard_liquidity", chosen_main.copy()),
        (
            "sensitivity_only",
            chosen_sensitivity.loc[
                chosen_sensitivity["route_label"].astype(str).eq("soft_failure_sensitivity")
            ].copy(),
        ),
    ]:
        if frame.empty:
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "target_scope": target_scope,
                            "route_label": np.nan,
                            "confidence_tier": np.nan,
                            "route_source": np.nan,
                            "chosen_exit_count": 0,
                        }
                    ]
                )
            )
            continue
        grouped = (
            frame.groupby(["route_label", "confidence_tier", "route_source"], as_index=False)
            .size()
            .rename(columns={"size": "chosen_exit_count"})
        )
        grouped["target_scope"] = target_scope
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)[
        ["target_scope", "route_label", "confidence_tier", "route_source", "chosen_exit_count"]
    ].sort_values(["target_scope", "route_label", "confidence_tier", "route_source"]).reset_index(drop=True)


def split_pipe_values(value: object) -> list[str]:
    text = str(value).strip() if pd.notna(value) else ""
    if not text:
        return []
    return [part for part in text.split("|") if part]


def target_file_key(target_name: object, universe: object) -> str:
    return f"{str(universe).strip()}__{str(target_name).strip()}"


def target_stage2_classes(spec: pd.Series | dict) -> list[str]:
    return split_pipe_values(spec["stage2_route_set"])


def target_confidence_allowed(confidence_tier: object, rule: object) -> bool:
    tier = str(confidence_tier).strip().lower() if pd.notna(confidence_tier) else ""
    confidence_rule = str(rule).strip().lower() if pd.notna(rule) else ""
    if confidence_rule == "high_only":
        return tier == "high"
    if confidence_rule in {"high_or_medium", "high_or_medium_or_synthetic"}:
        return tier in {"high", "medium", "synthetic"}
    if confidence_rule == "synthetic_only":
        return tier == "synthetic"
    return tier not in {"", "low"}


def target_event_observation_kind(route_source: object) -> str:
    source = str(route_source).strip().lower() if pd.notna(route_source) else ""
    if source == "synthetic_route_process":
        return "synthetic_dated_event"
    if source in {"crunchbase_ipo", "crunchbase_acquisition"}:
        return "direct_dated_event"
    if source == "preqin_buyout_transition":
        return "inferred_transition"
    if "partial" in source:
        return "partial_realization"
    if "proxy" in source:
        return "sensitivity_proxy"
    return "other"


def directness_class_from_route_source(route_source: object) -> str:
    source = str(route_source).strip().lower() if pd.notna(route_source) else ""
    if source == "synthetic_route_process":
        return "direct_dated"
    if source in {"crunchbase_ipo", "crunchbase_acquisition"}:
        return "direct_dated"
    if source == "preqin_buyout_transition":
        return "inferred_transition"
    if "proxy" in source:
        return "proxy_only"
    if "direct" in source:
        return "direct_dated"
    return "other"


def observation_kind_from_directness(
    route_source: object,
    directness_class: object | None = None,
) -> str:
    directness = str(directness_class).strip().lower() if pd.notna(directness_class) else ""
    if directness == "direct_dated":
        return "direct_dated_event"
    if directness == "direct_undated":
        return "direct_undated_event"
    if directness == "inferred_transition":
        return "inferred_transition"
    if directness == "proxy_only":
        return "sensitivity_proxy"
    return target_event_observation_kind(route_source)


def build_target_registry() -> pd.DataFrame:
    base_rows = [
        {
            "target_name": HARD_TIMELY_LIQUIDITY_TARGET,
            "universe": "venture_growth",
            "horizon_quarters": 8,
            "included_routes": "ipo|mna|sponsor_sale",
            "excluded_routes": "soft_failure_sensitivity|writeoff",
            "label_confidence_rule": "high_or_medium_or_synthetic",
            "allowed_source_rules": "crunchbase_ipo|crunchbase_acquisition|preqin_buyout_transition|synthetic_route_process",
            "partial_realizations_included": 0,
            "stage2_route_set": "pooled_strategic|sponsor_sale",
            "candidate_role": "locked_baseline",
            "benchmark_row": 1,
            "data_supported": 1,
            "support_note": "Locked venture/growth baseline target.",
        },
        {
            "target_name": HARD_TIMELY_LIQUIDITY_TARGET,
            "universe": "buyout_pe",
            "horizon_quarters": 8,
            "included_routes": "ipo|mna|sponsor_sale",
            "excluded_routes": "soft_failure_sensitivity|writeoff",
            "label_confidence_rule": "high_or_medium_or_synthetic",
            "allowed_source_rules": "crunchbase_ipo|crunchbase_acquisition|preqin_buyout_transition|synthetic_route_process",
            "partial_realizations_included": 0,
            "stage2_route_set": "pooled_strategic|sponsor_sale",
            "candidate_role": "benchmark_only",
            "benchmark_row": 1,
            "data_supported": 1,
            "support_note": "Current buyout/PE benchmark row for apples-to-apples comparison only.",
        },
        {
            "target_name": "sponsor_sale_or_mna_by_12q",
            "universe": "buyout_pe",
            "horizon_quarters": 12,
            "included_routes": "mna|sponsor_sale",
            "excluded_routes": "ipo|soft_failure_sensitivity|writeoff",
            "label_confidence_rule": "high_or_medium_or_synthetic",
            "allowed_source_rules": "crunchbase_acquisition|preqin_buyout_transition|synthetic_route_process",
            "partial_realizations_included": 0,
            "stage2_route_set": "mna|sponsor_sale",
            "candidate_role": "candidate",
            "benchmark_row": 0,
            "data_supported": 1,
            "support_note": "Buyout candidate that removes the thin IPO tail and extends the horizon to 12 quarters.",
        },
        {
            "target_name": "sponsor_sale_or_mna_by_16q",
            "universe": "buyout_pe",
            "horizon_quarters": 16,
            "included_routes": "mna|sponsor_sale",
            "excluded_routes": "ipo|soft_failure_sensitivity|writeoff",
            "label_confidence_rule": "high_or_medium_or_synthetic",
            "allowed_source_rules": "crunchbase_acquisition|preqin_buyout_transition|synthetic_route_process",
            "partial_realizations_included": 0,
            "stage2_route_set": "mna|sponsor_sale",
            "candidate_role": "candidate",
            "benchmark_row": 0,
            "data_supported": 1,
            "support_note": "Longer-horizon buyout realization candidate built from sponsor-sale and M&A routes only.",
        },
        {
            "target_name": "hard_liquidity_by_12q",
            "universe": "buyout_pe",
            "horizon_quarters": 12,
            "included_routes": "ipo|mna|sponsor_sale",
            "excluded_routes": "soft_failure_sensitivity|writeoff",
            "label_confidence_rule": "high_or_medium_or_synthetic",
            "allowed_source_rules": "crunchbase_ipo|crunchbase_acquisition|preqin_buyout_transition|synthetic_route_process",
            "partial_realizations_included": 0,
            "stage2_route_set": "pooled_strategic|sponsor_sale",
            "candidate_role": "candidate",
            "benchmark_row": 0,
            "data_supported": 1,
            "support_note": "Buyout candidate with the same direct-liquidity principles but a longer 12-quarter horizon.",
        },
        {
            "target_name": "hard_liquidity_by_16q",
            "universe": "buyout_pe",
            "horizon_quarters": 16,
            "included_routes": "ipo|mna|sponsor_sale",
            "excluded_routes": "soft_failure_sensitivity|writeoff",
            "label_confidence_rule": "high_or_medium_or_synthetic",
            "allowed_source_rules": "crunchbase_ipo|crunchbase_acquisition|preqin_buyout_transition|synthetic_route_process",
            "partial_realizations_included": 0,
            "stage2_route_set": "pooled_strategic|sponsor_sale",
            "candidate_role": "candidate",
            "benchmark_row": 0,
            "data_supported": 1,
            "support_note": "Longer-horizon buyout candidate that keeps the full hard-liquidity route set.",
        },
        {
            "target_name": "partial_or_full_realization_by_12q",
            "universe": "buyout_pe",
            "horizon_quarters": 12,
            "included_routes": "ipo|mna|sponsor_sale|partial_realization",
            "excluded_routes": "soft_failure_sensitivity|writeoff",
            "label_confidence_rule": "high_or_medium_or_synthetic",
            "allowed_source_rules": "crunchbase_ipo|crunchbase_acquisition|preqin_buyout_transition|synthetic_route_process",
            "partial_realizations_included": 1,
            "stage2_route_set": "pooled_strategic|sponsor_sale|partial_realization",
            "candidate_role": "candidate_unsupported",
            "benchmark_row": 0,
            "data_supported": 0,
            "support_note": "No dated partial-realization fields are loaded from the staged local Preqin extracts, so this candidate remains definition-only.",
        },
        {
            "target_name": "partial_or_full_realization_by_16q",
            "universe": "buyout_pe",
            "horizon_quarters": 16,
            "included_routes": "ipo|mna|sponsor_sale|partial_realization",
            "excluded_routes": "soft_failure_sensitivity|writeoff",
            "label_confidence_rule": "high_or_medium_or_synthetic",
            "allowed_source_rules": "crunchbase_ipo|crunchbase_acquisition|preqin_buyout_transition|synthetic_route_process",
            "partial_realizations_included": 1,
            "stage2_route_set": "pooled_strategic|sponsor_sale|partial_realization",
            "candidate_role": "candidate_unsupported",
            "benchmark_row": 0,
            "data_supported": 0,
            "support_note": "No dated partial-realization fields are loaded from the staged local Preqin extracts, so this candidate remains definition-only.",
        },
        {
            "target_name": "recap_or_secondary_or_exit_by_12q",
            "universe": "buyout_pe",
            "horizon_quarters": 12,
            "included_routes": "ipo|mna|sponsor_sale|partial_realization",
            "excluded_routes": "soft_failure_sensitivity|writeoff",
            "label_confidence_rule": "high_or_medium_or_synthetic",
            "allowed_source_rules": "crunchbase_ipo|crunchbase_acquisition|preqin_buyout_transition|synthetic_route_process",
            "partial_realizations_included": 1,
            "stage2_route_set": "pooled_strategic|sponsor_sale|partial_realization",
            "candidate_role": "candidate_unsupported",
            "benchmark_row": 0,
            "data_supported": 0,
            "support_note": "The staged local extracts do not include dated recap, continuation, or secondary realization fields, so this candidate remains definition-only.",
        },
        {
            "target_name": "recap_or_secondary_or_exit_by_16q",
            "universe": "buyout_pe",
            "horizon_quarters": 16,
            "included_routes": "ipo|mna|sponsor_sale|partial_realization",
            "excluded_routes": "soft_failure_sensitivity|writeoff",
            "label_confidence_rule": "high_or_medium_or_synthetic",
            "allowed_source_rules": "crunchbase_ipo|crunchbase_acquisition|preqin_buyout_transition|synthetic_route_process",
            "partial_realizations_included": 1,
            "stage2_route_set": "pooled_strategic|sponsor_sale|partial_realization",
            "candidate_role": "candidate_unsupported",
            "benchmark_row": 0,
            "data_supported": 0,
            "support_note": "The staged local extracts do not include dated recap, continuation, or secondary realization fields, so this candidate remains definition-only.",
        },
    ]
    registry = pd.DataFrame(base_rows)
    registry["target_key"] = registry.apply(lambda row: target_file_key(row["target_name"], row["universe"]), axis=1)
    registry["canonical_feature_backbone"] = TARGET_BASE_FEATURE_BACKBONE
    registry["sponsor_fund_challenger_status"] = np.where(
        registry["universe"].astype(str).eq("buyout_pe") & registry["data_supported"].astype(int).eq(1),
        "buyout_market_quarter_candidate",
        "not_applicable",
    )
    return registry[
        [
            "target_key",
            "target_name",
            "universe",
            "horizon_quarters",
            "included_routes",
            "excluded_routes",
            "label_confidence_rule",
            "allowed_source_rules",
            "partial_realizations_included",
            "stage2_route_set",
            "candidate_role",
            "benchmark_row",
            "data_supported",
            "canonical_feature_backbone",
            "sponsor_fund_challenger_status",
            "support_note",
        ]
    ].copy()


def build_target_definition_from_spec(spec: pd.Series | dict) -> pd.DataFrame:
    included_rows = [
        {
            "target_key": spec["target_key"],
            "target_name": spec["target_name"],
            "universe": spec["universe"],
            "horizon_quarters": int(spec["horizon_quarters"]),
            "route_label": route_label,
            "included_in_target": 1,
            "label_confidence_rule": spec["label_confidence_rule"],
            "allowed_source_rules": spec["allowed_source_rules"],
            "partial_realizations_included": int(spec["partial_realizations_included"]),
            "stage2_route_set": spec["stage2_route_set"],
            "candidate_role": spec["candidate_role"],
            "data_supported": int(spec["data_supported"]),
            "note": spec["support_note"],
        }
        for route_label in split_pipe_values(spec["included_routes"])
    ]
    excluded_rows = [
        {
            "target_key": spec["target_key"],
            "target_name": spec["target_name"],
            "universe": spec["universe"],
            "horizon_quarters": int(spec["horizon_quarters"]),
            "route_label": route_label,
            "included_in_target": 0,
            "label_confidence_rule": spec["label_confidence_rule"],
            "allowed_source_rules": spec["allowed_source_rules"],
            "partial_realizations_included": int(spec["partial_realizations_included"]),
            "stage2_route_set": spec["stage2_route_set"],
            "candidate_role": spec["candidate_role"],
            "data_supported": int(spec["data_supported"]),
            "note": spec["support_note"],
        }
        for route_label in split_pipe_values(spec["excluded_routes"])
    ]
    return pd.DataFrame(included_rows + excluded_rows)


def write_target_definition_markdown(path: Path, definition: pd.DataFrame) -> None:
    if definition.empty:
        path.write_text("# Target Definition\n\nNo definition rows.\n", encoding="utf-8")
        return
    row = definition.iloc[0]
    included = definition[definition["included_in_target"].astype(int).eq(1)]["route_label"].astype(str).tolist()
    excluded = definition[definition["included_in_target"].astype(int).eq(0)]["route_label"].astype(str).tolist()
    lines = [
        f"# Target Definition: {row['target_name']}",
        "",
        f"- Target key: `{row['target_key']}`",
        f"- Universe: `{row['universe']}`",
        f"- Horizon quarters: `{int(row['horizon_quarters'])}`",
        f"- Included routes: `{ '|'.join(included) }`",
        f"- Excluded routes: `{ '|'.join(excluded) }`",
        f"- Label confidence rule: `{row['label_confidence_rule']}`",
        f"- Allowed source rules: `{row['allowed_source_rules']}`",
        f"- Partial realizations included: `{bool(row['partial_realizations_included'])}`",
        f"- Stage 2 route set: `{row['stage2_route_set']}`",
        f"- Data supported: `{bool(row['data_supported'])}`",
        f"- Note: {row['note']}",
        "",
        dataframe_to_markdown(definition),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def target_spec_allows_event(spec: pd.Series | dict, route_label: object, confidence_tier: object, route_source: object) -> bool:
    if int(spec["data_supported"]) != 1:
        return False
    if str(route_label) not in split_pipe_values(spec["included_routes"]):
        return False
    if not target_confidence_allowed(confidence_tier, spec["label_confidence_rule"]):
        return False
    allowed_sources = split_pipe_values(spec["allowed_source_rules"])
    return not allowed_sources or str(route_source) in allowed_sources


def build_target_candidate_panel(panel: pd.DataFrame, spec: pd.Series | dict) -> tuple[pd.DataFrame, str, str]:
    universe = str(spec["universe"])
    horizon_quarters = int(spec["horizon_quarters"])
    target_key = str(spec["target_key"])
    target_col = f"realized_{target_key}_by_horizon"
    realized_prefix = f"realized_{target_key}_by_h"
    subset = panel[panel["universe"].astype(str).eq(universe)].copy().reset_index(drop=True)
    if subset.empty:
        subset[target_col] = pd.Series(dtype=int)
        return subset, target_col, realized_prefix
    event_mask = np.array(
        [
            target_spec_allows_event(spec, route, confidence, source)
            for route, confidence, source in zip(
                subset["company_exit_route"].astype(str),
                subset["company_exit_confidence_tier"].astype(str),
                subset["company_exit_route_source"].astype(str),
            )
        ],
        dtype=bool,
    )
    exit_quarter_idx = pd.to_numeric(subset["exit_quarter_idx"], errors="coerce")
    quarter_idx = pd.to_numeric(subset["quarter_idx"], errors="coerce")
    subset[target_col] = (
        event_mask
        & exit_quarter_idx.notna()
        & exit_quarter_idx.ge(quarter_idx)
        & exit_quarter_idx.le(quarter_idx + horizon_quarters - 1)
    ).astype(int)
    subset["target_positive_route"] = np.where(subset[target_col].astype(int).eq(1), subset["company_exit_route"], np.nan)
    subset["target_positive_confidence_tier"] = np.where(
        subset[target_col].astype(int).eq(1),
        subset["company_exit_confidence_tier"],
        np.nan,
    )
    subset["target_positive_route_source"] = np.where(
        subset[target_col].astype(int).eq(1),
        subset["company_exit_route_source"],
        np.nan,
    )
    subset["target_positive_directness_class"] = np.where(
        subset[target_col].astype(int).eq(1),
        subset["company_exit_route_source"].map(directness_class_from_route_source),
        np.nan,
    )
    subset["target_event_quarter_idx"] = np.where(
        subset[target_col].astype(int).eq(1),
        exit_quarter_idx,
        np.nan,
    )
    subset["target_positive_observation_kind"] = np.where(
        subset[target_col].astype(int).eq(1),
        [
            observation_kind_from_directness(source, directness)
            for source, directness in zip(
                subset["company_exit_route_source"],
                subset["target_positive_directness_class"],
                strict=True,
            )
        ],
        None,
    )
    for horizon_step in range(1, horizon_quarters + 1):
        subset[f"{realized_prefix}{horizon_step}"] = (
            event_mask
            & exit_quarter_idx.notna()
            & exit_quarter_idx.ge(quarter_idx)
            & exit_quarter_idx.le(quarter_idx + horizon_step - 1)
        ).astype(int)
    return subset, target_col, realized_prefix


def build_target_event_frame(
    candidate_panel: pd.DataFrame,
    spec: pd.Series | dict,
    target_col: str | None = None,
) -> pd.DataFrame:
    if int(spec["data_supported"]) != 1:
        return pd.DataFrame(
            columns=[
                "split",
                "quarter_idx",
                "route_label",
                "company_id",
                "confidence_tier",
                "route_source",
                "event_observation_kind",
                "directness_class",
                "target_event_quarter_idx",
            ]
        )
    if candidate_panel.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "quarter_idx",
                "route_label",
                "company_id",
                "confidence_tier",
                "route_source",
                "event_observation_kind",
                "directness_class",
                "target_event_quarter_idx",
            ]
        )
    event_col = target_col or f"realized_{str(spec['target_key'])}_by_horizon"
    events = candidate_panel.loc[
        pd.to_numeric(candidate_panel.get(event_col), errors="coerce").fillna(0).astype(int).eq(1)
    ].copy()
    if events.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "quarter_idx",
                "route_label",
                "company_id",
                "confidence_tier",
                "route_source",
                "event_observation_kind",
                "directness_class",
                "target_event_quarter_idx",
            ]
        )
    directness = events.get("target_positive_directness_class", pd.Series(index=events.index, dtype=object))
    output = pd.DataFrame(
        {
            "split": events["split"].to_numpy(),
            "quarter_idx": pd.to_numeric(events["quarter_idx"], errors="coerce").to_numpy(),
            "route_label": events.get("target_positive_route", pd.Series(index=events.index, dtype=object)).to_numpy(),
            "company_id": events["company_id"].to_numpy(),
            "confidence_tier": events.get(
                "target_positive_confidence_tier",
                pd.Series(index=events.index, dtype=object),
            ).to_numpy(),
            "route_source": events.get(
                "target_positive_route_source",
                pd.Series(index=events.index, dtype=object),
            ).to_numpy(),
            "event_observation_kind": events.get(
                "target_positive_observation_kind",
                pd.Series(index=events.index, dtype=object),
            ).to_numpy(),
            "directness_class": directness.to_numpy(),
            "target_event_quarter_idx": pd.to_numeric(
                events.get("target_event_quarter_idx"),
                errors="coerce",
            ).to_numpy(),
        }
    )
    return output


def build_target_prevalence_by_split(
    candidate_panel: pd.DataFrame,
    target_col: str,
    spec: pd.Series | dict,
) -> pd.DataFrame:
    if candidate_panel.empty:
        return pd.DataFrame(
            columns=[
                "target_key",
                "target_name",
                "universe",
                "split",
                "rows",
                "positive_rows",
                "positive_companies",
                "prevalence",
            ]
        )
    grouped = candidate_panel.groupby("split", as_index=False).agg(
        rows=("company_id", "size"),
        positive_rows=(target_col, "sum"),
    )
    positive_companies = (
        candidate_panel.loc[candidate_panel[target_col].astype(int).eq(1)]
        .groupby("split", as_index=False)["company_id"]
        .nunique()
        .rename(columns={"company_id": "positive_companies"})
    )
    grouped = grouped.merge(positive_companies, on="split", how="left").fillna({"positive_companies": 0})
    grouped["positive_companies"] = grouped["positive_companies"].astype(int)
    grouped["prevalence"] = grouped["positive_rows"] / grouped["rows"].clip(lower=1)
    grouped["target_key"] = spec["target_key"]
    grouped["target_name"] = spec["target_name"]
    grouped["universe"] = spec["universe"]
    return grouped.sort_values("split").reset_index(drop=True)


def target_stage2_route_support_count(events: pd.DataFrame, stage2_route_label: str) -> tuple[int, int]:
    if events.empty:
        return 0, 0
    if stage2_route_label == "pooled_strategic":
        subset = events.loc[events["route_label"].astype(str).isin(["ipo", "mna"])].copy()
    else:
        subset = events.loc[events["route_label"].astype(str).eq(str(stage2_route_label))].copy()
    return int(len(subset)), int(subset["company_id"].nunique()) if not subset.empty else 0


def resolve_target_stage2_classes_from_events(
    candidate_events: pd.DataFrame,
    spec: pd.Series | dict,
) -> list[str]:
    requested = target_stage2_classes(spec)
    if not requested:
        return []
    train_events = candidate_events.loc[candidate_events["split"].astype(str).eq("train")].copy()
    resolved: list[str] = []
    for route_name in requested:
        support, _ = target_stage2_route_support_count(train_events, route_name)
        if support > 0 or len(requested) == 1:
            resolved.append(str(route_name))
    if resolved:
        return resolved
    return requested[:1]


def build_target_route_support_by_split(
    candidate_events: pd.DataFrame,
    spec: pd.Series | dict,
    config: dict,
    stage2_classes: list[str] | None = None,
) -> pd.DataFrame:
    routes = split_pipe_values(spec["included_routes"])
    splits = ["train", "validation", "test"]
    base = pd.MultiIndex.from_product([splits, routes], names=["split", "route_label"]).to_frame(index=False)
    if candidate_events.empty:
        base["positive_event_count"] = 0
        base["positive_companies"] = 0
    else:
        support = candidate_events.groupby(["split", "route_label"], as_index=False, observed=True).agg(
            positive_event_count=("company_id", "size"),
            positive_companies=("company_id", "nunique"),
        )
        base = base.merge(support, on=["split", "route_label"], how="left").fillna(0)
        base["positive_event_count"] = base["positive_event_count"].astype(int)
        base["positive_companies"] = base["positive_companies"].astype(int)
    ipo_train_events = int(
        base.loc[
            base["split"].astype(str).eq("train") & base["route_label"].astype(str).eq("ipo"),
            "positive_event_count",
        ].sum()
    )
    base["ipo_support_adequate_for_standalone"] = int(
        ipo_train_events >= int(config.get("stage2_min_route_support", 5))
    )
    base["partial_realizations_data_supported"] = int(spec["data_supported"]) if int(spec["partial_realizations_included"]) == 1 else 0
    base["target_key"] = spec["target_key"]
    base["target_name"] = spec["target_name"]
    base["universe"] = spec["universe"]
    base["support_scope"] = "raw_target_routes"
    actual_stage2_classes = stage2_classes or resolve_target_stage2_classes_from_events(candidate_events, spec)
    base["requested_stage2_route_set"] = "|".join(target_stage2_classes(spec))
    base["actual_stage2_route_set"] = "|".join(actual_stage2_classes)

    stage2_rows: list[dict[str, object]] = []
    for split in splits:
        split_events = candidate_events.loc[candidate_events["split"].astype(str).eq(split)].copy()
        for route_name in actual_stage2_classes:
            positive_event_count, positive_companies = target_stage2_route_support_count(split_events, str(route_name))
            stage2_rows.append(
                {
                    "split": split,
                    "route_label": str(route_name),
                    "positive_event_count": positive_event_count,
                    "positive_companies": positive_companies,
                    "ipo_support_adequate_for_standalone": int(
                        any(value == "ipo" for value in target_stage2_classes(spec))
                        and positive_event_count >= int(config.get("stage2_min_route_support", 5))
                    ),
                    "partial_realizations_data_supported": int(spec["data_supported"]) if int(spec["partial_realizations_included"]) == 1 else 0,
                    "target_key": spec["target_key"],
                    "target_name": spec["target_name"],
                    "universe": spec["universe"],
                    "support_scope": "stage2_actual_view",
                    "requested_stage2_route_set": "|".join(target_stage2_classes(spec)),
                    "actual_stage2_route_set": "|".join(actual_stage2_classes),
                }
            )
    stage2_frame = pd.DataFrame(stage2_rows)
    if stage2_frame.empty:
        stage2_frame = pd.DataFrame(
            columns=[
                "split",
                "route_label",
                "positive_event_count",
                "positive_companies",
                "ipo_support_adequate_for_standalone",
                "partial_realizations_data_supported",
                "target_key",
                "target_name",
                "universe",
                "support_scope",
                "requested_stage2_route_set",
                "actual_stage2_route_set",
            ]
        )
    return pd.concat([base, stage2_frame], ignore_index=True, sort=False)


def build_target_source_mix(candidate_events: pd.DataFrame, spec: pd.Series | dict) -> pd.DataFrame:
    if candidate_events.empty:
        return pd.DataFrame(
            columns=[
                "target_key",
                "target_name",
                "universe",
                "split",
                "route_source",
                "confidence_tier",
                "event_observation_kind",
                "positive_event_count",
                "positive_companies",
            ]
        )
    grouped = candidate_events.groupby(
        ["split", "route_source", "confidence_tier", "event_observation_kind"],
        as_index=False,
        observed=True,
    ).agg(
        positive_event_count=("company_id", "size"),
        positive_companies=("company_id", "nunique"),
    )
    grouped["target_key"] = spec["target_key"]
    grouped["target_name"] = spec["target_name"]
    grouped["universe"] = spec["universe"]
    return grouped.sort_values(["split", "route_source", "confidence_tier"]).reset_index(drop=True)


def build_target_label_confidence_audit(candidate_events: pd.DataFrame, spec: pd.Series | dict) -> pd.DataFrame:
    if candidate_events.empty:
        return pd.DataFrame(
            columns=[
                "target_key",
                "target_name",
                "universe",
                "split",
                "route_label",
                "confidence_tier",
                "route_source",
                "event_observation_kind",
                "positive_event_count",
            ]
        )
    grouped = candidate_events.groupby(
        ["split", "route_label", "confidence_tier", "route_source", "event_observation_kind"],
        as_index=False,
        observed=True,
    ).size().rename(columns={"size": "positive_event_count"})
    grouped["target_key"] = spec["target_key"]
    grouped["target_name"] = spec["target_name"]
    grouped["universe"] = spec["universe"]
    return grouped.sort_values(["split", "route_label", "confidence_tier", "route_source"]).reset_index(drop=True)


def build_target_time_distribution(candidate_events: pd.DataFrame, spec: pd.Series | dict) -> pd.DataFrame:
    if candidate_events.empty:
        return pd.DataFrame(
            columns=[
                "target_key",
                "target_name",
                "universe",
                "split",
                "exit_year",
                "exit_quarter_label",
                "route_label",
                "positive_event_count",
            ]
        )
    timed = candidate_events.copy()
    timed["exit_quarter_idx"] = pd.to_numeric(
        timed.get("target_event_quarter_idx", timed.get("exit_quarter_idx", timed["quarter_idx"])),
        errors="coerce",
    ).fillna(pd.to_numeric(timed["quarter_idx"], errors="coerce"))
    timed["exit_year"] = (pd.to_numeric(timed["exit_quarter_idx"], errors="coerce") // 4).astype(int)
    timed["exit_quarter_label"] = timed["exit_quarter_idx"].map(lambda value: quarter_label_from_idx(int(value)))
    grouped = timed.groupby(["split", "exit_year", "exit_quarter_label", "route_label"], as_index=False, observed=True).size()
    grouped = grouped.rename(columns={"size": "positive_event_count"})
    grouped["target_key"] = spec["target_key"]
    grouped["target_name"] = spec["target_name"]
    grouped["universe"] = spec["universe"]
    return grouped.sort_values(["split", "exit_year", "route_label"]).reset_index(drop=True)


def build_sample_route_audit(chosen_exits: pd.DataFrame) -> pd.DataFrame:
    audit = (
        chosen_exits.groupby(["route_label", "confidence_tier", "route_source"], as_index=False)
        .size()
        .rename(columns={"size": "chosen_exit_count"})
    )
    audit["candidate_count"] = audit["chosen_exit_count"]
    return audit[["route_label", "confidence_tier", "route_source", "candidate_count", "chosen_exit_count"]]


def resolve_patent_confidence_rank(label: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}[label]


def build_patent_match_inputs(company_master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for company in company_master.itertuples(index=False):
        alias_candidates = [
            ("primary_name", getattr(company, "company_name", None)),
            ("cb_name", getattr(company, "cb_company_name", None)),
            ("preqin_name", getattr(company, "preqin_company_name", None)),
        ]
        seen_aliases: set[str] = set()
        for alias_source, raw_name in alias_candidates:
            normalized = normalize_name(raw_name)
            if normalized is None or normalized in seen_aliases:
                continue
            seen_aliases.add(normalized)
            rows.append(
                {
                    "company_id": getattr(company, "company_id"),
                    "company_name": getattr(company, "company_name"),
                    "alias_source": alias_source,
                    "alias_name": raw_name,
                    "normalized_name": normalized,
                    "normalized_country_code": getattr(company, "normalized_country_code"),
                    "normalized_city": getattr(company, "normalized_city"),
                }
            )
    matcher = pd.DataFrame(rows)
    if matcher.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "company_name",
                "alias_source",
                "alias_name",
                "normalized_name",
                "normalized_country_code",
                "normalized_city",
            ]
        )
    return matcher.drop_duplicates(["company_id", "alias_source", "normalized_name"]).reset_index(drop=True)


def load_patent_matches(
    company_master: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aliases = read_path_aliases(config.get("paths_file"))
    patent_path = Path(aliases["wrds_patent_data_csv"]).resolve()
    matcher = build_patent_match_inputs(company_master)
    if matcher.empty:
        empty_matches = pd.DataFrame(
            columns=[
                "company_id",
                "patnum",
                "patnum_kpss",
                "ptype",
                "grantdate",
                "appldate",
                "applnum",
                "observable_date",
                "match_method",
                "confidence_tier",
            ]
        )
        empty_audit = pd.DataFrame(
            columns=[
                "alias_source",
                "confidence_tier",
                "match_method",
                "candidate_patent_rows",
                "candidate_patents",
                "used_patent_rows",
                "used_patents",
                "matched_companies",
                "ambiguous_patent_rows",
            ]
        )
        return empty_matches, empty_audit, empty_matches.copy()

    candidate_names = set(matcher["normalized_name"].dropna().unique().tolist())
    matched_frames: list[pd.DataFrame] = []
    min_rank = resolve_patent_confidence_rank(str(config.get("patent_min_confidence", "medium")).strip().lower())
    alias_rank = {"primary_name": 0, "cb_name": 1, "preqin_name": 1}
    for chunk in pd.read_csv(
        patent_path,
        usecols=[
            "patnum",
            "patnum_kpss",
            "ptype",
            "grantdate",
            "appldate",
            "applnum",
            "ee_number",
            "ee_name",
            "ee_role",
            "ee_role_desc",
            "ee_ind_fname",
            "ee_ind_lname",
            "ee_country",
            "ee_state",
            "ee_city",
            "backward_cites",
            "forward_cites",
        ],
        chunksize=200000,
        dtype={
            "patnum": "string",
            "patnum_kpss": "string",
            "ptype": "string",
            "applnum": "string",
            "ee_number": "string",
            "ee_name": "string",
            "ee_role": "string",
            "ee_role_desc": "string",
            "ee_ind_fname": "string",
            "ee_ind_lname": "string",
            "ee_country": "string",
            "ee_state": "string",
            "ee_city": "string",
        },
    ):
        chunk["grantdate"] = pd.to_datetime(chunk["grantdate"], errors="coerce")
        chunk["appldate"] = pd.to_datetime(chunk["appldate"], errors="coerce")
        chunk["observable_date"] = chunk["appldate"] + pd.DateOffset(months=18)
        chunk.loc[chunk["observable_date"].isna(), "observable_date"] = chunk["grantdate"]
        chunk["normalized_name"] = chunk["ee_name"].map(normalize_name)
        chunk = chunk[
            chunk["normalized_name"].isin(candidate_names)
            & chunk["observable_date"].notna()
        ].copy()
        if chunk.empty:
            continue
        chunk["assignee_country_code"] = chunk["ee_country"].map(normalize_country_code)
        chunk["assignee_city_key"] = chunk["ee_city"].map(normalize_location_key)
        joined = chunk.merge(
            matcher,
            on="normalized_name",
            how="inner",
            suffixes=("", "_company"),
        )
        if joined.empty:
            continue
        joined["country_match"] = (
            joined["assignee_country_code"].notna()
            & joined["normalized_country_code"].notna()
            & (joined["assignee_country_code"] == joined["normalized_country_code"])
        )
        joined["city_match"] = (
            joined["assignee_city_key"].notna()
            & joined["normalized_city"].notna()
            & (joined["assignee_city_key"] == joined["normalized_city"])
        )
        joined["match_method"] = np.select(
            [
                joined["country_match"] & joined["city_match"],
                joined["country_match"],
            ],
            ["exact_name_country_city", "exact_name_country"],
            default="exact_name_only",
        )
        joined["confidence_tier"] = np.select(
            [
                joined["country_match"] & joined["city_match"],
                joined["country_match"],
            ],
            ["high", "medium"],
            default="low",
        )
        joined["confidence_rank"] = joined["confidence_tier"].map(resolve_patent_confidence_rank).astype(int)
        joined["alias_rank"] = joined["alias_source"].map(alias_rank).fillna(2).astype(int)
        best_rank = joined.groupby("patnum")["confidence_rank"].transform("max")
        joined = joined[joined["confidence_rank"] == best_rank].copy()
        best_alias_rank = joined.groupby(["patnum", "company_id"])["alias_rank"].transform("min")
        joined = joined[joined["alias_rank"] == best_alias_rank].copy()
        joined["ambiguous_match"] = joined.groupby("patnum")["company_id"].transform("nunique") > 1
        joined["used_for_features"] = (~joined["ambiguous_match"]) & (joined["confidence_rank"] >= min_rank)
        matched_frames.append(
            joined[
                [
                    "company_id",
                    "company_name",
                    "alias_source",
                    "alias_name",
                    "alias_rank",
                    "patnum",
                    "patnum_kpss",
                    "ptype",
                    "grantdate",
                    "appldate",
                    "applnum",
                    "observable_date",
                    "match_method",
                    "confidence_tier",
                    "confidence_rank",
                    "used_for_features",
                    "ambiguous_match",
                ]
            ].copy()
        )

    if not matched_frames:
        empty_matches = pd.DataFrame(
            columns=[
                "company_id",
                "patnum",
                "patnum_kpss",
                "ptype",
                "grantdate",
                "appldate",
                "applnum",
                "observable_date",
                "match_method",
                "confidence_tier",
            ]
        )
        empty_audit = pd.DataFrame(
            columns=[
                "alias_source",
                "confidence_tier",
                "match_method",
                "candidate_patent_rows",
                "candidate_patents",
                "used_patent_rows",
                "used_patents",
                "matched_companies",
                "ambiguous_patent_rows",
            ]
        )
        return empty_matches, empty_audit, empty_matches.copy()

    matched = pd.concat(matched_frames, ignore_index=True)
    matched = matched.sort_values(
        ["used_for_features", "confidence_rank", "alias_rank", "company_id", "patnum"],
        ascending=[False, False, True, True, True],
    )
    matched = matched.drop_duplicates(["company_id", "patnum"], keep="first").reset_index(drop=True)
    audit = (
        matched.groupby(["alias_source", "confidence_tier", "match_method"], as_index=False)
        .agg(
            candidate_patent_rows=("patnum", "size"),
            candidate_patents=("patnum", "nunique"),
            used_patent_rows=("used_for_features", lambda values: int(np.sum(values))),
            ambiguous_patent_rows=("ambiguous_match", lambda values: int(np.sum(values))),
            matched_companies=("company_id", "nunique"),
        )
    )
    used = matched[matched["used_for_features"]].copy()
    baseline = used[used["alias_source"] == "primary_name"].copy()
    used_patents = (
        used.groupby(["alias_source", "confidence_tier", "match_method"], as_index=False)["patnum"]
        .nunique()
        .rename(columns={"patnum": "used_patents"})
    )
    audit = audit.merge(
        used_patents,
        on=["alias_source", "confidence_tier", "match_method"],
        how="left",
    ).fillna(0)
    audit["candidate_patent_rows"] = audit["candidate_patent_rows"].astype(int)
    audit["candidate_patents"] = audit["candidate_patents"].astype(int)
    audit["used_patent_rows"] = audit["used_patent_rows"].astype(int)
    audit["used_patents"] = audit["used_patents"].astype(int)
    audit["matched_companies"] = audit["matched_companies"].astype(int)
    audit["ambiguous_patent_rows"] = audit["ambiguous_patent_rows"].astype(int)
    return (
        used[
            [
                "company_id",
                "patnum",
                "patnum_kpss",
                "ptype",
                "grantdate",
                "appldate",
                "applnum",
                "observable_date",
                "match_method",
                "confidence_tier",
            ]
        ].copy(),
        audit.sort_values(["alias_source", "confidence_tier", "match_method"]).reset_index(drop=True),
        baseline[
            [
                "company_id",
                "patnum",
                "patnum_kpss",
                "ptype",
                "grantdate",
                "appldate",
                "applnum",
                "observable_date",
                "match_method",
                "confidence_tier",
            ]
        ].copy(),
    )


def build_patent_event_lookup(patent_matches: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    if patent_matches.empty:
        return {}
    events = patent_matches.copy()
    events["observable_quarter_idx"] = quarter_idx_from_dates(pd.to_datetime(events["observable_date"], errors="coerce"))
    events["grant_quarter_idx"] = quarter_idx_from_dates(pd.to_datetime(events["grantdate"], errors="coerce"))
    lookup: dict[str, dict[str, np.ndarray]] = {}
    for company_id, frame in events.groupby("company_id"):
        observable = np.sort(frame["observable_quarter_idx"].dropna().astype(int).to_numpy())
        grants = np.sort(frame["grant_quarter_idx"].dropna().astype(int).to_numpy())
        lookup[str(company_id)] = {
            "observable_quarters": observable,
            "grant_quarters": grants,
        }
    return lookup


def recent_event_counts(event_quarters: np.ndarray, lookup_quarters: np.ndarray, window_quarters: int = 4) -> np.ndarray:
    if event_quarters.size == 0:
        return np.zeros(len(lookup_quarters), dtype=np.float32)
    right = np.searchsorted(event_quarters, lookup_quarters, side="right")
    left = np.searchsorted(event_quarters, lookup_quarters - window_quarters, side="right")
    return (right - left).astype(np.float32)


def build_patent_feature_coverage(
    company_master: pd.DataFrame,
    panel: pd.DataFrame,
    patent_matches: pd.DataFrame,
) -> pd.DataFrame:
    if "company_id" in patent_matches.columns:
        matched_company_ids = set(patent_matches["company_id"].astype(str).unique().tolist())
    else:
        matched_company_ids = set()
    overall = pd.DataFrame(
        [
            {
                "split": "overall",
                "rows": int(len(panel)),
                "companies": int(panel["company_id"].nunique()),
                "matched_companies": int(company_master["company_id"].astype(str).isin(matched_company_ids).sum()),
                "share_companies_with_patents": float(
                    company_master["company_id"].astype(str).isin(matched_company_ids).mean()
                ),
                "rows_with_patent_signal": int(
                    (
                        (panel["patent_apps_visible_l4q"] > 0)
                        | (panel["patent_grants_l4q"] > 0)
                        | (panel["patent_stock_visible"] > 0)
                    ).sum()
                ),
                "mean_patent_apps_visible_l4q": float(panel["patent_apps_visible_l4q"].mean()),
                "mean_patent_grants_l4q": float(panel["patent_grants_l4q"].mean()),
                "mean_patent_stock_visible": float(panel["patent_stock_visible"].mean()),
            }
        ]
    )
    by_split = (
        panel.groupby("split", as_index=False)
        .agg(
            rows=("company_id", "size"),
            companies=("company_id", "nunique"),
            rows_with_patent_signal=(
                "patent_stock_visible",
                lambda values: int(np.sum(pd.to_numeric(values, errors="coerce").fillna(0.0) > 0.0)),
            ),
            mean_patent_apps_visible_l4q=("patent_apps_visible_l4q", "mean"),
            mean_patent_grants_l4q=("patent_grants_l4q", "mean"),
            mean_patent_stock_visible=("patent_stock_visible", "mean"),
        )
    )
    by_split["matched_companies"] = by_split["split"].map(
        {
            split: int(
                panel.loc[panel["split"] == split, "company_id"]
                .astype(str)
                .loc[lambda values: values.isin(matched_company_ids)]
                .nunique()
            )
            for split in by_split["split"].tolist()
        }
    )
    by_split["share_companies_with_patents"] = by_split["matched_companies"] / by_split["companies"].clip(lower=1)
    return pd.concat([overall, by_split], ignore_index=True)


def build_patent_coverage_comparison(
    company_master: pd.DataFrame,
    panel: pd.DataFrame,
    baseline_matches: pd.DataFrame,
    enhanced_matches: pd.DataFrame,
) -> pd.DataFrame:
    baseline = build_patent_feature_coverage(company_master, panel, baseline_matches).copy()
    baseline["coverage_mode"] = "baseline_primary_name_only"
    enhanced = build_patent_feature_coverage(company_master, panel, enhanced_matches).copy()
    enhanced["coverage_mode"] = "enhanced_alias_matching"
    comparison = pd.concat([baseline, enhanced], ignore_index=True)
    pivot_metrics = [
        "matched_companies",
        "share_companies_with_patents",
        "rows_with_patent_signal",
        "mean_patent_apps_visible_l4q",
        "mean_patent_grants_l4q",
        "mean_patent_stock_visible",
    ]
    delta = (
        comparison.pivot_table(
            index="split",
            columns="coverage_mode",
            values=pivot_metrics,
            aggfunc="first",
        )
        .sort_index(axis=1)
    )
    rows: list[dict[str, object]] = []
    for split in comparison["split"].drop_duplicates().tolist():
        row = {"split": split, "coverage_mode": "delta_enhanced_minus_baseline"}
        for metric in pivot_metrics:
            enhanced_value = delta.get((metric, "enhanced_alias_matching"), pd.Series(dtype=float)).get(split, np.nan)
            baseline_value = delta.get((metric, "baseline_primary_name_only"), pd.Series(dtype=float)).get(split, np.nan)
            row[metric] = (
                float(enhanced_value) - float(baseline_value)
                if pd.notna(enhanced_value) and pd.notna(baseline_value)
                else np.nan
            )
        rows.append(row)
    return pd.concat([comparison, pd.DataFrame(rows)], ignore_index=True, sort=False)


def build_company_spans(
    round_events: pd.DataFrame,
    chosen_exits: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    analysis_start_idx = quarter_idx_from_label(config["analysis_start_quarter"])
    analysis_end_idx = quarter_idx_from_label(config["analysis_end_quarter"])
    company_entry = round_events.groupby("company_id", as_index=False).agg(
        entry_quarter_idx=("quarter_idx", "min")
    )
    company_entry["entry_year"] = (company_entry["entry_quarter_idx"] // 4).astype(int)
    span = company_entry.merge(
        chosen_exits[["company_id", "exit_quarter_idx", "route_label"]],
        on="company_id",
        how="left",
    )
    span["panel_start_idx"] = np.maximum(span["entry_quarter_idx"] + 1, analysis_start_idx)
    span["analysis_end_idx"] = analysis_end_idx
    return span


def candidate_window_cut_points(config: dict) -> list[tuple[int, int, int]]:
    analysis_start_idx = quarter_idx_from_label(config["analysis_start_quarter"])
    analysis_end_idx = quarter_idx_from_label(config["analysis_end_quarter"])
    horizon = int(config["holdout_horizon_quarters"])
    latest_test_end_idx = analysis_end_idx - horizon + 1
    candidate_test_end = sorted(
        set(
            idx
            for idx in [
                latest_test_end_idx,
                min((latest_test_end_idx // 4) * 4 + 3, latest_test_end_idx),
                min(((latest_test_end_idx - 4) // 4) * 4 + 3, latest_test_end_idx),
            ]
            if analysis_start_idx + 11 <= idx <= latest_test_end_idx
        )
    )
    train_candidates = [idx for idx in range(analysis_start_idx + 11, latest_test_end_idx - 7) if idx % 4 == 3]
    validation_candidates = [idx for idx in range(analysis_start_idx + 15, latest_test_end_idx - 3) if idx % 4 == 3]
    windows: list[tuple[int, int, int]] = []
    for test_end_idx in candidate_test_end:
        for validation_end_idx in validation_candidates:
            if validation_end_idx >= test_end_idx:
                continue
            if test_end_idx - validation_end_idx < int(config.get("min_test_quarters", 4)):
                continue
            for train_end_idx in train_candidates:
                if train_end_idx >= validation_end_idx:
                    continue
                if validation_end_idx - train_end_idx < int(config.get("min_validation_quarters", 4)):
                    continue
                train_years = (train_end_idx - analysis_start_idx + 1) / 4.0
                if train_years < float(config.get("min_train_years", 3)):
                    continue
                windows.append((train_end_idx, validation_end_idx, test_end_idx))
    return sorted(set(windows))


def build_window_selection_grid(
    round_events: pd.DataFrame,
    chosen_exits: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    span = build_company_spans(round_events, chosen_exits, config)
    if span.empty:
        return pd.DataFrame()
    start_year = max(int(config["entry_year_floor"]), int(span["entry_year"].min()))
    end_year = min(int(span["entry_year"].max()), quarter_idx_from_label(config["analysis_end_quarter"]) // 4)
    candidate_windows = candidate_window_cut_points(config)
    stress_start_idx = quarter_idx_from_label(str(config.get("stress_slice_start_quarter", "2020Q1")))
    stress_end_idx = quarter_idx_from_label(str(config.get("stress_slice_end_quarter", "2020Q4")))
    rows: list[dict[str, object]] = []
    for min_entry_year in range(start_year, end_year + 1):
        sample = span[span["entry_year"] >= min_entry_year].copy()
        if sample.empty:
            continue
        for train_end_idx, validation_end_idx, test_end_idx in candidate_windows:
            panel_end = np.where(
                sample["exit_quarter_idx"].notna(),
                np.minimum(sample["exit_quarter_idx"], test_end_idx),
                test_end_idx,
            )
            approx_panel_rows = np.maximum(panel_end - sample["panel_start_idx"] + 1, 0).astype(int)
            modeled_exit = (
                sample["exit_quarter_idx"].notna()
                & (sample["exit_quarter_idx"] >= sample["panel_start_idx"])
                & (sample["exit_quarter_idx"] <= test_end_idx)
            )
            exit_quarter = sample["exit_quarter_idx"]
            route = sample["route_label"].fillna("no_exit")
            route_support = {}
            for split_name, lower, upper in [
                ("train", -np.inf, train_end_idx),
                ("validation", train_end_idx, validation_end_idx),
                ("test", validation_end_idx, test_end_idx),
            ]:
                for route_name in MAIN_DIRECT_ROUTES:
                    route_support[f"{split_name}_{route_name}_exits"] = int(
                        (
                            modeled_exit
                            & (route == route_name)
                            & (exit_quarter > lower)
                            & (exit_quarter <= upper)
                        ).sum()
                    )
            pooled_train_support = route_support["train_ipo_exits"] + route_support["train_mna_exits"]
            stress_overlap_quarters = max(min(test_end_idx, stress_end_idx) - max(validation_end_idx + 1, stress_start_idx) + 1, 0)
            within_memory_budget = int(int(approx_panel_rows.sum()) <= int(config["target_panel_rows"]))
            meets_train_exit_threshold = int(int((modeled_exit & (exit_quarter <= train_end_idx)).sum()) >= int(config["min_train_exits"]))
            meets_test_exit_threshold = int(
                int(((modeled_exit & (exit_quarter > validation_end_idx) & (exit_quarter <= test_end_idx)).sum()))
                >= int(config["min_test_exits"])
            )
            has_validation_events = int(
                int((modeled_exit & (exit_quarter > train_end_idx) & (exit_quarter <= validation_end_idx)).sum()) > 0
            )
            meets_route_support = int(
                min(route_support[f"train_{route_name}_exits"] for route_name in MAIN_DIRECT_ROUTES)
                >= int(config.get("min_train_route_support", 0))
            )
            feasible = int(
                within_memory_budget
                and meets_train_exit_threshold
                and has_validation_events
            )
            validation_only_score = (
                12.0 * route_support["train_ipo_exits"]
                + 6.0 * pooled_train_support
                + 3.0 * route_support["train_sponsor_sale_exits"]
                + 2.0 * route_support["validation_ipo_exits"]
                + 1.5 * route_support["validation_mna_exits"]
                + 1.0 * route_support["validation_sponsor_sale_exits"]
                + 0.01 * int((modeled_exit & (exit_quarter <= train_end_idx)).sum())
                + 0.005 * int(((modeled_exit & (exit_quarter > train_end_idx) & (exit_quarter <= validation_end_idx)).sum()))
            )
            if not within_memory_budget:
                validation_only_score -= 1000.0
            exploratory_score = validation_only_score + 40.0 * stress_overlap_quarters + 0.005 * int(
                ((modeled_exit & (exit_quarter > validation_end_idx) & (exit_quarter <= test_end_idx)).sum())
            )
            rows.append(
                {
                    "min_entry_year": min_entry_year,
                    "train_end_quarter": quarter_label_from_idx(train_end_idx),
                    "validation_end_quarter": quarter_label_from_idx(validation_end_idx),
                    "test_end_quarter": quarter_label_from_idx(test_end_idx),
                    "train_years": (train_end_idx - quarter_idx_from_label(config["analysis_start_quarter"]) + 1) / 4.0,
                    "validation_quarters": validation_end_idx - train_end_idx,
                    "test_quarters": test_end_idx - validation_end_idx,
                    "companies": int(sample["company_id"].nunique()),
                    "approx_panel_rows": int(approx_panel_rows.sum()),
                    "observed_exits": int(modeled_exit.sum()),
                    "train_exits": int((modeled_exit & (exit_quarter <= train_end_idx)).sum()),
                    "validation_exits": int(
                        (modeled_exit & (exit_quarter > train_end_idx) & (exit_quarter <= validation_end_idx)).sum()
                    ),
                    "test_exits": int(
                        (modeled_exit & (exit_quarter > validation_end_idx) & (exit_quarter <= test_end_idx)).sum()
                    ),
                    "ipo_exits": int(((route == "ipo") & modeled_exit).sum()),
                    "mna_exits": int(((route == "mna") & modeled_exit).sum()),
                    "sponsor_sale_exits": int(((route == "sponsor_sale") & modeled_exit).sum()),
                    "pooled_strategic_train_exits": pooled_train_support,
                    "stress_slice_overlap_quarters": int(stress_overlap_quarters),
                    "supports_stress_slice": int(stress_overlap_quarters > 0),
                    "within_memory_budget": within_memory_budget,
                    "meets_train_exit_threshold": meets_train_exit_threshold,
                    "has_validation_events": has_validation_events,
                    "meets_test_exit_threshold": meets_test_exit_threshold,
                    "meets_train_route_support": meets_route_support,
                    "validation_only_score": validation_only_score,
                    "objective_score": exploratory_score,
                    **route_support,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["validation_only_score", "train_exits", "validation_exits", "approx_panel_rows", "min_entry_year"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def select_actual_window(window_grid: pd.DataFrame, config: dict) -> tuple[pd.Series, pd.DataFrame]:
    if window_grid.empty:
        raise ValueError("Window selection grid is empty. Check the live inputs and analysis range.")
    full_grid = window_grid.copy()
    explicit = config.get("min_entry_year")
    if explicit is not None and not pd.isna(explicit):
        explicit_year = int(explicit)
        window_grid = window_grid[window_grid["min_entry_year"].astype(int) == explicit_year].copy()
        if window_grid.empty:
            available = ", ".join(map(str, sorted(full_grid["min_entry_year"].astype(int).unique().tolist())))
            raise ValueError(f"Configured min_entry_year={explicit_year} is not available. Choices: {available}")
    validation_events = pd.to_numeric(
        window_grid.get("has_validation_events", pd.Series(np.ones(len(window_grid)), index=window_grid.index)),
        errors="coerce",
    ).fillna(1).astype(int)
    feasible = window_grid[
        (window_grid["within_memory_budget"] == 1)
        & (window_grid["meets_train_exit_threshold"] == 1)
        & validation_events.eq(1)
    ].copy()
    if feasible.empty:
        feasible = window_grid[window_grid["within_memory_budget"] == 1].copy()
    if feasible.empty:
        feasible = window_grid.copy()
    route_feasible = feasible[feasible["meets_train_route_support"] == 1].copy()
    if route_feasible.empty:
        selected = feasible.sort_values(
            [
                "pooled_strategic_train_exits",
                "train_ipo_exits",
                "train_exits",
                "validation_exits",
                "validation_only_score",
                "min_entry_year",
            ],
            ascending=[False, False, False, False, False, True],
        ).iloc[0]
        fallback_reason = (
            "No validation-feasible split delivered the requested direct IPO support under the memory-safe row budget; "
            "window selection is locked on train-plus-validation support and pooled strategic exit remains an evaluation-only fallback."
        )
        used_fallback = True
    else:
        selected = route_feasible.sort_values(
            ["validation_only_score", "train_exits", "validation_exits", "min_entry_year"],
            ascending=[False, False, False, True],
        ).iloc[0]
        fallback_reason = "Direct-route support thresholds were met on the locked train-plus-validation window; test remains sealed for confirmation."
        used_fallback = False
    annotated = full_grid.copy()
    selected_key = (
        int(selected["min_entry_year"]),
        str(selected["train_end_quarter"]),
        str(selected["validation_end_quarter"]),
        str(selected["test_end_quarter"]),
    )
    annotated["selected"] = (
        (annotated["min_entry_year"].astype(int) == selected_key[0])
        & (annotated["train_end_quarter"].astype(str) == selected_key[1])
        & (annotated["validation_end_quarter"].astype(str) == selected_key[2])
        & (annotated["test_end_quarter"].astype(str) == selected_key[3])
    ).astype(int)
    annotated["selection_protocol"] = "validation_only_locked_test_confirmation"
    annotated["used_route_pooling_fallback"] = int(used_fallback)
    annotated["fallback_reason"] = fallback_reason
    return selected, annotated


def build_route_pooling_fallback_summary(selected_window: pd.Series, used_fallback: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "used_route_pooling_fallback": int(bool(used_fallback)),
                "fallback_route_label": "pooled_strategic_exit",
                "fallback_components": "ipo+mna",
                "train_ipo_exits": int(selected_window.get("train_ipo_exits", 0)),
                "train_mna_exits": int(selected_window.get("train_mna_exits", 0)),
                "pooled_strategic_train_exits": int(selected_window.get("pooled_strategic_train_exits", 0)),
                "selected_min_entry_year": int(selected_window.get("min_entry_year", 0)),
                "selected_train_end_quarter": str(selected_window.get("train_end_quarter", "")),
                "selected_validation_end_quarter": str(selected_window.get("validation_end_quarter", "")),
                "selected_test_end_quarter": str(selected_window.get("test_end_quarter", "")),
                "fallback_reason": (
                    "Direct IPO support remained below threshold in all feasible windows."
                    if used_fallback
                    else "Fallback was not required by the selected window."
                ),
            }
        ]
    )


def build_density_windows(
    round_events: pd.DataFrame,
    chosen_exits: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    analysis_start_idx = quarter_idx_from_label(config["analysis_start_quarter"])
    analysis_end_idx = quarter_idx_from_label(config["analysis_end_quarter"])
    train_end = quarter_idx_from_label(config["train_end_quarter"])
    validation_end = quarter_idx_from_label(config["validation_end_quarter"])
    test_end = quarter_idx_from_label(config["test_end_quarter"])

    company_entry = round_events.groupby("company_id", as_index=False).agg(
        entry_quarter_idx=("quarter_idx", "min")
    )
    company_entry["entry_year"] = (company_entry["entry_quarter_idx"] // 4).astype(int)
    span = company_entry.merge(
        chosen_exits[["company_id", "exit_quarter_idx", "route_label"]],
        on="company_id",
        how="left",
    )
    span["panel_start_idx"] = np.maximum(span["entry_quarter_idx"] + 1, analysis_start_idx)
    span["panel_end_idx"] = np.where(
        span["exit_quarter_idx"].notna(),
        np.minimum(span["exit_quarter_idx"], analysis_end_idx),
        analysis_end_idx,
    )
    span["approx_panel_rows"] = np.maximum(span["panel_end_idx"] - span["panel_start_idx"] + 1, 0).astype(int)

    start_year = max(int(config["entry_year_floor"]), int(span["entry_year"].min()))
    end_year = min(int(span["entry_year"].max()), analysis_end_idx // 4)
    rows = []
    for min_entry_year in range(start_year, end_year + 1):
        sample = span[span["entry_year"] >= min_entry_year].copy()
        if sample.empty:
            continue
        exit_quarter = sample["exit_quarter_idx"]
        route = sample["route_label"].fillna("no_exit")
        modeled_exit = exit_quarter.notna() & (exit_quarter >= sample["panel_start_idx"])
        train_route_support = {
            "ipo": int(((route == "ipo") & modeled_exit & (exit_quarter <= train_end)).sum()),
            "mna": int(((route == "mna") & modeled_exit & (exit_quarter <= train_end)).sum()),
            "sponsor_sale": int(((route == "sponsor_sale") & modeled_exit & (exit_quarter <= train_end)).sum()),
        }
        rows.append(
            {
                "min_entry_year": min_entry_year,
                "companies": int(sample["company_id"].nunique()),
                "approx_panel_rows": int(sample["approx_panel_rows"].sum()),
                "observed_exits": int(modeled_exit.sum()),
                "train_exits": int((modeled_exit & (exit_quarter <= train_end)).sum()),
                "validation_exits": int(
                    (modeled_exit & (exit_quarter > train_end) & (exit_quarter <= validation_end)).sum()
                ),
                "test_exits": int(
                    (modeled_exit & (exit_quarter > validation_end) & (exit_quarter <= test_end)).sum()
                ),
                "ipo_exits": int(((route == "ipo") & modeled_exit).sum()),
                "mna_exits": int(((route == "mna") & modeled_exit).sum()),
                "sponsor_sale_exits": int(((route == "sponsor_sale") & modeled_exit).sum()),
                "writeoff_exits": int(((route == "writeoff") & modeled_exit).sum()),
                "train_ipo_exits": train_route_support["ipo"],
                "train_mna_exits": train_route_support["mna"],
                "train_sponsor_sale_exits": train_route_support["sponsor_sale"],
                "train_min_main_route_support": int(min(train_route_support.values())),
                "meets_train_route_support": int(
                    min(train_route_support.values()) >= int(config.get("min_train_route_support", 0))
                ),
                "validation_ipo_exits": int(
                    ((route == "ipo") & modeled_exit & (exit_quarter > train_end) & (exit_quarter <= validation_end)).sum()
                ),
                "validation_mna_exits": int(
                    ((route == "mna") & modeled_exit & (exit_quarter > train_end) & (exit_quarter <= validation_end)).sum()
                ),
                "validation_sponsor_sale_exits": int(
                    (
                        (route == "sponsor_sale")
                        & modeled_exit
                        & (exit_quarter > train_end)
                        & (exit_quarter <= validation_end)
                    ).sum()
                ),
                "test_ipo_exits": int(
                    ((route == "ipo") & modeled_exit & (exit_quarter > validation_end) & (exit_quarter <= test_end)).sum()
                ),
                "test_mna_exits": int(
                    ((route == "mna") & modeled_exit & (exit_quarter > validation_end) & (exit_quarter <= test_end)).sum()
                ),
                "test_sponsor_sale_exits": int(
                    (
                        (route == "sponsor_sale")
                        & modeled_exit
                        & (exit_quarter > validation_end)
                        & (exit_quarter <= test_end)
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def select_min_entry_year(density_windows: pd.DataFrame, config: dict) -> int:
    explicit = config.get("min_entry_year")
    if explicit is not None:
        explicit_year = int(explicit)
        if explicit_year not in set(density_windows["min_entry_year"].astype(int)):
            available = ", ".join(map(str, density_windows["min_entry_year"].astype(int).tolist()))
            raise ValueError(f"Configured min_entry_year={explicit_year} is not available. Choices: {available}")
        return explicit_year

    route_support_cols = [f"train_{route}_exits" for route in MAIN_DIRECT_ROUTES]
    route_support_threshold = int(config.get("min_train_route_support", 0))
    route_support_ok = np.logical_and.reduce(
        [
            density_windows[column].fillna(0).astype(int) >= route_support_threshold
            for column in route_support_cols
        ]
    )
    eligible = density_windows[
        (density_windows["approx_panel_rows"] <= int(config["target_panel_rows"]))
        & (density_windows["train_exits"] >= int(config["min_train_exits"]))
        & (density_windows["test_exits"] >= int(config["min_test_exits"]))
        & route_support_ok
    ].copy()
    if not eligible.empty:
        return int(eligible["min_entry_year"].min())

    within_budget = density_windows[density_windows["approx_panel_rows"] <= int(config["target_panel_rows"])].copy()
    if not within_budget.empty:
        within_budget["min_route_support"] = within_budget[route_support_cols].min(axis=1)
        return int(
            within_budget.sort_values(
                ["min_route_support", "train_exits", "test_exits", "min_entry_year"],
                ascending=[False, False, False, True],
            )
            .iloc[0]["min_entry_year"]
        )

    return int(
        density_windows.sort_values(["approx_panel_rows", "min_entry_year"], ascending=[True, False]).iloc[0][
            "min_entry_year"
        ]
    )


def build_route_support_by_split(panel: pd.DataFrame) -> pd.DataFrame:
    support_frame = panel.loc[panel["route_label"] != "no_exit", ["split", "route_label", "company_id"]].copy()
    support_frame["route_label"] = support_frame["route_label"].astype(str)
    support = (
        support_frame.groupby(["split", "route_label"], as_index=False, observed=True)
        .agg(rows=("company_id", "size"), companies=("company_id", "nunique"))
    )
    return support.sort_values(["split", "route_label"]).reset_index(drop=True)


def build_company_universe_map(round_events: pd.DataFrame) -> pd.DataFrame:
    events = round_events.copy()
    source_text = pd.Series("", index=events.index, dtype=object)
    if "event_sources" in events.columns:
        source_text = events["event_sources"].fillna("").astype(str)
    elif "source" in events.columns:
        source_text = events["source"].fillna("").astype(str)
    stage_bucket = events.get("stage_or_type", pd.Series(index=events.index, dtype=object)).map(map_stage_bucket)
    grouped = (
        events.assign(
            has_buyout_source=source_text.str.contains("preqin_buyout", case=False, regex=False),
            stage_bucket_key=stage_bucket.fillna(STAGE_BUCKET_BASE).astype(str),
        )
        .sort_values(["company_id", "quarter_idx"])
        .groupby("company_id", as_index=False)
        .agg(
            has_buyout_source=("has_buyout_source", "max"),
            latest_stage_bucket=("stage_bucket_key", "last"),
        )
    )
    grouped["universe"] = np.where(
        grouped["has_buyout_source"].astype(bool) | grouped["latest_stage_bucket"].astype(str).eq("buyout_late"),
        "buyout_pe",
        "venture_growth",
    )
    return grouped[["company_id", "universe"]].copy()


def attach_universe_labels(
    panel: pd.DataFrame,
    company_master: pd.DataFrame,
    round_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe_map = build_company_universe_map(round_events)
    panel_with_universe = panel.merge(universe_map, on="company_id", how="left")
    panel_with_universe["universe"] = panel_with_universe["universe"].fillna("venture_growth").astype("category")
    company_master_with_universe = company_master.merge(universe_map, on="company_id", how="left")
    company_master_with_universe["universe"] = company_master_with_universe["universe"].fillna("venture_growth")
    return panel_with_universe, company_master_with_universe, universe_map


def add_redesigned_targets(
    panel: pd.DataFrame,
    chosen_exits_sensitivity: pd.DataFrame,
    horizon_quarters: int,
) -> pd.DataFrame:
    enriched = panel.copy()
    enriched["realized_hard_timely_liquidity_by_horizon"] = pd.to_numeric(
        enriched["realized_exit_by_horizon"],
        errors="coerce",
    ).fillna(0).astype(int)
    soft_failure = chosen_exits_sensitivity.loc[
        chosen_exits_sensitivity["route_label"].astype(str).eq("soft_failure_sensitivity"),
        ["company_id", "exit_quarter_idx", "route_source", "confidence_tier"],
    ].rename(
        columns={
            "route_source": "soft_failure_route_source",
            "confidence_tier": "soft_failure_confidence_tier",
        }
    )
    enriched = enriched.merge(soft_failure, on="company_id", how="left")
    enriched["realized_soft_failure_sensitivity_by_horizon"] = (
        enriched["exit_quarter_idx_y"].notna()
        & (pd.to_numeric(enriched["exit_quarter_idx_y"], errors="coerce") >= pd.to_numeric(enriched["quarter_idx"], errors="coerce"))
        & (
            pd.to_numeric(enriched["exit_quarter_idx_y"], errors="coerce")
            <= pd.to_numeric(enriched["quarter_idx"], errors="coerce") + int(horizon_quarters) - 1
        )
    ).astype(int)
    if "exit_quarter_idx_y" in enriched.columns:
        enriched = enriched.rename(columns={"exit_quarter_idx_y": "soft_failure_exit_quarter_idx"})
    if "exit_quarter_idx_x" in enriched.columns:
        enriched = enriched.rename(columns={"exit_quarter_idx_x": "exit_quarter_idx"})
    return enriched


def build_universe_support(panel: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        panel.groupby(["split", "universe"], as_index=False, observed=True)
        .agg(
            rows=("company_id", "size"),
            companies=("company_id", "nunique"),
            hard_timely_liquidity_events=("realized_hard_timely_liquidity_by_horizon", "sum"),
        )
    )
    return grouped.sort_values(["split", "universe"]).reset_index(drop=True)


def filter_modeled_universe(
    company_master: pd.DataFrame,
    round_events: pd.DataFrame,
    exit_candidates: pd.DataFrame,
    chosen_exits: pd.DataFrame,
    crosswalk: pd.DataFrame,
    min_entry_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    company_entry = round_events.groupby("company_id", as_index=False).agg(
        entry_quarter_idx=("quarter_idx", "min")
    )
    company_entry["entry_year"] = (company_entry["entry_quarter_idx"] // 4).astype(int)
    keep_company_ids = set(company_entry.loc[company_entry["entry_year"] >= int(min_entry_year), "company_id"])

    filtered_company_master = company_master[company_master["company_id"].isin(keep_company_ids)].copy()
    filtered_round_events = round_events[round_events["company_id"].isin(keep_company_ids)].copy()
    filtered_exit_candidates = exit_candidates[exit_candidates["company_id"].isin(keep_company_ids)].copy()
    filtered_chosen_exits = chosen_exits[chosen_exits["company_id"].isin(keep_company_ids)].copy()

    keep_preqin_ids = set(filtered_company_master["portfolio_company_id"].dropna().tolist())
    keep_cb_ids = set(filtered_company_master["cb_company_uuid"].dropna().tolist())
    filtered_crosswalk = crosswalk[
        crosswalk["portfolio_company_id"].isin(keep_preqin_ids)
        | crosswalk["company_uuid"].isin(keep_cb_ids)
    ].copy()
    return (
        filtered_company_master,
        filtered_round_events,
        filtered_exit_candidates,
        filtered_chosen_exits,
        filtered_crosswalk,
    )


def build_sample_company_master(sample_inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    company = sample_inputs["company"].copy()
    company["company_source"] = "sample"
    company["city"] = np.nan
    company["portfolio_company_id"] = np.nan
    company["match_method"] = np.nan
    company["match_confidence"] = np.nan
    company["company_uuid"] = np.nan
    company["cb_company_uuid"] = np.nan
    company["cb_operating_status"] = np.nan
    company["cb_last_funding_at"] = np.nan
    company["cb_num_funding_rounds"] = np.nan
    company["cb_collected_at"] = np.nan
    company["website"] = np.nan
    company["raw_sector_text"] = company["company_name"].map(sample_sector_token)
    company["sector_bucket"] = [
        map_sector_bucket(raw_text, company_name)
        for raw_text, company_name in zip(company["raw_sector_text"], company["company_name"], strict=True)
    ]
    company["normalized_name"] = company["company_name"].map(normalize_name)
    company["normalized_domain"] = company["website"].map(normalize_domain)
    company["normalized_country_code"] = company["country"].map(normalize_country_code)
    company["normalized_city"] = company["city"].map(normalize_location_key)
    return company[
        [
            "company_id",
            "company_name",
            "website",
            "country",
            "city",
            "region",
            "founded_date",
            "company_source",
            "portfolio_company_id",
            "match_method",
            "match_confidence",
            "company_uuid",
            "cb_company_uuid",
            "cb_operating_status",
            "cb_last_funding_at",
            "cb_num_funding_rounds",
            "cb_collected_at",
            "raw_sector_text",
            "sector_bucket",
            "normalized_name",
            "normalized_domain",
            "normalized_country_code",
            "normalized_city",
            "baseline_patent_flow_l4q",
            "baseline_sponsor_score",
        ]
    ].copy()


def build_sample_round_events(sample_inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rounds = sample_inputs["rounds"].copy()
    rounds["quarter_idx"] = quarter_idx_from_dates(rounds["round_date"])
    round_events = rounds.rename(columns={"investor_count": "num_investors"})
    return round_events[
        [
            "company_id",
            "round_date",
            "quarter_idx",
            "round_amount_usd",
            "num_investors",
            "stage_or_type",
            "lead_fund_id",
        ]
    ].copy()


def build_sample_exits(sample_inputs: dict[str, pd.DataFrame], analysis_end_idx: int) -> pd.DataFrame:
    exits = sample_inputs["exits"].copy()
    exits["exit_quarter_idx"] = quarter_idx_from_dates(exits["exit_date"])
    exits = exits[exits["exit_quarter_idx"] <= analysis_end_idx].copy()
    return exits[
        [
            "company_id",
            "exit_date",
            "exit_quarter_idx",
            "route_label",
            "confidence_tier",
            "route_source",
            "event_value_usd",
        ]
    ].copy()


def build_sample_macro_panel(sample_inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    macro = sample_inputs["regimes"].copy()
    macro["quarter_idx"] = macro["quarter_idx"].astype(int)
    return macro[["quarter_idx", "quarter_label", "market_regime"]].sort_values("quarter_idx").reset_index(drop=True)


def apply_patent_features_to_chunk(
    enriched: pd.DataFrame,
    company_frame: pd.DataFrame,
    patent_event_lookup: dict[str, dict[str, np.ndarray]] | None,
) -> pd.DataFrame:
    if patent_event_lookup:
        apps_visible_l4q = np.zeros(len(enriched), dtype=np.float32)
        patent_stock_visible = np.zeros(len(enriched), dtype=np.float32)
        patent_grants_l4q = np.zeros(len(enriched), dtype=np.float32)
        company_codes = enriched["company_code"].to_numpy(dtype=np.int32)
        lookup_quarters = enriched["lookup_idx"].to_numpy(dtype=np.int32)
        for row in company_frame[["company_code", "company_id"]].itertuples(index=False):
            patent_rows = patent_event_lookup.get(str(row.company_id))
            if patent_rows is None:
                continue
            mask = company_codes == int(row.company_code)
            company_lookup = lookup_quarters[mask]
            observable_quarters = patent_rows.get("observable_quarters", np.array([], dtype=int))
            grant_quarters = patent_rows.get("grant_quarters", np.array([], dtype=int))
            if observable_quarters.size:
                patent_stock_visible[mask] = np.searchsorted(
                    observable_quarters,
                    company_lookup,
                    side="right",
                ).astype(np.float32)
                apps_visible_l4q[mask] = recent_event_counts(observable_quarters, company_lookup)
            if grant_quarters.size:
                patent_grants_l4q[mask] = recent_event_counts(grant_quarters, company_lookup)
        enriched["patent_apps_visible_l4q"] = apps_visible_l4q
        enriched["patent_stock_visible"] = patent_stock_visible
        enriched["patent_grants_l4q"] = patent_grants_l4q
        enriched["patent_flow_l4q"] = enriched["patent_apps_visible_l4q"] + enriched["patent_grants_l4q"]
        return enriched

    if "baseline_patent_flow_l4q" in enriched.columns:
        patent_base = pd.to_numeric(enriched["baseline_patent_flow_l4q"], errors="coerce").fillna(0.0)
    else:
        patent_base = pd.Series(np.zeros(len(enriched), dtype=float), index=enriched.index)
    enriched["patent_apps_visible_l4q"] = patent_base.astype(np.float32)
    enriched["patent_grants_l4q"] = patent_base.astype(np.float32)
    enriched["patent_stock_visible"] = np.maximum(
        patent_base.to_numpy(dtype=float) * np.maximum(1.0, enriched["age_q"].to_numpy(dtype=float) / 8.0),
        patent_base.to_numpy(dtype=float),
    ).astype(np.float32)
    enriched["patent_flow_l4q"] = patent_base.astype(np.float32)
    return enriched


def build_company_quarter_panel_chunk(
    company_frame: pd.DataFrame,
    round_events: pd.DataFrame,
    analysis_start_idx: int,
    analysis_end_idx: int,
    global_round_median: float,
    patent_event_lookup: dict[str, dict[str, np.ndarray]] | None = None,
) -> pd.DataFrame:
    company_frame = company_frame.copy()
    company_frame["panel_start_idx"] = np.maximum(
        company_frame["entry_quarter_idx"] + 1, analysis_start_idx
    )
    company_frame["panel_end_idx"] = np.where(
        company_frame["exit_quarter_idx"].notna(),
        np.minimum(company_frame["exit_quarter_idx"], analysis_end_idx),
        analysis_end_idx,
    )
    company_frame = company_frame[company_frame["panel_start_idx"] <= company_frame["panel_end_idx"]].copy()
    if company_frame.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "company_name",
                "company_source",
                "raw_sector_text",
                "sector_bucket",
                "quarter_idx",
                "route_label",
                "raw_stage_label",
                "age_q",
                "time_since_last_round_q",
                "log_last_round_usd",
                "patent_apps_visible_l4q",
                "patent_stock_visible",
                "patent_grants_l4q",
                "patent_flow_l4q",
                "sponsor_score",
            ]
        )

    company_frame = company_frame.sort_values("company_id").reset_index(drop=True)
    company_frame["company_code"] = np.arange(len(company_frame), dtype=np.int32)

    lengths = (company_frame["panel_end_idx"] - company_frame["panel_start_idx"] + 1).astype(int).to_numpy()
    company_codes = np.repeat(company_frame["company_code"].to_numpy(dtype=np.int32), lengths)
    quarter_idx = np.concatenate(
        [
            np.arange(start, end + 1, dtype=np.int32)
            for start, end in zip(
                company_frame["panel_start_idx"].astype(int).to_numpy(),
                company_frame["panel_end_idx"].astype(int).to_numpy(),
                strict=True,
            )
        ]
    )
    exit_quarter = np.repeat(
        company_frame["exit_quarter_idx"].fillna(-1).astype(np.int32).to_numpy(), lengths
    )
    exit_route = np.repeat(
        company_frame["route_label"].fillna("no_exit").astype(str).to_numpy(), lengths
    )
    base = pd.DataFrame({"company_code": company_codes, "quarter_idx": quarter_idx})
    base["route_label"] = np.where(base["quarter_idx"].to_numpy() == exit_quarter, exit_route, "no_exit")
    base["lookup_idx"] = (base["quarter_idx"] - 1).astype(np.int32)

    company_code_map = company_frame.set_index("company_id")["company_code"]
    round_lookup = round_events.loc[
        round_events["company_id"].isin(company_code_map.index),
        [
            "company_id",
            "quarter_idx",
            "round_date",
            "round_amount_usd",
            "num_investors",
            "stage_or_type",
        ],
    ].copy()
    round_lookup["company_code"] = round_lookup["company_id"].map(company_code_map).astype(np.int32)
    round_lookup = round_lookup.sort_values(["company_code", "quarter_idx"]).copy()
    round_lookup["lookup_idx"] = round_lookup["quarter_idx"]

    base_for_merge = base.sort_values(["lookup_idx", "company_code"]).reset_index(drop=True)
    round_for_merge = round_lookup[
        [
            "company_code",
            "lookup_idx",
            "round_date",
            "round_amount_usd",
            "num_investors",
            "stage_or_type",
        ]
    ].sort_values(["lookup_idx", "company_code"]).reset_index(drop=True)
    enriched = pd.merge_asof(
        base_for_merge,
        round_for_merge,
        on="lookup_idx",
        by="company_code",
        direction="backward",
        allow_exact_matches=True,
    )
    enriched = enriched.sort_values(["company_code", "quarter_idx"]).reset_index(drop=True)
    enriched = enriched.merge(
        company_frame[
            [
                "company_code",
                "company_id",
                "company_name",
                "company_source",
                "raw_sector_text",
                "sector_bucket",
                "founded_quarter_idx",
                *(
                    ["baseline_patent_flow_l4q", "baseline_sponsor_score"]
                    if "baseline_patent_flow_l4q" in company_frame.columns
                    and "baseline_sponsor_score" in company_frame.columns
                    else []
                ),
            ]
        ],
        on="company_code",
        how="left",
    )
    enriched["age_q"] = (enriched["lookup_idx"] - enriched["founded_quarter_idx"]).clip(lower=0)
    enriched["time_since_last_round_q"] = (
        enriched["lookup_idx"] - quarter_idx_from_dates(pd.to_datetime(enriched["round_date"]))
    ).clip(lower=0)
    enriched["round_amount_usd"] = pd.to_numeric(enriched["round_amount_usd"], errors="coerce").fillna(
        global_round_median
    )
    enriched["log_last_round_usd"] = np.log1p(enriched["round_amount_usd"].clip(lower=0))
    investor_term = np.log1p(pd.to_numeric(enriched["num_investors"], errors="coerce").fillna(0.0))
    fallback_sponsor_score = enriched["company_source"].eq("preqin").astype(float) + 0.25 * investor_term
    if "baseline_sponsor_score" in enriched.columns:
        enriched["sponsor_score"] = pd.to_numeric(
            enriched["baseline_sponsor_score"], errors="coerce"
        ).fillna(fallback_sponsor_score)
    else:
        enriched["sponsor_score"] = fallback_sponsor_score
    enriched["raw_stage_label"] = enriched["stage_or_type"]
    enriched = apply_patent_features_to_chunk(enriched, company_frame, patent_event_lookup)
    enriched["quarter_idx"] = enriched["quarter_idx"].astype(np.int32)
    enriched["age_q"] = enriched["age_q"].astype(np.float32)
    enriched["time_since_last_round_q"] = enriched["time_since_last_round_q"].astype(np.float32)
    enriched["log_last_round_usd"] = enriched["log_last_round_usd"].astype(np.float32)
    enriched["patent_apps_visible_l4q"] = enriched["patent_apps_visible_l4q"].astype(np.float32)
    enriched["patent_stock_visible"] = enriched["patent_stock_visible"].astype(np.float32)
    enriched["patent_grants_l4q"] = enriched["patent_grants_l4q"].astype(np.float32)
    enriched["patent_flow_l4q"] = enriched["patent_flow_l4q"].astype(np.float32)
    enriched["sponsor_score"] = enriched["sponsor_score"].astype(np.float32)
    return enriched[
        [
            "company_id",
            "company_name",
            "company_source",
            "raw_sector_text",
            "sector_bucket",
            "quarter_idx",
            "route_label",
            "raw_stage_label",
            "age_q",
            "time_since_last_round_q",
            "log_last_round_usd",
            "patent_apps_visible_l4q",
            "patent_stock_visible",
            "patent_grants_l4q",
            "patent_flow_l4q",
            "sponsor_score",
        ]
    ].copy()


def build_company_quarter_panel(
    company_master: pd.DataFrame,
    round_events: pd.DataFrame,
    chosen_exits: pd.DataFrame,
    config: dict,
    patent_event_lookup: dict[str, dict[str, np.ndarray]] | None = None,
    entry_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    analysis_start_idx = quarter_idx_from_label(config["analysis_start_quarter"])
    panel_end_label = str(config.get("panel_end_quarter") or config["analysis_end_quarter"])
    analysis_end_idx = quarter_idx_from_label(panel_end_label)
    chunk_size = int(config.get("company_chunk_size", 5000))

    if entry_override is None:
        entry = round_events.groupby("company_id", as_index=False).agg(
            entry_date=("round_date", "min"),
            entry_quarter_idx=("quarter_idx", "min"),
        )
    else:
        entry = entry_override.copy()
        entry["entry_date"] = pd.to_datetime(entry["entry_date"], errors="coerce")
        entry["entry_quarter_idx"] = pd.to_numeric(entry["entry_quarter_idx"], errors="coerce")
        entry = entry.dropna(subset=["company_id", "entry_date", "entry_quarter_idx"]).copy()
    company_frame = company_master.merge(entry, on="company_id", how="inner").merge(
        chosen_exits, on="company_id", how="left"
    )
    company_frame["effective_founded_date"] = company_frame["founded_date"].fillna(company_frame["entry_date"])
    company_frame["founded_quarter_idx"] = quarter_idx_from_dates(
        pd.to_datetime(company_frame["effective_founded_date"])
    )
    company_frame = company_frame.sort_values("company_id").reset_index(drop=True)
    global_round_median = float(
        pd.to_numeric(round_events["round_amount_usd"], errors="coerce").dropna().median()
    )
    if math.isnan(global_round_median):
        global_round_median = 0.0

    chunks = []
    for start in range(0, len(company_frame), max(chunk_size, 1)):
        stop = min(start + max(chunk_size, 1), len(company_frame))
        chunk = build_company_quarter_panel_chunk(
            company_frame.iloc[start:stop].copy(),
            round_events,
            analysis_start_idx,
            analysis_end_idx,
            global_round_median,
            patent_event_lookup=patent_event_lookup,
        )
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(
            columns=[
                "company_id",
                "company_name",
                "company_source",
                "raw_sector_text",
                "sector_bucket",
                "quarter_idx",
                "route_label",
                "raw_stage_label",
                "age_q",
                "time_since_last_round_q",
                "log_last_round_usd",
                "patent_apps_visible_l4q",
                "patent_stock_visible",
                "patent_grants_l4q",
                "patent_flow_l4q",
                "sponsor_score",
            ]
        )

    panel = pd.concat(chunks, ignore_index=True)
    panel["route_label"] = pd.Categorical(panel["route_label"], categories=ROUTES)
    panel["company_id"] = panel["company_id"].astype("category")
    panel["company_name"] = panel["company_name"].astype("category")
    panel["company_source"] = panel["company_source"].astype("category")
    panel["sector_bucket"] = panel["sector_bucket"].astype("category")
    return panel


def build_macro_panel(panel: pd.DataFrame) -> pd.DataFrame:
    quarter_idx = (
        pd.Series(panel["quarter_idx"])
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
    )
    macro_panel = pd.DataFrame({"quarter_idx": quarter_idx})
    macro_panel["quarter_label"] = [quarter_label_from_idx(value) for value in macro_panel["quarter_idx"]]
    macro_panel["market_regime"] = 0.0
    macro_panel["macro_source"] = "neutral_quarter_bridge"
    return macro_panel


def attach_macro(panel: pd.DataFrame, macro_panel: pd.DataFrame) -> pd.DataFrame:
    return panel.merge(
        macro_panel[["quarter_idx", "market_regime"]],
        on="quarter_idx",
        how="left",
    )


def add_realized_exit_within_horizon(panel: pd.DataFrame, exits: pd.DataFrame, horizon_quarters: int) -> pd.DataFrame:
    with_exit = panel.merge(
        exits[
            ["company_id", "exit_quarter_idx", "route_label", "event_value_usd", "confidence_tier", "route_source"]
        ].rename(
            columns={
                "route_label": "company_exit_route",
                "event_value_usd": "company_exit_value_usd",
                "confidence_tier": "company_exit_confidence_tier",
                "route_source": "company_exit_route_source",
            }
        ),
        on="company_id",
        how="left",
    )
    with_exit["realized_exit_by_horizon"] = (
        with_exit["exit_quarter_idx"].notna()
        & (with_exit["exit_quarter_idx"] >= with_exit["quarter_idx"])
        & (with_exit["exit_quarter_idx"] <= with_exit["quarter_idx"] + horizon_quarters - 1)
    ).astype(int)
    return with_exit


def split_panel(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    train_end = quarter_idx_from_label(config["train_end_quarter"])
    validation_end = quarter_idx_from_label(config["validation_end_quarter"])
    test_end = quarter_idx_from_label(config["test_end_quarter"])
    split = np.where(
        panel["quarter_idx"] <= train_end,
        "train",
        np.where(
            panel["quarter_idx"] <= validation_end,
            "validation",
            np.where(panel["quarter_idx"] <= test_end, "test", "exclude"),
        ),
    )
    panel = panel.copy()
    panel["split"] = split
    return panel[panel["split"] != "exclude"].copy()


def sample_training_rows(train_panel: pd.DataFrame, max_train_rows: int, seed: int) -> pd.DataFrame:
    if len(train_panel) <= max_train_rows:
        result = train_panel.copy()
        result["row_weight"] = 1.0
        return result
    event_rows = train_panel[train_panel["route_label"] != "no_exit"].copy()
    no_exit_rows = train_panel[train_panel["route_label"] == "no_exit"].copy()
    remaining = max(max_train_rows - len(event_rows), 0)
    if remaining <= 0:
        sampled_no_exit = no_exit_rows.sample(n=min(len(no_exit_rows), max_train_rows // 3), random_state=seed)
    else:
        sampled_no_exit = no_exit_rows.sample(
            n=min(len(no_exit_rows), remaining),
            random_state=seed,
        )
    sampled_no_exit["row_weight"] = len(no_exit_rows) / max(len(sampled_no_exit), 1)
    event_rows["row_weight"] = 1.0
    return pd.concat([event_rows, sampled_no_exit], ignore_index=True)


def stratified_cap_panel(
    panel: pd.DataFrame,
    max_rows: int,
    seed: int,
    stratify_columns: list[str],
) -> pd.DataFrame:
    if len(panel) <= max_rows:
        return panel.copy()
    frame = panel.copy()
    available_columns = [column for column in stratify_columns if column in frame.columns]
    if not available_columns:
        return frame.sample(n=max_rows, random_state=seed).copy()
    strata = frame[available_columns].astype(str).agg("|".join, axis=1)
    frame = frame.assign(_feature_search_stratum=strata)
    counts = frame["_feature_search_stratum"].value_counts().sort_index()
    allocations = (counts / counts.sum() * max_rows).astype(int).clip(lower=1)
    while int(allocations.sum()) > max_rows:
        largest = allocations.idxmax()
        allocations.loc[largest] -= 1
    while int(allocations.sum()) < max_rows:
        slack = (counts - allocations).sort_values(ascending=False)
        candidates = slack[slack > 0]
        if candidates.empty:
            break
        allocations.loc[candidates.index[0]] += 1
    sampled_parts = []
    for offset, (stratum, target_n) in enumerate(allocations.items(), start=1):
        bucket = frame[frame["_feature_search_stratum"] == stratum].copy()
        sampled_parts.append(bucket.sample(n=min(int(target_n), len(bucket)), random_state=seed + offset))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.drop(columns=["_feature_search_stratum"], errors="ignore")
    return sampled.sort_values(["quarter_idx", "company_id"]).reset_index(drop=True)


def build_feature_analysis_panels(dataset: dict, config: dict) -> dict[str, pd.DataFrame]:
    validation_panel = dataset["panel"][dataset["panel"]["split"] == "validation"].copy()
    test_panel = dataset["panel"][dataset["panel"]["split"] == "test"].copy()
    if str(config.get("data_mode", "sample")).strip().lower() != "actual":
        return {"validation": validation_panel, "test": test_panel}
    return {
        "validation": stratified_cap_panel(
            validation_panel,
            int(config.get("feature_search_validation_max_rows", 120000)),
            int(config.get("random_seed", 42)) + 700,
            ["realized_exit_by_horizon", "sector_bucket"],
        ),
        "test": stratified_cap_panel(
            test_panel,
            int(config.get("feature_search_test_max_rows", 160000)),
            int(config.get("random_seed", 42)) + 900,
            ["realized_exit_by_horizon", "sector_bucket"],
        ),
    }


def resolve_feature_columns(
    panel: pd.DataFrame,
    feature_columns: list[str] | None = None,
    model_state: dict | None = None,
) -> list[str]:
    if feature_columns is not None:
        return [column for column in feature_columns if column in panel.columns]
    if model_state is not None and "feature_columns" in model_state:
        return [column for column in model_state["feature_columns"] if column in panel.columns]
    return [column for column in COMPANY_FEATURES if column in panel.columns]


def prepare_model_matrix(
    panel: pd.DataFrame,
    model_state: dict | None = None,
    use_quarter_fixed_effects: bool = False,
    feature_columns: list[str] | None = None,
    use_macro_feature: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    filled = panel.copy()
    resolved_feature_columns = resolve_feature_columns(filled, feature_columns=feature_columns, model_state=model_state)
    if model_state is None:
        medians = filled[resolved_feature_columns].median(numeric_only=True).reindex(resolved_feature_columns).fillna(0.0)
        means = filled[resolved_feature_columns].mean(numeric_only=True).reindex(resolved_feature_columns).fillna(0.0)
        stds = (
            filled[resolved_feature_columns]
            .std(numeric_only=True, ddof=0)
            .reindex(resolved_feature_columns)
            .replace(0.0, 1.0)
            .fillna(1.0)
        )
        if use_macro_feature:
            macro_mean = float(filled["market_regime"].mean())
            if not np.isfinite(macro_mean):
                macro_mean = 0.0
            macro_std = float(filled["market_regime"].std(ddof=0))
            if (not np.isfinite(macro_std)) or macro_std == 0.0:
                macro_std = 1.0
        else:
            macro_mean = 0.0
            macro_std = 1.0
        model_state = {
            "feature_medians": medians.to_dict(),
            "feature_means": means.to_dict(),
            "feature_stds": stds.to_dict(),
            "feature_columns": list(resolved_feature_columns),
            "macro_mean": macro_mean,
            "macro_std": macro_std,
            "use_macro_feature": bool(use_macro_feature),
            "use_quarter_fixed_effects": bool(use_quarter_fixed_effects),
            "quarter_effect_levels": (
                sorted(filled["quarter_idx"].dropna().astype(int).unique().tolist())[1:]
                if use_quarter_fixed_effects
                else []
            ),
        }
    resolved_feature_columns = [column for column in model_state.get("feature_columns", resolved_feature_columns) if column in filled.columns]
    for column in resolved_feature_columns:
        filled[column] = pd.to_numeric(filled[column], errors="coerce").fillna(
            model_state["feature_medians"][column]
        )
    filled["market_regime"] = pd.to_numeric(filled["market_regime"], errors="coerce").fillna(
        model_state["macro_mean"]
    )
    if resolved_feature_columns:
        x = np.column_stack(
            [
                (filled[column].to_numpy(dtype=float) - model_state["feature_means"][column])
                / model_state["feature_stds"][column]
                for column in resolved_feature_columns
            ]
        )
    else:
        x = np.zeros((len(filled), 0), dtype=float)
    if bool(model_state.get("use_macro_feature", True)):
        macro = (filled["market_regime"].to_numpy(dtype=float) - model_state["macro_mean"]) / model_state["macro_std"]
    else:
        macro = np.zeros(len(filled), dtype=float)
    if model_state.get("use_quarter_fixed_effects"):
        quarter_values = filled["quarter_idx"].astype(int).to_numpy(dtype=int)
        levels = [int(value) for value in model_state.get("quarter_effect_levels", [])]
        quarter_effects = (
            np.column_stack([(quarter_values == level).astype(float) for level in levels])
            if levels
            else np.zeros((len(filled), 0), dtype=float)
        )
    else:
        quarter_effects = np.zeros((len(filled), 0), dtype=float)
    y = filled["route_label"].map({label: idx for idx, label in enumerate(ROUTES)}).to_numpy(dtype=int)
    return x, macro, quarter_effects, y, model_state


def stable_route_probabilities(linear_terms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shift = np.maximum(0.0, np.max(linear_terms, axis=1))
    exp_exit = np.exp(linear_terms - shift[:, None])
    denom = np.exp(-shift) + np.sum(exp_exit, axis=1)
    p_no_exit = np.exp(-shift) / denom
    p_exit = exp_exit / denom[:, None]
    return p_no_exit, p_exit


def unpack_params(
    theta: np.ndarray,
    n_features: int,
    n_quarter_effects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = theta.reshape(len(EXIT_ROUTES), n_features + n_quarter_effects + 2)
    intercept = matrix[:, 0]
    beta = matrix[:, 1 : 1 + n_features]
    quarter_delta = matrix[:, 1 + n_features : 1 + n_features + n_quarter_effects]
    gamma = matrix[:, -1]
    return intercept, beta, quarter_delta, gamma


def multinomial_objective(
    theta: np.ndarray,
    x: np.ndarray,
    macro: np.ndarray,
    quarter_effects: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge_penalty: float,
) -> tuple[float, np.ndarray]:
    intercept, beta, quarter_delta, gamma = unpack_params(theta, x.shape[1], quarter_effects.shape[1])
    quarter_term = quarter_effects @ quarter_delta.T if quarter_effects.shape[1] else 0.0
    linear_terms = intercept[None, :] + x @ beta.T + quarter_term + macro[:, None] * gamma[None, :]
    p_no_exit, p_exit = stable_route_probabilities(linear_terms)
    log_p_no_exit = np.log(np.clip(p_no_exit, 1e-12, 1.0))
    log_p_exit = np.log(np.clip(p_exit, 1e-12, 1.0))

    exit_targets = np.maximum(y - 1, 0)
    target_log_prob = np.where(
        y == 0,
        log_p_no_exit,
        log_p_exit[np.arange(y.size), exit_targets],
    )
    nll = -np.sum(weights * target_log_prob)
    penalty = 0.5 * ridge_penalty * (np.sum(beta**2) + np.sum(quarter_delta**2) + np.sum(gamma**2))
    objective_value = float(nll + penalty)

    target_matrix = np.zeros_like(p_exit)
    exit_mask = y > 0
    target_matrix[np.arange(y.size)[exit_mask], exit_targets[exit_mask]] = 1.0
    residual = (p_exit - target_matrix) * weights[:, None]

    grad_intercept = residual.sum(axis=0)
    grad_beta = residual.T @ x + ridge_penalty * beta
    grad_quarter = (
        residual.T @ quarter_effects + ridge_penalty * quarter_delta
        if quarter_effects.shape[1]
        else np.zeros_like(quarter_delta)
    )
    grad_gamma = residual.T @ macro + ridge_penalty * gamma
    gradient = np.concatenate(
        [
            grad_intercept[:, None],
            grad_beta,
            grad_quarter,
            grad_gamma[:, None],
        ],
        axis=1,
    ).reshape(-1)
    return objective_value, gradient


def fit_multinomial_hazard(train_panel: pd.DataFrame, config: dict) -> dict:
    if train_panel.empty:
        raise ValueError("Training panel is empty. Lower min_entry_year or widen the train window.")
    if int((train_panel["route_label"] != "no_exit").sum()) == 0:
        raise ValueError("Training panel has no realized exits. Lower min_entry_year or widen the train window.")
    sampled = sample_training_rows(train_panel, int(config["max_train_rows"]), int(config["random_seed"]))
    feature_columns = resolve_feature_columns(
        sampled,
        feature_columns=config.get("feature_columns"),
    )
    use_quarter_fixed_effects = bool(config.get("use_quarter_fixed_effects", False))
    use_macro_feature = bool(config.get("use_macro_feature", True))
    x, macro, quarter_effects, y, model_state = prepare_model_matrix(
        sampled,
        use_quarter_fixed_effects=use_quarter_fixed_effects,
        feature_columns=feature_columns,
        use_macro_feature=use_macro_feature,
    )
    weights = sampled["row_weight"].to_numpy(dtype=float)
    theta0 = np.zeros(len(EXIT_ROUTES) * (x.shape[1] + quarter_effects.shape[1] + 2), dtype=float)

    def objective(theta: np.ndarray) -> float:
        return multinomial_objective(
            theta,
            x,
            macro,
            quarter_effects,
            y,
            weights,
            float(config["ridge_penalty"]),
        )[0]

    def gradient(theta: np.ndarray) -> np.ndarray:
        return multinomial_objective(
            theta,
            x,
            macro,
            quarter_effects,
            y,
            weights,
            float(config["ridge_penalty"]),
        )[1]

    maxiter = int(config.get("optimizer_maxiter", 250))
    result = minimize(
        objective,
        theta0,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )
    if (not result.success) and "ITERATION" in str(result.message).upper():
        retry_maxiter = max(maxiter * 2, maxiter + 250)
        result = minimize(
            objective,
            result.x,
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": retry_maxiter},
        )
    if not result.success:
        raise RuntimeError(f"Hazard fit failed: {result.message}")
    intercept, beta, quarter_delta, gamma = unpack_params(result.x, x.shape[1], quarter_effects.shape[1])
    return {
        "intercept": intercept,
        "beta": beta,
        "quarter_delta": quarter_delta,
        "gamma": gamma,
        "model_state": model_state,
        "optimization_message": result.message,
        "optimization_iterations": int(result.nit),
    }


def predict_route_probs(panel: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    x, macro, quarter_effects, _, _ = prepare_model_matrix(panel, fitted["model_state"])
    scenario_shift = (
        pd.to_numeric(panel.get("scenario_exit_logit_shift", 0.0), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        if isinstance(panel.get("scenario_exit_logit_shift", 0.0), pd.Series)
        else np.repeat(float(panel.get("scenario_exit_logit_shift", 0.0)), len(panel))
    )
    quarter_term = (
        quarter_effects @ fitted.get("quarter_delta", np.zeros((len(EXIT_ROUTES), 0), dtype=float)).T
        if quarter_effects.shape[1] and fitted.get("quarter_delta") is not None
        else 0.0
    )
    linear_terms = (
        fitted["intercept"][None, :]
        + x @ fitted["beta"].T
        + quarter_term
        + macro[:, None] * fitted["gamma"][None, :]
        + scenario_shift[:, None]
    )
    p_no_exit, p_exit = stable_route_probabilities(linear_terms)
    output = panel[["company_id", "quarter_idx"]].copy()
    output["p_no_exit"] = p_no_exit
    for route_idx, route in enumerate(EXIT_ROUTES):
        output[f"p_{route}"] = p_exit[:, route_idx]
    return output


def probability_path_summary_vectorized(
    panel: pd.DataFrame,
    fitted: dict,
    horizon_quarters: int,
    config: dict | None = None,
    scenario_name: str = "baseline",
) -> tuple[pd.DataFrame, np.ndarray]:
    feature_columns = resolve_feature_columns(panel, model_state=fitted.get("model_state"))
    current = panel[feature_columns].copy()
    current_market_regime = pd.to_numeric(panel["market_regime"], errors="coerce").fillna(0.0).copy()
    survival = np.ones(len(panel), dtype=float)
    route_cumulative = {route: np.zeros(len(panel), dtype=float) for route in EXIT_ROUTES}
    point_route_probs: list[np.ndarray] = []
    horizon_exit_probs: list[np.ndarray] = []
    config = config or {}
    regime_shift = float(config.get("freeze_regime_shift", 0.0)) if scenario_name == "exit_freeze" else 0.0
    route_shift = float(config.get("freeze_exit_logit_shift", 0.0)) if scenario_name == "exit_freeze" else 0.0
    for _ in range(horizon_quarters):
        scored = predict_route_probs(
            panel.assign(
                **{column: current[column] for column in feature_columns},
                market_regime=current_market_regime + regime_shift,
                scenario_exit_logit_shift=route_shift,
            ),
            fitted,
        )
        p0 = scored["p_no_exit"].to_numpy(dtype=float)
        step_route_probs = np.zeros((len(panel), len(EXIT_ROUTES)), dtype=float)
        for route_idx, route in enumerate(EXIT_ROUTES):
            route_prob = scored[f"p_{route}"].to_numpy(dtype=float)
            point_prob = survival * route_prob
            route_cumulative[route] += point_prob
            step_route_probs[:, route_idx] = point_prob
        survival = survival * p0
        point_route_probs.append(step_route_probs)
        horizon_exit_probs.append(1.0 - survival.copy())
        if "age_q" in current.columns:
            current["age_q"] = current["age_q"] + 1.0
        if "time_since_last_round_q" in current.columns:
            current["time_since_last_round_q"] = current["time_since_last_round_q"] + 1.0
    result = panel[["company_id", "quarter_idx"]].copy()
    for route in EXIT_ROUTES:
        result[f"cum_{route}"] = route_cumulative[route]
    result["survival_horizon"] = survival
    result["pred_exit_by_horizon"] = 1.0 - survival
    for horizon_step, values in enumerate(horizon_exit_probs, start=1):
        result[f"pred_exit_by_h{horizon_step}"] = values
    point_route_matrix = (
        np.stack(point_route_probs, axis=1)
        if point_route_probs
        else np.zeros((len(panel), 0, len(EXIT_ROUTES)), dtype=float)
    )
    return result, point_route_matrix


def stage1_feature_columns(panel: pd.DataFrame, include_patent_sector_conditional: bool = False) -> list[str]:
    columns = [
        "age_q",
        "time_since_last_round_q",
        "log_last_round_usd",
        "sponsor_score",
        *[column for column in sector_dummy_columns() if column in panel.columns],
        *[column for column in stage_dummy_columns() if column in panel.columns],
    ]
    if include_patent_sector_conditional:
        columns.extend(
            [
                column
                for column in [
                    "patent_apps_sector_conditional",
                    "patent_stock_sector_conditional",
                    "patent_grants_sector_conditional",
                ]
                if column in panel.columns
            ]
        )
    return list(dict.fromkeys([column for column in columns if column in panel.columns]))


def add_sector_conditional_patent_features(panel: pd.DataFrame) -> pd.DataFrame:
    enriched = panel.copy()
    patent_plausible = enriched["sector_bucket"].astype(str).isin(PATENT_PLAUSIBLE_BUCKETS).astype(float)
    enriched["patent_plausible_sector_flag"] = patent_plausible.astype(np.float32)
    enriched["patent_apps_sector_conditional"] = (
        pd.to_numeric(enriched.get("patent_apps_visible_l4q"), errors="coerce").fillna(0.0) * patent_plausible
    ).astype(np.float32)
    enriched["patent_stock_sector_conditional"] = (
        pd.to_numeric(enriched.get("patent_stock_visible"), errors="coerce").fillna(0.0) * patent_plausible
    ).astype(np.float32)
    enriched["patent_grants_sector_conditional"] = (
        pd.to_numeric(enriched.get("patent_grants_l4q"), errors="coerce").fillna(0.0) * patent_plausible
    ).astype(np.float32)
    return enriched


def prepare_binary_model_matrix(
    panel: pd.DataFrame,
    label_col: str,
    model_state: dict | None = None,
    use_quarter_fixed_effects: bool = False,
    feature_columns: list[str] | None = None,
    use_macro_feature: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    x, macro, quarter_effects, _, model_state = prepare_model_matrix(
        panel.assign(route_label=np.where(pd.to_numeric(panel[label_col], errors="coerce").fillna(0).astype(int).eq(1), "ipo", "no_exit")),
        model_state=model_state,
        use_quarter_fixed_effects=use_quarter_fixed_effects,
        feature_columns=feature_columns,
        use_macro_feature=use_macro_feature,
    )
    y = pd.to_numeric(panel[label_col], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    return x, macro, quarter_effects, y, model_state


def binary_hazard_objective(
    theta: np.ndarray,
    x: np.ndarray,
    macro: np.ndarray,
    quarter_effects: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge_penalty: float,
) -> tuple[float, np.ndarray]:
    intercept = float(theta[0])
    beta = theta[1 : 1 + x.shape[1]]
    quarter_delta = theta[1 + x.shape[1] : 1 + x.shape[1] + quarter_effects.shape[1]]
    gamma = float(theta[-1])
    quarter_term = quarter_effects @ quarter_delta if quarter_effects.shape[1] else 0.0
    linear = intercept + x @ beta + quarter_term + macro * gamma
    p = 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    nll = -np.sum(weights * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    penalty = 0.5 * ridge_penalty * (np.sum(beta**2) + np.sum(quarter_delta**2) + gamma**2)
    residual = (p - y) * weights
    grad_intercept = float(residual.sum())
    grad_beta = residual @ x + ridge_penalty * beta
    grad_quarter = residual @ quarter_effects + ridge_penalty * quarter_delta if quarter_effects.shape[1] else np.zeros_like(quarter_delta)
    grad_gamma = float(residual @ macro + ridge_penalty * gamma)
    gradient = np.concatenate([[grad_intercept], grad_beta, grad_quarter, [grad_gamma]])
    return float(nll + penalty), gradient


def fit_binary_hazard(
    train_panel: pd.DataFrame,
    label_col: str,
    config: dict,
    feature_columns: list[str],
) -> dict:
    if train_panel.empty:
        raise ValueError("Binary stage-1 training panel is empty.")
    if int(pd.to_numeric(train_panel[label_col], errors="coerce").fillna(0).sum()) == 0:
        raise ValueError(f"Binary stage-1 training panel has no positives for {label_col}.")
    sampled = sample_training_rows(
        train_panel.assign(route_label=np.where(pd.to_numeric(train_panel[label_col], errors="coerce").fillna(0).astype(int).eq(1), "ipo", "no_exit")),
        int(config["max_train_rows"]),
        int(config["random_seed"]),
    )
    sampled[label_col] = sampled["route_label"].astype(str).ne("no_exit").astype(int)
    x, macro, quarter_effects, y, model_state = prepare_binary_model_matrix(
        sampled,
        label_col=label_col,
        use_quarter_fixed_effects=bool(config.get("use_quarter_fixed_effects", False)),
        feature_columns=feature_columns,
        use_macro_feature=bool(config.get("use_macro_feature", True)),
    )
    weights = sampled["row_weight"].to_numpy(dtype=float)
    theta0 = np.zeros(1 + x.shape[1] + quarter_effects.shape[1] + 1, dtype=float)

    def objective(theta: np.ndarray) -> float:
        return binary_hazard_objective(theta, x, macro, quarter_effects, y, weights, float(config["ridge_penalty"]))[0]

    def gradient(theta: np.ndarray) -> np.ndarray:
        return binary_hazard_objective(theta, x, macro, quarter_effects, y, weights, float(config["ridge_penalty"]))[1]

    maxiter = int(config.get("optimizer_maxiter", 250))
    result = minimize(objective, theta0, jac=gradient, method="L-BFGS-B", options={"maxiter": maxiter})
    if (not result.success) and "ITERATION" in str(result.message).upper():
        result = minimize(objective, result.x, jac=gradient, method="L-BFGS-B", options={"maxiter": max(maxiter * 2, maxiter + 250)})
    if not result.success:
        raise RuntimeError(f"Binary stage-1 fit failed: {result.message}")
    intercept = float(result.x[0])
    beta = result.x[1 : 1 + x.shape[1]]
    quarter_delta = result.x[1 + x.shape[1] : 1 + x.shape[1] + quarter_effects.shape[1]]
    gamma = float(result.x[-1])
    return {
        "intercept": intercept,
        "beta": beta,
        "quarter_delta": quarter_delta,
        "gamma": gamma,
        "model_state": model_state,
        "label_col": label_col,
        "feature_columns": list(feature_columns),
        "optimization_message": result.message,
        "optimization_iterations": int(result.nit),
    }


def predict_binary_hazard_probs(panel: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    x, macro, quarter_effects, _, _ = prepare_binary_model_matrix(
        panel.assign(**{str(fitted["label_col"]): 0}),
        label_col=str(fitted["label_col"]),
        model_state=fitted["model_state"],
    )
    scenario_shift = (
        pd.to_numeric(panel.get("scenario_exit_logit_shift", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if isinstance(panel.get("scenario_exit_logit_shift", 0.0), pd.Series)
        else np.repeat(float(panel.get("scenario_exit_logit_shift", 0.0)), len(panel))
    )
    quarter_term = quarter_effects @ fitted["quarter_delta"] if quarter_effects.shape[1] else 0.0
    linear = fitted["intercept"] + x @ fitted["beta"] + quarter_term + macro * fitted["gamma"] + scenario_shift
    p_event = 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
    output = panel[["company_id", "quarter_idx"]].copy()
    output["p_no_event"] = 1.0 - p_event
    output["p_event"] = p_event
    return output


def binary_probability_path_summary_vectorized(
    panel: pd.DataFrame,
    fitted: dict,
    horizon_quarters: int,
    config: dict | None = None,
    scenario_name: str = "baseline",
    prediction_label: str = "pred_hard_timely_liquidity_by_horizon",
) -> tuple[pd.DataFrame, np.ndarray]:
    feature_columns = resolve_feature_columns(panel, model_state=fitted.get("model_state"))
    current = panel[feature_columns].copy()
    current_market_regime = pd.to_numeric(panel["market_regime"], errors="coerce").fillna(0.0).copy()
    survival = np.ones(len(panel), dtype=float)
    point_event_probs: list[np.ndarray] = []
    horizon_exit_probs: list[np.ndarray] = []
    config = config or {}
    regime_shift = float(config.get("freeze_regime_shift", 0.0)) if scenario_name == "exit_freeze" else 0.0
    route_shift = float(config.get("freeze_exit_logit_shift", 0.0)) if scenario_name == "exit_freeze" else 0.0
    for _ in range(horizon_quarters):
        scored = predict_binary_hazard_probs(
            panel.assign(
                **{column: current[column] for column in feature_columns},
                market_regime=current_market_regime + regime_shift,
                scenario_exit_logit_shift=route_shift,
            ),
            fitted,
        )
        step_prob = survival * scored["p_event"].to_numpy(dtype=float)
        survival = survival * scored["p_no_event"].to_numpy(dtype=float)
        point_event_probs.append(step_prob)
        horizon_exit_probs.append(1.0 - survival.copy())
        if "age_q" in current.columns:
            current["age_q"] = current["age_q"] + 1.0
        if "time_since_last_round_q" in current.columns:
            current["time_since_last_round_q"] = current["time_since_last_round_q"] + 1.0
    result = panel[["company_id", "quarter_idx"]].copy()
    result[prediction_label] = horizon_exit_probs[-1] if horizon_exit_probs else np.zeros(len(panel), dtype=float)
    result["survival_horizon"] = survival
    for horizon_step in range(1, horizon_quarters + 1):
        horizon_value = horizon_exit_probs[horizon_step - 1] if horizon_step <= len(horizon_exit_probs) else np.zeros(len(panel), dtype=float)
        result[f"pred_exit_by_h{horizon_step}"] = horizon_value
        result[f"{prediction_label}_h{horizon_step}"] = horizon_value
    point_matrix = np.column_stack(point_event_probs) if point_event_probs else np.zeros((len(panel), 0), dtype=float)
    return result, point_matrix


def fit_stage1_models_by_universe(
    train_panel: pd.DataFrame,
    config: dict,
    label_col: str,
    feature_columns: list[str],
) -> tuple[dict[str, dict], pd.DataFrame]:
    models: dict[str, dict] = {}
    support_rows: list[dict[str, object]] = []
    overall_model = fit_binary_hazard(train_panel, label_col, config, feature_columns)
    min_events = int(config.get("stage1_min_train_events_per_universe", 25))
    for universe in UNIVERSE_ORDER:
        subset = train_panel[train_panel["universe"].astype(str).eq(universe)].copy()
        positives = int(pd.to_numeric(subset.get(label_col), errors="coerce").fillna(0).sum())
        if len(subset) >= min_events and positives >= min_events:
            model = fit_binary_hazard(subset, label_col, config, feature_columns)
            model_status = "universe_specific"
        else:
            model = overall_model
            model_status = "fallback_overall"
        models[universe] = model
        support_rows.append(
            {
                "universe": universe,
                "train_rows": int(len(subset)),
                "train_events": positives,
                "model_status": model_status,
                "feature_columns": "|".join(feature_columns),
            }
        )
    models["_overall"] = overall_model
    return models, pd.DataFrame(support_rows)


def score_stage1_panel(
    panel: pd.DataFrame,
    stage1_models: dict[str, dict],
    horizon_quarters: int,
    config: dict,
    scenario_name: str = "baseline",
    prediction_label: str = "pred_hard_timely_liquidity_by_horizon",
) -> tuple[pd.DataFrame, np.ndarray]:
    if panel.empty:
        return pd.DataFrame(), np.zeros((0, horizon_quarters), dtype=float)
    working = panel.copy().reset_index(drop=True)
    working["_row_id"] = np.arange(len(working), dtype=int)
    summary_frames: list[pd.DataFrame] = []
    point_matrix = np.zeros((len(working), horizon_quarters), dtype=float)
    for universe, subset in working.groupby("universe", observed=True):
        model = stage1_models.get(str(universe), stage1_models.get("_overall"))
        scored, points = binary_probability_path_summary_vectorized(
            subset.drop(columns=["_row_id"]),
            model,
            horizon_quarters,
            config,
            scenario_name=scenario_name,
            prediction_label=prediction_label,
        )
        scored["_row_id"] = subset["_row_id"].to_numpy(dtype=int)
        summary_frames.append(scored)
        point_matrix[subset["_row_id"].to_numpy(dtype=int), : points.shape[1]] = points
    summary = pd.concat(summary_frames, ignore_index=True).sort_values("_row_id").drop(columns=["_row_id"])
    return summary.reset_index(drop=True), point_matrix


def select_stage2_route_classes(train_panel: pd.DataFrame, config: dict) -> tuple[list[str], pd.DataFrame]:
    hard_rows = train_panel[train_panel["route_label"].astype(str).isin(MAIN_DIRECT_ROUTES)].copy()
    counts = hard_rows["route_label"].astype(str).value_counts().to_dict()
    ipo_support = int(counts.get("ipo", 0))
    mna_support = int(counts.get("mna", 0))
    sponsor_support = int(counts.get("sponsor_sale", 0))
    pooled_support = ipo_support + mna_support
    threshold = int(config.get("stage2_min_route_support", config.get("min_train_route_support", 5)))
    if ipo_support >= threshold and mna_support >= threshold and sponsor_support >= threshold:
        classes = ["ipo", "mna", "sponsor_sale"]
        mode = "direct_routes"
    elif pooled_support >= threshold and sponsor_support >= threshold:
        classes = ["pooled_strategic", "sponsor_sale"]
        mode = "pooled_strategic_plus_sponsor"
    else:
        classes = ["pooled_strategic"]
        mode = "pooled_strategic_only"
    support = pd.DataFrame(
        [
            {"split": "train", "route_label": "ipo", "rows": ipo_support, "route_model_mode": mode},
            {"split": "train", "route_label": "mna", "rows": mna_support, "route_model_mode": mode},
            {"split": "train", "route_label": "sponsor_sale", "rows": sponsor_support, "route_model_mode": mode},
            {"split": "train", "route_label": "pooled_strategic", "rows": pooled_support, "route_model_mode": mode},
        ]
    )
    return classes, support


def map_stage2_route_label(route_label: object, stage2_classes: list[str]) -> str | None:
    route = str(route_label)
    if route in {"ipo", "mna"} and "pooled_strategic" in stage2_classes:
        return "pooled_strategic"
    if route in stage2_classes:
        return route
    return None


def build_stage2_probability_tables(
    train_panel: pd.DataFrame,
    stage2_classes: list[str],
    config: dict,
) -> dict[str, object]:
    exits = train_panel[train_panel["route_label"].astype(str).isin(MAIN_DIRECT_ROUTES)].copy()
    exits["stage2_route"] = exits["route_label"].map(lambda value: map_stage2_route_label(value, stage2_classes))
    exits = exits[exits["stage2_route"].notna()].copy()
    min_bucket_support = int(config.get("stage2_min_bucket_support", 20))
    alpha = 1.0
    tables: list[dict[str, object]] = []
    for level_name, keys in [
        ("universe_sector_stage", ["universe", "sector_bucket", "stage_bucket"]),
        ("universe_stage", ["universe", "stage_bucket"]),
        ("universe", ["universe"]),
        ("global", []),
    ]:
        if keys:
            grouped = (
                exits.groupby(keys + ["stage2_route"], as_index=False, observed=True)
                .size()
                .rename(columns={"size": "route_count"})
            )
            totals = grouped.groupby(keys, as_index=False, observed=True)["route_count"].sum().rename(columns={"route_count": "support"})
            pivot = grouped.pivot_table(
                index=keys,
                columns="stage2_route",
                values="route_count",
                fill_value=0,
                observed=True,
            ).reset_index()
            table = totals.merge(pivot, on=keys, how="left")
            table = table[table["support"] >= min_bucket_support].copy()
        else:
            counts = exits["stage2_route"].value_counts().to_dict()
            table = pd.DataFrame([{"support": int(sum(counts.values())), **{route: int(counts.get(route, 0)) for route in stage2_classes}}])
        if table.empty:
            continue
        for route in stage2_classes:
            if route not in table.columns:
                table[route] = 0.0
            table[route] = pd.to_numeric(table[route], errors="coerce").fillna(0.0)
        denom = table[stage2_classes].sum(axis=1) + alpha * len(stage2_classes)
        for route in stage2_classes:
            table[f"p_cond_{route}"] = (pd.to_numeric(table[route], errors="coerce").fillna(0.0) + alpha) / denom
        table["fallback_level"] = level_name
        tables.append({"level_name": level_name, "keys": keys, "table": table})
    return {
        "classes": stage2_classes,
        "tables": tables,
    }


def predict_stage2_route_probs(panel: pd.DataFrame, stage2_model: dict[str, object]) -> pd.DataFrame:
    result = panel[["company_id", "quarter_idx", "universe", "sector_bucket", "stage_bucket"]].copy()
    classes = list(stage2_model.get("classes", []))
    for route in classes:
        result[f"p_cond_{route}"] = np.nan
    result["stage2_fallback_level"] = pd.Series([None] * len(result), dtype=object)
    for payload in stage2_model.get("tables", []):
        keys = list(payload["keys"])
        table = payload["table"].copy()
        prob_columns = [f"p_cond_{route}" for route in classes]
        if keys:
            merged = result[keys].merge(table[keys + prob_columns + ["fallback_level"]], on=keys, how="left")
            fill_mask = result["stage2_fallback_level"].isna() & merged["fallback_level"].notna()
            for column in prob_columns:
                result.loc[fill_mask, column] = merged.loc[fill_mask, column].to_numpy()
            result.loc[fill_mask, "stage2_fallback_level"] = merged.loc[fill_mask, "fallback_level"].to_numpy()
        else:
            for column in prob_columns:
                result.loc[result[column].isna(), column] = float(table.iloc[0][column])
            result.loc[result["stage2_fallback_level"].isna(), "stage2_fallback_level"] = str(table.iloc[0]["fallback_level"])
    if classes:
        row_sum = result[[f"p_cond_{route}" for route in classes]].sum(axis=1)
        for route in classes:
            result[f"p_cond_{route}"] = pd.to_numeric(result[f"p_cond_{route}"], errors="coerce").fillna(0.0) / np.clip(row_sum, 1e-9, None)
    return result


def stage2_multiclass_metrics(
    frame: pd.DataFrame,
    stage2_classes: list[str],
    evaluation_view: str,
) -> pd.DataFrame:
    if frame.empty or not stage2_classes:
        return pd.DataFrame(
            [
                {
                    "evaluation_view": evaluation_view,
                    "rows": 0,
                    "class_set": "|".join(stage2_classes),
                    "accuracy": np.nan,
                    "macro_brier_score": np.nan,
                    "top_route_hit_rate": np.nan,
                }
            ]
        )
    actual = frame["stage2_route"].astype(str)
    predicted = frame[[f"p_cond_{route}" for route in stage2_classes]].idxmax(axis=1).str.replace("p_cond_", "", regex=False)
    accuracy = float((predicted == actual).mean()) if len(frame) else np.nan
    brier_values = []
    for route in stage2_classes:
        actual_binary = actual.eq(route).astype(int)
        predicted_prob = pd.to_numeric(frame[f"p_cond_{route}"], errors="coerce").fillna(0.0)
        brier_values.append(float(np.mean((predicted_prob - actual_binary) ** 2)))
    return pd.DataFrame(
        [
            {
                "evaluation_view": evaluation_view,
                "rows": int(len(frame)),
                "class_set": "|".join(stage2_classes),
                "accuracy": accuracy,
                "macro_brier_score": float(np.mean(brier_values)) if brier_values else np.nan,
                "top_route_hit_rate": accuracy,
            }
        ]
    )


def build_stage2_route_support(panel: pd.DataFrame, stage2_classes: list[str]) -> pd.DataFrame:
    exits = panel[panel["route_label"].astype(str).isin(MAIN_DIRECT_ROUTES)].copy()
    exits["stage2_route"] = exits["route_label"].map(lambda value: map_stage2_route_label(value, stage2_classes))
    exits = exits[exits["stage2_route"].notna()].copy()
    support = (
        exits.groupby(["split", "universe", "stage2_route"], as_index=False, observed=True)
        .agg(rows=("company_id", "size"), companies=("company_id", "nunique"))
    )
    return support.sort_values(["split", "universe", "stage2_route"]).reset_index(drop=True)


def build_stage2_route_metrics(
    scored_holdout: pd.DataFrame,
    stage2_classes: list[str],
) -> pd.DataFrame:
    hard_rows = scored_holdout.loc[
        scored_holdout["realized_hard_timely_liquidity_by_horizon"].astype(int).eq(1)
        & scored_holdout["stage2_route"].notna()
    ].copy()
    frames = [stage2_multiclass_metrics(hard_rows, stage2_classes, "full_test")]
    for universe in UNIVERSE_ORDER:
        frames.append(
            stage2_multiclass_metrics(
                hard_rows[hard_rows["universe"].astype(str).eq(universe)].copy(),
                stage2_classes,
                f"universe_{universe}",
            )
        )
    return pd.concat(frames, ignore_index=True)


def probability_by_horizon_vectorized(panel: pd.DataFrame, fitted: dict, horizon_quarters: int) -> pd.DataFrame:
    result, _ = probability_path_summary_vectorized(panel, fitted, horizon_quarters)
    return result


def realized_exit_paths(panel: pd.DataFrame, horizon_quarters: int) -> pd.DataFrame:
    exit_quarter = pd.to_numeric(panel.get("exit_quarter_idx"), errors="coerce")
    current_quarter = pd.to_numeric(panel["quarter_idx"], errors="coerce")
    realized = panel[["company_id", "quarter_idx"]].copy()
    for horizon_step in range(1, horizon_quarters + 1):
        realized[f"realized_exit_by_h{horizon_step}"] = (
            exit_quarter.notna()
            & (exit_quarter >= current_quarter)
            & (exit_quarter <= current_quarter + horizon_step - 1)
        ).astype(int)
    return realized


def calibration_by_decile(
    scored_panel: pd.DataFrame,
    prediction_col: str = "pred_exit_by_horizon",
    realized_col: str = "realized_exit_by_horizon",
) -> pd.DataFrame:
    ranked = scored_panel.copy()
    if ranked.empty:
        return pd.DataFrame(columns=["decile", "n", "mean_predicted_exit", "realized_exit_rate"])
    ranked["decile"] = pd.qcut(
        ranked[prediction_col].rank(method="first"),
        10,
        labels=False,
        duplicates="drop",
    )
    calibration = (
        ranked.groupby("decile", as_index=False)
        .agg(
            n=(prediction_col, "size"),
            mean_predicted_exit=(prediction_col, "mean"),
            realized_exit_rate=(realized_col, "mean"),
        )
        .sort_values("decile")
    )
    calibration["decile"] = calibration["decile"] + 1
    return calibration


def summarize_calibration(calibration: pd.DataFrame, label: str) -> pd.DataFrame:
    if calibration.empty:
        return pd.DataFrame(
            [
                {
                    "calibration_view": label,
                    "deciles": 0,
                    "rows": 0,
                    "mean_abs_gap": np.nan,
                    "max_abs_gap": np.nan,
                }
            ]
        )
    abs_gap = (calibration["realized_exit_rate"] - calibration["mean_predicted_exit"]).abs()
    return pd.DataFrame(
        [
            {
                "calibration_view": label,
                "deciles": int(len(calibration)),
                "rows": int(calibration["n"].sum()),
                "mean_abs_gap": float(abs_gap.mean()),
                "max_abs_gap": float(abs_gap.max()),
            }
        ]
    )


def safe_roc_auc(realized: pd.Series, predicted: pd.Series) -> float:
    y = pd.to_numeric(realized, errors="coerce").fillna(0).astype(int)
    p = pd.to_numeric(predicted, errors="coerce").fillna(0.0)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = p.rank(method="average").to_numpy(dtype=float)
    auc = (ranks[y.to_numpy(dtype=bool)].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def safe_pr_auc(realized: pd.Series, predicted: pd.Series) -> float:
    y = pd.to_numeric(realized, errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    p = pd.to_numeric(predicted, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.clip(tp + fp, 1, None)
    recall = tp / positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def fit_calibration_curve(realized: pd.Series, predicted: pd.Series) -> tuple[float, float, str, str]:
    y = pd.to_numeric(realized, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    p = np.clip(pd.to_numeric(predicted, errors="coerce").fillna(0.0).to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    if y.size == 0:
        return float("nan"), float("nan"), "empty_view", "empty_view"
    if np.unique(y).size < 2:
        return float("nan"), float("nan"), "insufficient_class_support", "insufficient_class_support"
    logits = np.log(p / (1.0 - p))
    if float(np.std(logits, ddof=0)) < 1e-8:
        base_rate = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
        intercept = math.log(base_rate / (1.0 - base_rate))
        return 0.0, float(intercept), "low_variance_probability_assumed_flat", "low_variance_probability_assumed_flat"

    def objective(theta: np.ndarray) -> float:
        linear = theta[0] + theta[1] * logits
        fitted_prob = 1.0 / (1.0 + np.exp(-linear))
        return float(
            -np.sum(
                y * np.log(np.clip(fitted_prob, 1e-9, 1.0))
                + (1.0 - y) * np.log(np.clip(1.0 - fitted_prob, 1e-9, 1.0))
            )
            + 1e-4 * 0.5 * np.sum(theta**2)
        )

    base_rate = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
    start = np.array([math.log(base_rate / (1.0 - base_rate)), 1.0], dtype=float)
    result = minimize(objective, start, method="BFGS")
    if not result.success:
        return float("nan"), float("nan"), f"optimizer_failed:{result.message}", f"optimizer_failed:{result.message}"
    intercept, slope = result.x
    if not (np.isfinite(intercept) and np.isfinite(slope)):
        return float("nan"), float("nan"), "nonfinite_estimate", "nonfinite_estimate"
    return float(slope), float(intercept), "ok", "ok"


def top_decile_realized_rate(realized: pd.Series, predicted: pd.Series) -> float:
    if len(predicted) == 0:
        return float("nan")
    ranked = pd.DataFrame({"realized": realized, "predicted": predicted}).sort_values("predicted", ascending=False)
    n_top = max(int(math.ceil(len(ranked) * 0.10)), 1)
    return float(pd.to_numeric(ranked.head(n_top)["realized"], errors="coerce").fillna(0.0).mean())


def top_decile_lift(realized: pd.Series, predicted: pd.Series) -> float:
    realized_series = pd.to_numeric(realized, errors="coerce").fillna(0.0)
    base_rate = float(realized_series.mean()) if len(realized_series) else float("nan")
    if (not np.isfinite(base_rate)) or base_rate <= 0.0:
        return float("nan")
    return float(top_decile_realized_rate(realized_series, predicted) / base_rate)


def integrated_brier_score(
    scored_panel: pd.DataFrame,
    horizon_quarters: int,
    prediction_prefix: str = "pred_exit_by_h",
    realized_prefix: str = "realized_exit_by_h",
) -> float:
    scores: list[float] = []
    for horizon_step in range(1, horizon_quarters + 1):
        pred_col = f"{prediction_prefix}{horizon_step}"
        realized_col = f"{realized_prefix}{horizon_step}"
        if pred_col not in scored_panel.columns or realized_col not in scored_panel.columns:
            continue
        predicted = pd.to_numeric(scored_panel[pred_col], errors="coerce").fillna(0.0)
        realized = pd.to_numeric(scored_panel[realized_col], errors="coerce").fillna(0.0)
        scores.append(float(np.mean((predicted - realized) ** 2)))
    return float(np.mean(scores)) if scores else float("nan")


def summarize_evaluation_view(
    scored_panel: pd.DataFrame,
    view_name: str,
    prediction_col: str,
    realized_col: str,
    horizon_quarters: int,
    prediction_prefix: str = "pred_exit_by_h",
    realized_prefix: str = "realized_exit_by_h",
) -> pd.DataFrame:
    calibration = calibration_by_decile(scored_panel, prediction_col=prediction_col, realized_col=realized_col)
    slope, intercept, slope_status, intercept_status = fit_calibration_curve(
        scored_panel[realized_col],
        scored_panel[prediction_col],
    )
    abs_gap = (
        (calibration["realized_exit_rate"] - calibration["mean_predicted_exit"]).abs()
        if not calibration.empty
        else pd.Series(dtype=float)
    )
    predicted = pd.to_numeric(scored_panel[prediction_col], errors="coerce").fillna(0.0)
    realized = pd.to_numeric(scored_panel[realized_col], errors="coerce").fillna(0.0)
    return pd.DataFrame(
        [
            {
                "evaluation_view": view_name,
                "rows": int(len(scored_panel)),
                "brier_score": float(np.mean((predicted - realized) ** 2)) if len(predicted) else np.nan,
                "integrated_brier_score": integrated_brier_score(
                    scored_panel,
                    horizon_quarters,
                    prediction_prefix=prediction_prefix,
                    realized_prefix=realized_prefix,
                ),
                "pr_auc": safe_pr_auc(realized, predicted),
                "roc_auc": safe_roc_auc(realized, predicted),
                "calibration_slope": slope,
                "calibration_slope_status": slope_status,
                "calibration_intercept": intercept,
                "calibration_intercept_status": intercept_status,
                "top_decile_realized_exit_rate": top_decile_realized_rate(realized, predicted),
                "top_decile_lift": top_decile_lift(realized, predicted),
                "mean_abs_calibration_gap": float(abs_gap.mean()) if not abs_gap.empty else np.nan,
                "max_abs_calibration_gap": float(abs_gap.max()) if not abs_gap.empty else np.nan,
            }
        ]
    )


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return 0.0
    return float(numerator) / denominator_value


def binary_confusion_counts(actual_positive: pd.Series | np.ndarray, predicted_positive: pd.Series | np.ndarray) -> dict[str, int]:
    actual = pd.Series(actual_positive).fillna(0).astype(int).to_numpy(dtype=bool)
    predicted = pd.Series(predicted_positive).fillna(0).astype(int).to_numpy(dtype=bool)
    return {
        "TP": int(np.sum(actual & predicted)),
        "FP": int(np.sum((~actual) & predicted)),
        "TN": int(np.sum((~actual) & (~predicted))),
        "FN": int(np.sum(actual & (~predicted))),
    }


def binary_confusion_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp = int(counts["TP"])
    fp = int(counts["FP"])
    tn = int(counts["TN"])
    fn = int(counts["FN"])
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = safe_ratio(2.0 * precision * recall, precision + recall)
    prevalence = safe_ratio(tp + fn, tp + fp + tn + fn)
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "F1": f1,
        "prevalence": prevalence,
    }


def resolve_probability_thresholds(config: dict) -> list[float]:
    thresholds = config.get("fixed_probability_thresholds", [0.01, 0.02, 0.03])
    values = sorted({round(float(value), 4) for value in thresholds})
    return values


def format_threshold_label(threshold: float) -> str:
    return f"{float(threshold):.2f}"


def build_binary_confusion_exports(
    frame: pd.DataFrame,
    prediction_col: str,
    actual_col: str,
    thresholds: list[float],
    data_mode: str,
    evaluation_view: str,
    target_label: str,
    prediction_label: str,
    join_key: str | None = None,
    join_value: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    predicted = pd.to_numeric(frame[prediction_col], errors="coerce").fillna(0.0)
    actual = pd.to_numeric(frame[actual_col], errors="coerce").fillna(0).astype(int)
    for threshold in thresholds:
        predicted_positive = predicted >= float(threshold)
        counts = binary_confusion_counts(actual, predicted_positive.astype(int))
        metrics = binary_confusion_metrics(counts)
        summary_row = {
            "data_mode": data_mode,
            "evaluation_view": evaluation_view,
            "target_label": target_label,
            "prediction_label": prediction_label,
            "threshold": float(threshold),
            "threshold_label": format_threshold_label(threshold),
            "eligible_observations": int(len(frame)),
            **counts,
            **metrics,
        }
        if join_key and join_value is not None:
            summary_row[join_key] = join_value
        summary_rows.append(summary_row)
        for metric_name, metric_value in {
            "eligible_observations": int(len(frame)),
            **counts,
            **metrics,
        }.items():
            long_row = {
                "data_mode": data_mode,
                "evaluation_view": evaluation_view,
                "target_label": target_label,
                "prediction_label": prediction_label,
                "threshold": float(threshold),
                "threshold_label": format_threshold_label(threshold),
                "metric_name": metric_name,
                "metric_value": float(metric_value) if isinstance(metric_value, (float, np.floating)) else int(metric_value),
            }
            if join_key and join_value is not None:
                long_row[join_key] = join_value
            long_rows.append(long_row)
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def build_policy_rules(decision_panel: pd.DataFrame, config: dict) -> list[dict[str, object]]:
    thresholds = resolve_probability_thresholds(config)
    ce_threshold = float(config.get("certainty_equivalent_threshold", 0.0))
    predicted_prob = pd.to_numeric(decision_panel["pred_exit_by_horizon"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    predicted_ce = pd.to_numeric(decision_panel["predicted_certainty_equivalent"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rules: list[dict[str, object]] = [
        {
            "policy_key": "no_filter",
            "decision_rule": "no_filter",
            "policy_family": "no_filter",
            "prob_threshold": np.nan,
            "certainty_equivalent_threshold": np.nan,
            "accept_mask": np.ones(len(decision_panel), dtype=bool),
        }
    ]
    for threshold in thresholds:
        threshold_label = format_threshold_label(threshold)
        rules.append(
            {
                "policy_key": f"prob_ge_{threshold_label}",
                "decision_rule": f"prob_exit_by_8q >= {threshold_label}",
                "policy_family": "probability_threshold",
                "prob_threshold": float(threshold),
                "certainty_equivalent_threshold": np.nan,
                "accept_mask": predicted_prob >= float(threshold),
            }
        )
    rules.append(
        {
            "policy_key": "ce_gt_0",
            "decision_rule": "CE > 0",
            "policy_family": "certainty_equivalent_threshold",
            "prob_threshold": np.nan,
            "certainty_equivalent_threshold": ce_threshold,
            "accept_mask": predicted_ce > ce_threshold,
        }
    )
    for threshold in thresholds:
        threshold_label = format_threshold_label(threshold)
        rules.append(
            {
                "policy_key": f"dual_prob_ge_{threshold_label}_ce_gt_0",
                "decision_rule": f"dual rule: CE > 0 and prob_exit_by_8q >= {threshold_label}",
                "policy_family": "dual_rule",
                "prob_threshold": float(threshold),
                "certainty_equivalent_threshold": ce_threshold,
                "accept_mask": (predicted_ce > ce_threshold) & (predicted_prob >= float(threshold)),
            }
        )
    rules.append(
        {
            "policy_key": "cash_benchmark",
            "decision_rule": "cash_benchmark",
            "policy_family": "cash_benchmark",
            "prob_threshold": np.nan,
            "certainty_equivalent_threshold": np.nan,
            "accept_mask": np.zeros(len(decision_panel), dtype=bool),
        }
    )
    return rules


def multiclass_confusion_table(
    actual_labels: pd.Series,
    predicted_labels: pd.Series,
    classes: list[str],
    data_mode: str,
    class_set_name: str,
    note: str,
) -> pd.DataFrame:
    matrix = (
        pd.crosstab(
            pd.Categorical(actual_labels, categories=classes),
            pd.Categorical(predicted_labels, categories=classes),
            dropna=False,
        )
        .reindex(index=classes, columns=classes, fill_value=0)
    )
    long_rows: list[dict[str, object]] = []
    for actual_class in classes:
        row_total = int(matrix.loc[actual_class].sum())
        for predicted_class in classes:
            count = int(matrix.loc[actual_class, predicted_class])
            long_rows.append(
                {
                    "data_mode": data_mode,
                    "class_set_name": class_set_name,
                    "actual_class": actual_class,
                    "predicted_class": predicted_class,
                    "count": count,
                    "row_normalized_pct": safe_ratio(count, row_total),
                    "note": note,
                }
            )
    return pd.DataFrame(long_rows)


def build_route_competing_risks_summary(
    views: dict[str, pd.DataFrame],
    include_pooled_strategic: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    route_labels = list(MAIN_DIRECT_ROUTES)
    if include_pooled_strategic:
        route_labels.append("pooled_strategic_exit")
    for view_name, frame in views.items():
        for route_name in route_labels:
            if route_name == "pooled_strategic_exit":
                predicted = pd.to_numeric(frame.get("cum_ipo"), errors="coerce").fillna(0.0) + pd.to_numeric(
                    frame.get("cum_mna"), errors="coerce"
                ).fillna(0.0)
                realized = (
                    frame["company_exit_route"].astype(str).isin(["ipo", "mna"])
                    & pd.to_numeric(frame["realized_exit_by_horizon"], errors="coerce").fillna(0).astype(int).eq(1)
                ).astype(int)
            else:
                predicted = pd.to_numeric(frame.get(f"cum_{route_name}"), errors="coerce").fillna(0.0)
                realized = (
                    frame["company_exit_route"].astype(str).eq(route_name)
                    & pd.to_numeric(frame["realized_exit_by_horizon"], errors="coerce").fillna(0).astype(int).eq(1)
                ).astype(int)
            rows.append(
                {
                    "evaluation_view": view_name,
                    "route_label": route_name,
                    "rows": int(len(frame)),
                    "mean_predicted_probability": float(predicted.mean()) if len(predicted) else np.nan,
                    "realized_rate": float(realized.mean()) if len(realized) else np.nan,
                    "brier_score": float(np.mean((predicted - realized) ** 2)) if len(predicted) else np.nan,
                    "roc_auc": safe_roc_auc(realized, predicted),
                }
            )
    return pd.DataFrame(rows)


def build_promotion_gate(
    summary_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    calibration_high_confidence: pd.DataFrame,
    route_mapping_comparison: pd.DataFrame,
) -> pd.DataFrame:
    baseline = summary_metrics.loc[summary_metrics["scenario"] == "baseline"].iloc[0]
    freeze = summary_metrics.loc[summary_metrics["scenario"] == "exit_freeze"].iloc[0]
    main_abs_gap = (calibration["realized_exit_rate"] - calibration["mean_predicted_exit"]).abs()
    high_abs_gap = (
        (calibration_high_confidence["realized_exit_rate"] - calibration_high_confidence["mean_predicted_exit"]).abs()
        if not calibration_high_confidence.empty
        else pd.Series(dtype=float)
    )
    main_mae = float(main_abs_gap.mean()) if not calibration.empty else np.nan
    high_mae = float(high_abs_gap.mean()) if not high_abs_gap.empty else np.nan
    sensitivity_counts = route_mapping_comparison.set_index(["mapping_scope", "route_label"])["chosen_exit_count"].to_dict()
    sensitivity_soft_failures = int(sensitivity_counts.get(("sensitivity", "soft_failure_sensitivity"), 0))
    total_sensitivity = int(
        route_mapping_comparison.loc[route_mapping_comparison["mapping_scope"] == "sensitivity", "chosen_exit_count"].sum()
    )
    sensitivity_share = sensitivity_soft_failures / max(total_sensitivity, 1)
    freeze_direction_ok = bool(
        float(baseline["prob_exit_by_horizon"]) > float(freeze["prob_exit_by_horizon"])
        and float(baseline["mean_npv"]) > float(freeze["mean_npv"])
    )
    calibration_improved = bool(np.isfinite(main_mae) and main_mae <= 0.10)
    stable_across_confidence = bool(
        np.isfinite(main_mae)
        and np.isfinite(high_mae)
        and abs(main_mae - high_mae) <= 0.03
    )
    chapter_evidence_ready = bool(
        calibration_improved
        and stable_across_confidence
        and freeze_direction_ok
        and sensitivity_share <= 0.40
    )
    return pd.DataFrame(
        [
            {
                "main_calibration_mean_abs_gap": main_mae,
                "high_confidence_calibration_mean_abs_gap": high_mae,
                "soft_failure_sensitivity_share": sensitivity_share,
                "freeze_direction_ok": freeze_direction_ok,
                "calibration_improved": calibration_improved,
                "stable_across_confidence_cuts": stable_across_confidence,
                "chapter_evidence_ready": chapter_evidence_ready,
            }
        ]
    )


def build_coverage_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_year = (
        panel.assign(year=lambda frame: frame["quarter_idx"] // 4)
        .groupby("year", as_index=False)
        .agg(
            rows=("company_id", "size"),
            companies=("company_id", "nunique"),
            exits=("route_label", lambda values: int(np.sum(values != "no_exit"))),
        )
    )
    by_split = (
        panel.groupby("split", as_index=False)
        .agg(
            rows=("company_id", "size"),
            companies=("company_id", "nunique"),
            exits=("route_label", lambda values: int(np.sum(values != "no_exit"))),
            ipo=("route_label", lambda values: int(np.sum(values == "ipo"))),
            mna=("route_label", lambda values: int(np.sum(values == "mna"))),
            sponsor_sale=("route_label", lambda values: int(np.sum(values == "sponsor_sale"))),
            writeoff=("route_label", lambda values: int(np.sum(values == "writeoff"))),
        )
    )
    return by_year, by_split


def calibrate_route_multiples(
    round_events: pd.DataFrame,
    chosen_exits: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    prior_round = round_events.sort_values(["company_id", "quarter_idx"]).copy()
    prior_round = prior_round.rename(columns={"quarter_idx": "prior_quarter_idx"})
    exits = chosen_exits.sort_values(["company_id", "exit_quarter_idx"]).copy()
    prior_round = prior_round.sort_values(["prior_quarter_idx", "company_id"]).reset_index(drop=True)
    exits = exits.sort_values(["exit_quarter_idx", "company_id"]).reset_index(drop=True)
    merged = pd.merge_asof(
        exits,
        prior_round[["company_id", "prior_quarter_idx", "round_amount_usd"]],
        left_on="exit_quarter_idx",
        right_on="prior_quarter_idx",
        by="company_id",
        direction="backward",
        allow_exact_matches=False,
    )
    merged["multiple"] = merged["event_value_usd"] / merged["round_amount_usd"].replace(0.0, np.nan)
    merged["multiple"] = merged["multiple"].where(
        merged["multiple"].between(0.05, 25.0),
        np.nan,
    )
    params = {route: values.copy() for route, values in FALLBACK_MULTIPLE_PARAMS.items()}
    for route in EXIT_ROUTES:
        sample = merged.loc[merged["route_label"] == route, "multiple"].dropna()
        if len(sample) >= 10 and route != "writeoff":
            log_sample = np.log(sample.to_numpy(dtype=float))
            params[route]["mu"] = float(log_sample.mean())
            params[route]["sigma"] = float(max(log_sample.std(ddof=0), 0.15))
    return params


def build_future_states(
    stylized_row: pd.Series,
    horizon_quarters: int,
    config: dict,
    scenario_name: str,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    baseline_regime = float(stylized_row["market_regime"])
    if scenario_name == "exit_freeze":
        macro_path = np.repeat(baseline_regime + float(config["freeze_regime_shift"]), horizon_quarters)
        route_shift_path = np.repeat(float(config["freeze_exit_logit_shift"]), horizon_quarters)
    else:
        macro_path = np.repeat(baseline_regime, horizon_quarters)
        route_shift_path = np.repeat(0.0, horizon_quarters)
    feature_columns = feature_columns or [column for column in COMPANY_FEATURES if column in stylized_row.index]
    rows = []
    for step in range(1, horizon_quarters + 1):
        row = {
            "company_id": stylized_row["company_id"],
            "quarter_idx": int(stylized_row["quarter_idx"]) + step,
            "market_regime": float(macro_path[step - 1]),
            "scenario_exit_logit_shift": float(route_shift_path[step - 1]),
        }
        for column in feature_columns:
            if column not in stylized_row.index:
                continue
            value = float(stylized_row[column]) if pd.notna(stylized_row[column]) else 0.0
            if column == "age_q":
                row[column] = value + (step - 1)
            elif column == "time_since_last_round_q":
                row[column] = value + (step - 1)
            else:
                row[column] = value
        rows.append(row)
    return pd.DataFrame(rows)


def cumulative_incidence(future_probs: pd.DataFrame) -> pd.DataFrame:
    survival = 1.0
    rows = []
    cumulative = {route: 0.0 for route in EXIT_ROUTES}
    for horizon_step, row in enumerate(future_probs.itertuples(index=False), start=1):
        point = {}
        for route in EXIT_ROUTES:
            route_prob = survival * getattr(row, f"p_{route}")
            cumulative[route] += route_prob
            point[f"cum_{route}"] = cumulative[route]
        survival = survival * row.p_no_exit
        point["survival"] = survival
        point["horizon_q"] = horizon_step
        rows.append(point)
    result = pd.DataFrame(rows)
    result["prob_exit_by_horizon"] = 1.0 - result["survival"]
    return result


def discount_factors(quarterly_rates: np.ndarray) -> np.ndarray:
    rates = np.asarray(quarterly_rates, dtype=float)
    return np.cumprod(1.0 / (1.0 + rates))


def draw_route_multiple(route: str, multiple_params: dict[str, dict[str, float]], rng: np.random.Generator, size: int) -> np.ndarray:
    params = multiple_params[route]
    return rng.lognormal(mean=float(params["mu"]), sigma=float(params["sigma"]), size=size)


def simulate_npv(
    stylized_row: pd.Series,
    future_probs: pd.DataFrame,
    multiple_params: dict[str, dict[str, float]],
    config: dict,
    scenario_name: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["random_seed"]) + (0 if scenario_name == "baseline" else 17))
    n_paths = int(config["n_simulations"])
    quarterly_rates = np.repeat(0.02 / 4.0, len(future_probs))
    discount = discount_factors(quarterly_rates)
    if scenario_name == "exit_freeze":
        kappa_multiplier = float(config["freeze_kappa_multiplier"])
        multiple_multiplier = float(config["freeze_multiple_multiplier"])
    else:
        kappa_multiplier = 1.0
        multiple_multiplier = 1.0

    v0 = float(np.expm1(stylized_row["log_last_round_usd"]))
    v0 = max(v0, 1.0)
    p0 = float(config["purchase_price_fraction_of_v0"]) * float(config["ownership"]) * v0
    tv_t = float(config["tv_fraction_of_v0"]) * float(config["ownership"]) * v0

    route_probability_matrix = future_probs[[f"p_{route}" for route in EXIT_ROUTES]].to_numpy(dtype=float)
    p_no_exit = future_probs["p_no_exit"].to_numpy(dtype=float)

    outcomes = []
    for _ in range(n_paths):
        realized_cash_flow = 0.0
        realized_route = "no_exit"
        realized_horizon = len(future_probs) + 1
        for step in range(len(future_probs)):
            step_probs = np.concatenate(([p_no_exit[step]], route_probability_matrix[step]))
            draw_idx = int(rng.choice(len(ROUTES), p=step_probs / step_probs.sum()))
            draw = ROUTES[draw_idx]
            if draw == "no_exit":
                continue
            multiple = float(draw_route_multiple(draw, multiple_params, rng, 1)[0]) * multiple_multiplier
            kappa = float(multiple_params[draw]["kappa"]) * kappa_multiplier
            realized_cash_flow = float(config["ownership"]) * kappa * multiple * v0 * discount[step]
            realized_route = draw
            realized_horizon = step + 1
            break
        if realized_route == "no_exit":
            realized_cash_flow = tv_t * discount[-1]
        npv = -p0 + realized_cash_flow
        outcomes.append(
            {
                "scenario": scenario_name,
                "route": realized_route,
                "horizon_q": realized_horizon,
                "npv": npv,
                "purchase_price": p0,
                "reference_value": v0,
            }
        )
    return pd.DataFrame(outcomes)


def decision_metrics(npv_paths: pd.DataFrame, incidence: pd.DataFrame, config: dict) -> pd.DataFrame:
    mean_npv = float(npv_paths["npv"].mean())
    var_npv = float(npv_paths["npv"].var(ddof=0))
    ce = mean_npv - 0.5 * float(config["gamma_risk_aversion"]) * var_npv
    return pd.DataFrame(
        [
            {
                "scenario": str(npv_paths["scenario"].iloc[0]),
                "mean_npv": mean_npv,
                "prob_npv_positive": float((npv_paths["npv"] > 0).mean()),
                "prob_exit_by_horizon": float(incidence["prob_exit_by_horizon"].iloc[-1]),
                "certainty_equivalent": ce,
            }
        ]
    )


def simulate_panel_value_summary(
    panel: pd.DataFrame,
    point_route_matrix: np.ndarray,
    multiple_params: dict[str, dict[str, float]],
    config: dict,
    scenario_name: str,
    n_paths: int | None = None,
) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "quarter_idx",
                "predicted_mean_npv",
                "predicted_certainty_equivalent",
                "predicted_prob_exit_by_horizon",
            ]
        )
    rng = np.random.default_rng(int(config["random_seed"]) + (101 if scenario_name == "exit_freeze" else 53))
    path_count = int(n_paths or config.get("decision_eval_paths", 64))
    horizon_quarters = point_route_matrix.shape[1]
    flattened = point_route_matrix.reshape(len(panel), horizon_quarters * len(EXIT_ROUTES))
    no_exit_prob = np.clip(1.0 - flattened.sum(axis=1), 0.0, 1.0)
    category_probs = np.concatenate([flattened, no_exit_prob[:, None]], axis=1)
    category_probs = category_probs / np.clip(category_probs.sum(axis=1, keepdims=True), 1e-12, None)
    category_cdf = np.cumsum(category_probs, axis=1)
    quarterly_rates = np.repeat(0.02 / 4.0, horizon_quarters)
    discount = discount_factors(quarterly_rates)
    if scenario_name == "exit_freeze":
        kappa_multiplier = float(config["freeze_kappa_multiplier"])
        multiple_multiplier = float(config["freeze_multiple_multiplier"])
    else:
        kappa_multiplier = 1.0
        multiple_multiplier = 1.0
    ownership = float(config["ownership"])
    purchase_fraction = float(config["purchase_price_fraction_of_v0"])
    terminal_fraction = float(config["tv_fraction_of_v0"])
    risk_aversion = float(config["gamma_risk_aversion"])
    rows = []
    for row_idx, row in enumerate(panel.itertuples(index=False)):
        v0 = max(float(np.expm1(getattr(row, "log_last_round_usd"))), 1.0)
        p0 = purchase_fraction * ownership * v0
        tv_t = terminal_fraction * ownership * v0
        draws = rng.random(path_count)
        category_idx = np.searchsorted(category_cdf[row_idx], draws, side="right")
        category_idx = np.clip(category_idx, 0, category_probs.shape[1] - 1)
        npvs = np.empty(path_count, dtype=float)
        no_exit_mask = category_idx == category_probs.shape[1] - 1
        npvs[no_exit_mask] = -p0 + tv_t * discount[-1]
        exit_idx = category_idx[~no_exit_mask]
        if exit_idx.size:
            step_idx = exit_idx // len(EXIT_ROUTES)
            route_idx = exit_idx % len(EXIT_ROUTES)
            realized_cash = np.empty(exit_idx.size, dtype=float)
            for local_route_idx, route_name in enumerate(EXIT_ROUTES):
                route_mask = route_idx == local_route_idx
                if not route_mask.any():
                    continue
                multiples = draw_route_multiple(route_name, multiple_params, rng, int(route_mask.sum())) * multiple_multiplier
                kappa = float(multiple_params[route_name]["kappa"]) * kappa_multiplier
                realized_cash[route_mask] = ownership * kappa * multiples * v0 * discount[step_idx[route_mask]]
            npvs[~no_exit_mask] = -p0 + realized_cash
        mean_npv = float(np.mean(npvs))
        var_npv = float(np.var(npvs, ddof=0))
        rows.append(
            {
                "company_id": getattr(row, "company_id"),
                "quarter_idx": int(getattr(row, "quarter_idx")),
                "predicted_mean_npv": mean_npv,
                "predicted_certainty_equivalent": mean_npv - 0.5 * risk_aversion * var_npv,
                "predicted_prob_exit_by_horizon": float(1.0 - no_exit_prob[row_idx]),
            }
        )
    return pd.DataFrame(rows)


def build_realized_value_proxy(
    panel: pd.DataFrame,
    multiple_params: dict[str, dict[str, float]],
    config: dict,
) -> pd.DataFrame:
    frame = panel[["company_id", "quarter_idx", "realized_exit_by_horizon", "company_exit_route"]].copy()
    horizon = int(config["holdout_horizon_quarters"])
    discount = discount_factors(np.repeat(0.02 / 4.0, horizon))
    ownership = float(config["ownership"])
    purchase_fraction = float(config["purchase_price_fraction_of_v0"])
    terminal_fraction = float(config["tv_fraction_of_v0"])
    v0 = np.maximum(np.expm1(pd.to_numeric(panel["log_last_round_usd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)), 1.0)
    purchase_price = purchase_fraction * ownership * v0
    terminal_value = terminal_fraction * ownership * v0 * discount[-1]
    realized_horizon = (
        pd.to_numeric(panel["exit_quarter_idx"], errors="coerce")
        - pd.to_numeric(panel["quarter_idx"], errors="coerce")
        + 1
    )
    realized_horizon = realized_horizon.clip(lower=1, upper=horizon).fillna(horizon).astype(int)
    observed_value = pd.to_numeric(panel.get("company_exit_value_usd"), errors="coerce").fillna(np.nan).to_numpy(dtype=float)
    realized_route = panel["company_exit_route"].astype(str).to_numpy()
    realized_cash = terminal_value.copy()
    exit_mask = pd.to_numeric(panel["realized_exit_by_horizon"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=bool)
    for route_name in EXIT_ROUTES:
        route_mask = exit_mask & (realized_route == route_name)
        if not route_mask.any():
            continue
        fallback_multiple = float(np.exp(multiple_params[route_name]["mu"]))
        route_multiple = np.where(
            np.isfinite(observed_value[route_mask]) & (observed_value[route_mask] > 0),
            observed_value[route_mask] / v0[route_mask],
            fallback_multiple,
        )
        route_multiple = np.clip(route_multiple, 0.01, 25.0)
        kappa = float(multiple_params[route_name]["kappa"])
        realized_cash[route_mask] = ownership * kappa * route_multiple * v0[route_mask] * discount[
            realized_horizon[route_mask] - 1
        ]
    frame["realized_npv_proxy"] = -purchase_price + realized_cash
    frame["realized_positive_value"] = (frame["realized_npv_proxy"] > 0).astype(int)
    return frame


def build_decision_backtest(decision_panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    if decision_panel.empty:
        return pd.DataFrame(
            columns=[
                "policy_key",
                "decision_rule",
                "policy_family",
                "confusion_target",
                "accepted_observations",
                "avg_predicted_prob_exit_by_8q",
                "realized_exit_by_8q",
                "avg_predicted_npv",
                "avg_predicted_certainty_equivalent",
                "realized_npv_proxy",
                "TP",
                "FP",
                "TN",
                "FN",
                "precision",
                "recall",
                "specificity",
                "balanced_accuracy",
                "F1",
                "prevalence",
                "false_positive_rate",
                "false_negative_rate",
                "prob_threshold",
                "certainty_equivalent_threshold",
                "decision_target",
            ]
        )
    realized_positive = decision_panel["realized_positive_value"].to_numpy(dtype=bool)
    rows = []
    for rule in build_policy_rules(decision_panel, config):
        accepted = np.asarray(rule["accept_mask"], dtype=bool)
        accepted_count = int(np.sum(accepted))
        accepted_frame = decision_panel.loc[accepted].copy()
        counts = binary_confusion_counts(realized_positive.astype(int), accepted.astype(int))
        metrics = binary_confusion_metrics(counts)
        rows.append(
            {
                "policy_key": str(rule["policy_key"]),
                "decision_rule": str(rule["decision_rule"]),
                "policy_family": str(rule["policy_family"]),
                "confusion_target": "realized_npv_proxy_positive",
                "accepted_observations": accepted_count,
                "avg_predicted_prob_exit_by_8q": float(accepted_frame["pred_exit_by_horizon"].mean()) if accepted_count else np.nan,
                "realized_exit_by_8q": float(accepted_frame["realized_exit_by_horizon"].mean()) if accepted_count else np.nan,
                "avg_predicted_npv": float(accepted_frame["predicted_mean_npv"].mean()) if accepted_count else np.nan,
                "avg_predicted_certainty_equivalent": float(accepted_frame["predicted_certainty_equivalent"].mean()) if accepted_count else np.nan,
                "realized_npv_proxy": float(accepted_frame["realized_npv_proxy"].mean()) if accepted_count else np.nan,
                **counts,
                **metrics,
                "false_positive_rate": metrics["specificity"] * 0.0 + safe_ratio(counts["FP"], counts["FP"] + counts["TN"]),
                "false_negative_rate": safe_ratio(counts["FN"], counts["FN"] + counts["TP"]),
                "prob_threshold": rule["prob_threshold"],
                "certainty_equivalent_threshold": rule["certainty_equivalent_threshold"],
                "decision_target": "realized_npv_proxy_positive",
            }
        )
    return pd.DataFrame(rows)


def build_decision_policy_confusion_exports(decision_panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decision_panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    long_frames = []
    summary_frames = []
    targets = [
        ("realized_npv_proxy_positive", "realized_positive_value"),
        ("realized_exit_by_8q", "realized_exit_by_horizon"),
    ]
    for rule in build_policy_rules(decision_panel, config):
        accepted = np.asarray(rule["accept_mask"], dtype=bool)
        for target_label, actual_col in targets:
            long_frame, summary_frame = build_binary_confusion_exports(
                decision_panel.assign(policy_accept=accepted.astype(int)),
                prediction_col="policy_accept",
                actual_col=actual_col,
                thresholds=[0.5],
                data_mode=str(config.get("data_mode", "sample")),
                evaluation_view="decision_policy",
                target_label=target_label,
                prediction_label=str(rule["decision_rule"]),
                join_key="policy_key",
                join_value=str(rule["policy_key"]),
            )
            long_frame["decision_rule"] = str(rule["decision_rule"])
            long_frame["policy_family"] = str(rule["policy_family"])
            long_frame["prob_threshold"] = rule["prob_threshold"]
            long_frame["certainty_equivalent_threshold"] = rule["certainty_equivalent_threshold"]
            long_frame["classification_threshold"] = 0.5
            summary_frame["decision_rule"] = str(rule["decision_rule"])
            summary_frame["policy_family"] = str(rule["policy_family"])
            summary_frame["prob_threshold"] = rule["prob_threshold"]
            summary_frame["certainty_equivalent_threshold"] = rule["certainty_equivalent_threshold"]
            summary_frame["classification_threshold"] = 0.5
            long_frames.append(long_frame)
            summary_frames.append(summary_frame)
    long_output = pd.concat(long_frames, ignore_index=True)
    summary_output = pd.concat(summary_frames, ignore_index=True)
    summary_output["threshold"] = summary_output["classification_threshold"]
    return long_output, summary_output


def score_stage1_holdout_panel(
    holdout_panel: pd.DataFrame,
    stage1_models: dict[str, dict],
    horizon_quarters: int,
    config: dict,
    company_master: pd.DataFrame,
    scenario_name: str = "baseline",
) -> tuple[pd.DataFrame, np.ndarray]:
    scored_summary, point_matrix = score_stage1_panel(
        holdout_panel,
        stage1_models,
        horizon_quarters,
        config,
        scenario_name=scenario_name,
        prediction_label="pred_hard_timely_liquidity_by_horizon",
    )
    keep_columns = [
        column
        for column in [
            "company_id",
            "quarter_idx",
            "route_label",
            "company_name",
            "realized_hard_timely_liquidity_by_horizon",
            "realized_soft_failure_sensitivity_by_horizon",
            "company_exit_route",
            "company_exit_value_usd",
            "company_exit_confidence_tier",
            "company_exit_route_source",
            "soft_failure_route_source",
            "soft_failure_confidence_tier",
            "exit_quarter_idx",
            "log_last_round_usd",
            "sector_bucket",
            "stage_bucket",
            "universe",
        ]
        if column in holdout_panel.columns
    ]
    scored_panel = scored_summary.merge(holdout_panel[keep_columns], on=["company_id", "quarter_idx"], how="left")
    scored_panel = scored_panel.merge(realized_exit_paths(holdout_panel, horizon_quarters), on=["company_id", "quarter_idx"], how="left")
    scored_panel = scored_panel.merge(
        company_master[["company_id", "match_confidence", "company_source", "universe"]].rename(
            columns={"match_confidence": "entity_match_confidence"}
        ),
        on=["company_id", "universe"],
        how="left",
    )
    return scored_panel, point_matrix


def build_evaluation_metrics_main(
    scored_holdout: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    horizon = int(config["holdout_horizon_quarters"])
    exit_label_high_conf_mask = target_positive_subset_mask(
        scored_holdout,
        "realized_hard_timely_liquidity_by_horizon",
        "exit_label_confidence_high",
    )
    if int(exit_label_high_conf_mask.sum()) == 0:
        exit_label_high_conf_mask = target_positive_subset_mask(
            scored_holdout,
            "realized_hard_timely_liquidity_by_horizon",
            "exit_label_confidence_high_or_medium",
        )
    entity_match_high_conf_mask = target_positive_subset_mask(
        scored_holdout,
        "realized_hard_timely_liquidity_by_horizon",
        "entity_match_confidence_high",
    )
    stress_start_idx = quarter_idx_from_label(str(config.get("stress_slice_start_quarter", "2020Q1")))
    stress_end_idx = quarter_idx_from_label(str(config.get("stress_slice_end_quarter", "2020Q4")))
    stress_mask = (
        pd.to_numeric(scored_holdout["quarter_idx"], errors="coerce").ge(stress_start_idx)
        & pd.to_numeric(scored_holdout["quarter_idx"], errors="coerce").le(stress_end_idx)
    )
    full_calibration = calibration_by_decile(
        scored_holdout,
        prediction_col="pred_hard_timely_liquidity_by_horizon",
        realized_col="realized_hard_timely_liquidity_by_horizon",
    )
    high_conf_calibration = calibration_by_decile(
        scored_holdout.loc[exit_label_high_conf_mask].copy(),
        prediction_col="pred_hard_timely_liquidity_by_horizon",
        realized_col="realized_hard_timely_liquidity_by_horizon",
    )
    metrics = pd.concat(
        [
            summarize_evaluation_view(
                scored_holdout,
                "full_test",
                "pred_hard_timely_liquidity_by_horizon",
                "realized_hard_timely_liquidity_by_horizon",
                horizon,
            ),
            summarize_evaluation_view(
                scored_holdout.loc[exit_label_high_conf_mask].copy(),
                "high_confidence_subset",
                "pred_hard_timely_liquidity_by_horizon",
                "realized_hard_timely_liquidity_by_horizon",
                horizon,
            ),
            summarize_evaluation_view(
                scored_holdout.loc[entity_match_high_conf_mask].copy(),
                "high_confidence_entity_match",
                "pred_hard_timely_liquidity_by_horizon",
                "realized_hard_timely_liquidity_by_horizon",
                horizon,
            ),
            summarize_evaluation_view(
                scored_holdout.loc[stress_mask].copy(),
                "stress_slice",
                "pred_hard_timely_liquidity_by_horizon",
                "realized_hard_timely_liquidity_by_horizon",
                horizon,
            ),
        ],
        ignore_index=True,
    )
    return metrics, full_calibration, high_conf_calibration, stress_mask


def build_evaluation_metrics_by_universe(
    scored_holdout: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    horizon = int(config["holdout_horizon_quarters"])
    frames = []
    for universe in UNIVERSE_ORDER:
        subset = scored_holdout[scored_holdout["universe"].astype(str).eq(universe)].copy()
        frames.append(
            summarize_evaluation_view(
                subset,
                universe,
                "pred_hard_timely_liquidity_by_horizon",
                "realized_hard_timely_liquidity_by_horizon",
                horizon,
            ).assign(universe=universe)
        )
    return pd.concat(frames, ignore_index=True)


def fit_probability_recalibrators(
    validation_scored: pd.DataFrame,
    prediction_col: str,
    realized_col: str,
) -> pd.DataFrame:
    rows = []
    full = validation_scored.copy()
    slope, intercept, slope_status, intercept_status = fit_calibration_curve(full[realized_col], full[prediction_col])
    rows.append(
        {
            "universe": "_overall",
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "calibration_slope_status": slope_status,
            "calibration_intercept_status": intercept_status,
        }
    )
    for universe in UNIVERSE_ORDER:
        subset = validation_scored[validation_scored["universe"].astype(str).eq(universe)].copy()
        if subset.empty:
            continue
        slope, intercept, slope_status, intercept_status = fit_calibration_curve(subset[realized_col], subset[prediction_col])
        rows.append(
            {
                "universe": universe,
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                "calibration_slope_status": slope_status,
                "calibration_intercept_status": intercept_status,
            }
        )
    return pd.DataFrame(rows)


def apply_probability_recalibrators(
    scored_panel: pd.DataFrame,
    recalibrators: pd.DataFrame,
    prediction_col: str,
) -> pd.DataFrame:
    if scored_panel.empty or recalibrators.empty:
        return scored_panel.copy()
    recalibrator_map = recalibrators.set_index("universe").to_dict(orient="index")
    output = scored_panel.copy()
    calibrated = pd.to_numeric(output[prediction_col], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=True)
    for universe in output["universe"].astype(str).unique().tolist():
        params = recalibrator_map.get(str(universe), recalibrator_map.get("_overall", {}))
        mask = output["universe"].astype(str).eq(str(universe)).to_numpy(dtype=bool)
        slope = params.get("calibration_slope")
        intercept = params.get("calibration_intercept")
        slope_status = str(params.get("calibration_slope_status", ""))
        intercept_status = str(params.get("calibration_intercept_status", ""))
        if not (np.isfinite(slope) and np.isfinite(intercept)) or slope_status != "ok" or intercept_status != "ok":
            continue
        base = np.clip(calibrated[mask], 1e-6, 1.0 - 1e-6)
        logits = np.log(base / (1.0 - base))
        calibrated[mask] = 1.0 / (1.0 + np.exp(-(float(intercept) + float(slope) * logits)))
    output[prediction_col] = calibrated
    return output


def build_sector_stage_metrics(
    scored_holdout: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    horizon = int(config["holdout_horizon_quarters"])
    grouped = []
    for (universe, sector_bucket, stage_bucket), subset in scored_holdout.groupby(
        ["universe", "sector_bucket", "stage_bucket"],
        observed=True,
    ):
        if len(subset) < 100:
            continue
        metrics = summarize_evaluation_view(
            subset.copy(),
            "sector_stage_bucket",
            "pred_hard_timely_liquidity_by_horizon",
            "realized_hard_timely_liquidity_by_horizon",
            horizon,
        ).iloc[0].to_dict()
        grouped.append(
            {
                "universe": str(universe),
                "sector_bucket": str(sector_bucket),
                "stage_bucket": str(stage_bucket),
                "rows": int(len(subset)),
                "hard_timely_liquidity_events": int(
                    pd.to_numeric(subset["realized_hard_timely_liquidity_by_horizon"], errors="coerce").fillna(0).sum()
                ),
                **metrics,
            }
        )
    return pd.DataFrame(grouped)


def build_feature_coverage_by_block(
    panel: pd.DataFrame,
    company_master: pd.DataFrame,
    patent_matches: pd.DataFrame,
) -> pd.DataFrame:
    patent_match_count = int(patent_matches["company_id"].nunique()) if not patent_matches.empty else 0
    buyout_panel = panel.loc[panel["universe"].astype(str).eq("buyout_pe")].copy() if "universe" in panel.columns else pd.DataFrame()
    sponsor_feature_columns = [column for column in BUYOUT_SPONSOR_FUND_FEATURES if column in buyout_panel.columns]
    sponsor_active = bool(
        sponsor_feature_columns
        and float(
            buyout_panel[sponsor_feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs().sum().sum()
        )
        > 0.0
    )
    return pd.DataFrame(
        [
            {
                "feature_block": "company_core",
                "active_status": "active",
                "available_columns": "age_q|sector_stage",
                "covered_companies": int(company_master["company_id"].nunique()),
                "coverage_share": 1.0,
                "note": "Core company timing and bucket features are available across the modeled panel.",
            },
            {
                "feature_block": "financing_trajectory",
                "active_status": "active",
                "available_columns": "time_since_last_round_q|log_last_round_usd",
                "covered_companies": int(company_master["company_id"].nunique()),
                "coverage_share": 1.0,
                "note": "Financing history is reconstructed from dated rounds.",
            },
            {
                "feature_block": "sponsor_fund",
                "active_status": "market_quarter_active" if sponsor_active else "proxy_only",
                "available_columns": (
                    "sponsor_score|" + "|".join(sponsor_feature_columns)
                    if sponsor_active
                    else "sponsor_score"
                ),
                "covered_companies": int(buyout_panel["company_id"].nunique()) if sponsor_active else int(company_master["company_id"].nunique()),
                "coverage_share": safe_ratio(
                    int(buyout_panel["company_id"].nunique()) if sponsor_active else int(company_master["company_id"].nunique()),
                    int(company_master["company_id"].nunique()),
                ),
                "note": (
                    "Buyout/PE now includes PIT-lagged market-quarter sponsor/fund/LP features built from dated Preqin fund, manager, cash-flow, and investor extracts."
                    if sponsor_active
                    else "Only the baseline sponsor proxy is PIT-safe in the staged local files."
                ),
            },
            {
                "feature_block": "lp_demand",
                "active_status": "unsupported",
                "available_columns": "",
                "covered_companies": 0,
                "coverage_share": 0.0,
                "note": "Investor-detail extracts exist, but no PIT-safe company or fund link is staged in the local deal extracts.",
            },
            {
                "feature_block": "patent_core",
                "active_status": "sector_conditional_diagnostic",
                "available_columns": "patent_apps_visible_l4q|patent_stock_visible|patent_grants_l4q",
                "covered_companies": patent_match_count,
                "coverage_share": safe_ratio(patent_match_count, int(company_master["company_id"].nunique())),
                "note": "Patent features are retained only as sector-conditional challengers in the redesign pass.",
            },
        ]
    )


def build_sponsor_fund_feature_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(
            columns=[
                "universe",
                "feature_name",
                "rows",
                "nonzero_rows",
                "coverage_share",
                "mean_value",
            ]
        )
    rows: list[dict[str, object]] = []
    for universe in UNIVERSE_ORDER:
        subset = panel.loc[panel["universe"].astype(str).eq(universe)].copy()
        if subset.empty:
            continue
        for column in BUYOUT_SPONSOR_FUND_FEATURES:
            if column not in subset.columns:
                continue
            values = pd.to_numeric(subset[column], errors="coerce").fillna(0.0)
            nonzero_rows = int(values.ne(0.0).sum())
            rows.append(
                {
                    "universe": universe,
                    "feature_name": column,
                    "rows": int(len(subset)),
                    "nonzero_rows": nonzero_rows,
                    "coverage_share": safe_ratio(nonzero_rows, int(len(subset))),
                    "mean_value": float(values.mean()),
                }
            )
    return pd.DataFrame(rows)


def build_sponsor_fund_join_audit(
    round_events: pd.DataFrame,
    sources: dict[str, pd.DataFrame] | None = None,
    buyout_market_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    has_lead_fund = int("lead_fund_id" in round_events.columns and round_events["lead_fund_id"].notna().any())
    has_deal_level_firm_id = int(any(column in round_events.columns for column in ["firm_id", "fund_id"]))
    fund_tables_present = 0
    manager_tables_present = 0
    investor_tables_present = 0
    if sources:
        fund_tables_present = int(
            any(not sources.get(key, pd.DataFrame()).empty for key in ["preqin_fund_details", "preqin_fund_performance", "preqin_fund_terms"])
        )
        manager_tables_present = int(not sources.get("preqin_manager_details", pd.DataFrame()).empty)
        investor_tables_present = int(not sources.get("preqin_investor_details", pd.DataFrame()).empty)
    market_quarter_panel_ready = int(buyout_market_panel is not None and not buyout_market_panel.empty)
    rows = [
        {
            "join_block": "sponsor_fund",
            "join_scope": "company_to_deal",
            "required_deal_key_present": 1,
            "sample_lead_fund_key_present": has_lead_fund,
            "point_in_time_join_supported": 1,
            "active_status": "active",
            "note": "Company snapshots are linked to dated deal events through the reconstructed company round panel.",
        },
        {
            "join_block": "sponsor_fund",
            "join_scope": "deal_to_fund_or_firm",
            "required_deal_key_present": has_deal_level_firm_id,
            "sample_lead_fund_key_present": has_lead_fund,
            "point_in_time_join_supported": int(has_deal_level_firm_id == 1),
            "active_status": "inactive_missing_deal_key" if has_deal_level_firm_id == 0 else "active",
            "note": (
                "The staged actual deal extracts do not carry firm_id or fund_id, so direct sponsor-level company joins remain unavailable."
                if has_deal_level_firm_id == 0
                else "A direct deal-to-fund join is available."
            ),
        },
        {
            "join_block": "sponsor_fund",
            "join_scope": "buyout_market_quarter",
            "required_deal_key_present": int(fund_tables_present == 1),
            "sample_lead_fund_key_present": int(manager_tables_present == 1 or investor_tables_present == 1),
            "point_in_time_join_supported": market_quarter_panel_ready,
            "active_status": "active_market_quarter" if market_quarter_panel_ready == 1 else "inactive",
            "note": (
                "PIT-lagged market-quarter sponsor/fund/LP features are active for buyout/PE using dated Preqin fund, manager, cash-flow, and investor extracts."
                if market_quarter_panel_ready == 1
                else "The dated fund and manager extracts were not sufficient to build a market-quarter sponsor/fund panel."
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_lp_demand_join_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "join_block": "lp_demand",
                "investor_detail_extract_present": 1,
                "company_or_fund_link_present": 0,
                "point_in_time_join_supported": 0,
                "active_status": "unsupported",
                "note": "The local LP investor-detail extract has dated allocation fields but no PIT-safe link into the staged company deal graph.",
            }
        ]
    )


def build_patent_crosswalk_confidence(
    patent_match_audit: pd.DataFrame,
) -> pd.DataFrame:
    if patent_match_audit.empty:
        return pd.DataFrame(
            columns=["confidence_tier", "match_method", "matched_companies", "used_patent_rows", "used_patents"]
        )
    summary = (
        patent_match_audit.groupby(["confidence_tier", "match_method"], as_index=False)
        .agg(
            matched_companies=("matched_companies", "sum"),
            used_patent_rows=("used_patent_rows", "sum"),
            used_patents=("used_patents", "sum"),
        )
    )
    return summary.sort_values(["confidence_tier", "match_method"]).reset_index(drop=True)


def build_patent_sector_model_comparison(
    dataset: dict,
    config: dict,
    base_feature_columns: list[str],
) -> pd.DataFrame:
    train_panel = dataset["panel"][dataset["panel"]["split"] == "train"].copy()
    test_panel = dataset["panel"][dataset["panel"]["split"] == "test"].copy()
    if train_panel.empty or test_panel.empty:
        return pd.DataFrame()
    main_label = "realized_hard_timely_liquidity_by_horizon"
    base_models, _ = fit_stage1_models_by_universe(train_panel, config, main_label, base_feature_columns)
    patent_columns = stage1_feature_columns(train_panel, include_patent_sector_conditional=True)
    challenger_models, _ = fit_stage1_models_by_universe(train_panel, config, main_label, patent_columns)
    base_scored, _ = score_stage1_holdout_panel(test_panel, base_models, int(config["holdout_horizon_quarters"]), config, dataset["company_master"])
    challenger_scored, _ = score_stage1_holdout_panel(
        test_panel,
        challenger_models,
        int(config["holdout_horizon_quarters"]),
        config,
        dataset["company_master"],
    )
    rows: list[dict[str, object]] = []
    for sector_bucket in SECTOR_BUCKET_ORDER:
        base_subset = base_scored[base_scored["sector_bucket"].astype(str).eq(sector_bucket)].copy()
        challenger_subset = challenger_scored[challenger_scored["sector_bucket"].astype(str).eq(sector_bucket)].copy()
        if len(base_subset) < 250:
            continue
        for variant_name, subset in [("baseline", base_subset), ("patent_sector_conditional", challenger_subset)]:
            metrics = summarize_evaluation_view(
                subset,
                variant_name,
                "pred_hard_timely_liquidity_by_horizon",
                "realized_hard_timely_liquidity_by_horizon",
                int(config["holdout_horizon_quarters"]),
            ).iloc[0]
            rows.append(
                {
                    "sector_bucket": sector_bucket,
                    "model_variant": variant_name,
                    "rows": int(len(subset)),
                    "hard_events": int(pd.to_numeric(subset["realized_hard_timely_liquidity_by_horizon"], errors="coerce").fillna(0).sum()),
                    "brier_score": metrics["brier_score"],
                    "mean_abs_calibration_gap": metrics["mean_abs_calibration_gap"],
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "included_in_main_live_model": int(variant_name == "baseline"),
                }
            )
    return pd.DataFrame(rows)


def build_stage2_cumulative_incidence(
    future_states: pd.DataFrame,
    stage1_point_probs: np.ndarray,
    stage2_model: dict[str, object],
) -> pd.DataFrame:
    route_probs = predict_stage2_route_probs(future_states, stage2_model)
    classes = list(stage2_model.get("classes", []))
    cumulative = {route: 0.0 for route in classes}
    survival = 1.0
    rows = []
    for step in range(stage1_point_probs.shape[1]):
        event_mass = float(stage1_point_probs[0, step]) if stage1_point_probs.ndim == 2 and len(stage1_point_probs) else 0.0
        survival = max(survival - event_mass, 0.0)
        row = {"horizon_q": step + 1, "survival": survival}
        for route in classes:
            route_mass = event_mass * float(route_probs.iloc[step][f"p_cond_{route}"])
            cumulative[route] += route_mass
            row[f"cum_{route}"] = cumulative[route]
        row["prob_hard_timely_liquidity_by_horizon"] = 1.0 - survival
        rows.append(row)
    return pd.DataFrame(rows)


def conservative_route_cash_multiple(route_name: str) -> float:
    if route_name in {"ipo", "mna", "pooled_strategic"}:
        return 0.30
    if route_name == "sponsor_sale":
        return 0.22
    return 0.05


def build_redesigned_decision_panel(
    split_panel: pd.DataFrame,
    stage1_models: dict[str, dict],
    stage2_model: dict[str, object],
    config: dict,
) -> pd.DataFrame:
    decision_candidates = split_panel[split_panel["route_label"].astype(str).eq("no_exit")].copy()
    if decision_candidates.empty:
        return pd.DataFrame()
    decision_quarter = int(decision_candidates["quarter_idx"].max())
    decision_panel = decision_candidates[decision_candidates["quarter_idx"] == decision_quarter].copy().reset_index(drop=True)
    stage1_summary, _ = score_stage1_panel(
        decision_panel,
        stage1_models,
        int(config["holdout_horizon_quarters"]),
        config,
        prediction_label="pred_hard_timely_liquidity_by_horizon",
    )
    decision_panel = decision_panel.merge(stage1_summary, on=["company_id", "quarter_idx"], how="left")
    route_mix = predict_stage2_route_probs(decision_panel, stage2_model)
    decision_panel = decision_panel.merge(
        route_mix.drop(columns=["universe", "sector_bucket", "stage_bucket"]),
        on=["company_id", "quarter_idx"],
        how="left",
    )
    v0 = np.maximum(np.expm1(pd.to_numeric(decision_panel["log_last_round_usd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)), 1.0)
    ownership = float(config["ownership"])
    conservative_expected_multiple = np.zeros(len(decision_panel), dtype=float)
    for route_name in stage2_model.get("classes", []):
        conservative_expected_multiple += (
            pd.to_numeric(decision_panel.get(f"p_cond_{route_name}"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            * conservative_route_cash_multiple(str(route_name))
        )
    decision_panel["predicted_conservative_proceeds_usd"] = (
        pd.to_numeric(decision_panel["pred_hard_timely_liquidity_by_horizon"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        * conservative_expected_multiple
        * v0
        * ownership
    )
    decision_panel["realized_conservative_proceeds_usd"] = np.where(
        pd.to_numeric(decision_panel["realized_hard_timely_liquidity_by_horizon"], errors="coerce").fillna(0).astype(int).eq(1),
        pd.to_numeric(decision_panel["company_exit_value_usd"], errors="coerce").where(
            pd.to_numeric(decision_panel["company_exit_value_usd"], errors="coerce").notna(),
            pd.Series(v0, index=decision_panel.index),
        )
        * ownership
        * np.vectorize(conservative_route_cash_multiple)(
            np.where(
                decision_panel["company_exit_route"].astype(str).isin(["ipo", "mna"])
                & ("pooled_strategic" in stage2_model.get("classes", [])),
                "pooled_strategic",
                decision_panel["company_exit_route"].astype(str),
            )
        ),
        0.0,
    )
    return decision_panel


def backtest_policy_rules(
    decision_panel: pd.DataFrame,
    rules: list[dict[str, object]],
    target_col: str,
    target_label: str,
    value_col: str,
) -> pd.DataFrame:
    rows = []
    if decision_panel.empty:
        return pd.DataFrame()
    actual = pd.to_numeric(decision_panel[target_col], errors="coerce").fillna(0).astype(int)
    for rule in rules:
        accepted = np.asarray(rule["accept_mask"], dtype=bool)
        accepted_frame = decision_panel.loc[accepted].copy()
        counts = binary_confusion_counts(actual, accepted.astype(int))
        metrics = binary_confusion_metrics(counts)
        rows.append(
            {
                "policy_key": str(rule["policy_key"]),
                "decision_rule": str(rule["decision_rule"]),
                "policy_family": str(rule["policy_family"]),
                "target_label": target_label,
                "accepted_observations": int(np.sum(accepted)),
                "acceptance_rate": safe_ratio(int(np.sum(accepted)), int(len(decision_panel))),
                "hit_rate_accepted": float(pd.to_numeric(accepted_frame[target_col], errors="coerce").fillna(0).mean()) if len(accepted_frame) else np.nan,
                "avg_predicted_prob_hard_timely_liquidity": float(
                    pd.to_numeric(accepted_frame["pred_hard_timely_liquidity_by_horizon"], errors="coerce").fillna(0.0).mean()
                )
                if len(accepted_frame)
                else np.nan,
                f"avg_{value_col}": float(pd.to_numeric(accepted_frame[value_col], errors="coerce").fillna(0.0).mean()) if len(accepted_frame) else np.nan,
                **counts,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def choose_active_policy(
    validation_backtest: pd.DataFrame,
    min_acceptance: float,
) -> str:
    if validation_backtest.empty:
        return ""
    eligible = validation_backtest.loc[
        validation_backtest["acceptance_rate"].astype(float).ge(min_acceptance)
        & validation_backtest["accepted_observations"].astype(int).gt(0)
    ].copy()
    if eligible.empty:
        return str(validation_backtest.sort_values(["accepted_observations", "policy_key"], ascending=[False, True]).iloc[0]["policy_key"])
    return str(
        eligible.sort_values(
            ["hit_rate_accepted", "acceptance_rate", "policy_key"],
            ascending=[False, False, True],
        ).iloc[0]["policy_key"]
    )


def build_redesigned_policy_backtests(
    validation_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    stage1_models: dict[str, dict],
    stage2_model: dict[str, object],
    config: dict,
    stage1_recalibrators: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_decision = build_redesigned_decision_panel(validation_panel, stage1_models, stage2_model, config)
    test_decision = build_redesigned_decision_panel(test_panel, stage1_models, stage2_model, config)
    if stage1_recalibrators is not None and not stage1_recalibrators.empty:
        validation_decision = apply_probability_recalibrators(
            validation_decision,
            stage1_recalibrators,
            "pred_hard_timely_liquidity_by_horizon",
        )
        test_decision = apply_probability_recalibrators(
            test_decision,
            stage1_recalibrators,
            "pred_hard_timely_liquidity_by_horizon",
        )
        for frame in [validation_decision, test_decision]:
            if not frame.empty and "predicted_conservative_proceeds_usd" in frame.columns:
                v0 = np.maximum(
                    np.expm1(pd.to_numeric(frame["log_last_round_usd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)),
                    1.0,
                )
                ownership = float(config["ownership"])
                conservative_expected_multiple = np.zeros(len(frame), dtype=float)
                for route_name in stage2_model.get("classes", []):
                    conservative_expected_multiple += (
                        pd.to_numeric(frame.get(f"p_cond_{route_name}"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
                        * conservative_route_cash_multiple(str(route_name))
                    )
                frame["predicted_conservative_proceeds_usd"] = (
                    pd.to_numeric(frame["pred_hard_timely_liquidity_by_horizon"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    * conservative_expected_multiple
                    * v0
                    * ownership
                )
    thresholds = resolve_probability_thresholds(config)
    screening_rules = [
        {
            "policy_key": f"screen_prob_ge_{format_threshold_label(threshold)}",
            "decision_rule": f"hard_timely_liquidity_prob >= {format_threshold_label(threshold)}",
            "policy_family": "screening_threshold",
            "accept_mask": pd.to_numeric(validation_decision.get("pred_hard_timely_liquidity_by_horizon"), errors="coerce").fillna(0.0).to_numpy(dtype=float) >= float(threshold),
        }
        for threshold in thresholds
    ]
    validation_screening = backtest_policy_rules(
        validation_decision,
        screening_rules,
        "realized_hard_timely_liquidity_by_horizon",
        HARD_TIMELY_LIQUIDITY_TARGET,
        "predicted_conservative_proceeds_usd",
    )
    screening_key = choose_active_policy(validation_screening, float(config.get("promotion_gate_min_policy_acceptance", 0.005)))
    screening_rules_test = [
        {
            "policy_key": rule["policy_key"],
            "decision_rule": rule["decision_rule"],
            "policy_family": rule["policy_family"],
            "accept_mask": pd.to_numeric(test_decision.get("pred_hard_timely_liquidity_by_horizon"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            >= float(rule["policy_key"].split("_")[-1]),
        }
        for rule in screening_rules
    ]
    screening_backtest = backtest_policy_rules(
        test_decision,
        screening_rules_test,
        "realized_hard_timely_liquidity_by_horizon",
        HARD_TIMELY_LIQUIDITY_TARGET,
        "predicted_conservative_proceeds_usd",
    )
    screening_backtest["selected_on_validation"] = screening_backtest["policy_key"].astype(str).eq(screening_key).astype(int)

    validation_threshold = float(
        pd.to_numeric(validation_decision.get("predicted_conservative_proceeds_usd"), errors="coerce").fillna(0.0).quantile(
            float(config.get("policy_validation_quantile", 0.90))
        )
    ) if not validation_decision.empty else 0.0
    economic_rule_validation = [
        {
            "policy_key": f"economic_proceeds_ge_{validation_threshold:.0f}",
            "decision_rule": f"predicted_conservative_proceeds_usd >= {validation_threshold:.0f}",
            "policy_family": "economic_screen",
            "accept_mask": pd.to_numeric(validation_decision.get("predicted_conservative_proceeds_usd"), errors="coerce").fillna(0.0).to_numpy(dtype=float) >= validation_threshold,
        }
    ]
    economic_key = choose_active_policy(
        backtest_policy_rules(
            validation_decision,
            economic_rule_validation,
            "realized_hard_timely_liquidity_by_horizon",
            HARD_TIMELY_LIQUIDITY_TARGET,
            "predicted_conservative_proceeds_usd",
        ),
        float(config.get("promotion_gate_min_policy_acceptance", 0.005)),
    )
    economic_backtest = backtest_policy_rules(
        test_decision,
        [
            {
                "policy_key": economic_key or f"economic_proceeds_ge_{validation_threshold:.0f}",
                "decision_rule": f"predicted_conservative_proceeds_usd >= {validation_threshold:.0f}",
                "policy_family": "economic_screen",
                "accept_mask": pd.to_numeric(test_decision.get("predicted_conservative_proceeds_usd"), errors="coerce").fillna(0.0).to_numpy(dtype=float) >= validation_threshold,
            }
        ],
        "realized_hard_timely_liquidity_by_horizon",
        HARD_TIMELY_LIQUIDITY_TARGET,
        "predicted_conservative_proceeds_usd",
    )
    economic_backtest["selected_on_validation"] = 1

    policy_activation = pd.concat(
        [
            screening_backtest[["policy_key", "policy_family", "accepted_observations", "acceptance_rate", "hit_rate_accepted", "selected_on_validation"]],
            economic_backtest[["policy_key", "policy_family", "accepted_observations", "acceptance_rate", "hit_rate_accepted", "selected_on_validation"]],
        ],
        ignore_index=True,
    )
    return screening_backtest, economic_backtest, policy_activation


def target_exploration_feature_columns(panel: pd.DataFrame, include_sponsor_fund: bool = False) -> list[str]:
    columns = [
        "age_q",
        "time_since_last_round_q",
        "log_last_round_usd",
        *[column for column in sector_dummy_columns() if column in panel.columns],
        *[column for column in stage_dummy_columns() if column in panel.columns],
    ]
    if include_sponsor_fund:
        columns.extend([column for column in BUYOUT_SPONSOR_FUND_FEATURES if column in panel.columns])
    return list(dict.fromkeys([column for column in columns if column in panel.columns]))


def target_positive_subset_mask(
    scored_panel: pd.DataFrame,
    target_col: str,
    subset_rule: str,
) -> pd.Series:
    if scored_panel.empty:
        return pd.Series(dtype=bool)
    target_positive = pd.to_numeric(scored_panel.get(target_col), errors="coerce").fillna(0).astype(int).eq(1)
    positive_confidence = scored_panel.get(
        "target_positive_confidence_tier",
        pd.Series(index=scored_panel.index, dtype=object),
    ).astype(str).str.lower()
    positive_kind = scored_panel.get(
        "target_positive_observation_kind",
        pd.Series(index=scored_panel.index, dtype=object),
    ).astype(str).str.lower()
    entity_match = scored_panel.get(
        "entity_match_confidence",
        pd.Series(index=scored_panel.index, dtype=object),
    ).astype(str).str.lower()
    if subset_rule == "exit_label_confidence_high":
        return (~target_positive) | positive_confidence.eq("high")
    if subset_rule == "exit_label_confidence_high_or_medium":
        return (~target_positive) | positive_confidence.isin(["high", "medium"])
    if subset_rule == "entity_match_confidence_high":
        return entity_match.eq("high")
    if subset_rule == "confidence_overlap":
        return entity_match.eq("high") & ((~target_positive) | positive_confidence.eq("high"))
    if subset_rule == "direct_dated_only":
        return (~target_positive) | positive_kind.eq("direct_dated_event")
    if subset_rule == "direct_plus_high_conf_inferred":
        return (~target_positive) | positive_kind.eq("direct_dated_event") | (
            positive_kind.eq("inferred_transition") & positive_confidence.eq("high")
        )
    return pd.Series(np.ones(len(scored_panel), dtype=bool), index=scored_panel.index)


def build_target_model_map(fitted_model: dict, universe: str) -> dict[str, dict]:
    return {
        str(universe): fitted_model,
        "_overall": fitted_model,
    }


def score_target_holdout_panel(
    holdout_panel: pd.DataFrame,
    fitted_model: dict,
    spec: pd.Series | dict,
    target_col: str,
    realized_prefix: str,
    config: dict,
    company_master: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    prediction_col = f"pred_{str(spec['target_key'])}_by_horizon"
    scored_summary, point_matrix = score_stage1_panel(
        holdout_panel,
        build_target_model_map(fitted_model, str(spec["universe"])),
        int(spec["horizon_quarters"]),
        config,
        prediction_label=prediction_col,
    )
    realized_cols = [f"{realized_prefix}{horizon_step}" for horizon_step in range(1, int(spec["horizon_quarters"]) + 1)]
    keep_columns = [
        column
        for column in [
            "company_id",
            "quarter_idx",
            "route_label",
            "company_name",
            target_col,
            "target_positive_route",
            "target_positive_confidence_tier",
            "target_positive_route_source",
            "target_positive_observation_kind",
            "target_positive_directness_class",
            "target_event_quarter_idx",
            "company_exit_route",
            "company_exit_value_usd",
            "company_exit_confidence_tier",
            "company_exit_route_source",
            "exit_quarter_idx",
            "log_last_round_usd",
            "sector_bucket",
            "stage_bucket",
            "universe",
        ]
        if column in holdout_panel.columns
    ] + [column for column in realized_cols if column in holdout_panel.columns]
    scored_panel = scored_summary.merge(holdout_panel[keep_columns], on=["company_id", "quarter_idx"], how="left")
    scored_panel = scored_panel.merge(
        company_master[["company_id", "match_confidence", "company_source", "universe"]].rename(
            columns={"match_confidence": "entity_match_confidence"}
        ),
        on=["company_id", "universe"],
        how="left",
    )
    return scored_panel, point_matrix, prediction_col


def target_high_confidence_mask(scored_panel: pd.DataFrame, target_col: str) -> pd.Series:
    if scored_panel.empty:
        return pd.Series(dtype=bool)
    high_conf_mask = target_positive_subset_mask(scored_panel, target_col, "exit_label_confidence_high")
    if int(high_conf_mask.sum()) == 0:
        high_conf_mask = target_positive_subset_mask(scored_panel, target_col, "exit_label_confidence_high_or_medium")
    return high_conf_mask


def resolve_target_stage2_classes(
    train_candidate_panel: pd.DataFrame,
    spec: pd.Series | dict,
) -> list[str]:
    requested = target_stage2_classes(spec)
    if not requested:
        return []
    events = build_target_event_frame(train_candidate_panel, spec)
    counts = events["route_label"].astype(str).value_counts().to_dict() if not events.empty else {}
    resolved: list[str] = []
    for route_name in requested:
        if route_name == "pooled_strategic":
            support = int(counts.get("ipo", 0) + counts.get("mna", 0))
        else:
            support = int(counts.get(route_name, 0))
        if support > 0 or len(requested) == 1:
            resolved.append(route_name)
    if resolved:
        return resolved
    return requested[:1]


def build_target_decision_panel(
    split_panel: pd.DataFrame,
    fitted_model: dict,
    stage2_model: dict[str, object],
    spec: pd.Series | dict,
    target_col: str,
    config: dict,
    stage1_recalibrators: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    decision_candidates = split_panel[split_panel["route_label"].astype(str).eq("no_exit")].copy()
    if decision_candidates.empty:
        return pd.DataFrame(), ""
    decision_quarter = int(pd.to_numeric(decision_candidates["quarter_idx"], errors="coerce").max())
    decision_panel = decision_candidates[
        pd.to_numeric(decision_candidates["quarter_idx"], errors="coerce").eq(decision_quarter)
    ].copy().reset_index(drop=True)
    if decision_panel.empty:
        return pd.DataFrame(), ""
    prediction_col = f"pred_{str(spec['target_key'])}_by_horizon"
    scored_summary, _ = score_stage1_panel(
        decision_panel,
        build_target_model_map(fitted_model, str(spec["universe"])),
        int(spec["horizon_quarters"]),
        config,
        prediction_label=prediction_col,
    )
    decision_panel = decision_panel.merge(scored_summary, on=["company_id", "quarter_idx"], how="left")
    if stage1_recalibrators is not None and not stage1_recalibrators.empty:
        decision_panel = apply_probability_recalibrators(decision_panel, stage1_recalibrators, prediction_col)
    route_mix = predict_stage2_route_probs(decision_panel, stage2_model) if stage2_model.get("classes") else pd.DataFrame()
    if not route_mix.empty:
        decision_panel = decision_panel.merge(
            route_mix.drop(columns=["universe", "sector_bucket", "stage_bucket"]),
            on=["company_id", "quarter_idx"],
            how="left",
        )
    v0 = np.maximum(
        np.expm1(pd.to_numeric(decision_panel["log_last_round_usd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)),
        1.0,
    )
    ownership = float(config["ownership"])
    purchase_price = float(config["purchase_price_fraction_of_v0"]) * ownership * v0
    conservative_expected_multiple = np.zeros(len(decision_panel), dtype=float)
    for route_name in stage2_model.get("classes", []):
        conservative_expected_multiple += (
            pd.to_numeric(decision_panel.get(f"p_cond_{route_name}"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            * conservative_route_cash_multiple(str(route_name))
        )
    if not stage2_model.get("classes"):
        conservative_expected_multiple = np.repeat(1.0, len(decision_panel))
    predicted_prob = pd.to_numeric(decision_panel[prediction_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    decision_panel["predicted_conservative_proceeds_usd"] = predicted_prob * conservative_expected_multiple * v0 * ownership
    decision_panel["predicted_npv_proxy"] = decision_panel["predicted_conservative_proceeds_usd"] - purchase_price
    exit_values = []
    exit_prob_columns = []
    if stage2_model.get("classes"):
        for route_name in stage2_model.get("classes", []):
            route_prob = (
                predicted_prob
                * pd.to_numeric(decision_panel.get(f"p_cond_{route_name}"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            )
            route_value = ownership * conservative_route_cash_multiple(str(route_name)) * v0 - purchase_price
            exit_prob_columns.append(route_prob)
            exit_values.append(route_value)
    else:
        exit_prob_columns.append(predicted_prob)
        exit_values.append(ownership * conservative_route_cash_multiple("pooled_strategic") * v0 - purchase_price)
    no_exit_prob = np.clip(1.0 - predicted_prob, 0.0, 1.0)
    no_exit_value = -purchase_price
    mean_npv = pd.to_numeric(decision_panel["predicted_npv_proxy"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    variance = no_exit_prob * np.square(no_exit_value - mean_npv)
    for route_prob, route_value in zip(exit_prob_columns, exit_values, strict=True):
        variance += route_prob * np.square(route_value - mean_npv)
    decision_panel["predicted_certainty_equivalent"] = mean_npv - 0.5 * float(config.get("gamma_risk_aversion", 2.0)) * variance
    decision_panel["predicted_exit_by_horizon"] = predicted_prob
    realized_positive = pd.to_numeric(decision_panel[target_col], errors="coerce").fillna(0).astype(int).eq(1).to_numpy(dtype=bool)
    observed_value = pd.to_numeric(decision_panel.get("company_exit_value_usd"), errors="coerce").fillna(np.nan).to_numpy(dtype=float)
    realized_route = decision_panel["company_exit_route"].map(
        lambda value: map_stage2_route_label(value, list(stage2_model.get("classes", []))) or str(value)
    )
    realized_proceeds = np.zeros(len(decision_panel), dtype=float)
    for route_name in pd.Series(realized_route).dropna().astype(str).unique().tolist():
        route_mask = realized_positive & pd.Series(realized_route).astype(str).eq(route_name).to_numpy(dtype=bool)
        if not route_mask.any():
            continue
        route_multiple = np.where(
            np.isfinite(observed_value[route_mask]) & (observed_value[route_mask] > 0),
            observed_value[route_mask] / v0[route_mask],
            conservative_route_cash_multiple(route_name),
        )
        realized_proceeds[route_mask] = ownership * conservative_route_cash_multiple(route_name) * np.clip(route_multiple, 0.01, 25.0) * v0[route_mask]
    decision_panel["realized_conservative_proceeds_usd"] = realized_proceeds
    decision_panel["realized_npv_proxy"] = decision_panel["realized_conservative_proceeds_usd"] - purchase_price
    decision_panel["realized_positive_value"] = (decision_panel["realized_npv_proxy"] > 0).astype(int)
    return decision_panel, prediction_col


def backtest_target_policy_rules(
    decision_panel: pd.DataFrame,
    rules: list[dict[str, object]],
    target_col: str,
    target_label: str,
    prediction_col: str,
) -> pd.DataFrame:
    if decision_panel.empty:
        return pd.DataFrame()
    actual = pd.to_numeric(decision_panel[target_col], errors="coerce").fillna(0).astype(int)
    rows = []
    for rule in rules:
        accepted = np.asarray(rule["accept_mask"], dtype=bool)
        accepted_frame = decision_panel.loc[accepted].copy()
        counts = binary_confusion_counts(actual, accepted.astype(int))
        metrics = binary_confusion_metrics(counts)
        rule_metadata = {
            key: value
            for key, value in rule.items()
            if key not in {"accept_mask", "policy_key", "decision_rule", "policy_family"}
        }
        rows.append(
            {
                "policy_key": str(rule["policy_key"]),
                "decision_rule": str(rule["decision_rule"]),
                "policy_family": str(rule["policy_family"]),
                "target_label": target_label,
                "accepted_observations": int(np.sum(accepted)),
                "acceptance_rate": safe_ratio(int(np.sum(accepted)), int(len(decision_panel))),
                "hit_rate_accepted": float(pd.to_numeric(accepted_frame[target_col], errors="coerce").fillna(0.0).mean()) if len(accepted_frame) else np.nan,
                "predicted_mean_prob": float(pd.to_numeric(accepted_frame[prediction_col], errors="coerce").fillna(0.0).mean()) if len(accepted_frame) else np.nan,
                "predicted_mean_npv_proxy": float(pd.to_numeric(accepted_frame["predicted_npv_proxy"], errors="coerce").fillna(0.0).mean()) if len(accepted_frame) else np.nan,
                "predicted_mean_certainty_equivalent": float(pd.to_numeric(accepted_frame["predicted_certainty_equivalent"], errors="coerce").fillna(0.0).mean()) if len(accepted_frame) and "predicted_certainty_equivalent" in accepted_frame.columns else np.nan,
                "predicted_mean_exit_by_horizon": float(pd.to_numeric(accepted_frame["predicted_exit_by_horizon"], errors="coerce").fillna(0.0).mean()) if len(accepted_frame) and "predicted_exit_by_horizon" in accepted_frame.columns else np.nan,
                "realized_mean_npv_proxy": float(pd.to_numeric(accepted_frame["realized_npv_proxy"], errors="coerce").fillna(0.0).mean()) if len(accepted_frame) else np.nan,
                "degenerate_rule": int(np.sum(accepted) == 0),
                "lift_over_prevalence": safe_ratio(metrics.get("precision", 0.0), metrics.get("prevalence", 0.0)) if float(metrics.get("prevalence", 0.0) or 0.0) > 0.0 else np.nan,
                **rule_metadata,
                **counts,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_buyout_policy_rule_specs(config: dict) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    exit_probability_thresholds = [0.05, 0.10, 0.15, 0.20]
    for threshold in BUYOUT_POLICY_PROBABILITY_THRESHOLDS:
        threshold_label = format_threshold_label(threshold)
        specs.append(
            {
                "policy_key": f"screen_prob_ge_{threshold_label}",
                "decision_rule": f"target_probability >= {threshold_label}",
                "policy_family": "probability_screen",
                "policy_form": "fixed_threshold",
                "prob_threshold": float(threshold),
                "top_quantile": np.nan,
                "npv_gate": 0,
                "certainty_equivalent_gate": 0,
                "exit_probability_threshold": np.nan,
            }
        )
        specs.append(
            {
                "policy_key": f"dual_prob_ge_{threshold_label}_npv_gt_0",
                "decision_rule": f"target_probability >= {threshold_label} and predicted_npv_proxy > 0",
                "policy_family": "dual_npv_screen",
                "policy_form": "threshold_and_positive_npv",
                "prob_threshold": float(threshold),
                "top_quantile": np.nan,
                "npv_gate": 1,
                "certainty_equivalent_gate": 0,
                "exit_probability_threshold": np.nan,
            }
        )
        specs.append(
            {
                "policy_key": f"dual_prob_ge_{threshold_label}_ce_gt_0",
                "decision_rule": f"target_probability >= {threshold_label} and certainty_equivalent > 0",
                "policy_family": "dual_ce_screen",
                "policy_form": "threshold_and_positive_ce",
                "prob_threshold": float(threshold),
                "top_quantile": np.nan,
                "npv_gate": 0,
                "certainty_equivalent_gate": 1,
                "exit_probability_threshold": np.nan,
            }
        )
        for exit_threshold in exit_probability_thresholds:
            exit_label = format_threshold_label(exit_threshold)
            specs.append(
                {
                    "policy_key": f"dual_prob_ge_{threshold_label}_exit_prob_ge_{exit_label}",
                    "decision_rule": f"target_probability >= {threshold_label} and predicted_exit_by_horizon >= {exit_label}",
                    "policy_family": "dual_exit_probability_screen",
                    "policy_form": "threshold_and_exit_prob",
                    "prob_threshold": float(threshold),
                    "top_quantile": np.nan,
                    "npv_gate": 0,
                    "certainty_equivalent_gate": 0,
                    "exit_probability_threshold": float(exit_threshold),
                }
            )
    for quantile in BUYOUT_POLICY_TOP_QUANTILES:
        quantile_pct = int(round(100 * float(quantile)))
        specs.append(
            {
                "policy_key": f"top_{quantile_pct:02d}pct_prob",
                "decision_rule": f"top {quantile_pct}% by target_probability",
                "policy_family": "top_quantile_screen",
                "policy_form": "top_quantile",
                "prob_threshold": np.nan,
                "top_quantile": float(quantile),
                "npv_gate": 0,
                "certainty_equivalent_gate": 0,
                "exit_probability_threshold": np.nan,
            }
        )
    return specs


def materialize_buyout_policy_rules(
    decision_panel: pd.DataFrame,
    prediction_col: str,
    rule_specs: list[dict[str, object]],
) -> list[dict[str, object]]:
    if decision_panel.empty:
        return []
    predicted_prob = pd.to_numeric(decision_panel.get(prediction_col), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    predicted_npv = pd.to_numeric(decision_panel.get("predicted_npv_proxy"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    predicted_ce = pd.to_numeric(decision_panel.get("predicted_certainty_equivalent"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    predicted_exit = pd.to_numeric(decision_panel.get("predicted_exit_by_horizon"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rank_frame = pd.DataFrame(
        {
            "_predicted_prob": predicted_prob,
            "_company_id": decision_panel.get("company_id", pd.Series(index=decision_panel.index, dtype=object)).astype(str),
            "_quarter_idx": pd.to_numeric(decision_panel.get("quarter_idx"), errors="coerce").fillna(-1).astype(int),
        },
        index=decision_panel.index,
    )
    ordered_index = rank_frame.sort_values(
        ["_predicted_prob", "_company_id", "_quarter_idx"],
        ascending=[False, True, True],
        kind="mergesort",
    ).index.tolist()
    rules: list[dict[str, object]] = []
    for spec in rule_specs:
        accept_mask = np.ones(len(decision_panel), dtype=bool)
        threshold = pd.to_numeric(pd.Series([spec.get("prob_threshold")]), errors="coerce").iloc[0]
        if pd.notna(threshold):
            accept_mask &= predicted_prob >= float(threshold)
        top_quantile = pd.to_numeric(pd.Series([spec.get("top_quantile")]), errors="coerce").iloc[0]
        if pd.notna(top_quantile):
            accepted_count = int(math.ceil(float(top_quantile) * len(decision_panel)))
            accepted_count = max(1, min(accepted_count, len(decision_panel))) if len(decision_panel) else 0
            selected_index = set(ordered_index[:accepted_count])
            accept_mask &= np.fromiter((idx in selected_index for idx in decision_panel.index), dtype=bool, count=len(decision_panel))
        if int(spec.get("npv_gate", 0)) == 1:
            accept_mask &= predicted_npv > 0.0
        if int(spec.get("certainty_equivalent_gate", 0)) == 1:
            accept_mask &= predicted_ce > 0.0
        exit_threshold = pd.to_numeric(pd.Series([spec.get("exit_probability_threshold")]), errors="coerce").iloc[0]
        if pd.notna(exit_threshold):
            accept_mask &= predicted_exit >= float(exit_threshold)
        rules.append({**spec, "accept_mask": accept_mask})
    return rules


def score_buyout_policy_search(
    validation_backtest: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if validation_backtest.empty:
        return validation_backtest.copy(), validation_backtest.copy(), {
            "selected_policy_key": "",
            "acceptance_min_used": float(config.get("buyout_policy_acceptance_min", 0.005)),
            "acceptance_max_used": float(config.get("buyout_policy_acceptance_max", 0.50)),
            "fallback_band_used": 0,
            "selection_status": "empty_validation_policy_surface",
        }
    minimum = float(config.get("buyout_policy_acceptance_min", 0.005))
    default_maximum = float(config.get("buyout_policy_acceptance_max", 0.50))
    fallback_maximum = float(config.get("buyout_policy_acceptance_max_fallback", 0.60))
    scored = validation_backtest.copy()
    scored["precision"] = pd.to_numeric(scored["precision"], errors="coerce").fillna(0.0)
    scored["recall"] = pd.to_numeric(scored["recall"], errors="coerce").fillna(0.0)
    scored["balanced_accuracy"] = pd.to_numeric(scored["balanced_accuracy"], errors="coerce").fillna(0.0)
    scored["acceptance_rate"] = pd.to_numeric(scored["acceptance_rate"], errors="coerce").fillna(0.0)
    scored["prevalence"] = pd.to_numeric(scored["prevalence"], errors="coerce").fillna(0.0)
    scored["degenerate_rule"] = pd.to_numeric(scored["degenerate_rule"], errors="coerce").fillna(1).astype(int)
    scored["lift_over_prevalence"] = np.where(
        scored["prevalence"].gt(0.0),
        scored["precision"] / scored["prevalence"],
        np.nan,
    )
    scored["feasible_default_band"] = (
        scored["degenerate_rule"].eq(0)
        & scored["acceptance_rate"].ge(minimum)
        & scored["acceptance_rate"].le(default_maximum)
    ).astype(int)
    scored["feasible_fallback_band"] = (
        scored["degenerate_rule"].eq(0)
        & scored["acceptance_rate"].ge(minimum)
        & scored["acceptance_rate"].le(fallback_maximum)
    ).astype(int)
    fallback_band_used = int(scored["feasible_default_band"].sum() == 0 and scored["feasible_fallback_band"].sum() > 0)
    acceptance_maximum = fallback_maximum if fallback_band_used else default_maximum
    scored["selection_acceptance_min"] = minimum
    scored["selection_acceptance_max"] = acceptance_maximum
    scored["selection_fallback_band_used"] = fallback_band_used
    scored["feasible_active_band"] = (
        scored["degenerate_rule"].eq(0)
        & scored["acceptance_rate"].ge(minimum)
        & scored["acceptance_rate"].le(acceptance_maximum)
    ).astype(int)
    midpoint = 0.5 * (minimum + acceptance_maximum)
    scored["acceptance_distance_to_midband"] = np.abs(scored["acceptance_rate"] - midpoint)
    ordered = scored.sort_values(
        [
            "degenerate_rule",
            "precision",
            "lift_over_prevalence",
            "balanced_accuracy",
            "acceptance_distance_to_midband",
            "acceptance_rate",
            "policy_key",
        ],
        ascending=[True, False, False, False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["validation_policy_rank"] = np.arange(1, len(ordered) + 1)
    feasible = ordered.loc[ordered["feasible_active_band"].eq(1)].copy().reset_index(drop=True)
    feasible["validation_feasible_rank"] = np.arange(1, len(feasible) + 1)
    selection_pool = feasible.copy()
    selection_status = "selected_within_default_band"
    if selection_pool.empty:
        selection_status = "selected_within_fallback_band" if fallback_band_used else "no_feasible_policy_after_fallback"
        selection_pool = ordered.loc[
            ordered["degenerate_rule"].eq(0) & ordered["accepted_observations"].astype(int).gt(0)
        ].copy()
        if selection_pool.empty:
            selection_pool = ordered.head(1).copy()
            selection_status = "all_rules_degenerate"
    selected_policy_key = str(selection_pool.iloc[0]["policy_key"]) if not selection_pool.empty else ""
    ordered["selected_on_validation"] = ordered["policy_key"].astype(str).eq(selected_policy_key).astype(int)
    feasible["selected_on_validation"] = feasible["policy_key"].astype(str).eq(selected_policy_key).astype(int)
    ordered["selection_status"] = selection_status
    feasible["selection_status"] = selection_status
    return ordered, feasible, {
        "selected_policy_key": selected_policy_key,
        "acceptance_min_used": minimum,
        "acceptance_max_used": acceptance_maximum,
        "fallback_band_used": fallback_band_used,
        "selection_status": selection_status,
    }


def build_target_policy_backtests(
    validation_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    fitted_model: dict,
    stage2_model: dict[str, object],
    spec: pd.Series | dict,
    target_col: str,
    config: dict,
    stage1_recalibrators: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str, dict[str, pd.DataFrame]]:
    validation_decision, prediction_col = build_target_decision_panel(
        validation_panel,
        fitted_model,
        stage2_model,
        spec,
        target_col,
        config,
        stage1_recalibrators=stage1_recalibrators,
    )
    test_decision, _ = build_target_decision_panel(
        test_panel,
        fitted_model,
        stage2_model,
        spec,
        target_col,
        config,
        stage1_recalibrators=stage1_recalibrators,
    )
    is_buyout_target = str(spec.get("universe", "")).strip() == "buyout_pe"
    threshold = float(config.get("target_exploration_probability_threshold", 0.03))
    if is_buyout_target:
        rule_specs = build_buyout_policy_rule_specs(config)
        validation_rules = materialize_buyout_policy_rules(validation_decision, prediction_col, rule_specs)
        test_rules = materialize_buyout_policy_rules(test_decision, prediction_col, rule_specs)
    else:
        validation_rules = [
            {
                "policy_key": f"screen_prob_ge_{format_threshold_label(threshold)}",
                "decision_rule": f"target_probability >= {format_threshold_label(threshold)}",
                "policy_family": "probability_screen",
                "policy_form": "fixed_threshold",
                "prob_threshold": float(threshold),
                "top_quantile": np.nan,
                "npv_gate": 0,
                "certainty_equivalent_gate": 0,
                "exit_probability_threshold": np.nan,
                "accept_mask": pd.to_numeric(validation_decision.get(prediction_col), errors="coerce").fillna(0.0).to_numpy(dtype=float) >= threshold,
            },
            {
                "policy_key": "economic_npv_gt_0",
                "decision_rule": "predicted_npv_proxy > 0",
                "policy_family": "economic_screen",
                "policy_form": "positive_npv",
                "prob_threshold": np.nan,
                "top_quantile": np.nan,
                "npv_gate": 1,
                "certainty_equivalent_gate": 0,
                "exit_probability_threshold": np.nan,
                "accept_mask": pd.to_numeric(validation_decision.get("predicted_npv_proxy"), errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0,
            },
            {
                "policy_key": f"dual_prob_ge_{format_threshold_label(threshold)}_npv_gt_0",
                "decision_rule": f"target_probability >= {format_threshold_label(threshold)} and predicted_npv_proxy > 0",
                "policy_family": "dual_screen",
                "policy_form": "threshold_and_positive_npv",
                "prob_threshold": float(threshold),
                "top_quantile": np.nan,
                "npv_gate": 1,
                "certainty_equivalent_gate": 0,
                "exit_probability_threshold": np.nan,
                "accept_mask": (
                    pd.to_numeric(validation_decision.get(prediction_col), errors="coerce").fillna(0.0).to_numpy(dtype=float) >= threshold
                )
                & (
                    pd.to_numeric(validation_decision.get("predicted_npv_proxy"), errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0
                ),
            },
        ]
        test_rules = [
            {
                **rule,
                "accept_mask": (
                    pd.to_numeric(test_decision.get(prediction_col), errors="coerce").fillna(0.0).to_numpy(dtype=float) >= threshold
                    if rule["policy_family"] == "probability_screen"
                    else (
                        pd.to_numeric(test_decision.get("predicted_npv_proxy"), errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0
                        if rule["policy_family"] == "economic_screen"
                        else (
                            pd.to_numeric(test_decision.get(prediction_col), errors="coerce").fillna(0.0).to_numpy(dtype=float) >= threshold
                        )
                        & (
                            pd.to_numeric(test_decision.get("predicted_npv_proxy"), errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0
                        )
                    )
                ),
            }
            for rule in validation_rules
        ]
    validation_backtest = backtest_target_policy_rules(
        validation_decision,
        validation_rules,
        target_col,
        str(spec["target_name"]),
        prediction_col,
    )
    test_backtest = backtest_target_policy_rules(
        test_decision,
        test_rules,
        target_col,
        str(spec["target_name"]),
        prediction_col,
    )
    if is_buyout_target:
        validation_leaderboard, validation_feasible, selection_meta = score_buyout_policy_search(validation_backtest, config)
        selected_policy_key = str(selection_meta.get("selected_policy_key", ""))
        validation_output = validation_leaderboard.copy()
        confirmation_test = test_backtest.loc[test_backtest["policy_key"].astype(str).eq(selected_policy_key)].copy()
        if confirmation_test.empty and not test_backtest.empty:
            confirmation_test = test_backtest.head(1).copy()
        validation_output["evaluation_split"] = "validation"
        validation_output["prediction_col"] = prediction_col
        validation_output["screening_threshold"] = threshold
        validation_output["selection_status"] = str(selection_meta.get("selection_status", ""))
        validation_output["selection_fallback_band_used"] = int(selection_meta.get("fallback_band_used", 0))
        validation_output["selection_acceptance_min"] = float(selection_meta.get("acceptance_min_used", np.nan))
        validation_output["selection_acceptance_max"] = float(selection_meta.get("acceptance_max_used", np.nan))

        validation_feasible_output = validation_feasible.copy()
        if not validation_feasible_output.empty:
            validation_feasible_output["evaluation_split"] = "validation"
            validation_feasible_output["prediction_col"] = prediction_col
            validation_feasible_output["screening_threshold"] = threshold
            validation_feasible_output["selection_status"] = str(selection_meta.get("selection_status", ""))
            validation_feasible_output["selection_fallback_band_used"] = int(selection_meta.get("fallback_band_used", 0))
            validation_feasible_output["selection_acceptance_min"] = float(selection_meta.get("acceptance_min_used", np.nan))
            validation_feasible_output["selection_acceptance_max"] = float(selection_meta.get("acceptance_max_used", np.nan))

        confirmation_test["evaluation_split"] = "test"
        confirmation_test["selected_on_validation"] = confirmation_test["policy_key"].astype(str).eq(selected_policy_key).astype(int)
        confirmation_test["prediction_col"] = prediction_col
        confirmation_test["screening_threshold"] = threshold
        confirmation_test["selection_status"] = str(selection_meta.get("selection_status", ""))
        confirmation_test["selection_fallback_band_used"] = int(selection_meta.get("fallback_band_used", 0))
        confirmation_test["selection_acceptance_min"] = float(selection_meta.get("acceptance_min_used", np.nan))
        confirmation_test["selection_acceptance_max"] = float(selection_meta.get("acceptance_max_used", np.nan))

        output = pd.concat([validation_output, confirmation_test], ignore_index=True, sort=False)
        return output, selected_policy_key, {
            "validation": validation_output,
            "validation_feasible": validation_feasible_output,
            "confirmation_test": confirmation_test,
        }

    selected_policy_key = choose_active_policy(
        validation_backtest,
        float(config.get("promotion_gate_min_policy_acceptance", 0.005)),
    ) if not validation_backtest.empty else ""
    validation_backtest["evaluation_split"] = "validation"
    test_backtest["evaluation_split"] = "test"
    output = pd.concat([validation_backtest, test_backtest], ignore_index=True)
    output["selected_on_validation"] = output["policy_key"].astype(str).eq(selected_policy_key).astype(int)
    output["prediction_col"] = prediction_col
    output["screening_threshold"] = threshold
    return output, selected_policy_key, {
        "validation": pd.DataFrame(),
        "validation_feasible": pd.DataFrame(),
        "confirmation_test": pd.DataFrame(),
    }


def evaluate_target_candidate(
    candidate_panel: pd.DataFrame,
    spec: pd.Series | dict,
    target_col: str,
    realized_prefix: str,
    company_master: pd.DataFrame,
    config: dict,
    feature_columns_override: list[str] | None = None,
    feature_backbone: str = TARGET_BASE_FEATURE_BACKBONE,
) -> dict[str, pd.DataFrame | str]:
    effective_config = DEFAULT_CONFIG.copy()
    effective_config.update(config)
    definition = build_target_definition_from_spec(spec)
    candidate_events = build_target_event_frame(candidate_panel, spec, target_col=target_col)
    stage2_classes = resolve_target_stage2_classes_from_events(candidate_events, spec)
    prevalence = build_target_prevalence_by_split(candidate_panel, target_col, spec)
    route_support = build_target_route_support_by_split(candidate_events, spec, effective_config, stage2_classes=stage2_classes)
    source_mix = build_target_source_mix(candidate_events, spec)
    label_audit = build_target_label_confidence_audit(candidate_events, spec)
    time_distribution = build_target_time_distribution(candidate_events, spec)
    base_fields = {
        "target_key": str(spec["target_key"]),
        "target_name": str(spec["target_name"]),
        "universe": str(spec["universe"]),
        "candidate_role": str(spec["candidate_role"]),
        "benchmark_row": int(spec["benchmark_row"]),
        "data_supported": int(spec["data_supported"]),
        "feature_backbone": str(feature_backbone),
    }
    calibration_summary = pd.DataFrame()
    evaluation_metrics = pd.DataFrame()
    decision_backtest = pd.DataFrame()
    confusion_summary = pd.DataFrame()
    prediction_col = f"pred_{str(spec['target_key'])}_by_horizon"
    if int(spec["data_supported"]) != 1:
        evaluation_metrics = pd.DataFrame(
            [
                {
                    **base_fields,
                    "evaluation_view": evaluation_view,
                    "rows": 0,
                    "brier_score": np.nan,
                    "integrated_brier_score": np.nan,
                    "pr_auc": np.nan,
                    "roc_auc": np.nan,
                    "calibration_slope": np.nan,
                    "calibration_slope_status": "unsupported",
                    "calibration_intercept": np.nan,
                    "calibration_intercept_status": "unsupported",
                    "top_decile_realized_exit_rate": np.nan,
                    "top_decile_lift": np.nan,
                    "mean_abs_calibration_gap": np.nan,
                    "max_abs_calibration_gap": np.nan,
                    "estimation_status": "unsupported_definition_only",
                    "estimation_note": str(spec["support_note"]),
                    "prediction_col": prediction_col,
                }
                for evaluation_view in ["validation_selection", "full_test", "high_confidence_subset"]
            ]
        )
        return {
            "definition": definition,
            "prevalence": prevalence,
            "route_support": route_support,
            "source_mix": source_mix,
            "label_audit": label_audit,
            "time_distribution": time_distribution,
            "calibration_summary": calibration_summary,
            "evaluation_metrics": evaluation_metrics,
            "decision_backtest": decision_backtest,
            "confusion_summary": confusion_summary,
            "prediction_col": prediction_col,
            "selected_policy_key": "",
            "policy_search_validation": pd.DataFrame(),
            "policy_search_validation_feasible": pd.DataFrame(),
            "policy_confirmation_test": pd.DataFrame(),
            "selected_stage2_classes": "|".join(stage2_classes),
        }
    train_panel = candidate_panel[candidate_panel["split"].astype(str).eq("train")].copy()
    validation_panel = candidate_panel[candidate_panel["split"].astype(str).eq("validation")].copy()
    test_panel = candidate_panel[candidate_panel["split"].astype(str).eq("test")].copy()
    feature_columns = feature_columns_override or target_exploration_feature_columns(candidate_panel)
    train_positive_rows = int(pd.to_numeric(train_panel.get(target_col), errors="coerce").fillna(0).sum())
    if train_panel.empty or validation_panel.empty or test_panel.empty or train_positive_rows == 0:
        evaluation_metrics = pd.DataFrame(
            [
                {
                    **base_fields,
                    "evaluation_view": evaluation_view,
                    "rows": int(len(test_panel)) if evaluation_view != "validation_selection" else int(len(validation_panel)),
                    "brier_score": np.nan,
                    "integrated_brier_score": np.nan,
                    "pr_auc": np.nan,
                    "roc_auc": np.nan,
                    "calibration_slope": np.nan,
                    "calibration_slope_status": "insufficient_support",
                    "calibration_intercept": np.nan,
                    "calibration_intercept_status": "insufficient_support",
                    "top_decile_realized_exit_rate": np.nan,
                    "top_decile_lift": np.nan,
                    "mean_abs_calibration_gap": np.nan,
                    "max_abs_calibration_gap": np.nan,
                    "estimation_status": "insufficient_support",
                    "estimation_note": "Train, validation, or test support is insufficient for a stable target comparison fit.",
                    "prediction_col": prediction_col,
                }
                for evaluation_view in ["validation_selection", "full_test", "high_confidence_subset"]
            ]
        )
        return {
            "definition": definition,
            "prevalence": prevalence,
            "route_support": route_support,
            "source_mix": source_mix,
            "label_audit": label_audit,
            "time_distribution": time_distribution,
            "calibration_summary": calibration_summary,
            "evaluation_metrics": evaluation_metrics,
            "decision_backtest": decision_backtest,
            "confusion_summary": confusion_summary,
            "prediction_col": prediction_col,
            "selected_policy_key": "",
            "policy_search_validation": pd.DataFrame(),
            "policy_search_validation_feasible": pd.DataFrame(),
            "policy_confirmation_test": pd.DataFrame(),
            "selected_stage2_classes": "|".join(stage2_classes),
        }
    fitted_model = fit_binary_hazard(train_panel, target_col, effective_config, feature_columns)
    validation_scored, _, prediction_col = score_target_holdout_panel(
        validation_panel,
        fitted_model,
        spec,
        target_col,
        realized_prefix,
        effective_config,
        company_master,
    )
    recalibrators = fit_probability_recalibrators(validation_scored, prediction_col, target_col)
    validation_scored = apply_probability_recalibrators(validation_scored, recalibrators, prediction_col)
    test_scored, _, prediction_col = score_target_holdout_panel(
        test_panel,
        fitted_model,
        spec,
        target_col,
        realized_prefix,
        effective_config,
        company_master,
    )
    test_scored = apply_probability_recalibrators(test_scored, recalibrators, prediction_col)
    validation_eval = summarize_evaluation_view(
        validation_scored,
        "validation_selection",
        prediction_col,
        target_col,
        int(spec["horizon_quarters"]),
        prediction_prefix="pred_exit_by_h",
        realized_prefix=realized_prefix,
    )
    high_conf_mask = target_high_confidence_mask(test_scored, target_col)
    entity_match_mask = target_positive_subset_mask(test_scored, target_col, "entity_match_confidence_high")
    overlap_mask = target_positive_subset_mask(test_scored, target_col, "confidence_overlap")
    direct_only_mask = target_positive_subset_mask(test_scored, target_col, "direct_dated_only")
    direct_plus_high_conf_mask = target_positive_subset_mask(test_scored, target_col, "direct_plus_high_conf_inferred")
    evaluation_metrics = pd.concat(
        [
            validation_eval,
            summarize_evaluation_view(
                test_scored,
                "full_test",
                prediction_col,
                target_col,
                int(spec["horizon_quarters"]),
                prediction_prefix="pred_exit_by_h",
                realized_prefix=realized_prefix,
            ),
            summarize_evaluation_view(
                test_scored.loc[high_conf_mask].copy(),
                "high_confidence_subset",
                prediction_col,
                target_col,
                int(spec["horizon_quarters"]),
                prediction_prefix="pred_exit_by_h",
                realized_prefix=realized_prefix,
            ),
            summarize_evaluation_view(
                test_scored.loc[high_conf_mask].copy(),
                "high_confidence_exit_label_only",
                prediction_col,
                target_col,
                int(spec["horizon_quarters"]),
                prediction_prefix="pred_exit_by_h",
                realized_prefix=realized_prefix,
            ),
            summarize_evaluation_view(
                test_scored.loc[entity_match_mask].copy(),
                "high_confidence_entity_match_only",
                prediction_col,
                target_col,
                int(spec["horizon_quarters"]),
                prediction_prefix="pred_exit_by_h",
                realized_prefix=realized_prefix,
            ),
            summarize_evaluation_view(
                test_scored.loc[overlap_mask].copy(),
                "high_confidence_overlap",
                prediction_col,
                target_col,
                int(spec["horizon_quarters"]),
                prediction_prefix="pred_exit_by_h",
                realized_prefix=realized_prefix,
            ),
            summarize_evaluation_view(
                test_scored.loc[direct_only_mask].copy(),
                "direct_dated_only",
                prediction_col,
                target_col,
                int(spec["horizon_quarters"]),
                prediction_prefix="pred_exit_by_h",
                realized_prefix=realized_prefix,
            ),
            summarize_evaluation_view(
                test_scored.loc[direct_plus_high_conf_mask].copy(),
                "direct_dated_plus_high_conf_inferred",
                prediction_col,
                target_col,
                int(spec["horizon_quarters"]),
                prediction_prefix="pred_exit_by_h",
                realized_prefix=realized_prefix,
            ),
        ],
        ignore_index=True,
    )
    evaluation_metrics["estimation_status"] = "estimated"
    evaluation_metrics["estimation_note"] = ""
    for key, value in base_fields.items():
        evaluation_metrics[key] = value
    validation_calibration = calibration_by_decile(validation_scored, prediction_col=prediction_col, realized_col=target_col)
    full_calibration = calibration_by_decile(test_scored, prediction_col=prediction_col, realized_col=target_col)
    high_conf_calibration = calibration_by_decile(
        test_scored.loc[high_conf_mask].copy(),
        prediction_col=prediction_col,
        realized_col=target_col,
    )
    entity_match_calibration = calibration_by_decile(
        test_scored.loc[entity_match_mask].copy(),
        prediction_col=prediction_col,
        realized_col=target_col,
    )
    direct_only_calibration = calibration_by_decile(
        test_scored.loc[direct_only_mask].copy(),
        prediction_col=prediction_col,
        realized_col=target_col,
    )
    direct_plus_high_conf_calibration = calibration_by_decile(
        test_scored.loc[direct_plus_high_conf_mask].copy(),
        prediction_col=prediction_col,
        realized_col=target_col,
    )
    calibration_summary = pd.concat(
        [
            summarize_calibration(validation_calibration, "validation_selection"),
            summarize_calibration(full_calibration, "full_test"),
            summarize_calibration(high_conf_calibration, "high_confidence_subset"),
            summarize_calibration(high_conf_calibration, "high_confidence_exit_label_only"),
            summarize_calibration(entity_match_calibration, "high_confidence_entity_match_only"),
            summarize_calibration(direct_only_calibration, "direct_dated_only"),
            summarize_calibration(direct_plus_high_conf_calibration, "direct_dated_plus_high_conf_inferred"),
        ],
        ignore_index=True,
    )
    for key, value in base_fields.items():
        calibration_summary[key] = value
    _, confusion_summary = build_binary_confusion_exports(
        test_scored,
        prediction_col=prediction_col,
        actual_col=target_col,
        thresholds=[float(effective_config.get("target_exploration_probability_threshold", 0.03))],
        data_mode=str(effective_config.get("data_mode", "sample")),
        evaluation_view="target_exploration",
        target_label=str(spec["target_name"]),
        prediction_label=prediction_col,
        join_key="target_key",
        join_value=str(spec["target_key"]),
    )
    if not confusion_summary.empty:
        confusion_row = confusion_summary.iloc[0]
        mask = evaluation_metrics["evaluation_view"].astype(str).eq("full_test")
        for column in ["threshold", "TP", "FP", "TN", "FN", "precision", "recall", "balanced_accuracy", "F1", "prevalence"]:
            evaluation_metrics.loc[mask, f"screen_{column}"] = confusion_row[column]
    stage2_model = build_stage2_probability_tables(train_panel, stage2_classes, effective_config) if stage2_classes else {"classes": [], "tables": []}
    decision_backtest, selected_policy_key, policy_search_artifacts = build_target_policy_backtests(
        validation_panel,
        test_panel,
        fitted_model,
        stage2_model,
        spec,
        target_col,
        effective_config,
        stage1_recalibrators=recalibrators,
    )
    if not decision_backtest.empty:
        for key, value in base_fields.items():
            decision_backtest[key] = value
        decision_backtest["selected_stage2_classes"] = "|".join(stage2_classes)
    for artifact_key in ["validation", "validation_feasible", "confirmation_test"]:
        artifact_frame = policy_search_artifacts.get(artifact_key, pd.DataFrame())
        if isinstance(artifact_frame, pd.DataFrame) and not artifact_frame.empty:
            for key, value in base_fields.items():
                artifact_frame[key] = value
            artifact_frame["selected_stage2_classes"] = "|".join(stage2_classes)
            policy_search_artifacts[artifact_key] = artifact_frame
    evaluation_metrics["selected_policy_key"] = selected_policy_key
    evaluation_metrics["prediction_col"] = prediction_col
    evaluation_metrics["feature_backbone"] = str(feature_backbone)
    evaluation_metrics["selected_stage2_classes"] = "|".join(stage2_classes)
    return {
        "definition": definition,
        "prevalence": prevalence,
        "route_support": route_support,
        "source_mix": source_mix,
        "label_audit": label_audit,
        "time_distribution": time_distribution,
        "calibration_summary": calibration_summary,
        "evaluation_metrics": evaluation_metrics,
        "decision_backtest": decision_backtest,
        "confusion_summary": confusion_summary,
        "prediction_col": prediction_col,
        "selected_policy_key": selected_policy_key,
        "policy_search_validation": policy_search_artifacts.get("validation", pd.DataFrame()),
        "policy_search_validation_feasible": policy_search_artifacts.get("validation_feasible", pd.DataFrame()),
        "policy_confirmation_test": policy_search_artifacts.get("confirmation_test", pd.DataFrame()),
        "selected_stage2_classes": "|".join(stage2_classes),
    }


def build_calibration_status_notes(
    evaluation_metrics_main: pd.DataFrame,
) -> str:
    lines = [
        "# Calibration Status Notes",
        "",
        "Calibration remains the headline criterion for the redesigned main target.",
        "",
    ]
    for row in evaluation_metrics_main.itertuples(index=False):
        lines.append(
            f"- {row.evaluation_view}: slope status `{row.calibration_slope_status}`, intercept status `{row.calibration_intercept_status}`, mean abs gap `{float(row.mean_abs_calibration_gap) if pd.notna(row.mean_abs_calibration_gap) else float('nan'):.4f}`."
        )
    return "\n".join(lines) + "\n"


def build_promotion_gate_v2(
    evaluation_metrics_main: pd.DataFrame,
    label_confidence_audit: pd.DataFrame,
    stage2_route_support: pd.DataFrame,
    policy_activation_summary: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    eval_lookup = evaluation_metrics_main.set_index("evaluation_view")
    full_gap = float(eval_lookup.loc["full_test", "mean_abs_calibration_gap"]) if "full_test" in eval_lookup.index else np.nan
    high_gap = float(eval_lookup.loc["high_confidence_subset", "mean_abs_calibration_gap"]) if "high_confidence_subset" in eval_lookup.index else np.nan
    main_labels = label_confidence_audit[label_confidence_audit["target_scope"].astype(str).eq("main_hard_liquidity")].copy()
    supported_main = int(
        main_labels.loc[main_labels["confidence_tier"].astype(str).isin(["high", "medium"]), "chosen_exit_count"].sum()
    )
    total_main = int(main_labels["chosen_exit_count"].sum())
    label_conf_share = safe_ratio(supported_main, total_main)
    enough_route_support = bool(
        not stage2_route_support.empty
        and int(stage2_route_support.loc[stage2_route_support["split"].astype(str).eq("train"), "rows"].sum()) >= int(config.get("stage2_min_route_support", 5))
    )
    enough_policy_activation = bool(
        not policy_activation_summary.empty
        and float(policy_activation_summary["acceptance_rate"].max()) >= float(config.get("promotion_gate_min_policy_acceptance", 0.005))
    )
    acceptable_label_confidence = bool(label_conf_share >= float(config.get("promotion_gate_min_label_confidence_share", 0.85)))
    acceptable_calibration_full = bool(np.isfinite(full_gap) and full_gap <= float(config.get("promotion_gate_calibration_gap_max", 0.05)))
    acceptable_calibration_high_confidence = bool(np.isfinite(high_gap) and high_gap <= float(config.get("promotion_gate_high_conf_gap_max", 0.08)))
    chapter_evidence_ready = bool(
        enough_route_support
        and enough_policy_activation
        and acceptable_label_confidence
        and acceptable_calibration_full
        and acceptable_calibration_high_confidence
    )
    return pd.DataFrame(
        [
            {
                "target_name": HARD_TIMELY_LIQUIDITY_TARGET,
                "enough_route_support": enough_route_support,
                "enough_policy_activation": enough_policy_activation,
                "acceptable_label_confidence": acceptable_label_confidence,
                "acceptable_calibration_full": acceptable_calibration_full,
                "acceptable_calibration_high_confidence": acceptable_calibration_high_confidence,
                "chapter_evidence_ready": chapter_evidence_ready,
                "label_confidence_share": label_conf_share,
                "full_test_mean_abs_calibration_gap": full_gap,
                "high_confidence_mean_abs_calibration_gap": high_gap,
            }
        ]
    )


def write_promotion_gate_v2_explanation(path: Path, gate: pd.DataFrame) -> None:
    row = gate.iloc[0]
    lines = [
        "# Promotion Gate V2 Explanation",
        "",
        f"- Target: `{row['target_name']}`.",
        f"- Enough route support: `{bool(row['enough_route_support'])}`.",
        f"- Enough policy activation: `{bool(row['enough_policy_activation'])}`.",
        f"- Acceptable label confidence: `{bool(row['acceptable_label_confidence'])}`.",
        f"- Acceptable full-test calibration: `{bool(row['acceptable_calibration_full'])}`.",
        f"- Acceptable high-confidence calibration: `{bool(row['acceptable_calibration_high_confidence'])}`.",
        f"- Chapter evidence ready: `{bool(row['chapter_evidence_ready'])}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_route_multiclass_diagnostics(scored_holdout: pd.DataFrame, data_mode: str, route_pooling_used: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if str(data_mode).strip().lower() == "sample":
        classes = ["no_exit", "ipo", "mna", "sponsor_sale", "writeoff"]
        predicted_frame = pd.DataFrame(
            {
                "no_exit": pd.to_numeric(scored_holdout["survival_horizon"], errors="coerce").fillna(0.0),
                "ipo": pd.to_numeric(scored_holdout["cum_ipo"], errors="coerce").fillna(0.0),
                "mna": pd.to_numeric(scored_holdout["cum_mna"], errors="coerce").fillna(0.0),
                "sponsor_sale": pd.to_numeric(scored_holdout["cum_sponsor_sale"], errors="coerce").fillna(0.0),
                "writeoff": pd.to_numeric(scored_holdout["cum_writeoff"], errors="coerce").fillna(0.0),
            }
        )
        actual_labels = np.where(
            pd.to_numeric(scored_holdout["realized_exit_by_horizon"], errors="coerce").fillna(0).astype(int).eq(1),
            scored_holdout["company_exit_route"].astype(str),
            "no_exit",
        )
        note = "Sample mode uses the full route class set."
    else:
        classes = ["no_exit", "pooled_strategic_exit", "sponsor_sale"]
        predicted_frame = pd.DataFrame(
            {
                "no_exit": pd.to_numeric(scored_holdout["survival_horizon"], errors="coerce").fillna(0.0),
                "pooled_strategic_exit": pd.to_numeric(scored_holdout["cum_ipo"], errors="coerce").fillna(0.0)
                + pd.to_numeric(scored_holdout["cum_mna"], errors="coerce").fillna(0.0),
                "sponsor_sale": pd.to_numeric(scored_holdout["cum_sponsor_sale"], errors="coerce").fillna(0.0),
            }
        )
        actual_route = scored_holdout["company_exit_route"].astype(str)
        actual_labels = np.where(
            pd.to_numeric(scored_holdout["realized_exit_by_horizon"], errors="coerce").fillna(0).astype(int).eq(1),
            np.where(actual_route.isin(["ipo", "mna"]), "pooled_strategic_exit", np.where(actual_route.eq("sponsor_sale"), "sponsor_sale", "no_exit")),
            "no_exit",
        )
        note = (
            "Actual-mode multiclass diagnostics omit soft_failure_sensitivity because the main supervised holdout panel excludes sensitivity-only routes."
            if route_pooling_used
            else "Actual mode uses the pooled strategic diagnostic class set."
        )
    predicted_labels = predicted_frame.idxmax(axis=1)
    table = multiclass_confusion_table(
        pd.Series(actual_labels),
        pd.Series(predicted_labels),
        classes,
        str(data_mode),
        "route_multiclass",
        note,
    )
    status = pd.DataFrame(
        [
            {
                "data_mode": str(data_mode),
                "status": "ok",
                "class_set": "|".join(classes),
                "note": note,
            }
        ]
    )
    return table, status


def choose_display_view(
    panel: pd.DataFrame,
    fitted: dict,
    multiple_params: dict[str, dict[str, float]],
    config: dict,
) -> dict:
    latest_test_quarter = int(panel.loc[panel["split"] == "test", "quarter_idx"].max())
    candidates = panel[
        (panel["split"] == "test")
        & (panel["quarter_idx"] == latest_test_quarter)
        & (panel["route_label"] == "no_exit")
    ].copy().reset_index(drop=True)
    candidates["candidate_idx"] = np.arange(len(candidates), dtype=int)
    if candidates.empty:
        raise ValueError("No eligible test-quarter company snapshots are available for the display view.")
    horizon = int(config["holdout_horizon_quarters"])
    baseline_summary, baseline_points = probability_path_summary_vectorized(candidates, fitted, horizon, config, "baseline")
    freeze_summary, freeze_points = probability_path_summary_vectorized(candidates, fitted, horizon, config, "exit_freeze")
    target_quantile = 0.75 if str(config.get("data_mode", "sample")).strip().lower() == "actual" else 0.50
    target_prob = float(baseline_summary["pred_exit_by_horizon"].quantile(target_quantile))
    audit = candidates.merge(
        baseline_summary[["company_id", "quarter_idx", "pred_exit_by_horizon"]].rename(
            columns={"pred_exit_by_horizon": "baseline_pred_exit_by_horizon"}
        ),
        on=["company_id", "quarter_idx"],
        how="left",
    ).merge(
        freeze_summary[["company_id", "quarter_idx", "pred_exit_by_horizon"]].rename(
            columns={"pred_exit_by_horizon": "freeze_pred_exit_by_horizon"}
        ),
        on=["company_id", "quarter_idx"],
        how="left",
    )
    audit["decision_boundary_distance"] = (audit["baseline_pred_exit_by_horizon"] - target_prob).abs()
    audit["prob_delta"] = audit["baseline_pred_exit_by_horizon"] - audit["freeze_pred_exit_by_horizon"]
    shortlist_size = max(int(config.get("stylized_bucket_size", 24)) * 4, 48)
    shortlist = audit.sort_values(
        ["decision_boundary_distance", "prob_delta", "company_name"],
        ascending=[True, False, True],
    ).head(shortlist_size).copy()
    shortlist_panel = candidates.merge(
        shortlist[["company_id", "quarter_idx"]],
        on=["company_id", "quarter_idx"],
        how="inner",
    ).sort_values(["company_id", "quarter_idx"]).reset_index(drop=True)
    shortlist_idx = shortlist_panel["candidate_idx"].to_numpy(dtype=int)
    shortlist_baseline = baseline_points[shortlist_idx]
    shortlist_freeze = freeze_points[shortlist_idx]
    baseline_values = simulate_panel_value_summary(
        shortlist_panel,
        shortlist_baseline,
        multiple_params,
        config,
        "baseline",
        n_paths=int(config.get("decision_eval_paths", 64)),
    )
    freeze_values = simulate_panel_value_summary(
        shortlist_panel,
        shortlist_freeze,
        multiple_params,
        config,
        "exit_freeze",
        n_paths=int(config.get("decision_eval_paths", 64)),
    )
    shortlist = shortlist.merge(
        baseline_values.rename(
            columns={
                "predicted_mean_npv": "baseline_predicted_mean_npv",
                "predicted_certainty_equivalent": "baseline_predicted_certainty_equivalent",
            }
        )[["company_id", "quarter_idx", "baseline_predicted_mean_npv", "baseline_predicted_certainty_equivalent"]],
        on=["company_id", "quarter_idx"],
        how="left",
    ).merge(
        freeze_values.rename(
            columns={
                "predicted_mean_npv": "freeze_predicted_mean_npv",
                "predicted_certainty_equivalent": "freeze_predicted_certainty_equivalent",
            }
        )[["company_id", "quarter_idx", "freeze_predicted_mean_npv", "freeze_predicted_certainty_equivalent"]],
        on=["company_id", "quarter_idx"],
        how="left",
    )
    shortlist["npv_delta"] = shortlist["baseline_predicted_mean_npv"] - shortlist["freeze_predicted_mean_npv"]
    shortlist["prob_rank"] = shortlist["prob_delta"].rank(method="dense", ascending=False, pct=True)
    shortlist["npv_rank"] = shortlist["npv_delta"].rank(method="dense", ascending=False, pct=True)
    shortlist["boundary_rank"] = shortlist["decision_boundary_distance"].rank(method="dense", ascending=True, pct=True)
    shortlist["selection_score"] = 0.45 * shortlist["prob_rank"] + 0.35 * shortlist["npv_rank"] + 0.20 * shortlist["boundary_rank"]
    best = shortlist.sort_values(["selection_score", "company_name"], ascending=[False, True]).iloc[0]
    visible_single = bool(
        float(best["prob_delta"]) >= float(config.get("stress_visibility_prob_gap", 0.0025))
        or float(best["npv_delta"]) >= float(config.get("stress_visibility_npv_gap", 0.01))
    )
    if visible_single:
        selected_panel = candidates[
            (candidates["company_id"].astype(str) == str(best["company_id"]))
            & (candidates["quarter_idx"].astype(int) == int(best["quarter_idx"]))
        ].copy()
        display_mode = "single_company"
        display_label = str(best["company_name"])
        selected_ids = {(str(best["company_id"]), int(best["quarter_idx"]))}
    else:
        bucket_size = max(int(config.get("stylized_bucket_size", 24)), 4)
        bucket = shortlist.sort_values(["selection_score", "company_name"], ascending=[False, True]).head(bucket_size).copy()
        selected_ids = set(zip(bucket["company_id"].astype(str), bucket["quarter_idx"].astype(int), strict=True))
        selected_panel = candidates[
            [
                (str(company_id), int(quarter_idx)) in selected_ids
                for company_id, quarter_idx in zip(candidates["company_id"], candidates["quarter_idx"], strict=True)
            ]
        ].copy()
        display_mode = "portfolio_bucket"
        display_label = f"Top {len(selected_panel)} latest-test candidates"
    audit = audit.merge(
        shortlist[
            [
                "company_id",
                "quarter_idx",
                "baseline_predicted_mean_npv",
                "freeze_predicted_mean_npv",
                "npv_delta",
                "selection_score",
            ]
        ],
        on=["company_id", "quarter_idx"],
        how="left",
    )
    audit["display_mode"] = display_mode
    if display_mode == "single_company":
        audit["selected_for_display"] = (
            (audit["company_id"].astype(str) == str(best["company_id"]))
            & (audit["quarter_idx"].astype(int) == int(best["quarter_idx"]))
        ).astype(int)
    else:
        bucket_pairs = {
            (company_id, quarter_idx)
            for company_id, quarter_idx in zip(
                selected_panel["company_id"].astype(str),
                selected_panel["quarter_idx"].astype(int),
                strict=True,
            )
        }
        audit["selected_for_display"] = [
            int((str(company_id), int(quarter_idx)) in bucket_pairs)
            for company_id, quarter_idx in zip(audit["company_id"], audit["quarter_idx"], strict=True)
        ]
    return {
        "display_mode": display_mode,
        "display_label": display_label,
        "selected_panel": selected_panel.reset_index(drop=True),
        "selected_row": selected_panel.sort_values(["company_name", "company_id"]).iloc[0],
        "audit": audit.sort_values(["selected_for_display", "selection_score", "company_name"], ascending=[False, False, True]).reset_index(drop=True),
    }


def save_figure_with_svg(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_calibration_deciles(
    calibration: pd.DataFrame,
    stress_calibration: pd.DataFrame,
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True, sharey=True)
    for axis, frame, title in [
        (axes[0], calibration, "Full test window"),
        (axes[1], stress_calibration, "Stress slice"),
    ]:
        axis.plot([0.0, 1.0], [0.0, 1.0], color="#777777", linestyle="--", linewidth=1.2)
        if not frame.empty:
            axis.scatter(
                frame["mean_predicted_exit"],
                frame["realized_exit_rate"],
                s=np.maximum(frame["n"].to_numpy(dtype=float), 12.0) * 1.4,
                color="#2451B7",
                alpha=0.75,
            )
            for row in frame.itertuples(index=False):
                axis.text(
                    float(row.mean_predicted_exit),
                    float(row.realized_exit_rate),
                    str(int(row.decile)),
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )
        else:
            axis.text(0.5, 0.5, "No supported stress slice", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Realized event rate")
    fig.suptitle("Out-of-time 8-quarter exit calibration", y=1.02)
    path = output_dir / "vcpe-calibration-deciles.png"
    save_figure_with_svg(fig, path)
    return path


def plot_cumulative_incidence(
    incidence_map: dict[str, pd.DataFrame],
    output_dir: Path,
    display_label: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for scenario_name, frame in incidence_map.items():
        ax.plot(
            frame["horizon_q"],
            frame["prob_exit_by_horizon"],
            linewidth=2.2,
            label=f"{scenario_name} total exit",
        )
        ax.plot(
            frame["horizon_q"],
            frame["cum_sponsor_sale"],
            linewidth=1.5,
            linestyle="--",
            label=f"{scenario_name} sponsor sale",
        )
    ax.set_xlabel("Forward quarter")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"{display_label} cumulative incidence by scenario")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    path = output_dir / "vcpe-cumulative-incidence.png"
    save_figure_with_svg(fig, path)
    return path


def plot_stage2_cumulative_incidence(
    incidence_map: dict[str, pd.DataFrame],
    output_dir: Path,
    display_label: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for scenario_name, frame in incidence_map.items():
        strategic = (
            pd.to_numeric(frame.get("cum_pooled_strategic"), errors="coerce").fillna(0.0)
            if "cum_pooled_strategic" in frame.columns
            else pd.to_numeric(frame.get("cum_ipo"), errors="coerce").fillna(0.0)
            + pd.to_numeric(frame.get("cum_mna"), errors="coerce").fillna(0.0)
        )
        sponsor = pd.to_numeric(frame.get("cum_sponsor_sale"), errors="coerce").fillna(0.0)
        ax.plot(frame["horizon_q"], strategic, linewidth=2.4, label=f"{scenario_name} pooled strategic")
        ax.plot(frame["horizon_q"], sponsor, linewidth=1.8, linestyle="--", label=f"{scenario_name} sponsor sale")
    ax.set_xlabel("Forward quarter")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"{display_label} hard-liquidity route incidence")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    path = output_dir / "vcpe-cumulative-incidence.png"
    save_figure_with_svg(fig, path)
    return path


def plot_npv_distribution(npv_map: dict[str, pd.DataFrame], output_dir: Path, display_label: str) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    bins = 40
    for scenario_name, frame in npv_map.items():
        ax.hist(
            frame["npv"],
            bins=bins,
            alpha=0.45,
            density=True,
            label=scenario_name,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("NPV")
    ax.set_ylabel("Density")
    ax.set_title(f"{display_label} NPV distribution")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path = output_dir / "vcpe-npv-distribution.png"
    save_figure_with_svg(fig, path)
    return path


def plot_route_waterfall(incidence_map: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    scenarios = list(incidence_map.keys())
    route_matrix = np.array(
        [
            [
                frame[f"cum_{route}"].iloc[-1]
                for route in EXIT_ROUTES
            ]
            for frame in incidence_map.values()
        ]
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bottom = np.zeros(len(scenarios))
    colors = {
        "ipo": "#2451B7",
        "mna": "#4F86C6",
        "sponsor_sale": "#2A7F62",
        "writeoff": "#B63A2B",
    }
    for route_idx, route in enumerate(EXIT_ROUTES):
        values = route_matrix[:, route_idx]
        ax.bar(scenarios, values, bottom=bottom, label=route, color=colors[route], alpha=0.85)
        bottom = bottom + values
    ax.set_ylabel("Probability by horizon")
    ax.set_title("Route mix by horizon and scenario")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    path = output_dir / "vcpe-route-waterfall.png"
    save_figure_with_svg(fig, path)
    return path


def plot_binary_confusion_heatmap(summary_row: pd.Series, title: str, path: Path) -> Path:
    counts = np.array(
        [
            [int(summary_row["TN"]), int(summary_row["FP"])],
            [int(summary_row["FN"]), int(summary_row["TP"])],
        ],
        dtype=float,
    )
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, np.clip(row_totals, 1.0, None))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=max(float(normalized.max()), 0.25))
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Predicted no", "Predicted yes"])
    ax.set_yticks([0, 1], labels=["Actual no", "Actual yes"])
    ax.set_title(title)
    for row_idx in range(2):
        for col_idx in range(2):
            ax.text(
                col_idx,
                row_idx,
                f"{int(counts[row_idx, col_idx])}\n{normalized[row_idx, col_idx] * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="#0B2239" if normalized[row_idx, col_idx] < 0.6 else "white",
            )
    ax.set_xlabel("Model classification")
    ax.set_ylabel("Observed outcome")
    save_figure_with_svg(fig, path)
    return path


def plot_multiclass_confusion_heatmap(
    confusion_table: pd.DataFrame,
    classes: list[str],
    title: str,
    path: Path,
) -> Path:
    matrix = (
        confusion_table.pivot(index="actual_class", columns="predicted_class", values="count")
        .reindex(index=classes, columns=classes, fill_value=0)
        .to_numpy(dtype=float)
    )
    normalized = np.divide(matrix, np.clip(matrix.sum(axis=1, keepdims=True), 1.0, None))
    fig, ax = plt.subplots(figsize=(1.35 * len(classes) + 2.8, 1.2 * len(classes) + 2.2))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=max(float(normalized.max()), 0.25))
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)), labels=classes, rotation=25, ha="right")
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set_title(title)
    for row_idx, actual_class in enumerate(classes):
        for col_idx, predicted_class in enumerate(classes):
            count = int(matrix[row_idx, col_idx])
            pct = normalized[row_idx, col_idx] * 100.0
            ax.text(
                col_idx,
                row_idx,
                f"{count}\n{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="#0B2239" if normalized[row_idx, col_idx] < 0.6 else "white",
            )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    save_figure_with_svg(fig, path)
    return path


def plot_policy_backtest(
    screening_backtest: pd.DataFrame,
    economic_backtest: pd.DataFrame,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = []
    labels = []
    for frame, label in [(screening_backtest, "screening"), (economic_backtest, "economic")]:
        if frame.empty:
            continue
        selected = frame.loc[frame["selected_on_validation"].astype(int).eq(1)].copy()
        if selected.empty:
            selected = frame.iloc[[0]].copy()
        row = selected.iloc[0]
        labels.append(label)
        bars.append(float(row["hit_rate_accepted"]) if pd.notna(row["hit_rate_accepted"]) else 0.0)
    if not bars:
        bars = [0.0]
        labels = ["no_active_policy"]
    ax.bar(labels, bars, color=["#2451B7", "#D16A00"][: len(bars)])
    ax.set_ylabel("Held-out hit rate among accepted names")
    ax.set_title("Active policy backtest")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "vcpe-policy-backtest.png"
    save_figure_with_svg(fig, path)
    return path


def plot_feature_importance_groups(feature_group_importance: pd.DataFrame, output_dir: Path) -> Path:
    chart = feature_group_importance[
        feature_group_importance["status"].astype(str).eq("ok")
    ].sort_values("mean_abs_calibration_gap_delta", ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.55 * max(len(chart), 1))))
    ax.barh(chart["feature_group"], chart["mean_abs_calibration_gap_delta"], color="#2451B7", alpha=0.85)
    ax.set_xlabel("Calibration-gap delta after grouped permutation")
    ax.set_title("Grouped permutation importance (test)")
    ax.grid(axis="x", alpha=0.25)
    path = output_dir / "vcpe-feature-importance-groups.png"
    save_figure_with_svg(fig, path)
    return path


def plot_feature_group_ablation(ablation_test: pd.DataFrame, output_dir: Path) -> Path:
    chart = ablation_test[
        ablation_test["status"].astype(str).eq("ok")
    ].sort_values("mean_abs_calibration_gap_delta", ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.55 * max(len(chart), 1))))
    ax.barh(chart["feature_group"], chart["mean_abs_calibration_gap_delta"], color="#2A7F62", alpha=0.85)
    ax.set_xlabel("Calibration-gap delta after leave-group-out retrain")
    ax.set_title("Feature-group ablation impact (test)")
    ax.grid(axis="x", alpha=0.25)
    path = output_dir / "vcpe-feature-group-ablation.png"
    save_figure_with_svg(fig, path)
    return path


def plot_feature_combo_heatmap(combo_leaderboard: pd.DataFrame, output_dir: Path) -> Path:
    chart = combo_leaderboard.copy()
    metrics = [
        "validation_mean_abs_calibration_gap",
        "validation_brier_score",
        "validation_pr_auc",
        "validation_top_decile_lift",
        "validation_roc_auc",
    ]
    matrix = chart[metrics].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, max(4.6, 0.6 * len(chart) + 1.5)))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(metrics)), labels=[metric.replace("validation_", "") for metric in metrics], rotation=20, ha="right")
    ax.set_yticks(range(len(chart)), labels=chart["combo_key"])
    ax.set_title("Validation feature-combination leaderboard")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.4f}", ha="center", va="center", fontsize=8, color="white" if matrix[row_idx, col_idx] > np.nanmean(matrix) else "#0B2239")
    path = output_dir / "vcpe-feature-combo-heatmap.png"
    save_figure_with_svg(fig, path)
    return path


def plot_sector_feature_importance(sector_feature_importance: pd.DataFrame, output_dir: Path) -> Path:
    if sector_feature_importance.empty or "bucket_dimension" not in sector_feature_importance.columns:
        chart = pd.DataFrame()
    else:
        chart = sector_feature_importance[
            sector_feature_importance["bucket_dimension"].astype(str).eq("sector")
            & sector_feature_importance["status"].astype(str).eq("ok")
        ].copy()
    if chart.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.text(0.5, 0.5, "No supported sector buckets", ha="center", va="center", transform=ax.transAxes)
        path = output_dir / "vcpe-sector-feature-importance.png"
        save_figure_with_svg(fig, path)
        return path
    pivot = chart.pivot(index="bucket_name", columns="feature_group", values="mean_abs_calibration_gap_delta").fillna(0.0)
    fig, ax = plt.subplots(figsize=(1.5 * pivot.shape[1] + 2.8, 0.65 * pivot.shape[0] + 2.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(pivot.shape[1]), labels=pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(pivot.shape[0]), labels=pivot.index)
    ax.set_title("Sector-by-feature-group importance")
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = float(pivot.iloc[row_idx, col_idx])
            ax.text(col_idx, row_idx, f"{value:.4f}", ha="center", va="center", fontsize=8, color="white" if value > float(np.nanmean(pivot.to_numpy(dtype=float))) else "#0B2239")
    path = output_dir / "vcpe-sector-feature-importance.png"
    save_figure_with_svg(fig, path)
    return path


def plot_patent_value_by_sector(patent_value_by_sector: pd.DataFrame, output_dir: Path) -> Path:
    chart = patent_value_by_sector.copy()
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.55 * max(len(chart), 1))))
    if chart.empty:
        ax.text(0.5, 0.5, "No supported patent-by-sector result", ha="center", va="center", transform=ax.transAxes)
    else:
        colors = ["#2451B7" if int(value) == 1 else "#8AA9D6" for value in chart["patent_plausible"]]
        ax.barh(chart["sector_bucket"], chart["mean_abs_calibration_gap_delta"], color=colors, alpha=0.85)
        ax.set_xlabel("Patent-core calibration-gap delta")
    ax.set_title("Patent value by sector bucket")
    ax.grid(axis="x", alpha=0.25)
    path = output_dir / "vcpe-patent-value-by-sector.png"
    save_figure_with_svg(fig, path)
    return path


def build_evaluation_view_definitions(data_mode: str, route_pooling_used: bool) -> pd.DataFrame:
    rows = [
        {
            "evaluation_view": "any_exit_aggregate",
            "reporting_role": "headline",
            "target_definition": "realized exit within 8 quarters",
            "support_policy_note": "Primary chapter evaluation view; calibration remains the leading diagnostic.",
        },
        {
            "evaluation_view": "competing_risk_aggregate",
            "reporting_role": "supporting",
            "target_definition": "direct-route aggregate within 8 quarters",
            "support_policy_note": "Competing-risk view over the modeled direct routes.",
        },
        {
            "evaluation_view": "high_confidence_subset",
            "reporting_role": "supporting",
            "target_definition": "realized exit within 8 quarters on high-confidence exit labels",
            "support_policy_note": "Default robustness slice now uses exit-label confidence rather than entity-match confidence.",
        },
        {
            "evaluation_view": "high_confidence_entity_match",
            "reporting_role": "supporting",
            "target_definition": "realized exit within 8 quarters on high-confidence entity matches",
            "support_policy_note": "Separate subset diagnostic; entity-match confidence is not the default high-confidence robustness slice.",
        },
        {
            "evaluation_view": "stress_regime_subperiod",
            "reporting_role": "supporting",
            "target_definition": "realized exit within 8 quarters in the stress slice",
            "support_policy_note": "Stress-only slice used for calibration and decision diagnostics when the selected split supports it.",
        },
        {
            "evaluation_view": "sponsor_sale",
            "reporting_role": "diagnostic",
            "target_definition": "sponsor-sale competing-risk event",
            "support_policy_note": "Reported as a route diagnostic, not as the headline chapter metric.",
        },
        {
            "evaluation_view": "direct_ipo_diagnostic",
            "reporting_role": "diagnostic",
            "target_definition": "direct IPO event",
            "support_policy_note": "Demoted when train IPO support is thin.",
        },
        {
            "evaluation_view": "direct_mna_diagnostic",
            "reporting_role": "diagnostic",
            "target_definition": "direct M&A event",
            "support_policy_note": "Direct M&A remains exported but is not the headline metric.",
        },
        {
            "evaluation_view": "soft_failure_sensitivity_view",
            "reporting_role": "sensitivity_only",
            "target_definition": "sensitivity-only soft-failure proxy",
            "support_policy_note": "Not a supervised main-panel class; kept as a sensitivity audit rather than a chapter headline.",
        },
    ]
    if route_pooling_used:
        rows.append(
            {
                "evaluation_view": "pooled_strategic_fallback",
                "reporting_role": "supporting",
                "target_definition": "ipo + mna pooled strategic exit",
                "support_policy_note": "Evaluation-only fallback activated because direct IPO train support remains too thin.",
            }
        )
    return pd.DataFrame(rows)


def write_confusion_matrix_notes(
    output_dir: Path,
    data_mode: str,
    primary_threshold: float,
    primary_policy_key: str,
) -> Path:
    lines = [
        "# Confusion Matrix Notes",
        "",
        "- Confusion matrices in this folder are threshold-dependent diagnostics.",
        "- Raw accuracy is misleading in rare-event settings because most observations are negatives.",
        "- Balanced accuracy, precision, and recall are more informative than plain accuracy here.",
        "- Calibration remains the primary evaluation criterion; confusion matrices are supplemental decision diagnostics.",
        f"- The primary binary threshold for this run is `pred_exit_by_8q >= {primary_threshold:.2f}`.",
        f"- The primary policy heatmap uses policy key `{primary_policy_key}`.",
        f"- Data mode: `{data_mode}`.",
        "",
    ]
    path = output_dir / "confusion_matrix_notes.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_promotion_gate_explanation(
    output_dir: Path,
    promotion_gate: pd.DataFrame,
    route_pooling_used: bool,
) -> Path:
    gate = promotion_gate.iloc[0]
    lines = [
        "# Promotion Gate Explanation",
        "",
        "- Calibration is the chapter headline metric.",
        "- Confusion matrices supplement the evaluation package but do not replace calibration.",
        f"- Route-pooling fallback used: `{bool(route_pooling_used)}`.",
        f"- Main calibration mean absolute gap: `{float(gate['main_calibration_mean_abs_gap']):.4f}`.",
        f"- High-confidence calibration mean absolute gap: `{float(gate['high_confidence_calibration_mean_abs_gap']):.4f}`.",
        f"- Soft-failure sensitivity share: `{float(gate['soft_failure_sensitivity_share']):.4f}`.",
        f"- Freeze direction check passed: `{bool(gate['freeze_direction_ok'])}`.",
        f"- Chapter evidence ready: `{bool(gate['chapter_evidence_ready'])}`.",
        "",
        "- The gate remains conservative: strong calibration is necessary but not sufficient.",
    ]
    path = output_dir / "promotion_gate_explanation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def feature_group_columns(panel: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "macro_time": [],
        "company_core": [column for column in ["age_q"] if column in panel.columns],
        "sector_stage": [column for column in [*sector_dummy_columns(), *stage_dummy_columns()] if column in panel.columns],
        "financing_trajectory": [
            column
            for column in ["time_since_last_round_q", "log_last_round_usd"]
            if column in panel.columns
        ],
        "sponsor_fund": [column for column in ["sponsor_score"] if column in panel.columns],
        "lp_demand": [],
        "patent_core": [column for column in PATENT_FEATURE_COLUMNS if column in panel.columns],
        "patent_quality": [],
        "network_team": [],
        "interaction_bundle": [],
    }


def interaction_bundle_columns(panel: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "sector_bucket_x_patent_core": [
            column
            for column in [
                "interaction_sector_patent_apps_plausible",
                "interaction_sector_patent_grants_plausible",
                "interaction_sector_patent_stock_plausible",
            ]
            if column in panel.columns
        ],
        "stage_bucket_x_patent_core": [
            column
            for column in [
                "interaction_stage_patent_apps_growth",
                "interaction_stage_patent_stock_growth",
            ]
            if column in panel.columns
        ],
        "sector_bucket_x_sponsor_fund": [
            column
            for column in [
                "interaction_sector_sponsor_financial",
                "interaction_sector_sponsor_generic_services",
            ]
            if column in panel.columns
        ],
        "stage_bucket_x_financing_trajectory": [
            column
            for column in [
                "interaction_stage_financing_round_buyout",
                "interaction_stage_financing_tslr_growth",
            ]
            if column in panel.columns
        ],
        "macro_time_x_sponsor_fund": [
            column for column in ["interaction_macro_sponsor"] if column in panel.columns
        ],
    }


def build_feature_registry(panel: pd.DataFrame, data_mode: str) -> pd.DataFrame:
    active_groups = feature_group_columns(panel)
    registry_rows = [
        {
            "feature_name": "market_regime",
            "feature_group": "macro_time",
            "source_family": "macro_bridge",
            "pit_timing_rule": "quarter_t_observable_bridge",
            "active_in_actual_mode": "yes",
            "active_in_sample_mode": "yes",
            "placeholder_status": "active",
            "requires_sector_interaction": "no",
            "notes": "Time-varying macro bridge used directly; actual mode also keeps quarter fixed effects as structural time controls.",
        },
        {
            "feature_name": "quarter_fixed_effects",
            "feature_group": "macro_time",
            "source_family": "macro_bridge",
            "pit_timing_rule": "quarter_index_only",
            "active_in_actual_mode": "yes",
            "active_in_sample_mode": "no",
            "placeholder_status": "active_structural",
            "requires_sector_interaction": "no",
            "notes": "Actual mode uses quarter fixed effects as structural time controls; this term is handled in ablations rather than single-feature permutation.",
        },
        {
            "feature_name": "age_q",
            "feature_group": "company_core",
            "source_family": "company_history",
            "pit_timing_rule": "quarter_t_minus_1_age",
            "active_in_actual_mode": "yes",
            "active_in_sample_mode": "yes",
            "placeholder_status": "active",
            "requires_sector_interaction": "no",
            "notes": "Company age in quarters since effective founding date.",
        },
        {
            "feature_name": "sector_bucket",
            "feature_group": "sector_stage",
            "source_family": "company_sector_text",
            "pit_timing_rule": "time_invariant_company_bucket",
            "active_in_actual_mode": "yes",
            "active_in_sample_mode": "yes",
            "placeholder_status": "active",
            "requires_sector_interaction": "yes",
            "notes": "Modeled through governed coarse-bucket dummies; actual mode maps Crunchbase description text and sample mode maps synthetic name tokens.",
        },
        {
            "feature_name": "stage_bucket",
            "feature_group": "sector_stage",
            "source_family": "financing_round_stage",
            "pit_timing_rule": "last_observed_round_stage_by_t_minus_1",
            "active_in_actual_mode": "yes",
            "active_in_sample_mode": "yes",
            "placeholder_status": "active",
            "requires_sector_interaction": "no",
            "notes": "Stage bucket derived from the latest observed round stage or investment type by quarter-end t-1.",
        },
        {
            "feature_name": "time_since_last_round_q",
            "feature_group": "financing_trajectory",
            "source_family": "financing_history",
            "pit_timing_rule": "quarter_t_minus_1_gap",
            "active_in_actual_mode": "yes",
            "active_in_sample_mode": "yes",
            "placeholder_status": "active",
            "requires_sector_interaction": "no",
            "notes": "Gap in quarters since the latest observed round.",
        },
        {
            "feature_name": "log_last_round_usd",
            "feature_group": "financing_trajectory",
            "source_family": "financing_history",
            "pit_timing_rule": "latest_round_amount_by_t_minus_1",
            "active_in_actual_mode": "yes",
            "active_in_sample_mode": "yes",
            "placeholder_status": "active",
            "requires_sector_interaction": "no",
            "notes": "Log one plus last observed round amount.",
        },
        {
            "feature_name": "sponsor_score",
            "feature_group": "sponsor_fund",
            "source_family": "investor_proxy",
            "pit_timing_rule": "latest_observed_investor_proxy_by_t_minus_1",
            "active_in_actual_mode": "yes",
            "active_in_sample_mode": "yes",
            "placeholder_status": "baseline_proxy_active",
            "requires_sector_interaction": "no",
            "notes": "Baseline sponsor/fund proxy; not the deferred full sponsor-state layer.",
        },
        {
            "feature_name": "patent_apps_visible_l4q",
            "feature_group": "patent_core",
            "source_family": "wrds_patents",
            "pit_timing_rule": "application_visible_at_appldate_plus_18m",
            "active_in_actual_mode": "yes",
            "active_in_sample_mode": "yes",
            "placeholder_status": "active",
            "requires_sector_interaction": "yes",
            "notes": "Visible applications in the last four quarters.",
        },
        {
            "feature_name": "patent_grants_l4q",
            "feature_group": "patent_core",
            "source_family": "wrds_patents",
            "pit_timing_rule": "grantdate_visible",
            "active_in_actual_mode": "yes",
            "active_in_sample_mode": "yes",
            "placeholder_status": "active",
            "requires_sector_interaction": "yes",
            "notes": "Recent grants in the last four quarters.",
        },
        {
            "feature_name": "patent_stock_visible",
            "feature_group": "patent_core",
            "source_family": "wrds_patents",
            "pit_timing_rule": "cumulative_visible_application_stock",
            "active_in_actual_mode": "yes",
            "active_in_sample_mode": "yes",
            "placeholder_status": "active",
            "requires_sector_interaction": "yes",
            "notes": "Visible application stock by quarter-end t-1.",
        },
        {
            "feature_name": "patent_citation_flow_l4q",
            "feature_group": "patent_quality",
            "source_family": "wrds_patents",
            "pit_timing_rule": "deferred_pending_citation_lags",
            "active_in_actual_mode": "no",
            "active_in_sample_mode": "no",
            "placeholder_status": "deferred_placeholder",
            "requires_sector_interaction": "yes",
            "notes": "Forward and backward citation timing remains intentionally deferred.",
        },
        {
            "feature_name": "lp_demand_score",
            "feature_group": "lp_demand",
            "source_family": "lp_commitment_data",
            "pit_timing_rule": "deferred_pending_lp_links",
            "active_in_actual_mode": "no",
            "active_in_sample_mode": "no",
            "placeholder_status": "deferred_placeholder",
            "requires_sector_interaction": "no",
            "notes": "No PIT-safe LP coverage is staged in the current local files.",
        },
        {
            "feature_name": "company_network_signal",
            "feature_group": "network_team",
            "source_family": "network_enrichment",
            "pit_timing_rule": "deferred_pending_team_network_links",
            "active_in_actual_mode": "no",
            "active_in_sample_mode": "no",
            "placeholder_status": "deferred_placeholder",
            "requires_sector_interaction": "no",
            "notes": "Team and network enrichment remain outside the current Chapter 9 build.",
        },
        {
            "feature_name": "governed_interaction_bundle",
            "feature_group": "interaction_bundle",
            "source_family": "derived_interactions",
            "pit_timing_rule": "derived_from_active_pit_safe_features",
            "active_in_actual_mode": "screened_only",
            "active_in_sample_mode": "screened_only",
            "placeholder_status": "screened_separately",
            "requires_sector_interaction": "yes",
            "notes": "Interaction bundles are screened separately and are not part of the default combo search baseline.",
        },
    ]
    registry = pd.DataFrame(registry_rows)
    registry["data_mode"] = str(data_mode)
    registry["group_has_active_columns"] = registry["feature_group"].map(
        lambda value: "yes" if bool(active_groups.get(str(value), [])) or value == "macro_time" else "no"
    )
    return registry[
        [
            "data_mode",
            "feature_name",
            "feature_group",
            "source_family",
            "pit_timing_rule",
            "active_in_actual_mode",
            "active_in_sample_mode",
            "placeholder_status",
            "requires_sector_interaction",
            "group_has_active_columns",
            "notes",
        ]
    ].copy()


def build_sector_bucket_mapping(company_master: pd.DataFrame) -> pd.DataFrame:
    if company_master.empty:
        return pd.DataFrame(columns=["raw_sector_text", "sector_bucket", "companies"])
    mapping = company_master.copy()
    mapping["raw_sector_text"] = mapping["raw_sector_text"].fillna("unknown")
    return (
        mapping.groupby(["raw_sector_text", "sector_bucket"], as_index=False)
        .agg(companies=("company_id", "nunique"))
        .sort_values(["sector_bucket", "companies", "raw_sector_text"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def build_sector_stage_support(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dimension, column, min_rows, min_exits in [
        ("sector", "sector_bucket", int(config.get("sector_support_min_rows", 5000)), int(config.get("sector_support_min_exits", 25))),
        ("stage", "stage_bucket", int(config.get("stage_support_min_rows", 2500)), int(config.get("stage_support_min_exits", 20))),
    ]:
        if column not in panel.columns:
            continue
        grouped = (
            panel.groupby(["split", column], as_index=False, observed=True)
            .agg(
                rows=("company_id", "size"),
                companies=("company_id", "nunique"),
                exits=("realized_exit_by_horizon", "sum"),
            )
            .rename(columns={column: "bucket_name"})
        )
        grouped["bucket_dimension"] = dimension
        grouped["supported_for_bucket_analysis"] = (
            grouped["rows"].astype(int).ge(min_rows)
            & grouped["exits"].astype(int).ge(min_exits)
        ).astype(int)
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(
            columns=["split", "bucket_name", "rows", "companies", "exits", "bucket_dimension", "supported_for_bucket_analysis"]
        )
    return pd.concat(rows, ignore_index=True)[
        ["bucket_dimension", "bucket_name", "split", "rows", "companies", "exits", "supported_for_bucket_analysis"]
    ].sort_values(["bucket_dimension", "split", "bucket_name"]).reset_index(drop=True)


def build_feature_importance_target_definition(route_pooling_used: bool) -> pd.DataFrame:
    rows = [
        {
            "target_name": "any_exit_aggregate",
            "target_label": "realized_exit_by_8q",
            "reporting_role": "primary",
            "note": "Primary calibration-first feature-importance target for Chapter 9.",
        },
        {
            "target_name": "pooled_strategic_exit",
            "target_label": "realized_pooled_strategic_exit",
            "reporting_role": "supporting" if route_pooling_used else "inactive",
            "note": "Supporting honesty view when direct IPO support is too thin.",
        },
    ]
    return pd.DataFrame(rows)


def build_route_support_for_importance(route_support_by_split: pd.DataFrame, route_pooling_used: bool) -> pd.DataFrame:
    support = route_support_by_split.copy()
    support["route_pooling_used"] = int(bool(route_pooling_used))
    return support


def prediction_metrics_snapshot(
    scored_panel: pd.DataFrame,
    prediction_col: str,
    realized_col: str,
    horizon_quarters: int,
) -> dict[str, float]:
    summary = summarize_evaluation_view(scored_panel, "snapshot", prediction_col, realized_col, horizon_quarters).iloc[0]
    return {
        "brier_score": float(summary["brier_score"]) if pd.notna(summary["brier_score"]) else np.nan,
        "integrated_brier_score": float(summary["integrated_brier_score"]) if pd.notna(summary["integrated_brier_score"]) else np.nan,
        "mean_abs_calibration_gap": float(summary["mean_abs_calibration_gap"]) if pd.notna(summary["mean_abs_calibration_gap"]) else np.nan,
        "pr_auc": float(summary["pr_auc"]) if pd.notna(summary["pr_auc"]) else np.nan,
        "roc_auc": float(summary["roc_auc"]) if pd.notna(summary["roc_auc"]) else np.nan,
        "top_decile_realized_exit_rate": float(summary["top_decile_realized_exit_rate"]) if pd.notna(summary["top_decile_realized_exit_rate"]) else np.nan,
        "top_decile_lift": float(summary["top_decile_lift"]) if pd.notna(summary["top_decile_lift"]) else np.nan,
    }


def positive_help_delta(base_value: float, perturbed_value: float, lower_is_better: bool) -> float:
    if not (np.isfinite(base_value) and np.isfinite(perturbed_value)):
        return np.nan
    return float(perturbed_value - base_value) if lower_is_better else float(base_value - perturbed_value)


def permute_panel_columns(
    frame: pd.DataFrame,
    columns: list[str],
    seed: int,
    conditional_key: str | None = None,
) -> pd.DataFrame:
    if not columns:
        return frame.copy()
    permuted = frame.copy()
    rng = np.random.default_rng(int(seed))
    if conditional_key and conditional_key in frame.columns:
        for _, group_index in frame.groupby(conditional_key, sort=True, observed=True).groups.items():
            idx = np.asarray(list(group_index), dtype=int)
            if idx.size <= 1:
                continue
            shuffled = rng.permutation(idx)
            permuted.loc[idx, columns] = frame.loc[shuffled, columns].to_numpy()
        return permuted
    shuffled = rng.permutation(len(frame))
    permuted.loc[:, columns] = frame.iloc[shuffled][columns].to_numpy()
    return permuted


def score_model_panel(
    panel: pd.DataFrame,
    fitted: dict,
    horizon_quarters: int,
    company_master: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    scored_panel, point_matrix = probability_path_summary_vectorized(panel, fitted, horizon_quarters)
    keep_columns = [
        column
        for column in [
            "company_id",
            "quarter_idx",
            "route_label",
            "company_name",
            "realized_exit_by_horizon",
            "company_exit_route",
            "company_exit_value_usd",
            "company_exit_confidence_tier",
            "company_exit_route_source",
            "exit_quarter_idx",
            "log_last_round_usd",
            "sector_bucket",
            "stage_bucket",
        ]
        if column in panel.columns
    ]
    scored_panel = scored_panel.merge(panel[keep_columns], on=["company_id", "quarter_idx"], how="left")
    scored_panel = scored_panel.merge(realized_exit_paths(panel, horizon_quarters), on=["company_id", "quarter_idx"], how="left")
    scored_panel = scored_panel.merge(
        company_master[["company_id", "match_confidence", "company_source"]].rename(
            columns={"match_confidence": "entity_match_confidence"}
        ),
        on="company_id",
        how="left",
    )
    scored_panel["pred_direct_exit_by_horizon"] = (
        pd.to_numeric(scored_panel.get("cum_ipo"), errors="coerce").fillna(0.0)
        + pd.to_numeric(scored_panel.get("cum_mna"), errors="coerce").fillna(0.0)
        + pd.to_numeric(scored_panel.get("cum_sponsor_sale"), errors="coerce").fillna(0.0)
    )
    scored_panel["pred_pooled_strategic_exit"] = (
        pd.to_numeric(scored_panel.get("cum_ipo"), errors="coerce").fillna(0.0)
        + pd.to_numeric(scored_panel.get("cum_mna"), errors="coerce").fillna(0.0)
    )
    scored_panel["realized_direct_exit_by_horizon"] = pd.to_numeric(
        scored_panel.get("realized_exit_by_horizon"),
        errors="coerce",
    ).fillna(0).astype(int)
    scored_panel["realized_pooled_strategic_exit"] = (
        scored_panel["company_exit_route"].astype(str).isin(["ipo", "mna"])
        & pd.to_numeric(scored_panel["realized_exit_by_horizon"], errors="coerce").fillna(0).astype(int).eq(1)
    ).astype(int)
    return scored_panel, point_matrix


def build_decision_panel_for_split(
    split_panel: pd.DataFrame,
    fitted: dict,
    route_multiple_params: dict[str, dict[str, float]],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    decision_candidates = split_panel[split_panel["route_label"] == "no_exit"].copy()
    if decision_candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    decision_quarter = int(decision_candidates["quarter_idx"].max())
    decision_panel_raw = decision_candidates[decision_candidates["quarter_idx"] == decision_quarter].copy()
    horizon = int(config["holdout_horizon_quarters"])
    decision_summary, decision_points = probability_path_summary_vectorized(
        decision_panel_raw,
        fitted,
        horizon,
        config,
        "baseline",
    )
    decision_panel = decision_panel_raw.merge(decision_summary, on=["company_id", "quarter_idx"], how="left")
    decision_panel = decision_panel.merge(
        realized_exit_paths(decision_panel_raw, horizon),
        on=["company_id", "quarter_idx"],
        how="left",
    )
    decision_config = config.copy()
    decision_config["decision_eval_paths"] = int(config.get("feature_search_decision_eval_paths", config.get("decision_eval_paths", 24)))
    decision_panel = decision_panel.merge(
        simulate_panel_value_summary(
            decision_panel_raw,
            decision_points,
            route_multiple_params,
            decision_config,
            "baseline",
            n_paths=int(decision_config["decision_eval_paths"]),
        ),
        on=["company_id", "quarter_idx"],
        how="left",
    )
    decision_panel = decision_panel.merge(
        build_realized_value_proxy(decision_panel_raw, route_multiple_params, decision_config),
        on=["company_id", "quarter_idx", "realized_exit_by_horizon", "company_exit_route"],
        how="left",
    )
    return decision_panel, build_decision_backtest(decision_panel, decision_config)


def primary_policy_key(config: dict) -> str:
    threshold = float(config.get("primary_confusion_threshold", 0.02))
    return f"dual_prob_ge_{format_threshold_label(threshold)}_ce_gt_0"


def extract_primary_decision_metrics(decision_backtest: pd.DataFrame, config: dict) -> tuple[float, float]:
    if decision_backtest.empty:
        return np.nan, np.nan
    key = primary_policy_key(config)
    row = decision_backtest.loc[decision_backtest["policy_key"].astype(str).eq(key)]
    if row.empty:
        return np.nan, np.nan
    row = row.iloc[0]
    acceptance_rate = safe_ratio(
        float(row["accepted_observations"]),
        float(row["TP"] + row["FP"] + row["TN"] + row["FN"]),
    )
    return acceptance_rate, float(row["realized_exit_by_8q"]) if pd.notna(row["realized_exit_by_8q"]) else np.nan


def evaluate_feature_model(
    dataset: dict,
    config: dict,
    feature_columns: list[str],
    use_macro_feature: bool,
    use_quarter_fixed_effects: bool,
) -> dict[str, object]:
    search_config = config.copy()
    search_config["feature_columns"] = list(feature_columns)
    search_config["use_macro_feature"] = bool(use_macro_feature)
    search_config["use_quarter_fixed_effects"] = bool(use_quarter_fixed_effects)
    search_config["max_train_rows"] = min(
        int(config.get("max_train_rows", DEFAULT_CONFIG["max_train_rows"])),
        int(config.get("feature_search_max_train_rows", 50000)),
    )
    search_config["optimizer_maxiter"] = max(
        int(config.get("optimizer_maxiter", DEFAULT_CONFIG["optimizer_maxiter"])),
        1000,
    )
    horizon = int(search_config["holdout_horizon_quarters"])
    train_panel = dataset["panel"][dataset["panel"]["split"] == "train"].copy()
    analysis_panels = dataset.get("feature_analysis_panels", {})
    validation_panel = analysis_panels.get("validation", dataset["panel"][dataset["panel"]["split"] == "validation"].copy())
    test_panel = analysis_panels.get("test", dataset["panel"][dataset["panel"]["split"] == "test"].copy())
    fitted = fit_multinomial_hazard(train_panel, search_config)
    route_multiple_params = calibrate_route_multiples(dataset["round_events"], dataset["chosen_exits"])
    validation_scored, _ = score_model_panel(validation_panel, fitted, horizon, dataset["company_master"])
    test_scored, _ = score_model_panel(test_panel, fitted, horizon, dataset["company_master"])
    high_conf_mask = test_scored["entity_match_confidence"].astype(str).eq("high")
    if int(high_conf_mask.sum()) == 0:
        high_conf_mask = test_scored["company_exit_confidence_tier"].isna() | test_scored["company_exit_confidence_tier"].eq("high")
    stress_start_idx = quarter_idx_from_label(str(search_config.get("stress_slice_start_quarter", "2020Q1")))
    stress_end_idx = quarter_idx_from_label(str(search_config.get("stress_slice_end_quarter", "2020Q4")))
    stress_mask = (
        pd.to_numeric(test_scored["quarter_idx"], errors="coerce").ge(stress_start_idx)
        & pd.to_numeric(test_scored["quarter_idx"], errors="coerce").le(stress_end_idx)
    )
    validation_decision_panel, validation_decision_backtest = build_decision_panel_for_split(
        validation_panel,
        fitted,
        route_multiple_params,
        search_config,
    )
    test_decision_panel, test_decision_backtest = build_decision_panel_for_split(
        test_panel,
        fitted,
        route_multiple_params,
        search_config,
    )
    validation_acceptance_rate, validation_realized_exit = extract_primary_decision_metrics(validation_decision_backtest, search_config)
    test_acceptance_rate, test_realized_exit = extract_primary_decision_metrics(test_decision_backtest, search_config)
    return {
        "fitted": fitted,
        "route_multiple_params": route_multiple_params,
        "validation_scored": validation_scored,
        "test_scored": test_scored,
        "validation_metrics": prediction_metrics_snapshot(validation_scored, "pred_exit_by_horizon", "realized_exit_by_horizon", horizon),
        "test_metrics": prediction_metrics_snapshot(test_scored, "pred_exit_by_horizon", "realized_exit_by_horizon", horizon),
        "stress_metrics": prediction_metrics_snapshot(test_scored.loc[stress_mask].copy(), "pred_exit_by_horizon", "realized_exit_by_horizon", horizon),
        "high_confidence_metrics": prediction_metrics_snapshot(
            test_scored.loc[high_conf_mask].copy(),
            "pred_exit_by_horizon",
            "realized_exit_by_horizon",
            horizon,
        ),
        "validation_decision_panel": validation_decision_panel,
        "validation_decision_backtest": validation_decision_backtest,
        "test_decision_panel": test_decision_panel,
        "test_decision_backtest": test_decision_backtest,
        "validation_acceptance_rate": validation_acceptance_rate,
        "validation_realized_exit_rate_accepted": validation_realized_exit,
        "test_acceptance_rate": test_acceptance_rate,
        "test_realized_exit_rate_accepted": test_realized_exit,
    }


def feature_importance_items(panel: pd.DataFrame) -> list[dict[str, object]]:
    items = [
        {"feature_name": "age_q", "feature_group": "company_core", "columns": ["age_q"], "conditional_key": None},
        {"feature_name": "time_since_last_round_q", "feature_group": "financing_trajectory", "columns": ["time_since_last_round_q"], "conditional_key": None},
        {"feature_name": "log_last_round_usd", "feature_group": "financing_trajectory", "columns": ["log_last_round_usd"], "conditional_key": None},
        {"feature_name": "sector_bucket", "feature_group": "sector_stage", "columns": sector_dummy_columns(), "conditional_key": None},
        {"feature_name": "stage_bucket", "feature_group": "sector_stage", "columns": stage_dummy_columns(), "conditional_key": None},
        {"feature_name": "sponsor_score", "feature_group": "sponsor_fund", "columns": ["sponsor_score"], "conditional_key": None},
        {"feature_name": "patent_apps_visible_l4q", "feature_group": "patent_core", "columns": ["patent_apps_visible_l4q"], "conditional_key": "sector_bucket"},
        {"feature_name": "patent_grants_l4q", "feature_group": "patent_core", "columns": ["patent_grants_l4q"], "conditional_key": "sector_bucket"},
        {"feature_name": "patent_stock_visible", "feature_group": "patent_core", "columns": ["patent_stock_visible"], "conditional_key": "sector_bucket"},
        {"feature_name": "market_regime", "feature_group": "macro_time", "columns": ["market_regime"], "conditional_key": None},
    ]
    output = []
    for item in items:
        columns = [column for column in item["columns"] if column in panel.columns]
        if not columns:
            continue
        output.append({**item, "columns": columns})
    return output


def permutation_delta_row(
    item_name: str,
    feature_group: str,
    conditional_key: str | None,
    base_metrics: dict[str, float],
    permuted_metrics: dict[str, float],
    evaluation_target: str,
    data_mode: str,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "data_mode": data_mode,
        "evaluation_target": evaluation_target,
        "feature_name": item_name,
        "feature_group": feature_group,
        "conditional_permutation_key": conditional_key or "",
        "status": status,
        "base_brier_score": base_metrics["brier_score"],
        "base_integrated_brier_score": base_metrics["integrated_brier_score"],
        "base_mean_abs_calibration_gap": base_metrics["mean_abs_calibration_gap"],
        "base_pr_auc": base_metrics["pr_auc"],
        "base_roc_auc": base_metrics["roc_auc"],
        "base_top_decile_realized_exit_rate": base_metrics["top_decile_realized_exit_rate"],
        "base_top_decile_lift": base_metrics["top_decile_lift"],
        "permuted_brier_score": permuted_metrics["brier_score"],
        "permuted_integrated_brier_score": permuted_metrics["integrated_brier_score"],
        "permuted_mean_abs_calibration_gap": permuted_metrics["mean_abs_calibration_gap"],
        "permuted_pr_auc": permuted_metrics["pr_auc"],
        "permuted_roc_auc": permuted_metrics["roc_auc"],
        "permuted_top_decile_realized_exit_rate": permuted_metrics["top_decile_realized_exit_rate"],
        "permuted_top_decile_lift": permuted_metrics["top_decile_lift"],
        "brier_score_delta": positive_help_delta(base_metrics["brier_score"], permuted_metrics["brier_score"], True),
        "integrated_brier_score_delta": positive_help_delta(base_metrics["integrated_brier_score"], permuted_metrics["integrated_brier_score"], True),
        "mean_abs_calibration_gap_delta": positive_help_delta(base_metrics["mean_abs_calibration_gap"], permuted_metrics["mean_abs_calibration_gap"], True),
        "pr_auc_delta": positive_help_delta(base_metrics["pr_auc"], permuted_metrics["pr_auc"], False),
        "roc_auc_delta": positive_help_delta(base_metrics["roc_auc"], permuted_metrics["roc_auc"], False),
        "top_decile_realized_exit_rate_delta": positive_help_delta(
            base_metrics["top_decile_realized_exit_rate"],
            permuted_metrics["top_decile_realized_exit_rate"],
            False,
        ),
        "top_decile_lift_delta": positive_help_delta(base_metrics["top_decile_lift"], permuted_metrics["top_decile_lift"], False),
    }


def build_permutation_importance_exports(
    dataset: dict,
    config: dict,
    full_result: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_panel = dataset["panel"][dataset["panel"]["split"] == "test"].copy()
    if test_panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    horizon = int(config["holdout_horizon_quarters"])
    data_mode = str(config.get("data_mode", "sample"))
    base_metrics = dict(full_result["test_metrics"])
    feature_rows: list[dict[str, object]] = []
    for item_idx, item in enumerate(feature_importance_items(test_panel), start=1):
        permuted_panel = permute_panel_columns(
            test_panel,
            list(item["columns"]),
            int(config["random_seed"]) + 1000 + item_idx,
            str(item["conditional_key"]) if item["conditional_key"] else None,
        )
        permuted_scored, _ = score_model_panel(permuted_panel, full_result["fitted"], horizon, dataset["company_master"])
        permuted_metrics = prediction_metrics_snapshot(
            permuted_scored,
            "pred_exit_by_horizon",
            "realized_exit_by_horizon",
            horizon,
        )
        feature_rows.append(
            permutation_delta_row(
                str(item["feature_name"]),
                str(item["feature_group"]),
                str(item["conditional_key"]) if item["conditional_key"] else None,
                base_metrics,
                permuted_metrics,
                "any_exit_aggregate",
                data_mode,
            )
        )
    group_rows: list[dict[str, object]] = []
    group_columns = feature_group_columns(test_panel)
    for group_idx, group_name in enumerate([*BASELINE_FEATURE_GROUPS, *OPTIONAL_FEATURE_GROUPS, *PLACEHOLDER_FEATURE_GROUPS], start=1):
        columns = list(group_columns.get(group_name, []))
        if group_name == "macro_time":
            columns = ["market_regime"] if "market_regime" in test_panel.columns else []
        if not columns and group_name in PLACEHOLDER_FEATURE_GROUPS:
            group_rows.append(
                permutation_delta_row(
                    group_name,
                    group_name,
                    None,
                    base_metrics,
                    {key: np.nan for key in base_metrics},
                    "any_exit_aggregate",
                    data_mode,
                    status="placeholder",
                )
            )
            continue
        conditional_key = "sector_bucket" if group_name.startswith("patent") else None
        permuted_panel = permute_panel_columns(
            test_panel,
            columns,
            int(config["random_seed"]) + 2000 + group_idx,
            conditional_key,
        )
        permuted_scored, _ = score_model_panel(permuted_panel, full_result["fitted"], horizon, dataset["company_master"])
        permuted_metrics = prediction_metrics_snapshot(
            permuted_scored,
            "pred_exit_by_horizon",
            "realized_exit_by_horizon",
            horizon,
        )
        group_rows.append(
            permutation_delta_row(
                group_name,
                group_name,
                conditional_key,
                base_metrics,
                permuted_metrics,
                "any_exit_aggregate",
                data_mode,
            )
        )
    return pd.DataFrame(feature_rows), pd.DataFrame(group_rows)


def write_feature_importance_notes(output_dir: Path) -> Path:
    lines = [
        "# Feature Importance Notes",
        "",
        "- Positive deltas mean the metric worsened after permutation, so the feature or group was helping before it was shuffled.",
        "- Brier and calibration-gap deltas use the lower-is-better sign convention.",
        "- PR-AUC, ROC-AUC, top-decile realized exit rate, and top-decile lift deltas use the higher-is-better sign convention.",
        "- Patent features use conditional permutation within sector buckets to avoid unrealistic reshuffling across clearly different sectors.",
        "- Quarter fixed effects are handled mainly through ablations; single-feature permutation for the macro-time group only perturbs the explicit market-regime bridge.",
        "",
    ]
    path = output_dir / "feature_importance_notes.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def empty_feature_search_exports() -> dict[str, object]:
    empty_ablation = pd.DataFrame(
        columns=[
            "feature_group",
            "feature_columns_removed",
            "status",
            "brier_score",
            "mean_abs_calibration_gap",
            "pr_auc",
            "roc_auc",
            "top_decile_realized_exit_rate",
            "top_decile_lift",
            "acceptance_rate",
            "realized_exit_rate_accepted",
            "brier_score_delta",
            "mean_abs_calibration_gap_delta",
            "pr_auc_delta",
            "roc_auc_delta",
            "top_decile_realized_exit_rate_delta",
            "top_decile_lift_delta",
        ]
    )
    empty_feature_importance = pd.DataFrame(
        columns=[
            "feature_name",
            "feature_group",
            "conditional_key",
            "status",
            "brier_score_delta",
            "mean_abs_calibration_gap_delta",
            "pr_auc_delta",
            "roc_auc_delta",
            "top_decile_realized_exit_rate_delta",
            "top_decile_lift_delta",
            "evaluation_view",
            "data_mode",
        ]
    )
    empty_combo_validation = pd.DataFrame(
        columns=[
            "combo_key",
            "feature_groups",
            "n_optional_groups",
            "selection_split",
            "validation_brier_score",
            "validation_mean_abs_calibration_gap",
            "validation_pr_auc",
            "validation_roc_auc",
            "validation_top_decile_realized_exit_rate",
            "validation_top_decile_lift",
            "validation_acceptance_rate",
            "validation_realized_exit_rate_accepted",
            "test_brier_score",
            "test_mean_abs_calibration_gap",
            "test_pr_auc",
            "test_roc_auc",
            "test_top_decile_realized_exit_rate",
            "test_top_decile_lift",
            "test_acceptance_rate",
            "test_realized_exit_rate_accepted",
            "procedures_hit",
            "validation_rank",
            "selected_by_validation",
            "ordered_by_validation_rank",
            "validation_selected_feature_groups",
        ]
    )
    return {
        "analysis_panels": {},
        "ablation_exports": {
            "validation": empty_ablation.copy(),
            "test": empty_ablation.copy(),
            "stress": empty_ablation.copy(),
            "high_confidence": empty_ablation.copy(),
        },
        "feature_importance_permutation": empty_feature_importance.copy(),
        "feature_group_importance_permutation": empty_feature_importance.copy(),
        "search_group_columns": {},
        "full_search_feature_columns": [],
        "feature_combo_validation_leaderboard": empty_combo_validation.copy(),
        "feature_combo_test_leaderboard": empty_combo_validation.copy(),
        "feature_combo_pareto_frontier": empty_combo_validation.copy(),
        "chosen_feature_combo_summary": pd.DataFrame(
            columns=["combo_key", "feature_groups", "selected_by_validation", "selection_reason"]
        ),
        "combo_cache": {},
        "sector_feature_importance": pd.DataFrame(
            columns=[
                "bucket_dimension",
                "bucket_name",
                "rows",
                "exits",
                "feature_name",
                "feature_group",
                "status",
                "mean_abs_calibration_gap_delta",
                "pr_auc_delta",
            ]
        ),
        "patent_value_by_sector": pd.DataFrame(
            columns=[
                "sector_bucket",
                "rows",
                "exits",
                "patent_plausible",
                "assessment",
                "mean_abs_calibration_gap_delta",
                "pr_auc_delta",
            ]
        ),
        "sector_combo_challengers": pd.DataFrame(
            columns=[
                "sector_bucket",
                "combo_key",
                "feature_groups",
                "rows",
                "brier_score",
                "mean_abs_calibration_gap",
                "pr_auc",
                "roc_auc",
                "top_decile_lift",
                "best_for_bucket",
            ]
        ),
        "interaction_screen_results": pd.DataFrame(
            columns=[
                "bundle_name",
                "status",
                "feature_columns",
                "validation_brier_score",
                "validation_mean_abs_calibration_gap",
                "validation_pr_auc",
                "validation_roc_auc",
                "validation_top_decile_lift",
                "delta_mean_abs_calibration_gap",
                "delta_brier_score",
                "delta_pr_auc",
                "keep",
            ]
        ),
        "interaction_keep_drop_summary": pd.DataFrame(
            columns=["bundle_name", "status", "keep", "reason"]
        ),
        "top_combo_confusion_summary": pd.DataFrame(columns=["combo_key", "combo_rank", "threshold", "TP", "FP", "TN", "FN"]),
        "top_combo_decision_backtest": pd.DataFrame(columns=["combo_key", "combo_rank", "policy_key"]),
        "top_combo_summary_metrics": pd.DataFrame(
            columns=["scenario", "combo_key", "combo_rank", "mean_npv", "prob_exit_by_horizon"]
        ),
    }


def ablation_row_from_metrics(
    feature_group: str,
    feature_columns_removed: list[str],
    metrics: dict[str, float],
    base_metrics: dict[str, float],
    acceptance_rate: float,
    realized_exit_rate_accepted: float,
    status: str,
) -> dict[str, object]:
    return {
        "feature_group": feature_group,
        "feature_columns_removed": "|".join(feature_columns_removed),
        "status": status,
        "brier_score": metrics["brier_score"],
        "mean_abs_calibration_gap": metrics["mean_abs_calibration_gap"],
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "top_decile_realized_exit_rate": metrics["top_decile_realized_exit_rate"],
        "top_decile_lift": metrics["top_decile_lift"],
        "decision_acceptance_rate": acceptance_rate,
        "decision_realized_exit_rate_accepted": realized_exit_rate_accepted,
        "brier_score_delta": positive_help_delta(base_metrics["brier_score"], metrics["brier_score"], True),
        "mean_abs_calibration_gap_delta": positive_help_delta(base_metrics["mean_abs_calibration_gap"], metrics["mean_abs_calibration_gap"], True),
        "pr_auc_delta": positive_help_delta(base_metrics["pr_auc"], metrics["pr_auc"], False),
        "roc_auc_delta": positive_help_delta(base_metrics["roc_auc"], metrics["roc_auc"], False),
        "top_decile_realized_exit_rate_delta": positive_help_delta(
            base_metrics["top_decile_realized_exit_rate"],
            metrics["top_decile_realized_exit_rate"],
            False,
        ),
        "top_decile_lift_delta": positive_help_delta(base_metrics["top_decile_lift"], metrics["top_decile_lift"], False),
    }


def run_feature_group_ablation(
    dataset: dict,
    config: dict,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]], dict[str, list[str]], list[str]]:
    group_columns = feature_group_columns(dataset["panel"])
    full_feature_columns = list(
        dict.fromkeys(sum([group_columns[group] for group in [*BASELINE_FEATURE_GROUPS, *OPTIONAL_FEATURE_GROUPS]], []))
    )
    full_result = evaluate_feature_model(
        dataset,
        config,
        full_feature_columns,
        use_macro_feature=True,
        use_quarter_fixed_effects=bool(config.get("use_quarter_fixed_effects", False)),
    )
    results_by_group: dict[str, dict[str, object]] = {"full_active": full_result}
    output_frames = {"validation": [], "test": [], "stress": [], "high_confidence": []}
    for group_name in [*BASELINE_FEATURE_GROUPS, *OPTIONAL_FEATURE_GROUPS, *PLACEHOLDER_FEATURE_GROUPS]:
        columns = list(group_columns.get(group_name, []))
        if group_name in PLACEHOLDER_FEATURE_GROUPS:
            result = None
            status = "placeholder"
        else:
            ablated_feature_columns = [column for column in full_feature_columns if column not in columns]
            use_macro_feature = False if group_name == "macro_time" else True
            use_qfe = False if group_name == "macro_time" else bool(config.get("use_quarter_fixed_effects", False))
            result = evaluate_feature_model(
                dataset,
                config,
                ablated_feature_columns,
                use_macro_feature=use_macro_feature,
                use_quarter_fixed_effects=use_qfe,
            )
            results_by_group[group_name] = result
            status = "ok"
        for slice_name, metrics_key, accept_rate_key, exit_rate_key in [
            ("validation", "validation_metrics", "validation_acceptance_rate", "validation_realized_exit_rate_accepted"),
            ("test", "test_metrics", "test_acceptance_rate", "test_realized_exit_rate_accepted"),
            ("stress", "stress_metrics", None, None),
            ("high_confidence", "high_confidence_metrics", None, None),
        ]:
            metrics = result[metrics_key] if result is not None else {key: np.nan for key in full_result["test_metrics"]}
            output_frames[slice_name].append(
                ablation_row_from_metrics(
                    group_name,
                    columns,
                    metrics,
                    full_result[metrics_key] if metrics_key in full_result else full_result["test_metrics"],
                    result[accept_rate_key] if (result is not None and accept_rate_key) else np.nan,
                    result[exit_rate_key] if (result is not None and exit_rate_key) else np.nan,
                    status,
                )
            )
    return (
        {key: pd.DataFrame(value) for key, value in output_frames.items()},
        results_by_group,
        group_columns,
        full_feature_columns,
    )


def combo_key_from_groups(groups: list[str]) -> str:
    ordered = [group for group in [*BASELINE_FEATURE_GROUPS, *OPTIONAL_FEATURE_GROUPS] if group in groups]
    return "+".join(ordered)


def compute_pareto_frontier(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in leaderboard.itertuples(index=False):
        dominated = False
        for other in leaderboard.itertuples(index=False):
            if other.combo_key == row.combo_key:
                continue
            if (
                float(other.validation_mean_abs_calibration_gap) <= float(row.validation_mean_abs_calibration_gap)
                and float(other.validation_brier_score) <= float(row.validation_brier_score)
                and float(other.validation_pr_auc) >= float(row.validation_pr_auc)
                and (
                    float(other.validation_mean_abs_calibration_gap) < float(row.validation_mean_abs_calibration_gap)
                    or float(other.validation_brier_score) < float(row.validation_brier_score)
                    or float(other.validation_pr_auc) > float(row.validation_pr_auc)
                )
            ):
                dominated = True
                break
        rows.append({"combo_key": row.combo_key, "pareto_frontier": int(not dominated)})
    return pd.DataFrame(rows)


def run_feature_combo_search(
    dataset: dict,
    config: dict,
    group_columns: dict[str, list[str]],
    precomputed_cache: dict[str, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    optional_groups = [group for group in OPTIONAL_FEATURE_GROUPS if bool(group_columns.get(group))]
    combo_cache: dict[str, dict[str, object]] = dict(precomputed_cache or {})
    candidate_rows = []
    procedures: dict[str, set[str]] = {}
    exhaustive_group_sets = [
        list(BASELINE_FEATURE_GROUPS) + list(optional_subset)
        for r in range(len(optional_groups) + 1)
        for optional_subset in combinations(optional_groups, r)
    ]
    for groups in exhaustive_group_sets:
        combo_key = combo_key_from_groups(groups)
        feature_columns = list(dict.fromkeys(sum([group_columns[group] for group in groups if group != "macro_time"], [])))
        if combo_key not in combo_cache:
            result = evaluate_feature_model(
                dataset,
                config,
                feature_columns,
                use_macro_feature=True,
                use_quarter_fixed_effects=bool(config.get("use_quarter_fixed_effects", False)),
            )
            combo_cache[combo_key] = {**result, "feature_columns": feature_columns, "feature_groups": groups}
        else:
            combo_cache[combo_key] = {
                **combo_cache[combo_key],
                "feature_columns": feature_columns,
                "feature_groups": groups,
            }
        result = combo_cache[combo_key]
        procedures.setdefault(combo_key, set()).add("exhaustive")
        candidate_rows.append(
            {
                "combo_key": combo_key,
                "feature_groups": "|".join(groups),
                "n_optional_groups": int(sum(group in OPTIONAL_FEATURE_GROUPS for group in groups)),
                "selection_split": "validation",
                "validation_brier_score": result["validation_metrics"]["brier_score"],
                "validation_mean_abs_calibration_gap": result["validation_metrics"]["mean_abs_calibration_gap"],
                "validation_pr_auc": result["validation_metrics"]["pr_auc"],
                "validation_roc_auc": result["validation_metrics"]["roc_auc"],
                "validation_top_decile_realized_exit_rate": result["validation_metrics"]["top_decile_realized_exit_rate"],
                "validation_top_decile_lift": result["validation_metrics"]["top_decile_lift"],
                "validation_acceptance_rate": result["validation_acceptance_rate"],
                "validation_realized_exit_rate_accepted": result["validation_realized_exit_rate_accepted"],
                "test_brier_score": result["test_metrics"]["brier_score"],
                "test_mean_abs_calibration_gap": result["test_metrics"]["mean_abs_calibration_gap"],
                "test_pr_auc": result["test_metrics"]["pr_auc"],
                "test_roc_auc": result["test_metrics"]["roc_auc"],
                "test_top_decile_realized_exit_rate": result["test_metrics"]["top_decile_realized_exit_rate"],
                "test_top_decile_lift": result["test_metrics"]["top_decile_lift"],
                "test_acceptance_rate": result["test_acceptance_rate"],
                "test_realized_exit_rate_accepted": result["test_realized_exit_rate_accepted"],
            }
        )
    current_groups = list(BASELINE_FEATURE_GROUPS)
    for _ in optional_groups:
        candidates = [current_groups + [group] for group in optional_groups if group not in current_groups]
        if not candidates:
            break
        ranked = sorted(
            [combo_key_from_groups(groups) for groups in candidates],
            key=lambda key: next(
                (
                    float(row["validation_mean_abs_calibration_gap"]),
                    float(row["validation_brier_score"]),
                    -float(row["validation_pr_auc"]) if np.isfinite(row["validation_pr_auc"]) else 1.0,
                    -float(row["validation_top_decile_lift"]) if np.isfinite(row["validation_top_decile_lift"]) else 1.0,
                    -float(row["validation_roc_auc"]) if np.isfinite(row["validation_roc_auc"]) else 1.0,
                    int(row["n_optional_groups"]),
                    str(row["combo_key"]),
                )
                for row in candidate_rows
                if row["combo_key"] == key
            ),
        )
        if not ranked:
            break
        current_groups = combo_cache[ranked[0]]["feature_groups"]
        procedures.setdefault(ranked[0], set()).add("forward_selection")
    current_groups = list(BASELINE_FEATURE_GROUPS) + optional_groups
    if optional_groups:
        start_key = combo_key_from_groups(current_groups)
        procedures.setdefault(start_key, set()).add("backward_elimination")
        while len([group for group in current_groups if group in OPTIONAL_FEATURE_GROUPS]) > 0:
            candidates = [[value for value in current_groups if value != group] for group in current_groups if group in OPTIONAL_FEATURE_GROUPS]
            ranked = sorted(
                [combo_key_from_groups(groups) for groups in candidates],
                key=lambda key: next(
                    (
                        float(row["validation_mean_abs_calibration_gap"]),
                        float(row["validation_brier_score"]),
                        -float(row["validation_pr_auc"]) if np.isfinite(row["validation_pr_auc"]) else 1.0,
                        -float(row["validation_top_decile_lift"]) if np.isfinite(row["validation_top_decile_lift"]) else 1.0,
                        -float(row["validation_roc_auc"]) if np.isfinite(row["validation_roc_auc"]) else 1.0,
                        int(row["n_optional_groups"]),
                        str(row["combo_key"]),
                    )
                    for row in candidate_rows
                    if row["combo_key"] == key
                ),
            )
            if not ranked:
                break
            current_groups = combo_cache[ranked[0]]["feature_groups"]
            procedures.setdefault(ranked[0], set()).add("backward_elimination")
            if len([group for group in current_groups if group in OPTIONAL_FEATURE_GROUPS]) == 0:
                break
    leaderboard = pd.DataFrame(candidate_rows)
    leaderboard["procedures_hit"] = leaderboard["combo_key"].map(lambda value: "|".join(sorted(procedures.get(value, {"exhaustive"}))))
    leaderboard = leaderboard.sort_values(
        by=[
            "validation_mean_abs_calibration_gap",
            "validation_brier_score",
            "validation_pr_auc",
            "validation_top_decile_lift",
            "validation_roc_auc",
            "n_optional_groups",
            "combo_key",
        ],
        ascending=[True, True, False, False, False, True, True],
    ).reset_index(drop=True)
    leaderboard["validation_rank"] = np.arange(1, len(leaderboard) + 1)
    leaderboard["selected_by_validation"] = (leaderboard["validation_rank"] == 1).astype(int)
    test_leaderboard = leaderboard[
        [
            "combo_key",
            "feature_groups",
            "n_optional_groups",
            "procedures_hit",
            "validation_rank",
            "selected_by_validation",
            "test_brier_score",
            "test_mean_abs_calibration_gap",
            "test_pr_auc",
            "test_roc_auc",
            "test_top_decile_realized_exit_rate",
            "test_top_decile_lift",
            "test_acceptance_rate",
            "test_realized_exit_rate_accepted",
        ]
    ].copy()
    test_leaderboard["ordered_by_validation_rank"] = 1
    pareto = compute_pareto_frontier(leaderboard)
    chosen = leaderboard.iloc[[0]].copy()
    chosen["selection_rationale"] = "validation_calibration_first_lexicographic"
    return leaderboard, test_leaderboard, pareto, chosen, combo_cache


def build_sector_feature_importance(
    dataset: dict,
    config: dict,
    full_result: dict[str, object],
    group_columns: dict[str, list[str]],
    sector_stage_support: pd.DataFrame,
) -> pd.DataFrame:
    horizon = int(config["holdout_horizon_quarters"])
    rows: list[dict[str, object]] = []
    test_panel = dataset["panel"][dataset["panel"]["split"] == "test"].copy()
    support_rows = sector_stage_support[
        (sector_stage_support["split"].astype(str) == "test")
        & (sector_stage_support["supported_for_bucket_analysis"].astype(int) == 1)
    ].copy()
    for support_row in support_rows.itertuples(index=False):
        column_name = "sector_bucket" if support_row.bucket_dimension == "sector" else "stage_bucket"
        subset_panel = test_panel[test_panel[column_name].astype(str).eq(str(support_row.bucket_name))].copy()
        if subset_panel.empty:
            continue
        base_scored, _ = score_model_panel(subset_panel, full_result["fitted"], horizon, dataset["company_master"])
        base_metrics = prediction_metrics_snapshot(base_scored, "pred_exit_by_horizon", "realized_exit_by_horizon", horizon)
        for group_name in ["company_core", "financing_trajectory", "sponsor_fund", "patent_core"]:
            columns = list(group_columns.get(group_name, []))
            if not columns:
                rows.append(
                    {
                        "bucket_dimension": support_row.bucket_dimension,
                        "bucket_name": support_row.bucket_name,
                        "feature_group": group_name,
                        "status": "placeholder",
                    }
                )
                continue
            permuted_panel = permute_panel_columns(
                subset_panel,
                columns,
                int(config["random_seed"]) + len(rows) + 5000,
                "sector_bucket" if group_name.startswith("patent") and support_row.bucket_dimension == "stage" else None,
            )
            permuted_scored, _ = score_model_panel(permuted_panel, full_result["fitted"], horizon, dataset["company_master"])
            permuted_metrics = prediction_metrics_snapshot(permuted_scored, "pred_exit_by_horizon", "realized_exit_by_horizon", horizon)
            row = permutation_delta_row(
                group_name,
                group_name,
                "sector_bucket" if group_name.startswith("patent") and support_row.bucket_dimension == "stage" else None,
                base_metrics,
                permuted_metrics,
                "any_exit_aggregate",
                str(config.get("data_mode", "sample")),
            )
            row.update(
                {
                    "bucket_dimension": support_row.bucket_dimension,
                    "bucket_name": support_row.bucket_name,
                    "rows": int(support_row.rows),
                    "exits": int(support_row.exits),
                }
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "bucket_dimension",
                "bucket_name",
                "rows",
                "exits",
                "feature_name",
                "feature_group",
                "status",
                "mean_abs_calibration_gap_delta",
                "pr_auc_delta",
            ]
        )
    return pd.DataFrame(rows)


def build_patent_value_by_sector(sector_feature_importance: pd.DataFrame) -> pd.DataFrame:
    if sector_feature_importance.empty:
        return pd.DataFrame(
            columns=["sector_bucket", "rows", "exits", "patent_plausible", "assessment", "mean_abs_calibration_gap_delta", "pr_auc_delta"]
        )
    patent_rows = sector_feature_importance[
        (sector_feature_importance["bucket_dimension"].astype(str) == "sector")
        & (sector_feature_importance["feature_group"].astype(str) == "patent_core")
    ].copy()
    if patent_rows.empty:
        return pd.DataFrame(
            columns=["sector_bucket", "rows", "exits", "patent_plausible", "assessment", "mean_abs_calibration_gap_delta", "pr_auc_delta"]
        )
    patent_rows["patent_plausible"] = patent_rows["bucket_name"].isin(PATENT_PLAUSIBLE_BUCKETS).astype(int)
    patent_rows["assessment"] = np.where(
        patent_rows["mean_abs_calibration_gap_delta"].fillna(0.0).gt(0)
        & patent_rows["pr_auc_delta"].fillna(0.0).gt(0),
        "positive",
        np.where(
            patent_rows["mean_abs_calibration_gap_delta"].fillna(0.0).abs().lt(1e-4)
            & patent_rows["pr_auc_delta"].fillna(0.0).abs().lt(1e-4),
            "neutral",
            "weak_or_negative",
        ),
    )
    return patent_rows.rename(columns={"bucket_name": "sector_bucket"})[
        ["sector_bucket", "rows", "exits", "patent_plausible", "assessment", "mean_abs_calibration_gap_delta", "pr_auc_delta"]
    ].sort_values(["patent_plausible", "mean_abs_calibration_gap_delta"], ascending=[False, False]).reset_index(drop=True)


def build_sector_combo_challengers(
    combo_cache: dict[str, dict[str, object]],
    sector_stage_support: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    supported = sector_stage_support[
        (sector_stage_support["bucket_dimension"].astype(str) == "sector")
        & (sector_stage_support["split"].astype(str) == "validation")
        & (sector_stage_support["supported_for_bucket_analysis"].astype(int) == 1)
    ].copy()
    for support_row in supported.itertuples(index=False):
        for combo_key, payload in combo_cache.items():
            validation_scored = payload["validation_scored"]
            subset = validation_scored[validation_scored["sector_bucket"].astype(str).eq(str(support_row.bucket_name))].copy()
            if subset.empty:
                continue
            metrics = prediction_metrics_snapshot(subset, "pred_exit_by_horizon", "realized_exit_by_horizon", 8)
            rows.append(
                {
                    "sector_bucket": support_row.bucket_name,
                    "combo_key": combo_key,
                    "feature_groups": "|".join(payload["feature_groups"]),
                    "rows": int(len(subset)),
                    "brier_score": metrics["brier_score"],
                    "mean_abs_calibration_gap": metrics["mean_abs_calibration_gap"],
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "top_decile_lift": metrics["top_decile_lift"],
                }
            )
    if not rows:
        return pd.DataFrame()
    output = pd.DataFrame(rows)
    output["best_for_bucket"] = 0
    for _, bucket_frame in output.groupby("sector_bucket", sort=True):
        ranked = bucket_frame.sort_values(
            ["mean_abs_calibration_gap", "brier_score", "pr_auc", "top_decile_lift", "roc_auc", "combo_key"],
            ascending=[True, True, False, False, False, True],
        )
        output.loc[ranked.index[0], "best_for_bucket"] = 1
    return output.sort_values(["sector_bucket", "best_for_bucket", "mean_abs_calibration_gap"], ascending=[True, False, True]).reset_index(drop=True)


def run_interaction_screen(
    dataset: dict,
    config: dict,
    chosen_combo_summary: pd.DataFrame,
    combo_cache: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if chosen_combo_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    base_key = str(chosen_combo_summary["combo_key"].iloc[0])
    base_payload = combo_cache[base_key]
    base_columns = list(base_payload["feature_columns"])
    bundles = interaction_bundle_columns(dataset["panel"])
    rows = []
    summary_rows = []
    for bundle_name, bundle_columns in bundles.items():
        if not bundle_columns or float(dataset["panel"][bundle_columns].abs().sum().sum()) == 0.0:
            rows.append({"bundle_name": bundle_name, "status": "unsupported", "keep": 0})
            summary_rows.append({"bundle_name": bundle_name, "status": "unsupported", "keep": 0, "reason": "all_zero_or_missing"})
            continue
        payload = evaluate_feature_model(
            dataset,
            config,
            list(dict.fromkeys(base_columns + bundle_columns)),
            use_macro_feature=True,
            use_quarter_fixed_effects=bool(config.get("use_quarter_fixed_effects", False)),
        )
        metrics = payload["validation_metrics"]
        keep = int(
            np.isfinite(metrics["mean_abs_calibration_gap"])
            and metrics["mean_abs_calibration_gap"] <= base_payload["validation_metrics"]["mean_abs_calibration_gap"]
            and (
                metrics["pr_auc"] >= base_payload["validation_metrics"]["pr_auc"]
                or metrics["brier_score"] < base_payload["validation_metrics"]["brier_score"]
            )
        )
        rows.append(
            {
                "bundle_name": bundle_name,
                "status": "ok",
                "feature_columns": "|".join(bundle_columns),
                "validation_brier_score": metrics["brier_score"],
                "validation_mean_abs_calibration_gap": metrics["mean_abs_calibration_gap"],
                "validation_pr_auc": metrics["pr_auc"],
                "validation_roc_auc": metrics["roc_auc"],
                "validation_top_decile_lift": metrics["top_decile_lift"],
                "delta_mean_abs_calibration_gap": positive_help_delta(
                    base_payload["validation_metrics"]["mean_abs_calibration_gap"],
                    metrics["mean_abs_calibration_gap"],
                    True,
                ),
                "delta_brier_score": positive_help_delta(
                    base_payload["validation_metrics"]["brier_score"],
                    metrics["brier_score"],
                    True,
                ),
                "delta_pr_auc": positive_help_delta(
                    base_payload["validation_metrics"]["pr_auc"],
                    metrics["pr_auc"],
                    False,
                ),
                "keep": keep,
            }
        )
        summary_rows.append(
            {
                "bundle_name": bundle_name,
                "status": "ok",
                "keep": keep,
                "reason": "validation_calibration_first_improvement" if keep else "no_validation_calibration_first_gain",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def build_top_combo_economic_diagnostics(
    dataset: dict,
    config: dict,
    top_combo_keys: list[str],
    combo_cache: dict[str, dict[str, object]],
    reference_display_selection: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    confusion_rows = []
    decision_rows = []
    summary_rows = []
    display_panel = reference_display_selection["selected_panel"].copy()
    if display_panel.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for rank, combo_key in enumerate(top_combo_keys, start=1):
        payload = combo_cache[combo_key]
        _, exit_confusion_summary = build_binary_confusion_exports(
            payload["test_scored"],
            prediction_col="pred_exit_by_horizon",
            actual_col="realized_exit_by_horizon",
            thresholds=resolve_probability_thresholds(config),
            data_mode=str(config.get("data_mode", "sample")),
            evaluation_view="top_combo_test",
            target_label="realized_exit_by_8q",
            prediction_label="predicted_prob_exit_by_8q",
            join_key="combo_key",
            join_value=combo_key,
        )
        exit_confusion_summary["combo_rank"] = rank
        confusion_rows.append(exit_confusion_summary)
        top_decision = payload["test_decision_backtest"].copy()
        if not top_decision.empty:
            top_decision["combo_key"] = combo_key
            top_decision["combo_rank"] = rank
            decision_rows.append(top_decision)
        combo_selection = {
            "display_mode": reference_display_selection["display_mode"],
            "display_label": reference_display_selection["display_label"],
            "selected_panel": display_panel,
            "selected_row": reference_display_selection["selected_row"],
            "audit": reference_display_selection["audit"],
        }
        _, _, summary_metrics = build_display_outputs(
            combo_selection,
            payload["fitted"],
            payload["route_multiple_params"],
            config,
        )
        summary_metrics["combo_key"] = combo_key
        summary_metrics["combo_rank"] = rank
        summary_rows.append(summary_metrics)
    return (
        pd.concat(confusion_rows, ignore_index=True) if confusion_rows else pd.DataFrame(),
        pd.concat(decision_rows, ignore_index=True) if decision_rows else pd.DataFrame(),
        pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame(),
    )


def aggregate_incidence_from_point_matrix(point_route_matrix: np.ndarray) -> pd.DataFrame:
    if point_route_matrix.size == 0:
        return pd.DataFrame(columns=["horizon_q", *[f"cum_{route}" for route in EXIT_ROUTES], "prob_exit_by_horizon", "survival"])
    mean_point = point_route_matrix.mean(axis=0)
    cumulative = np.cumsum(mean_point, axis=0)
    rows = []
    for horizon_idx in range(cumulative.shape[0]):
        row = {"horizon_q": horizon_idx + 1}
        total_exit = 0.0
        for route_idx, route in enumerate(EXIT_ROUTES):
            value = float(cumulative[horizon_idx, route_idx])
            row[f"cum_{route}"] = value
            total_exit += value
        row["prob_exit_by_horizon"] = total_exit
        row["survival"] = 1.0 - total_exit
        rows.append(row)
    return pd.DataFrame(rows)


def build_display_outputs(
    display_selection: dict,
    fitted: dict,
    multiple_params: dict[str, dict[str, float]],
    config: dict,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    display_panel = display_selection["selected_panel"].copy()
    if display_panel.empty:
        raise ValueError("Display selection produced an empty panel.")
    horizon = int(config["holdout_horizon_quarters"])
    incidence_map: dict[str, pd.DataFrame] = {}
    npv_map: dict[str, pd.DataFrame] = {}
    metric_rows = []
    paths_per_company = (
        int(config["n_simulations"])
        if len(display_panel) == 1
        else max(int(config["n_simulations"]) // max(len(display_panel), 1), 32)
    )
    for scenario_name in ("baseline", "exit_freeze"):
        summary, point_matrix = probability_path_summary_vectorized(
            display_panel,
            fitted,
            horizon,
            config,
            scenario_name,
        )
        incidence = aggregate_incidence_from_point_matrix(point_matrix)
        incidence_map[scenario_name] = incidence
        scenario_paths = []
        for row in display_panel.itertuples(index=False):
            future_states = build_future_states(
                pd.Series(row._asdict()),
                horizon,
                config,
                scenario_name,
                feature_columns=fitted.get("model_state", {}).get("feature_columns"),
            )
            future_probs = predict_route_probs(future_states.assign(route_label="no_exit"), fitted)
            path_config = config.copy()
            path_config["n_simulations"] = paths_per_company
            scenario_paths.append(simulate_npv(pd.Series(row._asdict()), future_probs, multiple_params, path_config, scenario_name))
        npv_map[scenario_name] = pd.concat(scenario_paths, ignore_index=True)
        metrics = decision_metrics(npv_map[scenario_name], incidence, config)
        metrics["company_id"] = str(display_selection["selected_row"]["company_id"])
        metrics["company_name"] = str(display_selection["display_label"])
        metrics["display_mode"] = str(display_selection["display_mode"])
        metric_rows.append(metrics)
    return incidence_map, npv_map, pd.concat(metric_rows, ignore_index=True)


def build_feature_placeholder_status() -> pd.DataFrame:
    return pd.DataFrame(DEFERRED_FEATURE_BLOCKS)


def build_run_metadata(
    config: dict,
    dataset: dict,
    fitted: dict,
    stylized: pd.Series,
) -> pd.DataFrame:
    partition = dataset["partition_summary"].set_index("split")

    def metric(split: str, column: str) -> int:
        if split not in partition.index:
            return 0
        value = partition.loc[split, column]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return int(value)

    return pd.DataFrame(
        [
            {
                "data_mode": str(config["data_mode"]),
                "pack_label": str(config["pack_label"]),
                "macro_spec": (
                    "quarter_fixed_effects_neutral_bridge"
                    if bool(config.get("use_quarter_fixed_effects", False))
                    else "explicit_market_regime"
                ),
                "selected_min_entry_year": int(dataset["selected_min_entry_year"]),
                "panel_rows": int(len(dataset["panel"])),
                "train_rows": metric("train", "rows"),
                "validation_rows": metric("validation", "rows"),
                "test_rows": metric("test", "rows"),
                "train_exits": metric("train", "exits"),
                "validation_exits": metric("validation", "exits"),
                "test_exits": metric("test", "exits"),
                "matched_patent_companies": int(dataset["patent_matches"]["company_id"].nunique())
                if not dataset["patent_matches"].empty
                else 0,
                "company_chunk_size": int(config["company_chunk_size"]),
                "max_train_rows": int(config["max_train_rows"]),
                "min_train_route_support": int(config.get("min_train_route_support", 0)),
                "use_quarter_fixed_effects": int(bool(config.get("use_quarter_fixed_effects", False))),
                "n_simulations": int(config["n_simulations"]),
                "optimization_iterations": int(fitted["optimization_iterations"]),
                "optimization_message": str(fitted["optimization_message"]),
                "stylized_company_id": str(stylized["company_id"]),
                "stylized_company_name": str(stylized["company_name"]),
                "stylized_quarter": quarter_label_from_idx(int(stylized["quarter_idx"])),
            }
        ]
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_chapter_summary(
    output_dir: Path,
    run_metadata: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    route_audit: pd.DataFrame,
    placeholders: pd.DataFrame,
) -> Path:
    metadata_row = run_metadata.iloc[0]
    data_mode = str(metadata_row.get("data_mode", "sample")).strip().lower()
    mode_label = "Sample" if data_mode == "sample" else "Live"
    baseline = summary_metrics[summary_metrics["scenario"] == "baseline"].iloc[0]
    freeze = summary_metrics[summary_metrics["scenario"] == "exit_freeze"].iloc[0]
    lines = [
        f"# Chapter 9 {mode_label} Run Summary",
        "",
        "## Cohort",
        "",
        f"- Selected minimum entry year: {int(metadata_row['selected_min_entry_year'])}",
        f"- Selected train / validation / test end quarters: {metadata_row.get('selected_train_end_quarter', '')} / {metadata_row.get('selected_validation_end_quarter', '')} / {metadata_row.get('selected_test_end_quarter', '')}",
        f"- Panel rows: {int(metadata_row['panel_rows'])}",
        f"- Train / validation / test rows: {int(metadata_row['train_rows'])} / {int(metadata_row['validation_rows'])} / {int(metadata_row['test_rows'])}",
        f"- Train / validation / test exits: {int(metadata_row['train_exits'])} / {int(metadata_row['validation_exits'])} / {int(metadata_row['test_exits'])}",
        f"- Route-pooling fallback used: {bool(metadata_row.get('used_route_pooling_fallback', False))}",
        f"- Chapter evidence ready: {bool(metadata_row.get('chapter_evidence_ready', False))}",
        f"- Primary confusion threshold: {float(metadata_row.get('primary_confusion_threshold', 0.02)):.2f}",
        "",
        "## Display View",
        "",
        f"- Company: {metadata_row['stylized_company_name']} ({metadata_row['stylized_company_id']})",
        f"- Quarter: {metadata_row['stylized_quarter']}",
        f"- Display mode: {metadata_row.get('display_mode', 'single_company')}",
        f"- Display label: {metadata_row.get('display_label', metadata_row['stylized_company_name'])}",
        "",
        "## Scenario Metrics",
        "",
        f"- Baseline mean NPV: {float(baseline['mean_npv']):.4f}",
        f"- Baseline exit-by-horizon probability: {float(baseline['prob_exit_by_horizon']):.4f}",
        f"- Exit-freeze mean NPV: {float(freeze['mean_npv']):.4f}",
        f"- Exit-freeze exit-by-horizon probability: {float(freeze['prob_exit_by_horizon']):.4f}",
        "- Confusion matrices in this folder are threshold-dependent supplements; calibration remains the chapter headline diagnostic.",
        "",
        "## Route Audit",
        "",
    ]
    for row in route_audit.itertuples(index=False):
        lines.append(
            f"- {row.route_label} / {row.confidence_tier} / {row.route_source}: candidates={int(row.candidate_count)}, chosen={int(row.chosen_exit_count)}"
        )
    lines.extend(
        [
            "",
            "## Deferred Feature Blocks",
            "",
        ]
    )
    for row in placeholders.itertuples(index=False):
        lines.append(f"- {row.feature_name}: {row.status}; {row.dependency_note}")
    path = output_dir / "chapter_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def dataframe_to_markdown(frame: pd.DataFrame, decimals: int = 4) -> str:
    table = frame.copy()
    for column in table.columns:
        if isinstance(table[column].dtype, CategoricalDtype):
            table[column] = table[column].astype(object)
    numeric_columns = table.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{decimals}f}"
            )
        else:
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else str(int(value)))
    table = table.fillna("")
    columns = list(table.columns)
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in table.iterrows()
    ]
    return "\n".join([header, rule, *rows])


def build_target_recommendation_table(
    registry: pd.DataFrame,
    evaluation_metrics_targets: pd.DataFrame,
    decision_backtest_targets: pd.DataFrame,
    route_support_targets: pd.DataFrame,
    source_mix_targets: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    full_test = evaluation_metrics_targets.loc[
        evaluation_metrics_targets["evaluation_view"].astype(str).eq("full_test")
    ].copy()
    high_conf = (
        evaluation_metrics_targets.loc[
            evaluation_metrics_targets["evaluation_view"].astype(str).eq("high_confidence_subset"),
            ["target_key", "mean_abs_calibration_gap"],
        ]
        .rename(columns={"mean_abs_calibration_gap": "high_confidence_mean_abs_calibration_gap"})
        .drop_duplicates("target_key")
    )
    selected_policy = decision_backtest_targets.loc[
        decision_backtest_targets["evaluation_split"].astype(str).eq("test")
        & decision_backtest_targets["selected_on_validation"].astype(int).eq(1)
    ].copy()
    if selected_policy.empty:
        selected_policy = pd.DataFrame(
            columns=[
                "target_key",
                "policy_key",
                "policy_family",
                "acceptance_rate",
                "hit_rate_accepted",
                "precision",
                "recall",
                "degenerate_rule",
                "predicted_mean_npv_proxy",
                "realized_mean_npv_proxy",
            ]
        )
    selected_policy = (
        selected_policy[
            [
                "target_key",
                "policy_key",
                "policy_family",
                "acceptance_rate",
                "hit_rate_accepted",
                "precision",
                "recall",
                "degenerate_rule",
                "predicted_mean_npv_proxy",
                "realized_mean_npv_proxy",
            ]
        ]
        .rename(
            columns={
                "policy_key": "selected_policy_key",
                "policy_family": "selected_policy_family",
                "acceptance_rate": "selected_policy_acceptance_rate",
                "hit_rate_accepted": "selected_policy_hit_rate",
                "precision": "selected_policy_precision",
                "recall": "selected_policy_recall",
                "degenerate_rule": "selected_policy_degenerate",
            }
        )
        .drop_duplicates("target_key")
    )
    train_support = (
        route_support_targets.loc[route_support_targets["split"].astype(str).eq("train")]
        .groupby("target_key", as_index=False)
        .agg(
            train_positive_events=("positive_event_count", "sum"),
            train_positive_companies=("positive_companies", "sum"),
            ipo_support_adequate_for_standalone=("ipo_support_adequate_for_standalone", "max"),
            partial_realizations_data_supported=("partial_realizations_data_supported", "max"),
        )
    )
    source_rows = []
    for target_key, frame in source_mix_targets.groupby("target_key", observed=True):
        total = int(pd.to_numeric(frame["positive_event_count"], errors="coerce").fillna(0).sum())
        direct = int(
            pd.to_numeric(
                frame.loc[frame["event_observation_kind"].astype(str).eq("direct_dated_event"), "positive_event_count"],
                errors="coerce",
            ).fillna(0).sum()
        )
        inferred = int(
            pd.to_numeric(
                frame.loc[frame["event_observation_kind"].astype(str).eq("inferred_transition"), "positive_event_count"],
                errors="coerce",
            ).fillna(0).sum()
        )
        synthetic = int(
            pd.to_numeric(
                frame.loc[frame["event_observation_kind"].astype(str).eq("synthetic_dated_event"), "positive_event_count"],
                errors="coerce",
            ).fillna(0).sum()
        )
        confidence_good = int(
            pd.to_numeric(
                frame.loc[
                    frame["confidence_tier"].astype(str).isin(["high", "medium", "synthetic"]),
                    "positive_event_count",
                ],
                errors="coerce",
            ).fillna(0).sum()
        )
        source_rows.append(
            {
                "target_key": str(target_key),
                "positive_event_total": total,
                "direct_dated_share": safe_ratio(direct, total),
                "inferred_transition_share": safe_ratio(inferred, total),
                "synthetic_dated_share": safe_ratio(synthetic, total),
                "label_confidence_share": safe_ratio(confidence_good, total),
            }
        )
    source_summary = pd.DataFrame(source_rows)
    table = registry.merge(
        full_test,
        on=["target_key", "target_name", "universe", "candidate_role", "benchmark_row", "data_supported"],
        how="left",
    )
    table = table.merge(high_conf, on="target_key", how="left")
    table = table.merge(selected_policy, on="target_key", how="left")
    table = table.merge(train_support, on="target_key", how="left")
    table = table.merge(source_summary, on="target_key", how="left")
    table["selected_policy_degenerate"] = pd.to_numeric(table["selected_policy_degenerate"], errors="coerce").fillna(1).astype(int)
    table["train_positive_events"] = pd.to_numeric(table["train_positive_events"], errors="coerce").fillna(0).astype(int)
    table["train_positive_companies"] = pd.to_numeric(table["train_positive_companies"], errors="coerce").fillna(0).astype(int)
    table["recommendation_rank"] = np.nan
    table["recommended_for_universe"] = 0
    table["chapter_reporting_status"] = "comparison_only"
    table["headline_target_group"] = "separate_by_universe"
    candidate_priority = {
        "locked_baseline": 0,
        "candidate": 1,
        "optional_candidate": 2,
        "benchmark_only": 3,
        "candidate_unsupported": 4,
    }
    table["_candidate_priority"] = table["candidate_role"].map(candidate_priority).fillna(9).astype(int)
    table["_supported_flag"] = (
        table["data_supported"].astype(int).eq(1)
        & table["estimation_status"].astype(str).eq("estimated")
    ).astype(int)
    table["_calibration_rank"] = pd.to_numeric(table["mean_abs_calibration_gap"], errors="coerce").fillna(np.inf)
    table["_policy_hit_rank"] = pd.to_numeric(table["selected_policy_hit_rate"], errors="coerce").fillna(-1.0)
    table["_label_conf_rank"] = pd.to_numeric(table["label_confidence_share"], errors="coerce").fillna(-1.0)
    table["_policy_precision_rank"] = pd.to_numeric(table["selected_policy_precision"], errors="coerce").fillna(-1.0)
    table["_buyout_provisional"] = (
        table["universe"].astype(str).eq("buyout_pe")
        & (
            table["_supported_flag"].eq(0)
            | table["selected_policy_degenerate"].astype(int).eq(1)
            | pd.to_numeric(table["mean_abs_calibration_gap"], errors="coerce").fillna(np.inf).gt(
                float(config.get("promotion_gate_calibration_gap_max", 0.05))
            )
        )
    ).astype(int)
    for universe in UNIVERSE_ORDER:
        subset = table.loc[table["universe"].astype(str).eq(universe)].copy()
        if subset.empty:
            continue
        if universe == "venture_growth":
            ordered = subset.sort_values(
                ["_candidate_priority", "benchmark_row", "target_name"],
                ascending=[True, False, True],
            )
        else:
            ordered = subset.sort_values(
                [
                    "_supported_flag",
                    "selected_policy_degenerate",
                    "_calibration_rank",
                    "train_positive_events",
                    "_label_conf_rank",
                    "_policy_hit_rank",
                    "_policy_precision_rank",
                    "target_name",
                ],
                ascending=[False, True, True, False, False, False, False, True],
            )
        for rank, idx in enumerate(ordered.index.tolist(), start=1):
            table.loc[idx, "recommendation_rank"] = rank
        selected_idx = ordered.index[0]
        table.loc[selected_idx, "recommended_for_universe"] = 1
        if universe == "venture_growth":
            table.loc[selected_idx, "chapter_reporting_status"] = "recommended"
        elif int(table.loc[selected_idx, "_buyout_provisional"]) == 1:
            table.loc[selected_idx, "chapter_reporting_status"] = "provisional"
        else:
            table.loc[selected_idx, "chapter_reporting_status"] = "recommended"
    reasons = []
    caveats = []
    for row in table.itertuples(index=False):
        if int(row.data_supported) != 1:
            reasons.append("Definition recorded, but the available dated fields do not support empirical estimation.")
            caveats.append(str(row.support_note))
            continue
        if str(row.universe) == "venture_growth" and int(row.recommended_for_universe) == 1:
            reasons.append("Locked venture/growth baseline retained as the primary headline target.")
        elif int(row.recommended_for_universe) == 1 and str(row.chapter_reporting_status) == "provisional":
            reasons.append("Best available buyout/PE target on current direct-liquidity evidence, but still provisional.")
        elif int(row.recommended_for_universe) == 1:
            reasons.append("Best available target on calibration, direct-liquidity support, and policy usefulness.")
        elif int(row.benchmark_row) == 1:
            reasons.append("Benchmark row preserved for apples-to-apples comparison.")
        else:
            reasons.append("Comparison candidate.")
        if int(row.selected_policy_degenerate) == 1:
            caveats.append("Selected policy rule is degenerate on the test slice.")
        elif pd.isna(row.mean_abs_calibration_gap):
            caveats.append("Calibration is not estimable for this candidate.")
        elif float(row.mean_abs_calibration_gap) > float(config.get("promotion_gate_calibration_gap_max", 0.05)):
            caveats.append("Full-test calibration remains above the current chapter gate.")
        elif int(row.train_positive_events) < int(config.get("min_train_exits", 100)):
            caveats.append("Train support remains thin for a stable empirical chapter claim.")
        else:
            caveats.append("")
    table["recommendation_reason"] = reasons
    table["unresolved_caveat"] = caveats
    return table.sort_values(["universe", "recommendation_rank", "target_name"]).reset_index(drop=True)


def source_class_from_route_source(route_source: object) -> str:
    source = str(route_source).strip().lower() if pd.notna(route_source) else ""
    if source.startswith("crunchbase_"):
        return "source_crunchbase_only"
    if source.startswith("preqin_"):
        return "source_preqin_only"
    if "both" in source:
        return "source_both"
    if source.startswith("manual_"):
        return "source_manual_override"
    return "source_unknown"


def build_target_source_summary(source_mix_targets: pd.DataFrame) -> pd.DataFrame:
    if source_mix_targets.empty:
        return pd.DataFrame(
            columns=[
                "target_key",
                "target_name",
                "universe",
                "positive_event_total",
                "direct_dated_events",
                "direct_undated_events",
                "inferred_transition_events",
                "synthetic_dated_events",
                "sensitivity_proxy_events",
                "label_confidence_events",
                "source_preqin_only",
                "source_crunchbase_only",
                "source_both",
                "source_manual_override",
                "source_unknown",
                "direct_dated_share",
                "direct_undated_share",
                "inferred_transition_share",
                "synthetic_dated_share",
                "label_confidence_share",
            ]
        )
    frame = source_mix_targets.copy()
    frame["positive_event_count"] = pd.to_numeric(frame["positive_event_count"], errors="coerce").fillna(0).astype(int)
    frame["source_class"] = frame["route_source"].map(source_class_from_route_source)
    rows: list[dict[str, object]] = []
    for (target_key, target_name, universe), subset in frame.groupby(["target_key", "target_name", "universe"], observed=True):
        total = int(subset["positive_event_count"].sum())
        direct = int(subset.loc[subset["event_observation_kind"].astype(str).eq("direct_dated_event"), "positive_event_count"].sum())
        direct_undated = int(
            subset.loc[subset["event_observation_kind"].astype(str).eq("direct_undated_event"), "positive_event_count"].sum()
        )
        inferred = int(subset.loc[subset["event_observation_kind"].astype(str).eq("inferred_transition"), "positive_event_count"].sum())
        synthetic = int(subset.loc[subset["event_observation_kind"].astype(str).eq("synthetic_dated_event"), "positive_event_count"].sum())
        sensitivity = int(subset.loc[subset["event_observation_kind"].astype(str).eq("sensitivity_proxy"), "positive_event_count"].sum())
        confidence_good = int(
            subset.loc[subset["confidence_tier"].astype(str).isin(["high", "medium", "synthetic"]), "positive_event_count"].sum()
        )
        rows.append(
            {
                "target_key": str(target_key),
                "target_name": str(target_name),
                "universe": str(universe),
                "positive_event_total": total,
                "direct_dated_events": direct,
                "direct_undated_events": direct_undated,
                "inferred_transition_events": inferred,
                "synthetic_dated_events": synthetic,
                "sensitivity_proxy_events": sensitivity,
                "label_confidence_events": confidence_good,
                "source_preqin_only": int(subset.loc[subset["source_class"].eq("source_preqin_only"), "positive_event_count"].sum()),
                "source_crunchbase_only": int(subset.loc[subset["source_class"].eq("source_crunchbase_only"), "positive_event_count"].sum()),
                "source_both": int(subset.loc[subset["source_class"].eq("source_both"), "positive_event_count"].sum()),
                "source_manual_override": int(subset.loc[subset["source_class"].eq("source_manual_override"), "positive_event_count"].sum()),
                "source_unknown": int(subset.loc[subset["source_class"].eq("source_unknown"), "positive_event_count"].sum()),
                "direct_dated_share": safe_ratio(direct, total),
                "direct_undated_share": safe_ratio(direct_undated, total),
                "inferred_transition_share": safe_ratio(inferred, total),
                "synthetic_dated_share": safe_ratio(synthetic, total),
                "label_confidence_share": safe_ratio(confidence_good, total),
            }
        )
    return pd.DataFrame(rows).sort_values(["universe", "target_name"]).reset_index(drop=True)


def build_source_mix_by_universe(source_summary: pd.DataFrame) -> pd.DataFrame:
    if source_summary.empty:
        return pd.DataFrame()
    grouped = source_summary.groupby("universe", as_index=False).agg(
        positive_event_total=("positive_event_total", "sum"),
        source_preqin_only=("source_preqin_only", "sum"),
        source_crunchbase_only=("source_crunchbase_only", "sum"),
        source_both=("source_both", "sum"),
        source_manual_override=("source_manual_override", "sum"),
        source_unknown=("source_unknown", "sum"),
    )
    return grouped.sort_values("universe").reset_index(drop=True)


def build_directness_by_universe(source_summary: pd.DataFrame) -> pd.DataFrame:
    if source_summary.empty:
        return pd.DataFrame()
    working = source_summary.copy()
    if "label_confidence_events" not in working.columns:
        working["label_confidence_events"] = (
            pd.to_numeric(working.get("label_confidence_share"), errors="coerce").fillna(0.0)
            * pd.to_numeric(working.get("positive_event_total"), errors="coerce").fillna(0.0)
        ).round().astype(int)
    if "direct_undated_events" not in working.columns:
        working["direct_undated_events"] = 0
    grouped = working.groupby("universe", as_index=False).agg(
        positive_event_total=("positive_event_total", "sum"),
        direct_dated_events=("direct_dated_events", "sum"),
        direct_undated_events=("direct_undated_events", "sum"),
        inferred_transition_events=("inferred_transition_events", "sum"),
        synthetic_dated_events=("synthetic_dated_events", "sum"),
        sensitivity_proxy_events=("sensitivity_proxy_events", "sum"),
        label_confidence_events=("label_confidence_events", "sum"),
    )
    for numerator, share_col in [
        ("direct_dated_events", "direct_dated_share"),
        ("direct_undated_events", "direct_undated_share"),
        ("inferred_transition_events", "inferred_transition_share"),
        ("synthetic_dated_events", "synthetic_dated_share"),
        ("label_confidence_events", "label_confidence_share"),
    ]:
        grouped[share_col] = grouped.apply(
            lambda row: safe_ratio(row[numerator], row["positive_event_total"]),
            axis=1,
        )
    return grouped.sort_values("universe").reset_index(drop=True)


def collapse_selected_policy_rows(
    frame: pd.DataFrame,
    evaluation_split: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    if evaluation_split is not None and "evaluation_split" in working.columns:
        working = working.loc[working["evaluation_split"].astype(str).eq(str(evaluation_split))].copy()
    if "selected_on_validation" in working.columns:
        working = working.loc[working["selected_on_validation"].astype(int).eq(1)].copy()
    if working.empty:
        return working
    working["_degenerate"] = (
        pd.to_numeric(working["degenerate_rule"], errors="coerce").fillna(1).astype(int)
        if "degenerate_rule" in working.columns
        else pd.Series(1, index=working.index, dtype=int)
    )
    working["_precision"] = (
        pd.to_numeric(working["precision"], errors="coerce").fillna(-1.0)
        if "precision" in working.columns
        else pd.Series(-1.0, index=working.index, dtype=float)
    )
    working["_hit_rate"] = (
        pd.to_numeric(working["hit_rate_accepted"], errors="coerce").fillna(-1.0)
        if "hit_rate_accepted" in working.columns
        else pd.Series(-1.0, index=working.index, dtype=float)
    )
    working["_balanced_accuracy"] = (
        pd.to_numeric(working["balanced_accuracy"], errors="coerce").fillna(-1.0)
        if "balanced_accuracy" in working.columns
        else pd.Series(-1.0, index=working.index, dtype=float)
    )
    working["_acceptance_rate"] = (
        pd.to_numeric(working["acceptance_rate"], errors="coerce").fillna(-1.0)
        if "acceptance_rate" in working.columns
        else pd.Series(-1.0, index=working.index, dtype=float)
    )
    working["_policy_key"] = working.get("policy_key", pd.Series(index=working.index, dtype=object)).astype(str)
    selected_rows: list[pd.Series] = []
    for _, subset in working.groupby(["target_key", "feature_backbone"], observed=True):
        ordered = subset.sort_values(
            ["_degenerate", "_precision", "_hit_rate", "_balanced_accuracy", "_acceptance_rate", "_policy_key"],
            ascending=[True, False, False, False, False, True],
        )
        selected_rows.append(ordered.iloc[0])
    output = pd.DataFrame(selected_rows).reset_index(drop=True)
    return output.drop(
        columns=["_degenerate", "_precision", "_hit_rate", "_balanced_accuracy", "_acceptance_rate", "_policy_key"],
        errors="ignore",
    )


def select_target_feature_backbones(
    registry: pd.DataFrame,
    evaluation_metrics_targets: pd.DataFrame,
    decision_backtest_targets: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    validation_metrics = evaluation_metrics_targets.loc[
        evaluation_metrics_targets["evaluation_view"].astype(str).eq("validation_selection")
    ].copy()
    validation_policy = collapse_selected_policy_rows(decision_backtest_targets, evaluation_split="validation")
    if validation_policy.empty:
        validation_policy = pd.DataFrame(
            columns=["target_key", "feature_backbone", "policy_key", "acceptance_rate", "precision", "recall", "degenerate_rule"]
        )
    validation_policy = validation_policy[
        ["target_key", "feature_backbone", "policy_key", "acceptance_rate", "precision", "recall", "degenerate_rule"]
    ].rename(
        columns={
            "policy_key": "validation_policy_key",
            "acceptance_rate": "validation_policy_acceptance_rate",
            "precision": "validation_policy_precision",
            "recall": "validation_policy_recall",
            "degenerate_rule": "validation_policy_degenerate",
        }
    )
    merged = registry[["target_key", "target_name", "universe"]].merge(
        validation_metrics[["target_key", "feature_backbone", "mean_abs_calibration_gap", "estimation_status", "estimation_note"]],
        on="target_key",
        how="left",
    ).merge(
        validation_policy,
        on=["target_key", "feature_backbone"],
        how="left",
    )
    merged["_estimated"] = merged["estimation_status"].astype(str).eq("estimated").astype(int)
    merged["_policy_ok"] = (
        pd.to_numeric(merged["validation_policy_degenerate"], errors="coerce").fillna(1).eq(0)
        & pd.to_numeric(merged["validation_policy_acceptance_rate"], errors="coerce").fillna(0.0).ge(
            float(config.get("promotion_gate_min_policy_acceptance", 0.005))
        )
        & pd.to_numeric(merged["validation_policy_acceptance_rate"], errors="coerce").fillna(1.0).le(
            float(config.get("target_selection_max_policy_acceptance", 0.50))
        )
    ).astype(int)
    merged["_validation_gap"] = pd.to_numeric(merged[CANONICAL_TARGET_CALIBRATION_METRIC], errors="coerce").fillna(np.inf)
    merged["_validation_precision"] = pd.to_numeric(merged["validation_policy_precision"], errors="coerce").fillna(-1.0)
    rows: list[pd.Series] = []
    for _, subset in merged.groupby("target_key", observed=True):
        ordered = subset.sort_values(
            ["_estimated", "_policy_ok", "_validation_gap", "_validation_precision", "feature_backbone"],
            ascending=[False, False, True, False, True],
        )
        rows.append(ordered.iloc[0])
    selected = pd.DataFrame(rows).reset_index(drop=True)
    return selected.rename(
        columns={
            "feature_backbone": "selected_feature_backbone",
            "validation_policy_key": "selected_backbone_validation_policy_key",
            "validation_policy_acceptance_rate": "selected_backbone_validation_policy_acceptance_rate",
            "validation_policy_precision": "selected_backbone_validation_policy_precision",
            "validation_policy_recall": "selected_backbone_validation_policy_recall",
            "validation_policy_degenerate": "selected_backbone_validation_policy_degenerate",
            CANONICAL_TARGET_CALIBRATION_METRIC: "selected_backbone_validation_mean_abs_calibration_gap",
        }
    )[
        [
            "target_key",
            "target_name",
            "universe",
            "selected_feature_backbone",
            "selected_backbone_validation_mean_abs_calibration_gap",
            "selected_backbone_validation_policy_key",
            "selected_backbone_validation_policy_acceptance_rate",
            "selected_backbone_validation_policy_precision",
            "selected_backbone_validation_policy_recall",
            "selected_backbone_validation_policy_degenerate",
            "estimation_status",
            "estimation_note",
        ]
    ].drop_duplicates(subset=["target_key", "selected_feature_backbone"]).copy()


def build_target_selection_gates(
    registry: pd.DataFrame,
    selected_backbones: pd.DataFrame,
    evaluation_metrics_targets: pd.DataFrame,
    decision_backtest_targets: pd.DataFrame,
    route_support_targets: pd.DataFrame,
    source_summary: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    validation_metrics = evaluation_metrics_targets.loc[
        evaluation_metrics_targets["evaluation_view"].astype(str).eq("validation_selection")
    ].copy()
    validation_policy = collapse_selected_policy_rows(decision_backtest_targets, evaluation_split="validation").rename(
        columns={
            "policy_key": "selected_policy_key",
            "policy_family": "selected_policy_family",
            "acceptance_rate": "selected_policy_acceptance_rate",
            "hit_rate_accepted": "selected_policy_hit_rate",
            "precision": "selected_policy_precision",
            "recall": "selected_policy_recall",
            "degenerate_rule": "selected_policy_degenerate",
            "balanced_accuracy": "selected_policy_balanced_accuracy",
            "prevalence": "selected_policy_prevalence",
            "lift_over_prevalence": "selected_policy_lift_over_prevalence",
        }
    )
    train_support_source = route_support_targets.loc[
        route_support_targets["split"].astype(str).eq("train")
    ].copy()
    raw_support = train_support_source.loc[
        train_support_source.get("support_scope", pd.Series("raw_target_routes", index=train_support_source.index)).astype(str).eq("raw_target_routes")
    ].groupby(["target_key", "target_name", "universe"], as_index=False).agg(
        train_positive_events=("positive_event_count", "sum"),
        train_positive_companies=("positive_companies", "sum"),
        min_train_route_support_raw=("positive_event_count", "min"),
        ipo_support_adequate_for_standalone=("ipo_support_adequate_for_standalone", "max"),
        partial_realizations_data_supported=("partial_realizations_data_supported", "max"),
    )
    stage2_source = train_support_source.loc[
        train_support_source.get("support_scope", pd.Series(index=train_support_source.index, dtype=object)).astype(str).eq("stage2_actual_view")
    ].copy()
    for column in ["requested_stage2_route_set", "actual_stage2_route_set"]:
        if column not in stage2_source.columns:
            stage2_source[column] = ""
    if stage2_source.empty:
        stage2_support = pd.DataFrame(
            columns=[
                "target_key",
                "target_name",
                "universe",
                "train_positive_events_stage2",
                "train_positive_companies_stage2",
                "min_train_route_support_stage2",
                "requested_stage2_route_set",
                "actual_stage2_route_set",
            ]
        )
    else:
        stage2_support = stage2_source.groupby(["target_key", "target_name", "universe"], as_index=False).agg(
            train_positive_events_stage2=("positive_event_count", "sum"),
            train_positive_companies_stage2=("positive_companies", "sum"),
            min_train_route_support_stage2=("positive_event_count", "min"),
            requested_stage2_route_set=("requested_stage2_route_set", "max"),
            actual_stage2_route_set=("actual_stage2_route_set", "max"),
        )
    train_support = raw_support.merge(stage2_support, on=["target_key", "target_name", "universe"], how="left")
    train_support["route_support_scope_used"] = np.where(
        pd.to_numeric(train_support["min_train_route_support_stage2"], errors="coerce").notna(),
        "stage2_actual_view",
        "raw_target_routes",
    )
    train_support["min_train_route_support"] = np.where(
        train_support["route_support_scope_used"].astype(str).eq("stage2_actual_view"),
        pd.to_numeric(train_support["min_train_route_support_stage2"], errors="coerce"),
        pd.to_numeric(train_support["min_train_route_support_raw"], errors="coerce"),
    )
    gates = registry.merge(selected_backbones, on=["target_key", "target_name", "universe"], how="left")
    gates = gates.merge(
        validation_metrics[["target_key", "feature_backbone", "mean_abs_calibration_gap", "rows"]],
        left_on=["target_key", "selected_feature_backbone"],
        right_on=["target_key", "feature_backbone"],
        how="left",
    ).drop(columns=["feature_backbone"], errors="ignore")
    policy_columns = [
        "target_key",
        "feature_backbone",
        "selected_policy_key",
        "selected_policy_family",
        "selected_policy_acceptance_rate",
        "selected_policy_hit_rate",
        "selected_policy_precision",
        "selected_policy_recall",
        "selected_policy_degenerate",
        "selected_policy_balanced_accuracy",
        "selected_policy_prevalence",
        "selected_policy_lift_over_prevalence",
        "predicted_mean_npv_proxy",
        "realized_mean_npv_proxy",
    ]
    gates = gates.merge(
        validation_policy[[column for column in policy_columns if column in validation_policy.columns]],
        left_on=["target_key", "selected_feature_backbone"],
        right_on=["target_key", "feature_backbone"],
        how="left",
    ).drop(columns=["feature_backbone"], errors="ignore")
    gates = gates.merge(train_support, on=["target_key", "target_name", "universe"], how="left")
    gates = gates.merge(source_summary, on=["target_key", "target_name", "universe"], how="left")
    gates["selected_policy_degenerate"] = pd.to_numeric(gates["selected_policy_degenerate"], errors="coerce").fillna(1).astype(int)
    gates["train_positive_events"] = pd.to_numeric(gates["train_positive_events"], errors="coerce").fillna(0).astype(int)
    gates["train_positive_companies"] = pd.to_numeric(gates["train_positive_companies"], errors="coerce").fillna(0).astype(int)
    gates["train_positive_events_stage2"] = pd.to_numeric(gates["train_positive_events_stage2"], errors="coerce").fillna(0).astype(int)
    gates["train_positive_companies_stage2"] = pd.to_numeric(gates["train_positive_companies_stage2"], errors="coerce").fillna(0).astype(int)
    gates["min_train_route_support_raw"] = pd.to_numeric(gates["min_train_route_support_raw"], errors="coerce").fillna(0).astype(int)
    gates["min_train_route_support_stage2"] = pd.to_numeric(gates["min_train_route_support_stage2"], errors="coerce").fillna(0).astype(int)
    gates["min_train_route_support"] = pd.to_numeric(gates["min_train_route_support"], errors="coerce").fillna(0).astype(int)
    if "route_support_scope_used" not in gates.columns:
        gates["route_support_scope_used"] = "raw_target_routes"
    gates["min_train_support_pass"] = gates["train_positive_events"].ge(int(config.get("min_train_exits", 100))).astype(int)
    gates["min_direct_dated_share_pass"] = pd.to_numeric(gates["direct_dated_share"], errors="coerce").fillna(0.0).ge(
        float(config.get("target_selection_min_direct_dated_share", 0.25))
    ).astype(int)
    gates["min_label_confidence_pass"] = pd.to_numeric(gates["label_confidence_share"], errors="coerce").fillna(0.0).ge(
        float(config.get("promotion_gate_min_label_confidence_share", 0.85))
    ).astype(int)
    gates["acceptable_policy_activation_pass"] = (
        pd.to_numeric(gates["selected_policy_acceptance_rate"], errors="coerce").fillna(0.0).ge(
            float(config.get("promotion_gate_min_policy_acceptance", 0.005))
        )
        & pd.to_numeric(gates["selected_policy_acceptance_rate"], errors="coerce").fillna(1.0).le(
            float(config.get("target_selection_max_policy_acceptance", 0.50))
        )
        & pd.to_numeric(gates["selected_policy_degenerate"], errors="coerce").fillna(1).eq(0)
    ).astype(int)
    gates["acceptable_validation_calibration_pass"] = pd.to_numeric(
        gates[CANONICAL_TARGET_CALIBRATION_METRIC],
        errors="coerce",
    ).fillna(np.inf).le(float(config.get("promotion_gate_calibration_gap_max", 0.05))).astype(int)
    gates["acceptable_route_support_pass"] = gates["min_train_route_support"].ge(
        int(config.get("stage2_min_route_support", 5))
    ).astype(int)
    gates["all_selection_gates_pass"] = gates[
        [
            "min_train_support_pass",
            "min_direct_dated_share_pass",
            "min_label_confidence_pass",
            "acceptable_policy_activation_pass",
            "acceptable_validation_calibration_pass",
            "acceptable_route_support_pass",
        ]
    ].min(axis=1).astype(int)
    return gates.drop_duplicates(subset=["target_key", "selected_feature_backbone"]).reset_index(drop=True)


def build_target_leaderboard_validation(gates: pd.DataFrame) -> pd.DataFrame:
    table = gates.copy()
    table["validation_rank"] = np.nan
    table["selected_by_validation"] = 0
    table["retained_by_doctrine"] = 0
    table["selection_basis"] = "comparison_only"
    candidate_priority = {
        "locked_baseline": 0,
        "candidate": 1,
        "optional_candidate": 2,
        "benchmark_only": 3,
        "candidate_unsupported": 4,
    }
    table["_candidate_priority"] = table["candidate_role"].map(candidate_priority).fillna(9).astype(int)
    table["_validation_gap"] = pd.to_numeric(table[CANONICAL_TARGET_CALIBRATION_METRIC], errors="coerce").fillna(np.inf)
    table["_validation_precision"] = pd.to_numeric(table["selected_policy_precision"], errors="coerce").fillna(-1.0)
    table["_validation_hit_rate"] = pd.to_numeric(table["selected_policy_hit_rate"], errors="coerce").fillna(-1.0)
    table["_estimated_flag"] = table["estimation_status"].astype(str).eq("estimated").astype(int)
    table["_gate_pass_count"] = table[
        [
            "min_train_support_pass",
            "min_direct_dated_share_pass",
            "min_label_confidence_pass",
            "acceptable_policy_activation_pass",
            "acceptable_validation_calibration_pass",
            "acceptable_route_support_pass",
        ]
    ].sum(axis=1)
    for universe in UNIVERSE_ORDER:
        subset = table.loc[table["universe"].astype(str).eq(universe)].copy()
        if subset.empty:
            continue
        if universe == "venture_growth":
            ordered = subset.sort_values(
                ["_candidate_priority", "benchmark_row", "target_name"],
                ascending=[True, False, True],
            )
            table.loc[ordered.index[0], "retained_by_doctrine"] = 1
            table.loc[ordered.index[0], "selection_basis"] = "retained_by_doctrine"
        else:
            ordered = subset.sort_values(
                [
                    "_estimated_flag",
                    "_gate_pass_count",
                    "acceptable_policy_activation_pass",
                    "acceptable_validation_calibration_pass",
                    "min_direct_dated_share_pass",
                    "min_label_confidence_pass",
                    "_validation_gap",
                    "_validation_precision",
                    "_validation_hit_rate",
                    "train_positive_events",
                    "target_name",
                ],
                ascending=[False, False, False, False, False, False, True, False, False, False, True],
            )
            table.loc[ordered.index[0], "selected_by_validation"] = 1
            table.loc[ordered.index[0], "selection_basis"] = "selected_by_validation"
        for rank, idx in enumerate(ordered.index.tolist(), start=1):
            table.loc[idx, "validation_rank"] = rank
    return table.drop_duplicates(subset=["target_key", "selected_feature_backbone"]).sort_values(
        ["universe", "validation_rank", "target_name"]
    ).reset_index(drop=True)


def build_target_confirmation_test(
    leaderboard_validation: pd.DataFrame,
    evaluation_metrics_targets: pd.DataFrame,
    decision_backtest_targets: pd.DataFrame,
) -> pd.DataFrame:
    test_metrics = evaluation_metrics_targets.loc[
        evaluation_metrics_targets["evaluation_view"].astype(str).eq("full_test")
    ].copy()
    test_high_conf = evaluation_metrics_targets.loc[
        evaluation_metrics_targets["evaluation_view"].astype(str).eq("high_confidence_exit_label_only"),
        ["target_key", "feature_backbone", "mean_abs_calibration_gap"],
    ].rename(columns={"mean_abs_calibration_gap": "high_confidence_mean_abs_calibration_gap"})
    test_policy = collapse_selected_policy_rows(decision_backtest_targets, evaluation_split="test").rename(
        columns={
            "policy_key": "selected_policy_key",
            "policy_family": "selected_policy_family",
            "acceptance_rate": "selected_policy_acceptance_rate",
            "hit_rate_accepted": "selected_policy_hit_rate",
            "precision": "selected_policy_precision",
            "recall": "selected_policy_recall",
            "degenerate_rule": "selected_policy_degenerate",
            "balanced_accuracy": "selected_policy_balanced_accuracy",
            "prevalence": "selected_policy_prevalence",
            "lift_over_prevalence": "selected_policy_lift_over_prevalence",
        }
    )
    confirmation = leaderboard_validation.merge(
        test_metrics[
            [
                "target_key",
                "feature_backbone",
                "rows",
                "brier_score",
                "integrated_brier_score",
                "pr_auc",
                "roc_auc",
                "calibration_slope",
                "calibration_slope_status",
                "calibration_intercept",
                "calibration_intercept_status",
                "top_decile_realized_exit_rate",
                "top_decile_lift",
                "mean_abs_calibration_gap",
                "max_abs_calibration_gap",
            ]
        ],
        left_on=["target_key", "selected_feature_backbone"],
        right_on=["target_key", "feature_backbone"],
        how="left",
    ).drop(columns=["feature_backbone"], errors="ignore")
    confirmation = confirmation.merge(
        test_high_conf,
        left_on=["target_key", "selected_feature_backbone"],
        right_on=["target_key", "feature_backbone"],
        how="left",
    ).drop(columns=["feature_backbone"], errors="ignore")
    policy_columns = [
        "target_key",
        "feature_backbone",
        "selected_policy_key",
        "selected_policy_family",
        "selected_policy_acceptance_rate",
        "selected_policy_hit_rate",
        "selected_policy_precision",
        "selected_policy_recall",
        "selected_policy_degenerate",
        "selected_policy_balanced_accuracy",
        "selected_policy_prevalence",
        "selected_policy_lift_over_prevalence",
        "predicted_mean_npv_proxy",
        "realized_mean_npv_proxy",
    ]
    confirmation = confirmation.merge(
        test_policy[[column for column in policy_columns if column in test_policy.columns]],
        left_on=["target_key", "selected_feature_backbone"],
        right_on=["target_key", "feature_backbone"],
        how="left",
    ).drop(columns=["feature_backbone"], errors="ignore")
    rename_map = {
        "rows_y": "rows",
        "mean_abs_calibration_gap_y": "mean_abs_calibration_gap",
        "selected_policy_key_y": "selected_policy_key",
        "selected_policy_family_y": "selected_policy_family",
        "selected_policy_acceptance_rate_y": "selected_policy_acceptance_rate",
        "selected_policy_hit_rate_y": "selected_policy_hit_rate",
        "selected_policy_precision_y": "selected_policy_precision",
        "selected_policy_recall_y": "selected_policy_recall",
        "selected_policy_degenerate_y": "selected_policy_degenerate",
        "selected_policy_balanced_accuracy_y": "selected_policy_balanced_accuracy",
        "selected_policy_prevalence_y": "selected_policy_prevalence",
        "selected_policy_lift_over_prevalence_y": "selected_policy_lift_over_prevalence",
        "predicted_mean_npv_proxy_y": "predicted_mean_npv_proxy",
        "realized_mean_npv_proxy_y": "realized_mean_npv_proxy",
    }
    confirmation = confirmation.rename(columns={key: value for key, value in rename_map.items() if key in confirmation.columns})
    confirmation["confirmed_on_locked_test"] = (
        confirmation["selected_by_validation"].astype(int).eq(1)
        | confirmation["retained_by_doctrine"].astype(int).eq(1)
    ).astype(int)
    return confirmation.drop_duplicates(subset=["target_key", "selected_feature_backbone"]).sort_values(
        ["universe", "validation_rank", "target_name"]
    ).reset_index(drop=True)


def build_decision_usefulness_by_target(
    leaderboard_validation: pd.DataFrame,
    confirmation_test: pd.DataFrame,
    prevalence_by_split: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    split_prevalence = prevalence_by_split.rename(columns={"prevalence": "split_target_prevalence_all_rows"})[
        ["target_key", "target_name", "universe", "split", "split_target_prevalence_all_rows"]
    ].copy()
    validation_merge_keys = ["target_key", "target_name", "universe"]
    if "split" in leaderboard_validation.columns:
        validation_merge_keys.append("split")
    validation_rows = leaderboard_validation.merge(split_prevalence, on=validation_merge_keys, how="left")
    if "split" not in validation_rows.columns and "split_x" in validation_rows.columns:
        validation_rows["split"] = validation_rows["split_x"]
    validation_rows = validation_rows.loc[validation_rows["split"].astype(str).eq("validation")].copy()
    validation_rows["evaluation_split"] = "validation"
    validation_rows = validation_rows.rename(columns={"selected_policy_acceptance_rate": "selected_positive_rate"})
    test_merge_keys = ["target_key", "target_name", "universe"]
    if "split" in confirmation_test.columns:
        test_merge_keys.append("split")
    test_rows = confirmation_test.merge(split_prevalence, on=test_merge_keys, how="left")
    if "split" not in test_rows.columns and "split_x" in test_rows.columns:
        test_rows["split"] = test_rows["split_x"]
    test_rows = test_rows.loc[test_rows["split"].astype(str).eq("test")].copy()
    test_rows["evaluation_split"] = "test"
    test_rows = test_rows.rename(columns={"selected_policy_acceptance_rate": "selected_positive_rate"})
    combined = pd.concat([validation_rows, test_rows], ignore_index=True, sort=False)
    combined["target_prevalence"] = pd.to_numeric(
        combined.get("selected_policy_prevalence", pd.Series(index=combined.index, dtype=float)),
        errors="coerce",
    )
    combined["split_target_prevalence_all_rows"] = pd.to_numeric(
        combined.get("split_target_prevalence_all_rows", pd.Series(index=combined.index, dtype=float)),
        errors="coerce",
    )
    if "balanced_accuracy" not in combined.columns:
        combined["balanced_accuracy"] = pd.to_numeric(
            combined.get("selected_policy_balanced_accuracy", pd.Series(index=combined.index, dtype=float)),
            errors="coerce",
        )
    combined["lift_vs_prevalence"] = pd.to_numeric(
        combined.get("selected_policy_lift_over_prevalence", pd.Series(index=combined.index, dtype=float)),
        errors="coerce",
    )
    combined["metric_row_universe"] = "selected_policy_decision_panel"
    combined["acceptance_band_pass"] = (
        pd.to_numeric(combined["selected_positive_rate"], errors="coerce").fillna(0.0).ge(
            float(config.get("promotion_gate_min_policy_acceptance", 0.005))
        )
        & pd.to_numeric(combined["selected_positive_rate"], errors="coerce").fillna(1.0).le(
            float(config.get("target_selection_max_policy_acceptance", 0.50))
        )
    ).astype(int)
    return combined[
        [
            "target_key",
            "target_name",
            "universe",
            "selected_feature_backbone",
            "evaluation_split",
            "target_prevalence",
            "split_target_prevalence_all_rows",
            "metric_row_universe",
            "selected_positive_rate",
            "selected_policy_precision",
            "selected_policy_recall",
            "balanced_accuracy",
            "lift_vs_prevalence",
            "acceptance_band_pass",
            "selected_policy_key",
            "selection_basis",
            "selected_by_validation",
            "retained_by_doctrine",
        ]
    ].sort_values(["universe", "target_name", "evaluation_split"]).reset_index(drop=True)


def build_buyout_target_with_without_sponsor_fund(
    evaluation_metrics_targets: pd.DataFrame,
    decision_backtest_targets: pd.DataFrame,
) -> pd.DataFrame:
    validation_metrics = evaluation_metrics_targets.loc[
        evaluation_metrics_targets["evaluation_view"].astype(str).eq("validation_selection")
        & evaluation_metrics_targets["universe"].astype(str).eq("buyout_pe")
    ].copy()
    validation_policy = decision_backtest_targets.loc[
        decision_backtest_targets["evaluation_split"].astype(str).eq("validation")
        & decision_backtest_targets["selected_on_validation"].astype(int).eq(1)
        & decision_backtest_targets["universe"].astype(str).eq("buyout_pe")
    ].copy().rename(
        columns={
            "policy_key": "selected_policy_key",
            "acceptance_rate": "selected_policy_acceptance_rate",
            "precision": "selected_policy_precision",
            "recall": "selected_policy_recall",
        }
    )
    merged = validation_metrics.merge(
        validation_policy[
            ["target_key", "feature_backbone", "selected_policy_key", "selected_policy_acceptance_rate", "selected_policy_precision", "selected_policy_recall"]
        ],
        on=["target_key", "feature_backbone"],
        how="left",
    )
    base = merged.loc[merged["feature_backbone"].astype(str).eq(TARGET_BASE_FEATURE_BACKBONE)].copy()
    sponsor = merged.loc[merged["feature_backbone"].astype(str).eq(TARGET_SPONSOR_FUND_FEATURE_BACKBONE)].copy()
    comparison = base.merge(
        sponsor[
            ["target_key", "mean_abs_calibration_gap", "selected_policy_acceptance_rate", "selected_policy_precision", "selected_policy_recall"]
        ].rename(
            columns={
                "mean_abs_calibration_gap": "sponsor_fund_validation_gap",
                "selected_policy_acceptance_rate": "sponsor_fund_policy_acceptance_rate",
                "selected_policy_precision": "sponsor_fund_policy_precision",
                "selected_policy_recall": "sponsor_fund_policy_recall",
            }
        ),
        on="target_key",
        how="left",
    ).rename(
        columns={
            "mean_abs_calibration_gap": "baseline_validation_gap",
            "selected_policy_acceptance_rate": "baseline_policy_acceptance_rate",
            "selected_policy_precision": "baseline_policy_precision",
            "selected_policy_recall": "baseline_policy_recall",
        }
    )
    comparison["sponsor_fund_available"] = comparison["sponsor_fund_validation_gap"].notna().astype(int)
    comparison["validation_gap_delta_sponsor_minus_base"] = (
        pd.to_numeric(comparison["sponsor_fund_validation_gap"], errors="coerce")
        - pd.to_numeric(comparison["baseline_validation_gap"], errors="coerce")
    )
    return comparison.sort_values("target_name").reset_index(drop=True)


def build_target_recommendation_table_v2(
    leaderboard_validation: pd.DataFrame,
    confirmation_test: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    def buyout_blocker_text(row: pd.Series) -> str:
        if int(pd.to_numeric(pd.Series([row.get("min_direct_dated_share_pass")]), errors="coerce").fillna(0).iloc[0]) == 0:
            return "Validation-selected buyout target still fails the direct-dated-share gate."
        if int(pd.to_numeric(pd.Series([row.get("max_inferred_transition_share_pass")]), errors="coerce").fillna(1).iloc[0]) == 0:
            return "Validation-selected buyout target still relies too heavily on inferred-transition evidence."
        if int(pd.to_numeric(pd.Series([row.get("acceptable_policy_activation_pass")]), errors="coerce").fillna(0).iloc[0]) == 0:
            return "Validation-selected buyout target remains non-promotable because the selected policy is economically degenerate at the current acceptance band."
        if int(pd.to_numeric(pd.Series([row.get("acceptable_route_support_pass")]), errors="coerce").fillna(0).iloc[0]) == 0:
            return "Validation-selected buyout target remains non-promotable because route support is still too thin for a chapter-headline claim."
        if int(pd.to_numeric(pd.Series([row.get("acceptable_validation_calibration_pass")]), errors="coerce").fillna(0).iloc[0]) == 0:
            return "Validation-selected buyout target remains non-promotable because validation calibration is still outside the chapter gate."
        if int(pd.to_numeric(pd.Series([row.get("min_label_confidence_pass")]), errors="coerce").fillna(0).iloc[0]) == 0:
            return "Validation-selected buyout target remains non-promotable because exit-label confidence remains below the chapter gate."
        if int(pd.to_numeric(pd.Series([row.get("confirmed_on_locked_test")]), errors="coerce").fillna(0).iloc[0]) == 0:
            return "Validation-selected buyout target did not confirm on the locked test slice."
        return "Validation-selected buyout target still fails one or more chapter-promotion gates."

    table = leaderboard_validation.merge(
        confirmation_test[
            [
                "target_key",
                "high_confidence_mean_abs_calibration_gap",
                "rows",
                "brier_score",
                "integrated_brier_score",
                "pr_auc",
                "roc_auc",
                "calibration_slope",
                "calibration_slope_status",
                "calibration_intercept",
                "calibration_intercept_status",
                "top_decile_realized_exit_rate",
                "top_decile_lift",
                "mean_abs_calibration_gap",
                "max_abs_calibration_gap",
                "selected_policy_key",
                "selected_policy_family",
                "selected_policy_acceptance_rate",
                "selected_policy_hit_rate",
                "selected_policy_precision",
                "selected_policy_recall",
                "selected_policy_degenerate",
                "predicted_mean_npv_proxy",
                "realized_mean_npv_proxy",
                "confirmed_on_locked_test",
            ]
        ],
        on="target_key",
        how="left",
        suffixes=("_validation", ""),
    )
    table["recommendation_rank"] = table["validation_rank"]
    table["recommended_for_universe"] = (
        table["selected_by_validation"].astype(int).eq(1) | table["retained_by_doctrine"].astype(int).eq(1)
    ).astype(int)
    table["chapter_reporting_status"] = "comparison_only"
    table["headline_target_group"] = "separate_by_universe"
    if "headline_eligible" not in table.columns:
        table["headline_eligible"] = 1
    table.loc[
        table["universe"].astype(str).eq("venture_growth") & table["retained_by_doctrine"].astype(int).eq(1),
        "chapter_reporting_status",
    ] = "doctrinal_baseline"
    buyout_mask = table["universe"].astype(str).eq("buyout_pe") & table["selected_by_validation"].astype(int).eq(1)
    table.loc[buyout_mask, "chapter_reporting_status"] = np.where(
        table.loc[buyout_mask, "all_selection_gates_pass"].astype(int).eq(1)
        & pd.to_numeric(table.loc[buyout_mask, "headline_eligible"], errors="coerce").fillna(1).astype(int).eq(1),
        "recommended",
        "provisional",
    )
    table["recommendation_reason"] = np.where(
        table["retained_by_doctrine"].astype(int).eq(1),
        "Retained by doctrine as the venture/growth baseline while event support remains thin for broader search.",
        np.where(
            table["selected_by_validation"].astype(int).eq(1),
            "Selected on validation-only metrics and then carried to locked test confirmation.",
            "Validation-ranked comparison candidate.",
        ),
    )
    table["unresolved_caveat"] = ""
    table.loc[table["data_supported"].astype(int).eq(0), "unresolved_caveat"] = table["support_note"].astype(str)
    table.loc[
        pd.to_numeric(table["direct_dated_share"], errors="coerce").fillna(0.0).lt(
            float(config.get("target_selection_min_direct_dated_share", 0.25))
        ),
        "unresolved_caveat",
    ] = "Direct-dated share remains below the current chapter gate."
    table.loc[
        pd.to_numeric(table["label_confidence_share"], errors="coerce").fillna(0.0).lt(
            float(config.get("promotion_gate_min_label_confidence_share", 0.85))
        ),
        "unresolved_caveat",
    ] = "Exit-label confidence share remains below the current chapter gate."
    buyout_rows = table["universe"].astype(str).eq("buyout_pe")
    buyout_clear = bool(
        not table.loc[
            buyout_rows
            & table["selected_by_validation"].astype(int).eq(1)
            & table["chapter_reporting_status"].astype(str).eq("recommended")
        ].empty
    )
    buyout_doctrine = buyout_rows & table["target_name"].astype(str).eq("sponsor_sale_or_mna_by_12q")
    buyout_selected = table.loc[buyout_rows & table["selected_by_validation"].astype(int).eq(1)].copy()
    buyout_blocker = buyout_blocker_text(buyout_selected.iloc[0]) if not buyout_selected.empty else "No buyout candidate cleared the validation-first promotion protocol."
    if not buyout_clear and buyout_doctrine.any():
        table.loc[buyout_rows, "recommended_for_universe"] = 0
        table.loc[buyout_doctrine, "recommended_for_universe"] = 1
        table.loc[buyout_doctrine, "chapter_reporting_status"] = "provisional"
        table.loc[buyout_doctrine, "selection_basis"] = "retained_provisional_doctrine"
        table.loc[buyout_doctrine, "recommendation_reason"] = (
            "Retained as the provisional buyout/PE doctrine target because the validation-selected realization target still fails at least one hard promotion gate."
        )
        table.loc[buyout_doctrine, "unresolved_caveat"] = np.where(
            table.loc[buyout_doctrine, "unresolved_caveat"].astype(str).str.len().gt(0),
            table.loc[buyout_doctrine, "unresolved_caveat"],
            buyout_blocker,
        )
    return table.drop_duplicates(subset=["target_key", "selected_feature_backbone"]).sort_values(
        ["universe", "recommendation_rank", "target_name"]
    ).reset_index(drop=True)


def selected_universe_recommendations(recommendation_table: pd.DataFrame) -> pd.DataFrame:
    if recommendation_table.empty:
        return pd.DataFrame()
    recommended = recommendation_table.loc[
        recommendation_table["recommended_for_universe"].astype(int).eq(1)
    ].copy()
    if recommended.empty:
        return recommended
    recommended = recommended.sort_values(
        ["universe", "retained_by_doctrine", "selected_by_validation", "recommendation_rank", "target_name"],
        ascending=[True, False, False, True, True],
    )
    return recommended.drop_duplicates(subset=["universe"], keep="first").reset_index(drop=True)


def build_universe_claim_matrix(
    recommendation_table: pd.DataFrame,
    target_selection_gates: pd.DataFrame,
    target_confirmation_test: pd.DataFrame,
) -> pd.DataFrame:
    selected = selected_universe_recommendations(recommendation_table)
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "universe",
                "headline_target_name",
                "reporting_status",
                "selection_basis",
                "validation_pass",
                "locked_test_confirmation_pass",
                "directness_pass",
                "label_confidence_pass",
                "policy_activation_pass",
                "chapter_headline_eligible",
                "appendix_eligible",
                "main_limiting_factor",
            ]
        )
    gates = target_selection_gates.rename(
        columns={
            "min_direct_dated_share_pass": "directness_pass",
            "min_label_confidence_pass": "label_confidence_pass",
            "acceptable_policy_activation_pass": "policy_activation_pass",
        }
    )[
        [
            "target_key",
            "selected_feature_backbone",
            "all_selection_gates_pass",
            "directness_pass",
            "label_confidence_pass",
            "policy_activation_pass",
            "acceptable_validation_calibration_pass",
            "acceptable_route_support_pass",
        ]
    ].copy()
    confirmation = target_confirmation_test[
        [
            "target_key",
            "selected_feature_backbone",
            "confirmed_on_locked_test",
            "mean_abs_calibration_gap",
            "high_confidence_mean_abs_calibration_gap",
        ]
    ].rename(
        columns={
            "confirmed_on_locked_test": "confirmed_on_locked_test_source",
            "mean_abs_calibration_gap": "locked_test_mean_abs_calibration_gap",
            "high_confidence_mean_abs_calibration_gap": "locked_test_high_confidence_mean_abs_calibration_gap",
        }
    ).copy()
    matrix = selected.merge(
        gates,
        on=["target_key", "selected_feature_backbone"],
        how="left",
    ).merge(
        confirmation,
        on=["target_key", "selected_feature_backbone"],
        how="left",
    )
    doctrine_retained = matrix["selection_basis"].astype(str).eq("retained_provisional_doctrine")
    matrix["validation_pass"] = (
        matrix["selected_by_validation"].astype(int).eq(1)
        | matrix["retained_by_doctrine"].astype(int).eq(1)
        | doctrine_retained
    ).astype(int)
    locked_test_available = pd.to_numeric(
        matrix["locked_test_mean_abs_calibration_gap"], errors="coerce"
    ).notna()
    matrix["locked_test_confirmation_pass"] = np.where(
        doctrine_retained & locked_test_available,
        1,
        pd.to_numeric(matrix["confirmed_on_locked_test_source"], errors="coerce").fillna(0).astype(int),
    )
    matrix["directness_pass"] = pd.to_numeric(matrix["directness_pass"], errors="coerce").fillna(0).astype(int)
    matrix["label_confidence_pass"] = pd.to_numeric(matrix["label_confidence_pass"], errors="coerce").fillna(0).astype(int)
    matrix["policy_activation_pass"] = pd.to_numeric(matrix["policy_activation_pass"], errors="coerce").fillna(0).astype(int)
    matrix["chapter_headline_eligible"] = (
        matrix["chapter_reporting_status"].astype(str).eq("recommended")
        & matrix["validation_pass"].astype(int).eq(1)
        & matrix["locked_test_confirmation_pass"].astype(int).eq(1)
        & matrix["directness_pass"].astype(int).eq(1)
        & matrix["label_confidence_pass"].astype(int).eq(1)
        & matrix["policy_activation_pass"].astype(int).eq(1)
    ).astype(int)
    matrix["appendix_eligible"] = (
        matrix["chapter_reporting_status"].astype(str).isin(["doctrinal_baseline", "recommended", "provisional"])
        & (
            matrix["validation_pass"].astype(int).eq(1)
            | doctrine_retained
        )
    ).astype(int)

    def limiting_factor(row: pd.Series) -> str:
        validation_calibration_pass = pd.to_numeric(
            pd.Series([row.get("acceptable_validation_calibration_pass")]), errors="coerce"
        ).fillna(0).astype(int).iloc[0]
        route_support_pass = pd.to_numeric(
            pd.Series([row.get("acceptable_route_support_pass")]), errors="coerce"
        ).fillna(0).astype(int).iloc[0]
        status = str(row.get("chapter_reporting_status", ""))
        if status == "doctrinal_baseline":
            return "retained_by_doctrine_not_promoted_as_empirical_headline"
        if int(row.get("chapter_headline_eligible", 0)) == 1:
            return "none"
        if int(row.get("directness_pass", 0)) == 0:
            return "direct_dated_share_below_gate"
        if int(row.get("label_confidence_pass", 0)) == 0:
            return "exit_label_confidence_below_gate"
        if int(row.get("policy_activation_pass", 0)) == 0:
            return "policy_activation_outside_acceptance_band"
        if validation_calibration_pass == 0:
            return "validation_calibration_above_gate"
        if route_support_pass == 0:
            return "route_support_below_gate"
        if status == "provisional":
            return "buyout_provisional_pending_realization_mechanics"
        if int(row.get("locked_test_confirmation_pass", 0)) == 0:
            return "locked_test_confirmation_missing"
        return "unresolved"

    matrix["main_limiting_factor"] = matrix.apply(limiting_factor, axis=1)
    matrix = matrix.rename(
        columns={
            "target_name": "headline_target_name",
            "chapter_reporting_status": "reporting_status",
        }
    )
    return matrix[
        [
            "universe",
            "headline_target_name",
            "reporting_status",
            "selection_basis",
            "validation_pass",
            "locked_test_confirmation_pass",
            "directness_pass",
            "label_confidence_pass",
            "policy_activation_pass",
            "chapter_headline_eligible",
            "appendix_eligible",
            "main_limiting_factor",
        ]
    ].sort_values("universe").reset_index(drop=True)


def write_universe_claim_matrix(path: Path, claim_matrix: pd.DataFrame) -> None:
    lines = [
        "# Universe Claim Matrix",
        "",
        "- This matrix replaces a single global promotion interpretation with universe-specific reporting eligibility.",
        "- Venture/growth remains a doctrinal baseline in the current canonical pass.",
        "- Buyout/PE remains provisional until direct-dated realization support improves materially.",
        "",
        dataframe_to_markdown(claim_matrix),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_buyout_realization_field_audit(
    sources: dict[str, pd.DataFrame],
    round_events: pd.DataFrame,
) -> pd.DataFrame:
    buyout = sources.get("preqin_buyout", pd.DataFrame()).copy()
    fund_performance = sources.get("preqin_fund_performance", pd.DataFrame()).copy()
    cashflow = sources.get("preqin_cashflow", pd.DataFrame()).copy()
    fund_details = sources.get("preqin_fund_details", pd.DataFrame()).copy()
    manager_details = sources.get("preqin_manager_details", pd.DataFrame()).copy()
    investor_details = sources.get("preqin_investor_details", pd.DataFrame()).copy()

    def non_null_count(frame: pd.DataFrame, columns: list[str]) -> int:
        present = [column for column in columns if column in frame.columns]
        if not present or frame.empty:
            return 0
        return int(frame[present].notna().any(axis=1).sum())

    rows = [
        {
            "audit_area": "partial_realizations",
            "source_table": "preqin_buyout",
            "field_names": "deal_status|investment_status|deal_description",
            "dated_field_present": int("deal_date" in buyout.columns and buyout["deal_date"].notna().any()),
            "non_null_rows": non_null_count(buyout, ["deal_status", "investment_status", "deal_description"]),
            "candidate_supported": 0,
            "note": "No explicit dated partial-realization field is loaded in the staged local buyout extract.",
        },
        {
            "audit_area": "recapitalizations",
            "source_table": "preqin_buyout",
            "field_names": "deal_status|investment_status|deal_description",
            "dated_field_present": int("deal_date" in buyout.columns and buyout["deal_date"].notna().any()),
            "non_null_rows": non_null_count(buyout, ["deal_status", "investment_status", "deal_description"]),
            "candidate_supported": 0,
            "note": "No explicit dated recapitalization field is loaded; recap-like evidence would be text-only and remains unsupported.",
        },
        {
            "audit_area": "continuation_vehicle_events",
            "source_table": "preqin_buyout",
            "field_names": "deal_status|investment_status|deal_description",
            "dated_field_present": int("deal_date" in buyout.columns and buyout["deal_date"].notna().any()),
            "non_null_rows": non_null_count(buyout, ["deal_status", "investment_status", "deal_description"]),
            "candidate_supported": 0,
            "note": "No continuation-vehicle specific dated field is loaded in the staged local extracts.",
        },
        {
            "audit_area": "secondary_like_realizations",
            "source_table": "preqin_buyout",
            "field_names": "investment_type|deal_description",
            "dated_field_present": int("deal_date" in buyout.columns and buyout["deal_date"].notna().any()),
            "non_null_rows": non_null_count(buyout, ["investment_type", "deal_description"]),
            "candidate_supported": 0,
            "note": "Secondary-like realizations are not exposed as a clean dated field family in the staged local buyout extract.",
        },
        {
            "audit_area": "exit_realization_value",
            "source_table": "preqin_buyout",
            "field_names": "deal_size_usd|dealsizeequity_usd|enterprisevalue",
            "dated_field_present": int("deal_date" in buyout.columns and buyout["deal_date"].notna().any()),
            "non_null_rows": non_null_count(buyout, ["deal_size_usd", "dealsizeequity_usd", "enterprisevalue"]),
            "candidate_supported": int(non_null_count(buyout, ["deal_size_usd", "dealsizeequity_usd", "enterprisevalue"]) > 0),
            "note": "Exit-value style fields exist for some dated buyout events, but not as a clean realized-partial cash-out series.",
        },
        {
            "audit_area": "fund_performance_multiple_irr",
            "source_table": "preqin_fund_performance",
            "field_names": "date_reported|multiple|net_irr_pcent|distr_dpi_pcent|value_rvpi_pcent",
            "dated_field_present": int("date_reported" in fund_performance.columns and fund_performance["date_reported"].notna().any()),
            "non_null_rows": non_null_count(fund_performance, ["multiple", "net_irr_pcent", "distr_dpi_pcent", "value_rvpi_pcent"]),
            "candidate_supported": int(non_null_count(fund_performance, ["multiple", "net_irr_pcent", "distr_dpi_pcent", "value_rvpi_pcent"]) > 0),
            "note": "Fund-level performance fields are present and PIT-usable through report dates, but they are fund aggregates rather than direct company realization events.",
        },
        {
            "audit_area": "dated_cashflow",
            "source_table": "preqin_cashflow",
            "field_names": "transaction_date|transaction_type|transaction_amount|net_cashflow",
            "dated_field_present": int("transaction_date" in cashflow.columns and cashflow["transaction_date"].notna().any()),
            "non_null_rows": non_null_count(cashflow, ["transaction_type", "transaction_amount", "net_cashflow"]),
            "candidate_supported": int(non_null_count(cashflow, ["transaction_type", "transaction_amount", "net_cashflow"]) > 0),
            "note": "Cash-flow rows are available at the fund level with dates, supporting PIT-safe market-quarter aggregates.",
        },
        {
            "audit_area": "company_to_deal_link",
            "source_table": "round_events",
            "field_names": "company_id|event_date",
            "dated_field_present": int("event_date" in round_events.columns and pd.to_datetime(round_events["event_date"], errors="coerce").notna().any()),
            "non_null_rows": non_null_count(round_events, ["company_id", "event_date"]),
            "candidate_supported": int(non_null_count(round_events, ["company_id", "event_date"]) > 0),
            "note": "Company to dated deal-event linkage is active in the reconstructed round panel.",
        },
        {
            "audit_area": "deal_to_fund_or_firm_link",
            "source_table": "preqin_buyout",
            "field_names": "fund_id|firm_id|buyout_id",
            "dated_field_present": int("deal_date" in buyout.columns and buyout["deal_date"].notna().any()),
            "non_null_rows": non_null_count(buyout, ["fund_id", "firm_id", "buyout_id"]),
            "candidate_supported": int(non_null_count(buyout, ["fund_id", "firm_id"]) > 0),
            "note": "The staged buyout extract carries `buyout_id` but not a populated direct `fund_id` / `firm_id` bridge for company-level sponsor joins.",
        },
        {
            "audit_area": "fund_manager_gp_identifiers",
            "source_table": "preqin_fund_details|preqin_manager_details",
            "field_names": "fund_id|firm_id|lastupdated",
            "dated_field_present": int("lastupdated" in manager_details.columns and manager_details["lastupdated"].notna().any()),
            "non_null_rows": int(non_null_count(fund_details, ["fund_id", "firm_id"]) + non_null_count(manager_details, ["firm_id", "lastupdated"])),
            "candidate_supported": int(non_null_count(fund_details, ["fund_id", "firm_id"]) > 0 and non_null_count(manager_details, ["firm_id"]) > 0),
            "note": "Fund-to-manager identifiers are present and support PIT-safe buyout market-quarter sponsor/fund features.",
        },
        {
            "audit_area": "lp_plan_update_dates",
            "source_table": "preqin_investor_details",
            "field_names": "firm_id|next_12_months_quarter|next12monthsallocationmin_pe_usd|next12monthsallocationmax_pe_usd",
            "dated_field_present": int("next_12_months_quarter" in investor_details.columns and investor_details["next_12_months_quarter"].notna().any()),
            "non_null_rows": non_null_count(investor_details, ["firm_id", "next_12_months_quarter", "next12monthsallocationmin_pe_usd", "next12monthsallocationmax_pe_usd"]),
            "candidate_supported": int(non_null_count(investor_details, ["firm_id", "next_12_months_quarter"]) > 0),
            "note": "LP plan/update dates are present, but the staged graph lacks a PIT-safe company or deal-level LP join.",
        },
    ]
    return pd.DataFrame(rows)


def build_deal_fund_link_audit(
    sources: dict[str, pd.DataFrame],
    round_events: pd.DataFrame,
) -> pd.DataFrame:
    buyout = sources.get("preqin_buyout", pd.DataFrame()).copy()
    fund_details = sources.get("preqin_fund_details", pd.DataFrame()).copy()
    manager_details = sources.get("preqin_manager_details", pd.DataFrame()).copy()
    investor_details = sources.get("preqin_investor_details", pd.DataFrame()).copy()
    rows = [
        {
            "link_layer": "company_to_deal",
            "source_table": "round_events",
            "join_keys": "company_id|event_date",
            "keys_present": int(all(column in round_events.columns for column in ["company_id", "event_date"])),
            "dated_link_present": int("event_date" in round_events.columns and pd.to_datetime(round_events["event_date"], errors="coerce").notna().any()),
            "pit_safe_supported": 1,
            "active_status": "active",
            "note": "Reconstructed company-quarter panel links companies to dated deal events.",
        },
        {
            "link_layer": "deal_to_fund_or_firm",
            "source_table": "preqin_buyout",
            "join_keys": "buyout_id|fund_id|firm_id",
            "keys_present": int(any(column in buyout.columns for column in ["fund_id", "firm_id"])),
            "dated_link_present": int("deal_date" in buyout.columns and buyout["deal_date"].notna().any()),
            "pit_safe_supported": int(any(column in buyout.columns for column in ["fund_id", "firm_id"])),
            "active_status": "unsupported_missing_direct_deal_fund_key" if not any(column in buyout.columns for column in ["fund_id", "firm_id"]) else "active",
            "note": "Direct company-deal to sponsor/fund joins remain unavailable unless populated `fund_id` or `firm_id` fields appear in the staged buyout extract.",
        },
        {
            "link_layer": "fund_to_manager",
            "source_table": "preqin_fund_details|preqin_manager_details",
            "join_keys": "fund_id|firm_id|lastupdated",
            "keys_present": int(all(column in fund_details.columns for column in ["fund_id", "firm_id"]) and "firm_id" in manager_details.columns),
            "dated_link_present": int("lastupdated" in manager_details.columns and manager_details["lastupdated"].notna().any()),
            "pit_safe_supported": int(all(column in fund_details.columns for column in ["fund_id", "firm_id"]) and "firm_id" in manager_details.columns),
            "active_status": "active_market_quarter",
            "note": "Fund and manager tables support PIT-safe market-quarter sponsor/fund state features.",
        },
        {
            "link_layer": "manager_to_lp_plan",
            "source_table": "preqin_manager_details|preqin_investor_details",
            "join_keys": "firm_id|next_12_months_quarter",
            "keys_present": int("firm_id" in manager_details.columns and "firm_id" in investor_details.columns),
            "dated_link_present": int("next_12_months_quarter" in investor_details.columns and investor_details["next_12_months_quarter"].notna().any()),
            "pit_safe_supported": int("firm_id" in manager_details.columns and "firm_id" in investor_details.columns),
            "active_status": "active_market_quarter",
            "note": "LP demand can be aggregated to market-quarter sponsor state, but not linked back to specific company deals in the staged graph.",
        },
    ]
    return pd.DataFrame(rows)


def build_buyout_field_inventory(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    inventory_specs = [
        ("preqin_buyout", "deal_date", "dated buyout transaction row", "direct", "none", "portfolio_company_id|buyout_id", "direct_dated", "yes"),
        ("preqin_buyout", "buyout_id", "buyout deal identifier", "undated", "none", "portfolio_company_id|buyout_id", "key_only", "yes"),
        ("preqin_buyout", "fund_id", "fund identifier on buyout deal row", "undated", "none", "buyout_id|fund_id", "bridge_key", "partial"),
        ("preqin_buyout", "firm_id", "firm / sponsor identifier on buyout deal row", "undated", "none", "buyout_id|firm_id", "bridge_key", "partial"),
        ("preqin_buyout", "investment_type", "buyout transaction type", "indirect", "none", "portfolio_company_id|buyout_id", "direct_dated", "yes"),
        ("preqin_buyout", "deal_status", "transaction completion flag", "indirect", "none", "portfolio_company_id|buyout_id", "direct_dated", "yes"),
        ("preqin_buyout", "investment_status", "provider deal lifecycle status", "indirect", "none", "portfolio_company_id|buyout_id", "transition", "no"),
        ("preqin_buyout", "deal_description", "free-text realization mechanics", "indirect", "none", "portfolio_company_id|buyout_id", "text_inference", "no"),
        ("preqin_buyout", "pressreleaseurl", "press-release evidence link", "indirect", "none", "portfolio_company_id|buyout_id", "text_support", "no"),
        ("preqin_buyout", "deal_size_usd", "headline transaction value", "same_row", "amount", "portfolio_company_id|buyout_id", "direct_dated", "yes"),
        ("preqin_buyout", "dealsizeequity_usd", "headline equity value", "same_row", "amount", "portfolio_company_id|buyout_id", "direct_dated", "yes"),
        ("preqin_buyout", "enterprisevalue", "headline enterprise value", "same_row", "amount", "portfolio_company_id|buyout_id", "direct_dated", "yes"),
        ("preqin_buyout", "debtsize_usd", "headline debt size", "same_row", "amount", "portfolio_company_id|buyout_id", "direct_dated", "yes"),
        ("preqin_buyout", "acquired_share_pcent", "ownership transferred", "same_row", "amount", "portfolio_company_id|buyout_id", "direct_dated", "yes"),
        ("preqin_buyout", "firm_about", "sponsor narrative text", "undated", "none", "buyout_id", "weak_text", "no"),
        ("preqin_buyout", "firm_othernames", "alternate sponsor/company text", "undated", "none", "buyout_id", "weak_text", "no"),
        ("preqin_fund_details", "fund_id", "fund identifier", "undated", "none", "fund_id|firm_id", "bridge_key", "yes"),
        ("preqin_fund_details", "firm_id", "manager identifier on fund record", "undated", "none", "fund_id|firm_id", "bridge_key", "yes"),
        ("preqin_fund_details", "fundraising_launch_date", "fund launch date", "direct", "none", "fund_id|firm_id", "market_quarter", "yes"),
        ("preqin_fund_details", "latest_interim_close_date", "interim close date", "direct", "amount", "fund_id|firm_id", "market_quarter", "yes"),
        ("preqin_fund_details", "final_close_date", "final close date", "direct", "amount", "fund_id|firm_id", "market_quarter", "yes"),
        ("preqin_fund_details", "final_size_usd", "final fund size", "same_row", "amount", "fund_id|firm_id", "market_quarter", "yes"),
        ("preqin_fund_details", "latest_interim_close_size_usd", "interim close size", "same_row", "amount", "fund_id|firm_id", "market_quarter", "yes"),
        ("preqin_fund_details", "fund_number_overall", "fund sequence number", "undated", "none", "fund_id|firm_id", "fund_state", "partial"),
        ("preqin_fund_performance", "date_reported", "performance report date", "direct", "none", "fund_id", "market_quarter", "yes"),
        ("preqin_fund_performance", "distr_dpi_pcent", "distributed-to-paid-in", "same_row", "amount", "fund_id", "market_quarter", "yes_with_report_lag"),
        ("preqin_fund_performance", "value_rvpi_pcent", "remaining value to paid-in", "same_row", "amount", "fund_id", "market_quarter", "yes_with_report_lag"),
        ("preqin_fund_performance", "multiple", "fund multiple", "same_row", "amount", "fund_id", "market_quarter", "yes_with_report_lag"),
        ("preqin_manager_details", "firm_id", "manager identifier", "undated", "none", "firm_id", "bridge_key", "yes"),
        ("preqin_manager_details", "lastupdated", "manager record timestamp", "direct", "none", "firm_id", "market_quarter", "yes"),
        ("preqin_manager_details", "totalfundsraised10yearsmn", "sponsor raise history", "same_row", "amount", "firm_id", "market_quarter", "yes_with_update_lag"),
        ("preqin_manager_details", "investorcoinvestmentrights", "co-invest rights indicator", "same_row", "none", "firm_id", "market_quarter", "yes_with_update_lag"),
        ("preqin_investor_details", "firm_id", "investor identifier", "undated", "none", "firm_id", "bridge_key", "yes"),
        ("preqin_investor_details", "next_12_months_quarter", "LP plan date", "direct", "none", "firm_id", "market_quarter", "yes"),
        ("preqin_investor_details", "next12monthsallocationmax_pe_usd", "LP next-12m PE allocation", "same_row", "amount", "firm_id", "market_quarter", "yes"),
        ("preqin_investor_details", "coinvest_with_gp", "LP co-invest preference", "same_row", "none", "firm_id", "market_quarter", "yes"),
        ("preqin_investor_details", "current_pe_allocation_usd", "current LP PE allocation", "undated", "amount", "firm_id", "undated_snapshot", "no"),
        ("preqin_cashflow", "transaction_date", "fund cash-flow date", "direct", "none", "fund_id|firm_id", "market_quarter", "yes"),
        ("preqin_cashflow", "transaction_type", "fund cash-flow type", "same_row", "none", "fund_id|firm_id", "market_quarter", "yes"),
        ("preqin_cashflow", "transaction_amount", "fund cash-flow amount", "same_row", "amount", "fund_id|firm_id", "market_quarter", "yes"),
        ("preqin_cashflow", "net_cashflow", "fund net cash-flow amount", "same_row", "amount", "fund_id|firm_id", "market_quarter", "yes"),
        ("preqin_fund_terms", "fund_id", "fund identifier on terms", "undated", "none", "fund_id", "bridge_key", "yes"),
        ("preqin_fund_terms", "returninginvestorspcent", "returning LP share", "indirect", "amount", "fund_id", "market_quarter", "partial"),
        ("preqin_fund_terms", "investmentperiodyears", "investment period term", "undated", "none", "fund_id", "fund_state", "partial"),
    ]
    rows: list[dict[str, object]] = []
    for table_name, field_name, meaning, date_availability, value_availability, entity_keys, directness_role, pit_safe in inventory_specs:
        frame = sources.get(table_name, pd.DataFrame())
        available = int(field_name in frame.columns)
        non_null_rows = int(frame[field_name].notna().sum()) if available and not frame.empty else 0
        raw_type = str(frame[field_name].dtype) if available and not frame.empty else "missing"
        rows.append(
            {
                "table_name": table_name,
                "field_name": field_name,
                "inferred_economic_meaning": meaning,
                "raw_type": raw_type,
                "field_available": available,
                "non_null_rows": non_null_rows,
                "date_field_availability": date_availability,
                "value_field_availability": value_availability,
                "entity_keys_available": entity_keys,
                "directness_role": directness_role,
                "pit_safe_t_minus_1": pit_safe,
            }
        )
    return pd.DataFrame(rows).sort_values(["table_name", "field_name"]).reset_index(drop=True)


def write_buyout_field_inventory(path: Path, frame: pd.DataFrame) -> None:
    lines = [
        "# Buyout Field Inventory",
        "",
        "- This inventory covers staged local Preqin fields relevant to dated buyout realizations, sponsor/fund joins, and LP-demand joins.",
        "- `pit_safe_t_minus_1` distinguishes fields that can be used directly, only with date/report/update lags, or not safely at all.",
        "",
        dataframe_to_markdown(frame),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_bridge_company_deal(
    company_master: pd.DataFrame,
    buyout: pd.DataFrame,
    universe_map: pd.DataFrame,
) -> pd.DataFrame:
    preqin_map = company_master.dropna(subset=["portfolio_company_id"])[["portfolio_company_id", "company_id"]].drop_duplicates()
    bridge = buyout.merge(preqin_map, on="portfolio_company_id", how="inner")
    bridge = bridge.merge(universe_map, on="company_id", how="left")
    bridge = bridge.loc[bridge["universe"].astype(str).eq("buyout_pe")].copy()
    if bridge.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "deal_id",
                "deal_date",
                "deal_quarter_idx",
                "investment_type",
                "deal_status",
                "investment_status",
                "deal_value_usd",
                "join_confidence_tier",
                "provenance_note",
            ]
        )
    bridge["deal_id"] = bridge["buyout_id"].astype(str)
    bridge["deal_quarter_idx"] = quarter_idx_from_dates(pd.to_datetime(bridge["deal_date"], errors="coerce"))
    bridge["deal_value_usd"] = pd.to_numeric(bridge["deal_size_usd"], errors="coerce").fillna(
        pd.to_numeric(bridge["enterprisevalue"], errors="coerce")
    )
    bridge["join_confidence_tier"] = "deterministic"
    bridge["provenance_note"] = "Exact Preqin portfolio_company_id match."
    bridge = bridge.sort_values(["company_id", "deal_date", "deal_id"]).reset_index(drop=True)
    bridge["company_deal_sequence"] = bridge.groupby("company_id").cumcount() + 1
    return bridge[
        [
            "company_id",
            "portfolio_company_id",
            "deal_id",
            "fund_id",
            "firm_id",
            "deal_date",
            "deal_quarter_idx",
            "company_deal_sequence",
            "investment_type",
            "deal_status",
            "investment_status",
            "deal_description",
            "primary_industry",
            "sub_industries",
            "portfolio_company_country",
            "portfolio_company_region",
            "deal_value_usd",
            "join_confidence_tier",
            "provenance_note",
        ]
    ].copy()


def build_bridge_deal_fund(bridge_company_deal: pd.DataFrame) -> pd.DataFrame:
    if bridge_company_deal.empty or "fund_id" not in bridge_company_deal.columns:
        return pd.DataFrame(
            columns=["deal_id", "fund_id", "join_confidence_tier", "active_status", "provenance_note"]
        )
    frame = bridge_company_deal.dropna(subset=["deal_id", "fund_id"]).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=["deal_id", "fund_id", "join_confidence_tier", "active_status", "provenance_note"]
        )
    frame = frame[["deal_id", "fund_id"]].drop_duplicates().reset_index(drop=True)
    frame["join_confidence_tier"] = "deterministic"
    frame["active_status"] = "active"
    frame["provenance_note"] = "Exact Preqin buyout deal row carries fund_id."
    return frame


def empty_bridge_deal_fund() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["deal_id", "fund_id", "join_confidence_tier", "active_status", "provenance_note"]
    )


def build_bridge_deal_firm(bridge_company_deal: pd.DataFrame) -> pd.DataFrame:
    if bridge_company_deal.empty or "firm_id" not in bridge_company_deal.columns:
        return pd.DataFrame(
            columns=["deal_id", "firm_id", "join_confidence_tier", "active_status", "provenance_note"]
        )
    frame = bridge_company_deal.dropna(subset=["deal_id", "firm_id"]).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=["deal_id", "firm_id", "join_confidence_tier", "active_status", "provenance_note"]
        )
    frame = frame[["deal_id", "firm_id"]].drop_duplicates().reset_index(drop=True)
    frame["join_confidence_tier"] = "deterministic"
    frame["active_status"] = "active"
    frame["provenance_note"] = "Exact Preqin buyout deal row carries firm_id."
    return frame


def empty_bridge_deal_firm() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["deal_id", "firm_id", "join_confidence_tier", "active_status", "provenance_note"]
    )


def build_bridge_company_fund_active_window(
    bridge_company_deal: pd.DataFrame,
    analysis_end_quarter_idx: int | None = None,
) -> pd.DataFrame:
    columns = [
        "company_id",
        "fund_id",
        "active_window_start_quarter",
        "active_window_end_quarter",
        "join_confidence_tier",
        "active_status",
        "provenance_note",
    ]
    if bridge_company_deal.empty or "fund_id" not in bridge_company_deal.columns:
        return pd.DataFrame(columns=columns)
    frame = bridge_company_deal.dropna(subset=["company_id", "fund_id", "deal_quarter_idx"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["deal_quarter_idx"] = pd.to_numeric(frame["deal_quarter_idx"], errors="coerce")
    frame = frame.dropna(subset=["deal_quarter_idx"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["deal_quarter_idx"] = frame["deal_quarter_idx"].astype(int)
    frame = frame.sort_values(["company_id", "deal_quarter_idx", "deal_id"]).reset_index(drop=True)
    frame["next_company_deal_quarter"] = frame.groupby("company_id")["deal_quarter_idx"].shift(-1)
    frame["active_window_start_quarter"] = frame["deal_quarter_idx"] + 1
    frame["active_window_end_quarter"] = np.where(
        frame["next_company_deal_quarter"].notna(),
        pd.to_numeric(frame["next_company_deal_quarter"], errors="coerce") - 1,
        analysis_end_quarter_idx,
    )
    frame = frame.loc[
        pd.to_numeric(frame["active_window_start_quarter"], errors="coerce").notna()
        & pd.to_numeric(frame["active_window_end_quarter"], errors="coerce").notna()
        & (
            pd.to_numeric(frame["active_window_start_quarter"], errors="coerce")
            <= pd.to_numeric(frame["active_window_end_quarter"], errors="coerce")
        )
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["join_confidence_tier"] = "deterministic"
    frame["active_status"] = np.where(
        frame["next_company_deal_quarter"].notna(),
        "closed_on_next_company_deal",
        "active_until_analysis_end",
    )
    frame["provenance_note"] = "Company-active fund window implied by consecutive deterministic buyout deal rows."
    return frame[columns].drop_duplicates().reset_index(drop=True)


def empty_bridge_company_fund_active_window() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "company_id",
            "fund_id",
            "active_window_start_quarter",
            "active_window_end_quarter",
            "join_confidence_tier",
            "active_status",
            "provenance_note",
        ]
    )


def classify_buyout_realization_event(
    investment_type: object,
    deal_description: object,
    has_prior_buyout: bool,
    gap_quarters: int,
    min_gap_quarters: int,
) -> tuple[str | None, str | None, str | None, str, str, str]:
    if not has_prior_buyout or gap_quarters < min_gap_quarters:
        return None, None, None, "other", "unclassified", "Requires a prior dated buyout row and a minimum hold gap."
    investment_text = str(investment_type).strip().lower() if pd.notna(investment_type) else ""
    description_text = str(deal_description).strip().lower() if pd.notna(deal_description) else ""
    if any(term in description_text for term in ["continuation vehicle", "continuation fund", "gp-led continuation", "continuation"]):
        return "continuation_vehicle", "continuation_vehicle", "continuation_vehicle", "direct_dated", "medium", "Continuation-style realization text on a later dated buyout row."
    if any(term in investment_text for term in ["recapitalisation", "recapitalization"]) or any(
        term in description_text for term in ["recapitalis", "recapitaliz", "dividend recap", "dividend recapital", "refinanc"]
    ):
        return "recapitalization", "recapitalization", "recapitalization", "direct_dated", "high" if "recap" in investment_text else "medium", "Recapitalization-style later dated buyout row."
    if any(term in description_text for term in ["partial realization", "partial sale", "partial exit", "divest"]) or "partial" in investment_text:
        return "partial_realization", "partial_realization", "partial_realization", "direct_dated", "medium", "Partial-realization wording on a later dated buyout row."
    if investment_text == "merger" or any(term in description_text for term in ["trade sale", "strategic sale", "merger", "acquisition", "acquired by"]):
        return "mna", "merger_or_trade_sale", "mna", "direct_dated", "high" if investment_text == "merger" else "medium", "Explicit strategic sale / merger wording on a later dated row."
    if any(term in description_text for term in ["secondary buyout", "secondary sale", "sold to another sponsor", "sold to private equity", "sold to pe", "sold to sponsor"]):
        return "secondary_like_realization", "secondary_like_realization", "sponsor_sale", "direct_dated", "high", "Explicit secondary-style wording on a later dated buyout row."
    if investment_text in {"buyout", "public to private"}:
        return "sponsor_sale", "repeat_buyout_company", "sponsor_sale", "direct_dated", "medium", "Later completed buyout on the same company after a prior dated buyout row."
    return None, None, None, "other", "unclassified", "Later row exists but realization mechanics are not explicit enough."


def build_fact_buyout_realization_event(
    company_master: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    bridge_company_deal: pd.DataFrame,
    min_gap_quarters: int,
) -> pd.DataFrame:
    buyout_company_ids = set(bridge_company_deal["company_id"].astype(str))
    rows: list[dict[str, object]] = []
    cb_id_map = company_master.dropna(subset=["cb_company_uuid"])[["company_id", "cb_company_uuid"]].drop_duplicates()
    cb_id_map = cb_id_map.rename(columns={"cb_company_uuid": "company_uuid"})
    cb_ipos = sources.get("cb_ipos", pd.DataFrame()).merge(cb_id_map, on="company_uuid", how="inner")
    cb_ipos = cb_ipos.loc[cb_ipos["company_id"].astype(str).isin(buyout_company_ids)].copy()
    for row in cb_ipos.itertuples(index=False):
        event_date = pd.to_datetime(getattr(row, "went_public_on"), errors="coerce")
        if pd.isna(event_date):
            continue
        rows.append(
            {
                "company_id": getattr(row, "company_id"),
                "deal_id": np.nan,
                "fund_id": np.nan,
                "firm_id": np.nan,
                "event_id": f"cb_ipo::{getattr(row, 'company_id')}::{event_date.date()}",
                "event_date": event_date,
                "event_quarter": quarter_label_from_idx(int(quarter_idx_from_dates(pd.Series([event_date])).iloc[0])),
                "event_quarter_idx": int(quarter_idx_from_dates(pd.Series([event_date])).iloc[0]),
                "realization_type": "ipo",
                "realization_subtype": "crunchbase_ipo",
                "headline_route_family": "ipo",
                "value_amount": pd.to_numeric(getattr(row, "money_raised_usd"), errors="coerce"),
                "value_currency": "USD",
                "directness_class": "direct_dated",
                "exit_label_confidence": "high",
                "source_system": "crunchbase",
                "source_table": "cb_ipos",
                "source_field_date": "went_public_on",
                "source_field_value": "money_raised_usd",
                "proceeds_available_flag": int(pd.notna(getattr(row, "money_raised_usd"))),
                "ownership_link_available_flag": 0,
                "pit_safe_flag": 1,
                "provenance_note": "Direct dated IPO event from Crunchbase for a buyout-universe company.",
                "route_source": "crunchbase_ipo",
                "hold_period_q": np.nan,
            }
        )
    cb_acq = sources.get("cb_acquisitions", pd.DataFrame()).merge(
        cb_id_map,
        left_on="acquiree_uuid",
        right_on="company_uuid",
        how="inner",
    )
    cb_acq = cb_acq.loc[cb_acq["company_id"].astype(str).isin(buyout_company_ids)].copy()
    for row in cb_acq.itertuples(index=False):
        event_date = pd.to_datetime(getattr(row, "announced_on"), errors="coerce")
        if pd.isna(event_date):
            continue
        rows.append(
            {
                "company_id": getattr(row, "company_id"),
                "deal_id": np.nan,
                "fund_id": np.nan,
                "firm_id": np.nan,
                "event_id": f"cb_acq::{getattr(row, 'company_id')}::{event_date.date()}",
                "event_date": event_date,
                "event_quarter": quarter_label_from_idx(int(quarter_idx_from_dates(pd.Series([event_date])).iloc[0])),
                "event_quarter_idx": int(quarter_idx_from_dates(pd.Series([event_date])).iloc[0]),
                "realization_type": "mna",
                "realization_subtype": "crunchbase_acquisition",
                "headline_route_family": "mna",
                "value_amount": pd.to_numeric(getattr(row, "price_usd"), errors="coerce"),
                "value_currency": "USD",
                "directness_class": "direct_dated",
                "exit_label_confidence": "high",
                "source_system": "crunchbase",
                "source_table": "cb_acquisitions",
                "source_field_date": "announced_on",
                "source_field_value": "price_usd",
                "proceeds_available_flag": int(pd.notna(getattr(row, "price_usd"))),
                "ownership_link_available_flag": 0,
                "pit_safe_flag": 1,
                "provenance_note": "Direct dated acquisition event from Crunchbase for a buyout-universe company.",
                "route_source": "crunchbase_acquisition",
                "hold_period_q": np.nan,
            }
        )

    repeated = bridge_company_deal.copy()
    if not repeated.empty:
        repeated["deal_date"] = pd.to_datetime(repeated["deal_date"], errors="coerce")
        repeated = repeated.loc[repeated["deal_date"].notna()].copy()
        repeated = repeated.sort_values(["company_id", "deal_date", "deal_id"]).reset_index(drop=True)
        repeated["prior_buyout_quarter"] = repeated.groupby("company_id")["deal_quarter_idx"].shift(1)
        repeated["prior_buyout_date"] = repeated.groupby("company_id")["deal_date"].shift(1)
        repeated["gap_quarters"] = (
            pd.to_numeric(repeated["deal_quarter_idx"], errors="coerce")
            - pd.to_numeric(repeated["prior_buyout_quarter"], errors="coerce")
        ).fillna(0).astype(int)
        repeated["has_prior_buyout"] = repeated["prior_buyout_quarter"].notna().astype(int)
        repeated["deal_status_clean"] = repeated["deal_status"].astype(str).str.lower()
        repeated = repeated.loc[repeated["deal_status_clean"].str.contains("completed")].copy()
        for row in repeated.itertuples(index=False):
            realization_type, realization_subtype, headline_route_family, directness_class, confidence_tier, provenance = classify_buyout_realization_event(
                getattr(row, "investment_type"),
                getattr(row, "deal_description"),
                bool(getattr(row, "has_prior_buyout")),
                int(getattr(row, "gap_quarters", 0)),
                int(min_gap_quarters),
            )
            if realization_type is None:
                continue
            event_date = pd.to_datetime(getattr(row, "deal_date"), errors="coerce")
            event_quarter_idx = int(getattr(row, "deal_quarter_idx"))
            rows.append(
                {
                    "company_id": getattr(row, "company_id"),
                    "deal_id": getattr(row, "deal_id"),
                    "fund_id": getattr(row, "fund_id", np.nan),
                    "firm_id": getattr(row, "firm_id", np.nan),
                    "event_id": f"preqin_buyout::{getattr(row, 'company_id')}::{getattr(row, 'deal_id')}",
                    "event_date": event_date,
                    "event_quarter": quarter_label_from_idx(event_quarter_idx),
                    "event_quarter_idx": event_quarter_idx,
                    "realization_type": realization_type,
                    "realization_subtype": realization_subtype,
                    "headline_route_family": headline_route_family,
                    "value_amount": pd.to_numeric(getattr(row, "deal_value_usd"), errors="coerce"),
                    "value_currency": "USD",
                    "directness_class": directness_class,
                    "exit_label_confidence": confidence_tier,
                    "source_system": "preqin",
                    "source_table": "preqin_buyout",
                    "source_field_date": "deal_date",
                    "source_field_value": "deal_size_usd|enterprisevalue",
                    "proceeds_available_flag": int(pd.notna(getattr(row, "deal_value_usd"))),
                    "ownership_link_available_flag": int(
                        pd.notna(getattr(row, "fund_id", np.nan)) or pd.notna(getattr(row, "firm_id", np.nan))
                    ),
                    "pit_safe_flag": 1,
                    "provenance_note": provenance,
                    "route_source": f"preqin_buyout_{realization_subtype}",
                    "hold_period_q": int(getattr(row, "gap_quarters", 0)),
                }
            )
    events = pd.DataFrame(rows)
    if events.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "deal_id",
                "fund_id",
                "firm_id",
                "event_id",
                "event_date",
                "event_quarter",
                "event_quarter_idx",
                "realization_type",
                "realization_subtype",
                "headline_route_family",
                "value_amount",
                "value_currency",
                "directness_class",
                "exit_label_confidence",
                "source_system",
                "source_table",
                "source_field_date",
                "source_field_value",
                "proceeds_available_flag",
                "ownership_link_available_flag",
                "pit_safe_flag",
                "provenance_note",
                "route_source",
                "hold_period_q",
            ]
        )
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.sort_values(["company_id", "event_date", "event_id"]).drop_duplicates("event_id").reset_index(drop=True)
    if not events.empty:
        events["fund_id"] = events["fund_id"].where(events["fund_id"].notna(), np.nan)
        events["firm_id"] = events["firm_id"].where(events["firm_id"].notna(), np.nan)
        collapse_keys = ["company_id", "event_date", "headline_route_family", "realization_type", "realization_subtype"]
        collapsed_rows: list[dict[str, object]] = []
        for _, frame in events.groupby(collapse_keys, dropna=False):
            row = frame.sort_values(
                ["pit_safe_flag", "proceeds_available_flag", "ownership_link_available_flag", "exit_label_confidence"],
                ascending=[False, False, False, True],
            ).iloc[0].to_dict()
            row["source_system"] = "|".join(sorted({str(value) for value in frame["source_system"].dropna().tolist()}))
            row["source_table"] = "|".join(sorted({str(value) for value in frame["source_table"].dropna().tolist()}))
            row["source_field_date"] = "|".join(sorted({str(value) for value in frame["source_field_date"].dropna().tolist()}))
            row["source_field_value"] = "|".join(sorted({str(value) for value in frame["source_field_value"].dropna().tolist()}))
            row["route_source"] = "|".join(sorted({str(value) for value in frame["route_source"].dropna().tolist()}))
            row["provenance_note"] = " | ".join(sorted({str(value) for value in frame["provenance_note"].dropna().tolist()}))
            if frame["fund_id"].notna().any():
                row["fund_id"] = frame["fund_id"].dropna().astype(str).iloc[0]
            if frame["firm_id"].notna().any():
                row["firm_id"] = frame["firm_id"].dropna().astype(str).iloc[0]
            row["ownership_link_available_flag"] = int(frame["ownership_link_available_flag"].max())
            row["proceeds_available_flag"] = int(frame["proceeds_available_flag"].max())
            collapsed_rows.append(row)
        events = pd.DataFrame(collapsed_rows).sort_values(["company_id", "event_date", "event_id"]).reset_index(drop=True)
    return events


def build_buyout_realization_event_audit(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "realization_type",
                "headline_route_family",
                "directness_class",
                "exit_label_confidence",
                "source_system",
                "event_count",
                "proceeds_available_events",
                "ownership_link_available_events",
            ]
        )
    audit = (
        events.groupby(
            ["realization_type", "headline_route_family", "directness_class", "exit_label_confidence", "source_system"],
            as_index=False,
        )
        .agg(
            event_count=("event_id", "size"),
            proceeds_available_events=("proceeds_available_flag", "sum"),
            ownership_link_available_events=("ownership_link_available_flag", "sum"),
        )
        .sort_values(["realization_type", "directness_class", "source_system"])
        .reset_index(drop=True)
    )
    return audit


def write_buyout_realization_event_audit(path: Path, frame: pd.DataFrame) -> None:
    lines = [
        "# Buyout Realization Event Audit",
        "",
        "- `direct_dated` events are directly dated transactions from Crunchbase or later dated Preqin buyout rows on the same company.",
        "- The staged local export still lacks direct deal-to-fund ownership joins, so ownership-link coverage remains limited even when the event date is direct.",
        "",
        dataframe_to_markdown(frame),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_join_confidence_summary(
    bridge_company_deal: pd.DataFrame,
    bridge_deal_fund: pd.DataFrame,
    bridge_deal_firm: pd.DataFrame,
    bridge_company_fund_active_window: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for bridge_name, frame in [
        ("bridge_company_deal", bridge_company_deal),
        ("bridge_deal_fund", bridge_deal_fund),
        ("bridge_deal_firm", bridge_deal_firm),
        ("bridge_company_fund_active_window", bridge_company_fund_active_window),
    ]:
        if frame.empty:
            rows.append(
                {
                    "bridge_name": bridge_name,
                    "rows": 0,
                    "deterministic_rows": 0,
                    "fuzzy_rows": 0,
                    "unsupported_rows": 0,
                    "note": "No active rows emitted.",
                }
            )
            continue
        confidence = frame.get("join_confidence_tier", pd.Series(index=frame.index, dtype=object)).astype(str).str.lower()
        rows.append(
            {
                "bridge_name": bridge_name,
                "rows": int(len(frame)),
                "deterministic_rows": int(confidence.eq("deterministic").sum()),
                "fuzzy_rows": int(confidence.eq("fuzzy").sum()),
                "unsupported_rows": int(confidence.eq("unsupported").sum()),
                "note": "Bridge row counts by join confidence.",
            }
        )
    return pd.DataFrame(rows)


def attach_buyout_realization_features(
    buyout_panel: pd.DataFrame,
    bridge_company_deal: pd.DataFrame,
    fact_buyout_realization_event: pd.DataFrame,
    buyout_market_panel: pd.DataFrame,
) -> pd.DataFrame:
    enriched = buyout_panel.copy()
    if enriched.empty:
        return enriched
    first_deal = bridge_company_deal.groupby("company_id", as_index=False).agg(
        first_buyout_quarter=("deal_quarter_idx", "min")
    )
    enriched = enriched.merge(first_deal, on="company_id", how="left")
    enriched = enriched.loc[
        pd.to_numeric(enriched["first_buyout_quarter"], errors="coerce").notna()
        & (
            pd.to_numeric(enriched["quarter_idx"], errors="coerce")
            > pd.to_numeric(enriched["first_buyout_quarter"], errors="coerce")
        )
    ].copy()
    enriched["time_since_acquisition_q"] = (
        pd.to_numeric(enriched["quarter_idx"], errors="coerce")
        - pd.to_numeric(enriched["first_buyout_quarter"], errors="coerce")
    ).clip(lower=0)
    direct_events = fact_buyout_realization_event.loc[
        fact_buyout_realization_event["directness_class"].astype(str).eq("direct_dated")
        & fact_buyout_realization_event["hold_period_q"].notna()
    ].copy()
    if not direct_events.empty:
        hold_window = direct_events.groupby("event_quarter_idx", as_index=False).agg(
            historical_hold_window_lagged=("hold_period_q", "median")
        )
        hold_window["historical_hold_window_lagged"] = pd.to_numeric(
            hold_window["historical_hold_window_lagged"], errors="coerce"
        ).fillna(0.0)
    else:
        hold_window = pd.DataFrame(columns=["event_quarter_idx", "historical_hold_window_lagged"])
    market = buyout_market_panel.copy()
    if market.empty:
        market = pd.DataFrame(columns=["quarter_idx", *BUYOUT_SPONSOR_FUND_FEATURES])
    market["fund_dry_powder_proxy_lagged"] = np.maximum(
        pd.to_numeric(market.get("buyout_fund_final_close_usd_l4q"), errors="coerce").fillna(0.0)
        - pd.to_numeric(market.get("buyout_fund_net_cashflow_l4q"), errors="coerce").fillna(0.0),
        0.0,
    )
    market["lp_demand_index_lagged"] = np.log1p(
        pd.to_numeric(market.get("buyout_lp_next12m_allocation_usd_lagged"), errors="coerce").fillna(0.0)
    )
    market = market.merge(
        hold_window.rename(columns={"event_quarter_idx": "quarter_idx"}),
        on="quarter_idx",
        how="left",
    )
    base = enriched.reset_index(drop=False).rename(columns={"index": "_row_id"})
    base["lookup_idx"] = pd.to_numeric(base["quarter_idx"], errors="coerce").fillna(0).astype(int) - 1
    market = market.sort_values("quarter_idx").reset_index(drop=True)
    market["quarter_idx"] = pd.to_numeric(market["quarter_idx"], errors="coerce").fillna(0).astype(int)
    market_columns = [
        "historical_hold_window_lagged",
        "fund_dry_powder_proxy_lagged",
        "lp_demand_index_lagged",
        *[column for column in BUYOUT_SPONSOR_FUND_FEATURES if column in market.columns],
    ]
    market = market.rename(columns={column: f"{column}_market" for column in market_columns if column in market.columns})
    merged = pd.merge_asof(
        base.sort_values("lookup_idx"),
        market.rename(columns={"quarter_idx": "market_quarter_idx"}),
        left_on="lookup_idx",
        right_on="market_quarter_idx",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("_row_id").reset_index(drop=True)
    for column in market_columns:
        merged[column] = pd.to_numeric(
            merged.get(f"{column}_market", pd.Series(index=merged.index, dtype=float)),
            errors="coerce",
        ).fillna(0.0)
    return merged.drop(
        columns=["_row_id", "lookup_idx", "market_quarter_idx", *[f"{column}_market" for column in market_columns]],
        errors="ignore",
    )


def build_buyout_feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_name": "time_since_acquisition_q",
                "actual_column": "time_since_acquisition_q",
                "feature_block": "hold_period",
                "scope": "company_deal",
                "active_status": "active",
                "pit_rule": "Quarter t uses the first dated buyout quarter observed on or before t-1.",
                "note": "Direct company-level hold-period clock from the first staged buyout deal.",
            },
            {
                "feature_name": "historical_hold_window_lagged",
                "actual_column": "historical_hold_window_lagged",
                "feature_block": "hold_period",
                "scope": "buyout_market_quarter",
                "active_status": "active_market_quarter",
                "pit_rule": "Rolling median hold window from prior direct-dated buyout realizations, merged at quarter t-1.",
                "note": "Market-level hold-window context, not sponsor-specific.",
            },
            {
                "feature_name": "fund_dry_powder_proxy_lagged",
                "actual_column": "fund_dry_powder_proxy_lagged",
                "feature_block": "sponsor_fund_state",
                "scope": "buyout_market_quarter",
                "active_status": "active_market_quarter",
                "pit_rule": "Lagged market-quarter close cash minus fund cash-flow proxy, merged at quarter t-1.",
                "note": "Market-quarter proxy because direct deal-to-fund ownership links are unavailable.",
            },
            {
                "feature_name": "lp_demand_index_lagged",
                "actual_column": "lp_demand_index_lagged",
                "feature_block": "lp_demand_state",
                "scope": "buyout_market_quarter",
                "active_status": "active_market_quarter",
                "pit_rule": "Log next-12m LP PE allocation aggregated by quarter and lagged to t-1.",
                "note": "Market-quarter LP-demand state only; no company-linked LP join is available.",
            },
            {
                "feature_name": "sponsor_raise_10y_lagged",
                "actual_column": "buyout_sponsor_raise_10y_lagged",
                "feature_block": "sponsor_fund_state",
                "scope": "buyout_market_quarter",
                "active_status": "active_market_quarter",
                "pit_rule": "Manager update dates are lagged before quarter merges.",
                "note": "Sponsor-raise history aggregated to market quarter.",
            },
            {
                "feature_name": "co_invest_participation_proxy_lagged",
                "actual_column": "buyout_sponsor_coinvest_share_lagged",
                "feature_block": "lp_demand_state",
                "scope": "buyout_market_quarter",
                "active_status": "active_market_quarter",
                "pit_rule": "Manager co-invest rights share is lagged at manager update dates.",
                "note": "Market-quarter proxy for co-invest openness.",
            },
            {
                "feature_name": "returning_lp_share_lagged",
                "actual_column": "buyout_returning_lp_pct_lagged",
                "feature_block": "lp_demand_state",
                "scope": "buyout_market_quarter",
                "active_status": "active_market_quarter",
                "pit_rule": "Terms-derived returning LP share is dated conservatively from close/update fields and merged at t-1.",
                "note": "Fund-market aggregate only.",
            },
            {
                "feature_name": "fundraising_pressure_proxy_lagged",
                "actual_column": "buyout_fund_months_to_final_close_lagged",
                "feature_block": "lp_demand_state",
                "scope": "buyout_market_quarter",
                "active_status": "active_market_quarter",
                "pit_rule": "Quarter t uses the lagged market-quarter median months-to-final-close from dated fund launch and close fields.",
                "note": "Longer time to final close proxies tighter fundraising conditions at the market-quarter level.",
            },
            {
                "feature_name": "hold_period_percentile_vs_sponsor_history",
                "actual_column": "",
                "feature_block": "hold_period",
                "scope": "company_specific_sponsor",
                "active_status": "unsupported",
                "pit_rule": "Unsupported because the staged buyout extract lacks direct deal-to-sponsor ownership links.",
                "note": "Requires company-linked sponsor history.",
            },
            {
                "feature_name": "hold_period_percentile_vs_sector_region_history",
                "actual_column": "",
                "feature_block": "hold_period",
                "scope": "company_specific_sector_region",
                "active_status": "unsupported",
                "pit_rule": "Deferred to avoid a second heavy sweep in this pass.",
                "note": "Could be added later from the direct event spine if needed.",
            },
            {
                "feature_name": "sponsor_realization_rate_lagged",
                "actual_column": "",
                "feature_block": "sponsor_fund_state",
                "scope": "company_specific_sponsor",
                "active_status": "unsupported",
                "pit_rule": "Unsupported because the staged buyout extract lacks direct company-to-sponsor ownership joins.",
                "note": "Would require deal-to-firm links.",
            },
            {
                "feature_name": "sponsor_partial_realization_rate_lagged",
                "actual_column": "",
                "feature_block": "sponsor_fund_state",
                "scope": "company_specific_sponsor",
                "active_status": "unsupported",
                "pit_rule": "Unsupported because the staged buyout extract lacks direct company-to-sponsor ownership joins.",
                "note": "Would require deal-to-firm links.",
            },
            {
                "feature_name": "sponsor_secondary_like_rate_lagged",
                "actual_column": "",
                "feature_block": "sponsor_fund_state",
                "scope": "company_specific_sponsor",
                "active_status": "unsupported",
                "pit_rule": "Unsupported because the staged buyout extract lacks direct company-to-sponsor ownership joins.",
                "note": "Would require deal-to-firm links.",
            },
            {
                "feature_name": "fund_age_q",
                "actual_column": "",
                "feature_block": "sponsor_fund_state",
                "scope": "company_specific_fund",
                "active_status": "unsupported",
                "pit_rule": "Unsupported because direct company-to-fund links are missing.",
                "note": "Would require deal-to-fund joins.",
            },
            {
                "feature_name": "months_since_first_close",
                "actual_column": "",
                "feature_block": "sponsor_fund_state",
                "scope": "company_specific_fund",
                "active_status": "unsupported",
                "pit_rule": "Unsupported because direct company-to-fund links are missing.",
                "note": "Would require deal-to-fund joins.",
            },
            {
                "feature_name": "months_since_final_close",
                "actual_column": "",
                "feature_block": "sponsor_fund_state",
                "scope": "company_specific_fund",
                "active_status": "unsupported",
                "pit_rule": "Unsupported because direct company-to-fund links are missing.",
                "note": "Would require deal-to-fund joins.",
            },
            {
                "feature_name": "fund_sequence_number",
                "actual_column": "",
                "feature_block": "sponsor_fund_state",
                "scope": "company_specific_fund",
                "active_status": "unsupported",
                "pit_rule": "Unsupported because direct company-to-fund links are missing.",
                "note": "Would require deal-to-fund joins.",
            },
            {
                "feature_name": "sponsor_sector_match",
                "actual_column": "",
                "feature_block": "sponsor_fund_state",
                "scope": "company_specific_sponsor",
                "active_status": "unsupported",
                "pit_rule": "Unsupported because direct sponsor ownership links are missing.",
                "note": "Would require deal-to-firm links.",
            },
        ]
    )


def build_buyout_feature_coverage(
    buyout_panel: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row in feature_dictionary.itertuples(index=False):
        column = str(row.actual_column)
        if column and column in buyout_panel.columns:
            values = pd.to_numeric(buyout_panel[column], errors="coerce")
            non_null = int(values.notna().sum())
            nonzero = int(values.fillna(0.0).ne(0.0).sum())
            coverage_share = safe_ratio(non_null, int(len(buyout_panel)))
        else:
            non_null = 0
            nonzero = 0
            coverage_share = 0.0
        rows.append(
            {
                "feature_name": row.feature_name,
                "actual_column": column,
                "feature_block": row.feature_block,
                "scope": row.scope,
                "active_status": row.active_status,
                "rows": int(len(buyout_panel)),
                "non_null_rows": non_null,
                "nonzero_rows": nonzero,
                "coverage_share": coverage_share,
                "note": row.note,
            }
        )
    return pd.DataFrame(rows)


def build_buyout_feature_availability_by_quarter(
    buyout_panel: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if buyout_panel.empty:
        return pd.DataFrame(
            columns=["quarter_idx", "quarter_label", "feature_name", "active_rows", "coverage_share"]
        )
    total_rows = (
        buyout_panel.groupby("quarter_idx", as_index=False)
        .agg(rows=("company_id", "size"))
        .rename(columns={"rows": "quarter_rows"})
    )
    for row in feature_dictionary.itertuples(index=False):
        column = str(row.actual_column)
        if not column or column not in buyout_panel.columns:
            continue
        grouped = (
            buyout_panel.assign(_active=pd.to_numeric(buyout_panel[column], errors="coerce").fillna(0.0).ne(0.0).astype(int))
            .groupby("quarter_idx", as_index=False)
            .agg(active_rows=("_active", "sum"))
        ).merge(total_rows, on="quarter_idx", how="left")
        grouped["feature_name"] = row.feature_name
        grouped["quarter_label"] = grouped["quarter_idx"].map(lambda value: quarter_label_from_idx(int(value)))
        grouped["coverage_share"] = grouped.apply(
            lambda x: safe_ratio(x["active_rows"], x["quarter_rows"]),
            axis=1,
        )
        rows.extend(grouped[["quarter_idx", "quarter_label", "feature_name", "active_rows", "coverage_share"]].to_dict("records"))
    return pd.DataFrame(rows).sort_values(["quarter_idx", "feature_name"]).reset_index(drop=True)


def write_buyout_feature_pit_rules(path: Path, feature_dictionary: pd.DataFrame) -> None:
    lines = [
        "# Buyout Feature PIT Rules",
        "",
        "- Company-deal features are observable only after the dated buyout event quarter and are merged using quarter t-1 lookups where needed.",
        "- Fund, manager, terms, and LP-demand market-quarter features are merged using dated launch, close, report, update, or plan quarters lagged to t-1.",
        "- Unsupported company-linked sponsor/fund features are listed explicitly rather than approximated through unavailable ownership joins.",
        "",
    ]
    for row in feature_dictionary.itertuples(index=False):
        lines.append(f"- `{row.feature_name}`: {row.pit_rule}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
def build_buyout_missing_field_manifest(
    buyout_realization_field_audit: pd.DataFrame,
    deal_fund_link_audit: pd.DataFrame,
    sponsor_fund_join_audit: pd.DataFrame,
    lp_demand_join_audit: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "missing_component",
        "source_table",
        "required_fields",
        "active_status",
        "blocks_target_candidates",
        "reason",
    ]
    if buyout_realization_field_audit.empty and deal_fund_link_audit.empty and sponsor_fund_join_audit.empty and lp_demand_join_audit.empty:
        return pd.DataFrame(
            [
                {
                    "missing_component": "licensed_buyout_realization_support",
                    "source_table": "scenario_pack",
                    "required_fields": "not_applicable",
                    "active_status": "not_applicable_sample_mode",
                    "blocks_target_candidates": "",
                    "reason": "Sample mode does not audit licensed-data buyout realization mechanics.",
                }
            ],
            columns=columns,
        )

    rows: list[dict[str, object]] = []
    block_map = {
        "partial_realizations": "partial_or_full_realization_by_12q|partial_or_full_realization_by_16q",
        "recapitalizations": "recap_or_secondary_like_realization_by_12q|recap_or_secondary_like_realization_by_16q|recap_or_secondary_or_exit_by_12q|recap_or_secondary_or_exit_by_16q",
        "continuation_vehicle_events": "recap_or_secondary_like_realization_by_12q|recap_or_secondary_like_realization_by_16q|recap_or_secondary_or_exit_by_12q|recap_or_secondary_or_exit_by_16q",
        "secondary_like_realizations": "recap_or_secondary_like_realization_by_12q|recap_or_secondary_like_realization_by_16q|recap_or_secondary_or_exit_by_12q|recap_or_secondary_or_exit_by_16q",
        "deal_to_fund_or_firm": "company_specific_sponsor_or_fund_join_features",
        "deal_to_fund_or_firm_link": "company_specific_sponsor_or_fund_join_features",
        "lp_plan_update_dates": "company_or_deal_linked_lp_demand_features",
    }

    for row in buyout_realization_field_audit.itertuples(index=False):
        audit_area = str(row.audit_area)
        if int(getattr(row, "candidate_supported", 0)) == 1 or audit_area not in block_map:
            continue
        rows.append(
            {
                "missing_component": audit_area,
                "source_table": str(row.source_table),
                "required_fields": str(row.field_names),
                "active_status": "unsupported_missing_dated_field",
                "blocks_target_candidates": block_map[audit_area],
                "reason": str(row.note),
            }
        )

    existing_components = {str(row["missing_component"]) for row in rows}
    for row in deal_fund_link_audit.itertuples(index=False):
        link_layer = str(row.link_layer)
        if link_layer != "deal_to_fund_or_firm" or int(getattr(row, "pit_safe_supported", 0)) == 1:
            continue
        if "deal_to_fund_or_firm_link" in existing_components:
            continue
        rows.append(
            {
                "missing_component": link_layer,
                "source_table": str(row.source_table),
                "required_fields": str(row.join_keys),
                "active_status": str(row.active_status),
                "blocks_target_candidates": block_map[link_layer],
                "reason": str(row.note),
            }
        )
        existing_components.add(link_layer)

    if not sponsor_fund_join_audit.empty:
        join_row = sponsor_fund_join_audit.loc[
            sponsor_fund_join_audit["join_scope"].astype(str).eq("deal_to_fund_or_firm")
        ].head(1)
        if (
            not join_row.empty
            and int(join_row["point_in_time_join_supported"].iloc[0]) == 0
            and "deal_to_fund_or_firm_link" not in existing_components
            and "deal_to_fund_or_firm" not in existing_components
        ):
            rows.append(
                {
                    "missing_component": "direct_company_to_fund_join",
                    "source_table": "round_events|preqin_buyout",
                    "required_fields": "fund_id|firm_id|lead_fund_id",
                    "active_status": str(join_row["active_status"].iloc[0]),
                    "blocks_target_candidates": "company_specific_sponsor_or_fund_join_features",
                    "reason": str(join_row["note"].iloc[0]),
                }
            )

    if not lp_demand_join_audit.empty:
        lp_row = lp_demand_join_audit.head(1)
        if int(lp_row["point_in_time_join_supported"].iloc[0]) == 0:
            rows.append(
                {
                    "missing_component": "company_linked_lp_demand_join",
                    "source_table": "preqin_investor_details",
                    "required_fields": "firm_id|next_12_months_quarter",
                    "active_status": str(lp_row["active_status"].iloc[0]),
                    "blocks_target_candidates": "company_or_deal_linked_lp_demand_features",
                    "reason": str(lp_row["note"].iloc[0]),
                }
            )

    output = pd.DataFrame(rows, columns=columns)
    if output.empty:
        return pd.DataFrame(columns=columns)
    return output.drop_duplicates().reset_index(drop=True)


def write_buyout_missing_field_manifest(path: Path, frame: pd.DataFrame) -> None:
    lines = [
        "# Buyout Missing-Field Manifest",
        "",
        "- This manifest records unavailable or unsupported dated realization mechanics and join layers.",
        "- Unsupported fields are not approximated silently; blocked buyout candidates remain definition-only or provisional.",
        "",
        dataframe_to_markdown(frame),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_confidence_mask_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mask_name": "exit_label_confidence_high",
                "mask_scope": "target_positive_rows",
                "inclusion_rule": "All negative rows plus positive rows with high exit-label confidence.",
                "reporting_role": "default_high_confidence_target_robustness",
            },
            {
                "mask_name": "exit_label_confidence_high_or_medium",
                "mask_scope": "target_positive_rows",
                "inclusion_rule": "All negative rows plus positive rows with high or medium exit-label confidence.",
                "reporting_role": "supplemental_exit_label_slice",
            },
            {
                "mask_name": "entity_match_confidence_high",
                "mask_scope": "all_rows",
                "inclusion_rule": "Rows whose entity match is high-confidence.",
                "reporting_role": "separate_entity_match_diagnostic",
            },
            {
                "mask_name": "confidence_overlap",
                "mask_scope": "all_rows",
                "inclusion_rule": "Rows that satisfy both high entity-match confidence and high exit-label confidence when positive.",
                "reporting_role": "intersection_diagnostic",
            },
        ]
    )


def selected_target_view_frame(
    leaderboard_validation: pd.DataFrame,
    evaluation_metrics_targets: pd.DataFrame,
    evaluation_view: str,
) -> pd.DataFrame:
    selected = leaderboard_validation[["target_key", "selected_feature_backbone", "validation_rank", "selection_basis"]].drop_duplicates()
    subset = evaluation_metrics_targets.loc[
        evaluation_metrics_targets["evaluation_view"].astype(str).eq(evaluation_view)
    ].copy()
    output = selected.merge(
        subset,
        left_on=["target_key", "selected_feature_backbone"],
        right_on=["target_key", "feature_backbone"],
        how="left",
    ).drop(columns=["feature_backbone"], errors="ignore")
    return output.sort_values(["validation_rank", "target_key"]).reset_index(drop=True)


def write_markdown_table_report(
    path: Path,
    title: str,
    intro_lines: list[str],
    frame: pd.DataFrame,
) -> None:
    lines = [f"# {title}", ""]
    lines.extend(intro_lines)
    if intro_lines:
        lines.append("")
    lines.append(dataframe_to_markdown(frame))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_buyout_target_registry(
    fact_buyout_realization_event: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    direct_total = int(
        fact_buyout_realization_event["directness_class"].astype(str).eq("direct_dated").sum()
        if not fact_buyout_realization_event.empty
        else 0
    )
    include_fallback = direct_total < int(config.get("min_train_exits", 100))
    available_types = set(fact_buyout_realization_event["realization_type"].astype(str)) if not fact_buyout_realization_event.empty else set()
    candidate_rows = [
        {
            "target_name": "sponsor_sale_or_mna_by_12q",
            "universe": "buyout_pe",
            "horizon_quarters": 12,
            "included_routes": "mna|sponsor_sale",
            "label_confidence_rule": "high_or_medium",
            "allowed_source_rules": "crunchbase_acquisition|preqin_buyout_merger_or_trade_sale|preqin_buyout_repeat_buyout_company|preqin_buyout_secondary_like_realization",
            "allowed_directness_rules": "direct_dated",
            "stage2_route_set": "mna|sponsor_sale",
            "candidate_role": "candidate",
            "benchmark_row": 0,
            "headline_eligible": 1,
            "target_family": "buyout_event_spine",
            "support_note": "Buyout realization target from direct-dated sponsor-sale and M&A mechanics.",
        },
        {
            "target_name": "sponsor_sale_or_mna_by_16q",
            "universe": "buyout_pe",
            "horizon_quarters": 16,
            "included_routes": "mna|sponsor_sale",
            "label_confidence_rule": "high_or_medium",
            "allowed_source_rules": "crunchbase_acquisition|preqin_buyout_merger_or_trade_sale|preqin_buyout_repeat_buyout_company|preqin_buyout_secondary_like_realization",
            "allowed_directness_rules": "direct_dated",
            "stage2_route_set": "mna|sponsor_sale",
            "candidate_role": "candidate",
            "benchmark_row": 0,
            "headline_eligible": 1,
            "target_family": "buyout_event_spine",
            "support_note": "Longer-horizon direct-dated sponsor-sale and M&A target.",
        },
        {
            "target_name": "partial_or_full_realization_by_12q",
            "universe": "buyout_pe",
            "horizon_quarters": 12,
            "included_routes": "ipo|mna|sponsor_sale|partial_realization",
            "label_confidence_rule": "high_or_medium",
            "allowed_source_rules": "crunchbase_ipo|crunchbase_acquisition|preqin_buyout_merger_or_trade_sale|preqin_buyout_repeat_buyout_company|preqin_buyout_secondary_like_realization|preqin_buyout_partial_realization",
            "allowed_directness_rules": "direct_dated",
            "stage2_route_set": "pooled_strategic|sponsor_sale",
            "candidate_role": "candidate",
            "benchmark_row": 0,
            "headline_eligible": 1,
            "target_family": "buyout_event_spine",
            "support_note": "Direct-dated full or partial realization target where the staged extract supports it.",
        },
        {
            "target_name": "partial_or_full_realization_by_16q",
            "universe": "buyout_pe",
            "horizon_quarters": 16,
            "included_routes": "ipo|mna|sponsor_sale|partial_realization",
            "label_confidence_rule": "high_or_medium",
            "allowed_source_rules": "crunchbase_ipo|crunchbase_acquisition|preqin_buyout_merger_or_trade_sale|preqin_buyout_repeat_buyout_company|preqin_buyout_secondary_like_realization|preqin_buyout_partial_realization",
            "allowed_directness_rules": "direct_dated",
            "stage2_route_set": "pooled_strategic|sponsor_sale",
            "candidate_role": "candidate",
            "benchmark_row": 0,
            "headline_eligible": 1,
            "target_family": "buyout_event_spine",
            "support_note": "Longer-horizon direct-dated full or partial realization target.",
        },
        {
            "target_name": "recap_or_secondary_like_realization_by_12q",
            "universe": "buyout_pe",
            "horizon_quarters": 12,
            "included_routes": "recapitalization|secondary_like_realization|continuation_vehicle",
            "label_confidence_rule": "high_or_medium",
            "allowed_source_rules": "preqin_buyout_recapitalization|preqin_buyout_secondary_like_realization|preqin_buyout_continuation_vehicle",
            "allowed_directness_rules": "direct_dated",
            "stage2_route_set": "sponsor_sale",
            "candidate_role": "candidate",
            "benchmark_row": 0,
            "headline_eligible": 1,
            "target_family": "buyout_event_spine",
            "support_note": "Direct-dated recapitalization, secondary-like, or continuation events only.",
        },
        {
            "target_name": "recap_or_secondary_like_realization_by_16q",
            "universe": "buyout_pe",
            "horizon_quarters": 16,
            "included_routes": "recapitalization|secondary_like_realization|continuation_vehicle",
            "label_confidence_rule": "high_or_medium",
            "allowed_source_rules": "preqin_buyout_recapitalization|preqin_buyout_secondary_like_realization|preqin_buyout_continuation_vehicle",
            "allowed_directness_rules": "direct_dated",
            "stage2_route_set": "sponsor_sale",
            "candidate_role": "candidate",
            "benchmark_row": 0,
            "headline_eligible": 1,
            "target_family": "buyout_event_spine",
            "support_note": "Longer-horizon direct-dated recapitalization, secondary-like, or continuation events only.",
        },
        {
            "target_name": "any_direct_realization_by_12q",
            "universe": "buyout_pe",
            "horizon_quarters": 12,
            "included_routes": "ipo|mna|sponsor_sale|partial_realization|recapitalization|secondary_like_realization|continuation_vehicle",
            "label_confidence_rule": "high_or_medium",
            "allowed_source_rules": "",
            "allowed_directness_rules": "direct_dated",
            "stage2_route_set": "pooled_strategic|sponsor_sale",
            "candidate_role": "candidate",
            "benchmark_row": 0,
            "headline_eligible": 1,
            "target_family": "buyout_event_spine",
            "support_note": "All direct-dated realization mechanics combined into one buyout target.",
        },
        {
            "target_name": "any_direct_realization_by_16q",
            "universe": "buyout_pe",
            "horizon_quarters": 16,
            "included_routes": "ipo|mna|sponsor_sale|partial_realization|recapitalization|secondary_like_realization|continuation_vehicle",
            "label_confidence_rule": "high_or_medium",
            "allowed_source_rules": "",
            "allowed_directness_rules": "direct_dated",
            "stage2_route_set": "pooled_strategic|sponsor_sale",
            "candidate_role": "candidate",
            "benchmark_row": 0,
            "headline_eligible": 1,
            "target_family": "buyout_event_spine",
            "support_note": "Longer-horizon all-direct realization target.",
        },
    ]
    if include_fallback:
        candidate_rows.extend(
            [
                {
                    "target_name": "buyout_actionability_proxy_by_12q",
                    "universe": "buyout_pe",
                    "horizon_quarters": 12,
                    "included_routes": "mna|sponsor_sale",
                    "label_confidence_rule": "high_or_medium",
                    "allowed_source_rules": "crunchbase_acquisition|preqin_buyout_transition",
                    "allowed_directness_rules": "direct_dated|inferred_transition",
                    "stage2_route_set": "mna|sponsor_sale",
                    "candidate_role": "fallback_proxy",
                    "benchmark_row": 0,
                    "headline_eligible": 0,
                    "target_family": "fallback_proxy",
                    "support_note": "Fallback proxy target from the legacy company-exit surface; never headline-eligible.",
                },
                {
                    "target_name": "buyout_actionability_proxy_by_16q",
                    "universe": "buyout_pe",
                    "horizon_quarters": 16,
                    "included_routes": "mna|sponsor_sale",
                    "label_confidence_rule": "high_or_medium",
                    "allowed_source_rules": "crunchbase_acquisition|preqin_buyout_transition",
                    "allowed_directness_rules": "direct_dated|inferred_transition",
                    "stage2_route_set": "mna|sponsor_sale",
                    "candidate_role": "fallback_proxy",
                    "benchmark_row": 0,
                    "headline_eligible": 0,
                    "target_family": "fallback_proxy",
                    "support_note": "Longer-horizon fallback proxy target from the legacy company-exit surface; never headline-eligible.",
                },
            ]
        )
    registry = pd.DataFrame(candidate_rows)
    registry["target_key"] = registry.apply(lambda row: target_file_key(row["target_name"], row["universe"]), axis=1)
    registry["positive_type_count"] = registry["included_routes"].map(lambda value: len(split_pipe_values(value)))
    registry["excluded_routes"] = ""
    registry["partial_realizations_included"] = registry["included_routes"].astype(str).str.contains("partial_realization").astype(int)
    registry["canonical_feature_backbone"] = "buyout_realization_core"
    registry["sponsor_fund_challenger_status"] = "not_run_in_buyout_only_mode"
    supported_rows = []
    for row in registry.itertuples(index=False):
        included = set(split_pipe_values(row.included_routes))
        direct_filter = set(split_pipe_values(row.allowed_directness_rules))
        subset = fact_buyout_realization_event.copy()
        if row.target_family == "buyout_event_spine":
            subset = subset.loc[subset["realization_type"].astype(str).isin(included)].copy()
            if direct_filter:
                subset = subset.loc[subset["directness_class"].astype(str).isin(direct_filter)].copy()
        data_supported = int(not subset.empty)
        if row.target_family == "buyout_event_spine" and row.target_name.startswith("recap_or_secondary_like"):
            data_supported = int(any(name in available_types for name in {"recapitalization", "secondary_like_realization", "continuation_vehicle"}))
        supported_rows.append(data_supported)
    registry["data_supported"] = supported_rows
    return registry.sort_values(["candidate_role", "horizon_quarters", "target_name"]).reset_index(drop=True)


def write_buyout_target_registry(path: Path, frame: pd.DataFrame) -> None:
    lines = [
        "# Buyout Target Registry",
        "",
        "- Core candidates are built from the rebuilt buyout realization-event spine.",
        "- Fallback proxy candidates are included only when direct-dated support remains below the chapter support floor.",
        "- Fallback proxy rows are never headline-eligible.",
        "",
        dataframe_to_markdown(frame),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_buyout_target_candidate_panel(
    buyout_panel: pd.DataFrame,
    spec: pd.Series | dict,
    fact_buyout_realization_event: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    subset = buyout_panel.copy().reset_index(drop=True)
    target_key = str(spec["target_key"])
    target_col = f"realized_{target_key}_by_horizon"
    realized_prefix = f"realized_{target_key}_by_h"
    if subset.empty:
        subset[target_col] = pd.Series(dtype=int)
        return subset, target_col, realized_prefix
    allowed_routes = set(split_pipe_values(spec["included_routes"]))
    allowed_sources = set(split_pipe_values(spec.get("allowed_source_rules", "")))
    allowed_directness = set(split_pipe_values(spec.get("allowed_directness_rules", "")))
    events = fact_buyout_realization_event.loc[
        fact_buyout_realization_event["realization_type"].astype(str).isin(allowed_routes)
    ].copy()
    if allowed_sources:
        events = events.loc[events["route_source"].astype(str).isin(allowed_sources)].copy()
    if allowed_directness:
        events = events.loc[events["directness_class"].astype(str).isin(allowed_directness)].copy()
    events = events.loc[
        events["exit_label_confidence"].map(lambda value: target_confidence_allowed(value, spec["label_confidence_rule"]))
    ].copy()
    if events.empty:
        earliest = pd.DataFrame(
            columns=[
                "company_id",
                "target_event_quarter_idx",
                "target_positive_route",
                "target_positive_confidence_tier",
                "target_positive_route_source",
                "target_positive_directness_class",
                "target_positive_headline_route_family",
                "target_positive_value_amount",
                "target_positive_observation_kind",
            ]
        )
    else:
        earliest = (
            events.sort_values(["company_id", "event_date", "event_id"])
            .groupby("company_id", as_index=False)
            .first()
            .rename(
                columns={
                    "event_quarter_idx": "target_event_quarter_idx",
                    "realization_type": "target_positive_route",
                    "exit_label_confidence": "target_positive_confidence_tier",
                    "route_source": "target_positive_route_source",
                    "directness_class": "target_positive_directness_class",
                    "headline_route_family": "target_positive_headline_route_family",
                    "value_amount": "target_positive_value_amount",
                }
            )
        )
        earliest["target_positive_observation_kind"] = [
            observation_kind_from_directness(source, directness)
            for source, directness in zip(
                earliest["target_positive_route_source"],
                earliest["target_positive_directness_class"],
                strict=True,
            )
        ]
    subset = subset.merge(
        earliest[
            [
                "company_id",
                "target_event_quarter_idx",
                "target_positive_route",
                "target_positive_confidence_tier",
                "target_positive_route_source",
                "target_positive_directness_class",
                "target_positive_observation_kind",
                "target_positive_headline_route_family",
                "target_positive_value_amount",
            ]
        ],
        on="company_id",
        how="left",
    )
    horizon_quarters = int(spec["horizon_quarters"])
    target_event_q = pd.to_numeric(subset["target_event_quarter_idx"], errors="coerce")
    quarter_idx = pd.to_numeric(subset["quarter_idx"], errors="coerce")
    subset[target_col] = (
        target_event_q.notna()
        & target_event_q.ge(quarter_idx)
        & target_event_q.le(quarter_idx + horizon_quarters - 1)
    ).astype(int)
    subset["company_exit_route"] = np.where(
        subset[target_col].astype(int).eq(1),
        subset["target_positive_headline_route_family"],
        subset.get("company_exit_route"),
    )
    subset["company_exit_value_usd"] = np.where(
        subset[target_col].astype(int).eq(1),
        pd.to_numeric(subset["target_positive_value_amount"], errors="coerce"),
        pd.to_numeric(subset.get("company_exit_value_usd"), errors="coerce"),
    )
    subset["company_exit_confidence_tier"] = np.where(
        subset[target_col].astype(int).eq(1),
        subset["target_positive_confidence_tier"],
        subset.get("company_exit_confidence_tier"),
    )
    subset["company_exit_route_source"] = np.where(
        subset[target_col].astype(int).eq(1),
        subset["target_positive_route_source"],
        subset.get("company_exit_route_source"),
    )
    subset["exit_quarter_idx"] = np.where(
        subset[target_col].astype(int).eq(1),
        target_event_q,
        pd.to_numeric(subset.get("exit_quarter_idx"), errors="coerce"),
    )
    for horizon_step in range(1, horizon_quarters + 1):
        subset[f"{realized_prefix}{horizon_step}"] = (
            target_event_q.notna()
            & target_event_q.ge(quarter_idx)
            & target_event_q.le(quarter_idx + horizon_step - 1)
        ).astype(int)
    return subset, target_col, realized_prefix


def build_buyout_target_selection_gates(
    registry: pd.DataFrame,
    selected_backbones: pd.DataFrame,
    evaluation_metrics_targets: pd.DataFrame,
    decision_backtest_targets: pd.DataFrame,
    target_route_support_by_split: pd.DataFrame,
    target_source_summary: pd.DataFrame,
    target_prevalence_by_split: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    gates = build_target_selection_gates(
        registry,
        selected_backbones,
        evaluation_metrics_targets,
        decision_backtest_targets,
        target_route_support_by_split,
        target_source_summary,
        config,
    ).copy()
    validation_support = target_prevalence_by_split.loc[
        target_prevalence_by_split["split"].astype(str).eq("validation")
    ][["target_key", "positive_rows", "positive_companies"]].rename(
        columns={
            "positive_rows": "validation_positive_rows",
            "positive_companies": "validation_positive_companies",
        }
    )
    test_support = target_prevalence_by_split.loc[
        target_prevalence_by_split["split"].astype(str).eq("test")
    ][["target_key", "positive_rows", "positive_companies"]].rename(
        columns={
            "positive_rows": "test_positive_rows",
            "positive_companies": "test_positive_companies",
        }
    )
    gates = gates.merge(validation_support, on="target_key", how="left").merge(test_support, on="target_key", how="left")
    gates["validation_positive_rows"] = pd.to_numeric(gates["validation_positive_rows"], errors="coerce").fillna(0).astype(int)
    gates["test_positive_rows"] = pd.to_numeric(gates["test_positive_rows"], errors="coerce").fillna(0).astype(int)
    gates["min_validation_support_pass"] = gates["validation_positive_rows"].ge(
        int(config.get("buyout_min_validation_positives", 25))
    ).astype(int)
    gates["min_test_support_pass"] = gates["test_positive_rows"].ge(
        int(config.get("buyout_min_test_positives", 25))
    ).astype(int)
    gates["max_inferred_transition_share_pass"] = pd.to_numeric(
        gates["inferred_transition_share"], errors="coerce"
    ).fillna(0.0).le(float(config.get("buyout_max_inferred_transition_share", 0.75))).astype(int)
    gates["all_selection_gates_pass"] = gates[
        [
            "min_train_support_pass",
            "min_validation_support_pass",
            "min_direct_dated_share_pass",
            "max_inferred_transition_share_pass",
            "min_label_confidence_pass",
            "acceptable_policy_activation_pass",
            "acceptable_validation_calibration_pass",
            "acceptable_route_support_pass",
        ]
    ].min(axis=1).astype(int)
    return gates


def build_buyout_promotion_gate(
    leaderboard_validation: pd.DataFrame,
    confirmation_test: pd.DataFrame,
    target_selection_gates: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    selected = leaderboard_validation.loc[leaderboard_validation["selected_by_validation"].astype(int).eq(1)].copy()
    if selected.empty:
        return pd.DataFrame()
    selected = selected.merge(
        confirmation_test[
            [
                "target_key",
                "confirmed_on_locked_test",
                "mean_abs_calibration_gap",
                "high_confidence_mean_abs_calibration_gap",
                "selected_policy_acceptance_rate",
                "selected_policy_precision",
            ]
        ],
        on="target_key",
        how="left",
        suffixes=("", "_test"),
    )
    required_gate_columns = [
        "train_positive_events",
        "train_positive_events_stage2",
        "validation_positive_rows",
        "test_positive_rows",
        "direct_dated_share",
        "inferred_transition_share",
        "label_confidence_share",
        "min_train_support_pass",
        "min_validation_support_pass",
        "min_test_support_pass",
        "min_direct_dated_share_pass",
        "max_inferred_transition_share_pass",
        "min_label_confidence_pass",
        "acceptable_policy_activation_pass",
        "acceptable_validation_calibration_pass",
        "acceptable_route_support_pass",
        "route_support_scope_used",
        "min_train_route_support",
        "min_train_route_support_raw",
        "min_train_route_support_stage2",
        "requested_stage2_route_set",
        "actual_stage2_route_set",
    ]
    missing_gate_columns = [column for column in required_gate_columns if column not in selected.columns]
    if missing_gate_columns:
        available_gate_columns = [column for column in missing_gate_columns if column in target_selection_gates.columns]
        if available_gate_columns:
            selected = selected.merge(
                target_selection_gates[["target_key", *available_gate_columns]],
                on="target_key",
                how="left",
            )
        for column in missing_gate_columns:
            if column not in selected.columns:
                selected[column] = np.nan
    selected["locked_test_confirmation_pass"] = pd.to_numeric(
        selected["confirmed_on_locked_test"], errors="coerce"
    ).fillna(0).astype(int)
    selected["acceptable_calibration_high_confidence_pass"] = pd.to_numeric(
        selected["high_confidence_mean_abs_calibration_gap"], errors="coerce"
    ).fillna(np.inf).le(float(config.get("promotion_gate_high_conf_gap_max", 0.08))).astype(int)
    selected["chapter_headline_ready"] = (
        selected["headline_eligible"].astype(int).eq(1)
        & selected["min_train_support_pass"].astype(int).eq(1)
        & selected["min_validation_support_pass"].astype(int).eq(1)
        & selected["min_test_support_pass"].astype(int).eq(1)
        & selected["min_direct_dated_share_pass"].astype(int).eq(1)
        & selected["max_inferred_transition_share_pass"].astype(int).eq(1)
        & selected["min_label_confidence_pass"].astype(int).eq(1)
        & selected["acceptable_policy_activation_pass"].astype(int).eq(1)
        & selected["acceptable_validation_calibration_pass"].astype(int).eq(1)
        & selected["acceptable_calibration_high_confidence_pass"].astype(int).eq(1)
        & selected["acceptable_route_support_pass"].astype(int).eq(1)
        & selected["locked_test_confirmation_pass"].astype(int).eq(1)
    ).astype(int)
    return selected[
        [
            "target_key",
            "target_name",
            "train_positive_events",
            "train_positive_events_stage2",
            "validation_positive_rows",
            "test_positive_rows",
            "direct_dated_share",
            "inferred_transition_share",
            "label_confidence_share",
            "min_train_support_pass",
            "min_validation_support_pass",
            "min_test_support_pass",
            "min_direct_dated_share_pass",
            "max_inferred_transition_share_pass",
            "min_label_confidence_pass",
            "acceptable_policy_activation_pass",
            "acceptable_validation_calibration_pass",
            "acceptable_calibration_high_confidence_pass",
            "acceptable_route_support_pass",
            "route_support_scope_used",
            "min_train_route_support",
            "min_train_route_support_raw",
            "min_train_route_support_stage2",
            "requested_stage2_route_set",
            "actual_stage2_route_set",
            "locked_test_confirmation_pass",
            "chapter_headline_ready",
        ]
    ].copy()


def build_buyout_claim_matrix(
    recommendation_table: pd.DataFrame,
    promotion_gate: pd.DataFrame,
) -> pd.DataFrame:
    if recommendation_table.empty:
        return pd.DataFrame()
    selected = recommendation_table.loc[recommendation_table["recommended_for_universe"].astype(int).eq(1)].head(1).copy()
    if selected.empty:
        selected = recommendation_table.head(1).copy()
    validation_selected = recommendation_table.loc[
        recommendation_table["selected_by_validation"].astype(int).eq(1)
    ].head(1).copy()
    gate_source = selected.copy()
    if not validation_selected.empty:
        gate_source = validation_selected.copy()
    gate_values = gate_source.iloc[0].to_dict()
    if not promotion_gate.empty:
        gate_row = promotion_gate.loc[
            promotion_gate["target_key"].astype(str).eq(str(gate_source["target_key"].iloc[0]))
            & promotion_gate["target_name"].astype(str).eq(str(gate_source["target_name"].iloc[0]))
        ].head(1)
        if not gate_row.empty:
            gate_values.update(gate_row.iloc[0].to_dict())
    matrix = selected.copy()
    for column in [
        "direct_dated_share",
        "inferred_transition_share",
        "label_confidence_share",
        "chapter_headline_ready",
        "min_train_support_pass",
        "min_validation_support_pass",
        "min_test_support_pass",
        "min_direct_dated_share_pass",
        "max_inferred_transition_share_pass",
        "min_label_confidence_pass",
        "acceptable_policy_activation_pass",
        "acceptable_validation_calibration_pass",
        "acceptable_calibration_high_confidence_pass",
        "acceptable_route_support_pass",
        "locked_test_confirmation_pass",
    ]:
        if column in gate_values:
            matrix[column] = gate_values[column]
    if not validation_selected.empty:
        matrix["blocking_validation_target_name"] = str(validation_selected["target_name"].iloc[0])
        matrix["blocking_validation_selection_basis"] = str(validation_selected["selection_basis"].iloc[0])
    else:
        matrix["blocking_validation_target_name"] = str(selected["target_name"].iloc[0])
        matrix["blocking_validation_selection_basis"] = str(selected["selection_basis"].iloc[0])
    matrix["reporting_status"] = np.where(
        pd.to_numeric(matrix["chapter_headline_ready"], errors="coerce").fillna(0).astype(int).eq(1),
        "promoted",
        "provisional",
    )

    def limiting_factor(row: pd.Series) -> str:
        if int(pd.to_numeric(pd.Series([row.get("chapter_headline_ready")]), errors="coerce").fillna(0).iloc[0]) == 1:
            return "none"
        if int(pd.to_numeric(pd.Series([row.get("min_direct_dated_share_pass")]), errors="coerce").fillna(0).iloc[0]) == 0:
            return "direct_dated_share_below_gate"
        if int(pd.to_numeric(pd.Series([row.get("max_inferred_transition_share_pass")]), errors="coerce").fillna(0).iloc[0]) == 0:
            return "inferred_transition_share_above_gate"
        if int(pd.to_numeric(pd.Series([row.get("acceptable_validation_calibration_pass")]), errors="coerce").fillna(0).iloc[0]) == 0:
            return "validation_calibration_above_gate"
        if int(pd.to_numeric(pd.Series([row.get("locked_test_confirmation_pass")]), errors="coerce").fillna(0).iloc[0]) == 0:
            return "locked_test_confirmation_failed"
        if int(pd.to_numeric(pd.Series([row.get("acceptable_policy_activation_pass")]), errors="coerce").fillna(0).iloc[0]) == 0:
            return "policy_activation_outside_acceptance_band"
        if int(pd.to_numeric(pd.Series([row.get("acceptable_route_support_pass")]), errors="coerce").fillna(0).iloc[0]) == 0:
            return "route_support_below_gate"
        return "buyout_realization_support_still_incomplete"

    matrix["main_limiting_factor"] = matrix.apply(limiting_factor, axis=1)
    for column in ["direct_dated_share", "inferred_transition_share", "label_confidence_share"]:
        if column not in matrix.columns:
            matrix[column] = np.nan
    return matrix[
        [
            "target_key",
            "target_name",
            "selection_basis",
            "reporting_status",
            "chapter_headline_ready",
            "blocking_validation_target_name",
            "blocking_validation_selection_basis",
            "direct_dated_share",
            "inferred_transition_share",
            "label_confidence_share",
            "main_limiting_factor",
            "recommendation_reason",
            "unresolved_caveat",
        ]
    ].copy()

def write_target_recommendation_summary(
    path: Path,
    recommendation_table: pd.DataFrame,
) -> None:
    recommended = recommendation_table.loc[recommendation_table["recommended_for_universe"].astype(int).eq(1)].copy()
    venture_row = recommended.loc[recommended["universe"].astype(str).eq("venture_growth")].head(1)
    buyout_row = recommended.loc[recommended["universe"].astype(str).eq("buyout_pe")].head(1)
    lines = ["# Target Recommendation Summary", ""]
    if not venture_row.empty:
        row = venture_row.iloc[0]
        lines.append(f"- Venture/growth recommendation: `{row['target_name']}`")
        lines.append("  Status: `doctrinal_baseline`")
        lines.append(f"  Reason: {row['recommendation_reason']}")
    if not buyout_row.empty:
        row = buyout_row.iloc[0]
        lines.append(f"- Buyout/PE recommendation: `{row['target_name']}`")
        lines.append(f"  Status: `{row['chapter_reporting_status']}`")
        lines.append(f"  Reason: {row['recommendation_reason']}")
        if str(row["unresolved_caveat"]).strip():
            lines.append(f"  Caveat: {row['unresolved_caveat']}")
    lines.extend(
        [
            "",
            "Selection doctrine:",
            "- venture/growth is treated as a timed hard-liquidity milestone problem under strict PIT discipline",
            "- buyout/PE is treated as a realization-window problem and remains provisional until dated realization mechanics improve",
        ]
    )
    lines.extend(["", "## Ranked Table", "", dataframe_to_markdown(recommendation_table), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_chapter_target_doctrine(
    path: Path,
    recommendation_table: pd.DataFrame,
) -> None:
    recommended = recommendation_table.loc[recommendation_table["recommended_for_universe"].astype(int).eq(1)].copy()
    lines = [
        "# Chapter Target Doctrine",
        "",
        "The chapter should report separate empirical targets by universe unless a single target is simultaneously honest and useful in both slices.",
        "The current bundle is a diagnostic milestone for buyout/PE rather than the final empirical record.",
        "",
        "Literature-aligned interpretation for this chapter:",
        "- VC prediction is usually strongest on timed milestones such as IPO, acquisition, or similar hard events under strict PIT evaluation.",
        "- Provider-side Preqin logic already distinguishes venture from buyout objectives and horizons.",
        "- Buyout prediction is more naturally framed around realization windows, hold period, market conditions, and sponsor/fund state.",
        "- Commercial coverage is partial and selectively observed, so direct-dated labels outrank inferred transitions and snapshot status fields.",
        "",
    ]
    for universe in UNIVERSE_ORDER:
        row = recommended.loc[recommended["universe"].astype(str).eq(universe)].head(1)
        if row.empty:
            continue
        selected = row.iloc[0]
        lines.append(f"- `{universe}`: `{selected['target_name']}` with reporting status `{selected['chapter_reporting_status']}`.")
        lines.append(f"  Reason: {selected['recommendation_reason']}")
        if str(selected["unresolved_caveat"]).strip():
            lines.append(f"  Caveat: {selected['unresolved_caveat']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chapter_objective_definition_findings(
    path: Path,
    recommendation_table: pd.DataFrame,
    evaluation_metrics_targets: pd.DataFrame,
) -> None:
    full_test = evaluation_metrics_targets.loc[
        evaluation_metrics_targets["evaluation_view"].astype(str).eq("full_test")
    ].copy()
    recommended = recommendation_table.loc[recommendation_table["recommended_for_universe"].astype(int).eq(1)].copy()
    lines = [
        "# Chapter Objective Definition Findings",
        "",
        "This pass explored target definition directly rather than rerunning another broad feature search.",
        f"Validation decile `{CANONICAL_TARGET_CALIBRATION_METRIC}` remains the canonical selection metric; PR-AUC, ROC-AUC, and Brier-style metrics are supporting diagnostics only.",
        "",
    ]
    for universe in UNIVERSE_ORDER:
        row = recommended.loc[recommended["universe"].astype(str).eq(universe)].head(1)
        if row.empty:
            continue
        selected = row.iloc[0]
        metrics = full_test.loc[full_test["target_key"].astype(str).eq(str(selected["target_key"]))].head(1)
        lines.append(f"## {universe}")
        lines.append("")
        lines.append(f"- Recommended target: `{selected['target_name']}`")
        lines.append(f"- Reporting status: `{selected['chapter_reporting_status']}`")
        lines.append(f"- Reason: {selected['recommendation_reason']}")
        if not metrics.empty:
            metric_row = metrics.iloc[0]
            lines.append(
                f"- Full-test calibration gap / PR AUC / ROC AUC: `{metric_row['mean_abs_calibration_gap']:.4f}` / `{metric_row['pr_auc']:.4f}` / `{metric_row['roc_auc']:.4f}`"
            )
        if str(selected["unresolved_caveat"]).strip():
            lines.append(f"- Caveat: {selected['unresolved_caveat']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_chapter_target_tables(
    path: Path,
    recommendation_table: pd.DataFrame,
    evaluation_metrics_targets: pd.DataFrame,
    decision_backtest_targets: pd.DataFrame,
) -> None:
    selected_test_policies = decision_backtest_targets.loc[
        decision_backtest_targets["evaluation_split"].astype(str).eq("test")
        & decision_backtest_targets["selected_on_validation"].astype(int).eq(1)
    ].copy()
    lines = [
        "# Chapter Target Tables",
        "",
        "## Recommendation Table",
        "",
        dataframe_to_markdown(recommendation_table),
        "",
        "## Evaluation Metrics",
        "",
        dataframe_to_markdown(evaluation_metrics_targets),
        "",
        "## Selected Policy Backtests",
        "",
        dataframe_to_markdown(selected_test_policies),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_chapter_target_journey_update(
    path: Path,
    recommendation_table: pd.DataFrame,
) -> None:
    recommended = recommendation_table.loc[recommendation_table["recommended_for_universe"].astype(int).eq(1)].copy()
    lines = [
        "# Chapter Target Journey Update",
        "",
        "The original unified hard-liquidity framing remains acceptable for venture/growth, but the buyout/PE slice still needs a more conservative empirical framing.",
        "The doctrinal change in this stage is conceptual, not methodological: separate universes, separate objectives, and stricter provenance reporting.",
        "",
    ]
    for row in recommended.itertuples(index=False):
        lines.append(f"- `{row.universe}` currently reports `{row.target_name}` with status `{row.chapter_reporting_status}`.")
    lines.append("")
    lines.append("The main lesson is that objective definition matters more than another incremental feature sweep when route labels are thin or partially inferred.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_chapter_tables(
    output_dir: Path,
    run_metadata: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    window_selection: pd.DataFrame,
    partition_summary: pd.DataFrame,
    route_audit: pd.DataFrame,
    decision_backtest: pd.DataFrame,
    exit_confusion_summary: pd.DataFrame,
    decision_policy_confusion_summary: pd.DataFrame,
) -> Path:
    metadata = run_metadata.iloc[0]
    primary_threshold = round(float(metadata.get("primary_confusion_threshold", 0.02)), 4)
    primary_policy_key = str(metadata.get("primary_policy_key", ""))
    primary_exit_confusion = exit_confusion_summary.loc[
        exit_confusion_summary["threshold"].round(4).eq(primary_threshold)
    ].copy()
    primary_policy_confusion = decision_policy_confusion_summary.loc[
        decision_policy_confusion_summary["policy_key"].astype(str).eq(primary_policy_key)
        & decision_policy_confusion_summary["target_label"].astype(str).eq("realized_npv_proxy_positive")
    ].copy()
    lines = [
        "# Chapter 9 Tables",
        "",
        "Confusion matrices below are threshold-dependent diagnostic supplements. Calibration remains the primary model-evaluation criterion, and raw accuracy is not emphasized because the exit target is rare.",
        "",
        "## Run Metadata",
        "",
        dataframe_to_markdown(run_metadata),
        "",
        "## Selected Window",
        "",
        dataframe_to_markdown(window_selection),
        "",
        "## Scenario Metrics",
        "",
        dataframe_to_markdown(summary_metrics),
        "",
        "## Partition Summary",
        "",
        dataframe_to_markdown(partition_summary),
        "",
        "## Route Audit",
        "",
        dataframe_to_markdown(route_audit),
        "",
        "## Decision Backtest",
        "",
        dataframe_to_markdown(decision_backtest),
        "",
        "## Exit Confusion Supplement",
        "",
        dataframe_to_markdown(primary_exit_confusion),
        "",
        "## Policy Confusion Supplement",
        "",
        dataframe_to_markdown(primary_policy_confusion),
        "",
    ]
    path = output_dir / "chapter_tables.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_chapter_findings(
    output_dir: Path,
    summary_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    evaluation_metrics: pd.DataFrame,
    decision_backtest: pd.DataFrame,
    exit_confusion_summary: pd.DataFrame,
    decision_policy_confusion_summary: pd.DataFrame,
    run_metadata: pd.DataFrame,
    route_audit: pd.DataFrame,
) -> Path:
    baseline = summary_metrics[summary_metrics["scenario"] == "baseline"].iloc[0]
    freeze = summary_metrics[summary_metrics["scenario"] == "exit_freeze"].iloc[0]
    worst_gap = calibration.assign(
        abs_gap=(calibration["realized_exit_rate"] - calibration["mean_predicted_exit"]).abs()
    ).sort_values("abs_gap", ascending=False).iloc[0]
    dominant_route = route_audit.sort_values("chosen_exit_count", ascending=False).iloc[0]
    metadata = run_metadata.iloc[0]
    eval_lookup = evaluation_metrics.set_index("evaluation_view") if not evaluation_metrics.empty else pd.DataFrame()
    pooled_row = eval_lookup.loc["pooled_strategic_fallback"] if "pooled_strategic_fallback" in eval_lookup.index else None
    backtest_lookup = (
        decision_backtest.set_index("decision_rule")
        if not decision_backtest.empty and "decision_rule" in decision_backtest.columns
        else pd.DataFrame()
    )
    primary_threshold = round(float(metadata.get("primary_confusion_threshold", 0.02)), 4)
    primary_policy_key = str(metadata.get("primary_policy_key", ""))
    exit_confusion_lookup = (
        exit_confusion_summary.set_index("threshold")
        if not exit_confusion_summary.empty and "threshold" in exit_confusion_summary.columns
        else pd.DataFrame()
    )
    policy_confusion_lookup = (
        decision_policy_confusion_summary.set_index(["policy_key", "target_label"])
        if not decision_policy_confusion_summary.empty and {"policy_key", "target_label"}.issubset(decision_policy_confusion_summary.columns)
        else pd.DataFrame()
    )
    backtest_line = "- The decision back-test export was generated but did not produce a supported rule summary."
    if not backtest_lookup.empty and f"prob_exit_by_8q >= {primary_threshold:.2f}" in backtest_lookup.index:
        row = backtest_lookup.loc[f"prob_exit_by_8q >= {primary_threshold:.2f}"]
        backtest_line = (
            f"- In the latest-test decision back-test, the probability-threshold rule accepts {int(row['accepted_observations'])} observations "
            f"with realized exit-by-8-quarter rate {float(row['realized_exit_by_8q']):.4f}."
        )
    confusion_line = "- The confusion-matrix supplement was exported, but the primary threshold row was not available."
    if not exit_confusion_lookup.empty and primary_threshold in exit_confusion_lookup.index:
        row = exit_confusion_lookup.loc[primary_threshold]
        confusion_line = (
            f"- At the fixed threshold {primary_threshold:.2f}, balanced accuracy is {float(row['balanced_accuracy']):.4f}, "
            f"precision is {float(row['precision']):.4f}, and recall is {float(row['recall']):.4f}; these are threshold-dependent supplements, not replacements for calibration."
        )
    policy_confusion_line = "- The dual-rule policy confusion supplement was exported as a decision diagnostic."
    if not policy_confusion_lookup.empty and (primary_policy_key, "realized_npv_proxy_positive") in policy_confusion_lookup.index:
        row = policy_confusion_lookup.loc[(primary_policy_key, "realized_npv_proxy_positive")]
        policy_confusion_line = (
            f"- Under the primary dual rule, balanced accuracy is {float(row['balanced_accuracy']):.4f}, "
            f"precision is {float(row['precision']):.4f}, and recall is {float(row['recall']):.4f} against the realized NPV proxy."
        )
    data_mode = str(metadata.get("data_mode", "sample")).strip().lower()
    if data_mode == "sample":
        cohort_line = (
            f"- The successful sample run used the {int(metadata['selected_min_entry_year'])}-and-later synthetic cohort, "
            f"yielding {int(metadata['panel_rows'])} company-quarter rows."
        )
        route_caveat_line = (
            "- The sample route mix is fully synthetic and should be read as a teaching scaffold rather than an empirical claim."
        )
        placeholder_line = (
            "- Richer sponsor, LP, and patent blocks remain placeholders in the current sample-facing build."
        )
    else:
        cohort_line = (
            f"- The successful live run used the {int(metadata['selected_min_entry_year'])}-and-later cohort, "
            f"yielding {int(metadata['panel_rows'])} company-quarter rows."
        )
        route_caveat_line = (
            "- Soft-failure proxies are now held out of the primary route labels and tracked separately in the sensitivity audits."
        )
        placeholder_line = (
            "- The live build now uses a PIT-safe patent adapter; richer sponsor and LP blocks remain deferred."
        )
    lines = [
        "# Chapter 9 Findings",
        "",
        cohort_line,
        f"- The holdout exit-by-8-quarter probability for the stylized company falls from {float(baseline['prob_exit_by_horizon']):.4f} in baseline to {float(freeze['prob_exit_by_horizon']):.4f} in exit freeze.",
        f"- Mean NPV falls from {float(baseline['mean_npv']):.4f} to {float(freeze['mean_npv']):.4f} under exit freeze.",
        f"- The largest calibration gap appears in decile {int(worst_gap['decile'])}, with predicted exit {float(worst_gap['mean_predicted_exit']):.4f} versus realized {float(worst_gap['realized_exit_rate']):.4f}.",
        f"- The dominant chosen realized route in the current mapping is {dominant_route['route_label']} from {dominant_route['route_source']} with {int(dominant_route['chosen_exit_count'])} selected exits.",
        confusion_line,
        backtest_line,
        policy_confusion_line,
        (
            f"- The pooled strategic fallback view is active, with mean absolute calibration gap {float(pooled_row['mean_abs_calibration_gap']):.4f}."
            if pooled_row is not None
            else "- Direct-route evaluation remains the primary reporting view; pooled strategic exit is not required by the selected split."
        ),
        route_caveat_line,
        placeholder_line,
        (
            "- The current promotion gate flags the live build as chapter-evidence-ready."
            if bool(metadata.get("chapter_evidence_ready", False))
            else "- The current promotion gate does not yet mark the live build as chapter-evidence-ready."
        ),
        "",
    ]
    path = output_dir / "chapter_findings.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_appendix_confusion_notes(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Appendix Confusion Notes",
                "",
                "- Confusion matrices remain appendix-only diagnostics in the redesign pass.",
                "- The chapter-facing headline target is hard timely liquidity by 8 quarters, evaluated primarily by calibration.",
                "- Thresholded confusion and policy activation are useful for screening interpretation but do not replace calibration or label-confidence audits.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_chapter_summary_v2(
    output_dir: Path,
    run_metadata: pd.DataFrame,
    target_definition_main: pd.DataFrame,
    label_confidence_audit: pd.DataFrame,
    promotion_gate_v2: pd.DataFrame,
) -> Path:
    meta = run_metadata.iloc[0]
    gate = promotion_gate_v2.iloc[0]
    lines = [
        f"# Chapter 9 {'Sample' if str(meta['data_mode']) == 'sample' else 'Live'} Run Summary",
        "",
        "## Headline Target",
        "",
        f"- Target name: `{HARD_TIMELY_LIQUIDITY_TARGET}`.",
        "- Included routes: direct IPO, M&A/acquisition, and sponsor sale only.",
        "- Soft-failure proxies remain sensitivity-only evidence and are not part of the main supervised target.",
        f"- Venture/growth doctrine: `{meta.get('recommended_target_venture_growth', '')}` with status `{meta.get('venture_target_reporting_status', '')}`.",
        f"- Buyout/PE doctrine: `{meta.get('recommended_target_buyout_pe', '')}` with status `{meta.get('buyout_target_reporting_status', '')}`.",
        "",
        "## Cohort",
        "",
        f"- Selected minimum entry year: {int(meta['selected_min_entry_year'])}",
        f"- Selected train / validation / test end quarters: {meta.get('selected_train_end_quarter', '')} / {meta.get('selected_validation_end_quarter', '')} / {meta.get('selected_test_end_quarter', '')}",
        f"- Panel rows: {int(meta['panel_rows'])}",
        f"- Train / validation / test rows: {int(meta['train_rows'])} / {int(meta['validation_rows'])} / {int(meta['test_rows'])}",
        f"- Train / validation / test hard exits: {int(meta['train_exits'])} / {int(meta['validation_exits'])} / {int(meta['test_exits'])}",
        "",
        "## Promotion Gate V2",
        "",
        f"- Enough route support: {bool(gate['enough_route_support'])}",
        f"- Enough policy activation: {bool(gate['enough_policy_activation'])}",
        f"- Acceptable label confidence: {bool(gate['acceptable_label_confidence'])}",
        f"- Acceptable full-test calibration: {bool(gate['acceptable_calibration_full'])}",
        f"- Acceptable high-confidence calibration: {bool(gate['acceptable_calibration_high_confidence'])}",
        f"- Chapter evidence ready: {bool(gate['chapter_evidence_ready'])}",
        "- Use `universe_claim_matrix.csv` for the universe-specific reporting decision; the single global gate is diagnostic only.",
        "",
        "## Label Confidence Audit",
        "",
    ]
    for row in label_confidence_audit.itertuples(index=False):
        lines.append(
            f"- {row.target_scope} / {row.route_label} / {row.confidence_tier} / {row.route_source}: chosen={int(row.chosen_exit_count)}"
        )
    lines.extend(["", "## Target Definition Table", "", dataframe_to_markdown(target_definition_main)])
    path = output_dir / "chapter_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_chapter_tables_v2(
    output_dir: Path,
    universe_support: pd.DataFrame,
    label_confidence_audit: pd.DataFrame,
    evaluation_metrics_main: pd.DataFrame,
    sector_stage_metrics: pd.DataFrame,
    decision_backtest_screening: pd.DataFrame,
    decision_backtest_economic: pd.DataFrame,
) -> Path:
    lines = [
        "# Chapter 9 Tables",
        "",
        "## Table 1. Sample Construction and Label Confidence",
        "",
        dataframe_to_markdown(universe_support),
        "",
        dataframe_to_markdown(label_confidence_audit),
        "",
        "## Table 2. Main Evaluation Metrics",
        "",
        dataframe_to_markdown(evaluation_metrics_main),
        "",
        "## Table 3. Universe / Sector / Stage Heterogeneity",
        "",
        dataframe_to_markdown(sector_stage_metrics.head(24) if not sector_stage_metrics.empty else sector_stage_metrics),
        "",
        "## Table 4. Decision Backtest",
        "",
        dataframe_to_markdown(decision_backtest_screening),
        "",
        dataframe_to_markdown(decision_backtest_economic),
        "",
    ]
    path = output_dir / "chapter_tables.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_chapter_findings_v2(
    output_dir: Path,
    evaluation_metrics_main: pd.DataFrame,
    evaluation_metrics_by_universe: pd.DataFrame,
    stage2_route_metrics: pd.DataFrame,
    policy_activation_summary: pd.DataFrame,
    promotion_gate_v2: pd.DataFrame,
    patent_sector_model_comparison: pd.DataFrame,
) -> Path:
    eval_lookup = evaluation_metrics_main.set_index("evaluation_view")
    gate = promotion_gate_v2.iloc[0]
    lines = [
        "# Chapter 9 Findings",
        "",
        f"- The redesigned headline target is `{HARD_TIMELY_LIQUIDITY_TARGET}`, not the broader packaging-era any-exit framing.",
        "- Venture/growth remains on the doctrinal baseline target, while buyout/PE remains provisional in the current canonical pass.",
        f"- Full-test mean absolute calibration gap is {float(eval_lookup.loc['full_test', 'mean_abs_calibration_gap']):.4f}.",
        f"- High-confidence mean absolute calibration gap is {float(eval_lookup.loc['high_confidence_subset', 'mean_abs_calibration_gap']):.4f}.",
        f"- Stress-slice mean absolute calibration gap is {float(eval_lookup.loc['stress_slice', 'mean_abs_calibration_gap']):.4f}.",
    ]
    for row in evaluation_metrics_by_universe.itertuples(index=False):
        lines.append(
            f"- Universe `{row.universe}`: Brier {float(row.brier_score):.4f}, gap {float(row.mean_abs_calibration_gap):.4f}, PR-AUC {float(row.pr_auc) if pd.notna(row.pr_auc) else float('nan'):.4f}."
        )
    if not stage2_route_metrics.empty:
        top_route = stage2_route_metrics.iloc[0]
        lines.append(
            f"- The conditional stage-2 route model uses `{top_route['class_set']}` and reaches accuracy {float(top_route['accuracy']) if pd.notna(top_route['accuracy']) else float('nan'):.4f} on realized hard exits."
        )
    if not policy_activation_summary.empty:
        selected = policy_activation_summary[policy_activation_summary["selected_on_validation"].astype(int).eq(1)].copy()
        for row in selected.itertuples(index=False):
            lines.append(
                f"- Active {row.policy_family} policy `{row.policy_key}` accepts {int(row.accepted_observations)} names with hit rate {float(row.hit_rate_accepted) if pd.notna(row.hit_rate_accepted) else float('nan'):.4f}."
            )
    if not patent_sector_model_comparison.empty:
        positive = patent_sector_model_comparison[
            patent_sector_model_comparison["model_variant"].astype(str).eq("patent_sector_conditional")
        ].sort_values("mean_abs_calibration_gap").head(1)
        if not positive.empty:
            row = positive.iloc[0]
            lines.append(
                f"- Sector-conditional patents behave like challenger-only diagnostics; the strongest sector in this pass is `{row['sector_bucket']}`."
            )
    lines.append(
        "- The redesign promotion gate marks the chapter as evidence-ready."
        if bool(gate["chapter_evidence_ready"])
        else "- The redesign promotion gate does not yet mark the chapter as evidence-ready."
    )
    lines.append("")
    path = output_dir / "chapter_findings.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_live_dataset(config: dict) -> dict:
    sources = load_actual_inputs(config)
    preqin_master = build_preqin_company_master(sources["preqin_vc"], sources["preqin_buyout"])
    crosswalk = build_crosswalk(preqin_master, sources["cb_companies"])
    company_master = build_company_master(
        preqin_master,
        sources["cb_companies"],
        sources["cb_rounds"],
        crosswalk,
    )
    round_events = build_round_events(
        company_master,
        sources["preqin_vc"],
        sources["preqin_buyout"],
        sources["cb_rounds"],
    )
    direct_exit_candidates = build_direct_exit_candidates(
        company_master,
        round_events,
        sources["preqin_vc"],
        sources["preqin_buyout"],
        sources["cb_acquisitions"],
        sources["cb_ipos"],
    )
    sensitivity_exit_candidates = build_sensitivity_exit_candidates(
        company_master,
        round_events,
        sources["preqin_vc"],
        direct_exit_candidates,
    )
    chosen_exits_main = choose_first_exit(
        direct_exit_candidates,
        quarter_idx_from_label(config["analysis_end_quarter"]),
    )
    chosen_exits_sensitivity = choose_first_exit(
        pd.concat([direct_exit_candidates, sensitivity_exit_candidates], ignore_index=True),
        quarter_idx_from_label(config["analysis_end_quarter"]),
    )
    window_selection_grid = build_window_selection_grid(round_events, chosen_exits_main, config)
    selected_window, window_selection_grid = select_actual_window(window_selection_grid, config)
    selected_min_entry_year = int(selected_window["min_entry_year"])
    config["train_end_quarter"] = str(selected_window["train_end_quarter"])
    config["validation_end_quarter"] = str(selected_window["validation_end_quarter"])
    config["test_end_quarter"] = str(selected_window["test_end_quarter"])
    config["panel_end_quarter"] = str(selected_window["test_end_quarter"])
    company_master, round_events, direct_exit_candidates, chosen_exits_main, crosswalk = filter_modeled_universe(
        company_master,
        round_events,
        direct_exit_candidates,
        chosen_exits_main,
        crosswalk,
        selected_min_entry_year,
    )
    sensitivity_exit_candidates = sensitivity_exit_candidates[
        sensitivity_exit_candidates["company_id"].isin(set(company_master["company_id"]))
    ].copy()
    chosen_exits_sensitivity = chosen_exits_sensitivity[
        chosen_exits_sensitivity["company_id"].isin(set(company_master["company_id"]))
    ].copy()
    patent_matches, patent_match_audit, patent_matches_baseline = load_patent_matches(company_master, config)
    patent_event_lookup = build_patent_event_lookup(patent_matches)
    buyout_market_panel = build_buyout_sponsor_fund_market_panel(sources)
    panel = build_company_quarter_panel(
        company_master,
        round_events,
        chosen_exits_main,
        config,
        patent_event_lookup=patent_event_lookup,
    )
    macro_panel = build_macro_panel(panel)
    panel = attach_macro(panel, macro_panel)
    panel, company_master, universe_map = attach_universe_labels(panel, company_master, round_events)
    panel = attach_buyout_sponsor_fund_market_features(panel, buyout_market_panel)
    panel = add_bucket_feature_columns(panel)
    panel = add_interaction_candidate_columns(panel)
    panel = add_sector_conditional_patent_features(panel)
    panel = split_panel(panel, config)
    panel = add_realized_exit_within_horizon(panel, chosen_exits_main, int(config["holdout_horizon_quarters"]))
    panel = add_redesigned_targets(panel, chosen_exits_sensitivity, int(config["holdout_horizon_quarters"]))
    route_audit_main = build_route_audit(direct_exit_candidates, chosen_exits_main)
    route_audit_sensitivity = build_route_audit(
        sensitivity_exit_candidates,
        chosen_exits_sensitivity[chosen_exits_sensitivity["route_label"] == "soft_failure_sensitivity"].copy(),
    )
    route_confidence_summary = build_route_confidence_summary(
        direct_exit_candidates,
        chosen_exits_main,
        sensitivity_exit_candidates,
        chosen_exits_sensitivity,
    )
    route_mapping_comparison = build_route_mapping_comparison(chosen_exits_main, chosen_exits_sensitivity)
    coverage_by_year, partition_summary = build_coverage_tables(panel)
    route_support_by_split = build_route_support_by_split(panel)
    patent_feature_coverage = build_patent_feature_coverage(company_master, panel, patent_matches)
    patent_coverage_comparison = build_patent_coverage_comparison(
        company_master,
        panel,
        patent_matches_baseline,
        patent_matches,
    )
    sponsor_fund_feature_coverage = build_sponsor_fund_feature_coverage(panel)
    buyout_realization_field_audit = build_buyout_realization_field_audit(sources, round_events)
    deal_fund_link_audit = build_deal_fund_link_audit(sources, round_events)
    density_by_entry_year = window_selection_grid[
        (window_selection_grid["train_end_quarter"] == str(selected_window["train_end_quarter"]))
        & (window_selection_grid["validation_end_quarter"] == str(selected_window["validation_end_quarter"]))
        & (window_selection_grid["test_end_quarter"] == str(selected_window["test_end_quarter"]))
    ].copy()
    window_selection = window_selection_grid[window_selection_grid["selected"] == 1].reset_index(drop=True)
    route_pooling_fallback_summary = build_route_pooling_fallback_summary(
        selected_window,
        bool(window_selection["used_route_pooling_fallback"].iloc[0]) if not window_selection.empty else False,
    )
    return {
        "sources": sources,
        "panel": panel,
        "company_master": company_master,
        "round_events": round_events,
        "chosen_exits": chosen_exits_main,
        "chosen_exits_main": chosen_exits_main,
        "chosen_exits_sensitivity": chosen_exits_sensitivity,
        "macro_panel": macro_panel,
        "buyout_market_panel": buyout_market_panel,
        "route_audit": route_audit_main,
        "route_audit_main": route_audit_main,
        "route_audit_sensitivity": route_audit_sensitivity,
        "route_confidence_summary": route_confidence_summary,
        "route_mapping_comparison": route_mapping_comparison,
        "coverage_by_year": coverage_by_year,
        "partition_summary": partition_summary,
        "route_support_by_split": route_support_by_split,
        "crosswalk": crosswalk,
        "density_by_entry_year": density_by_entry_year,
        "window_selection_grid": window_selection_grid,
        "selected_min_entry_year": selected_min_entry_year,
        "window_selection": window_selection,
        "patent_matches": patent_matches,
        "patent_matches_baseline": patent_matches_baseline,
        "patent_match_audit": patent_match_audit,
        "patent_feature_coverage": patent_feature_coverage,
        "patent_coverage_comparison": patent_coverage_comparison,
        "sponsor_fund_feature_coverage": sponsor_fund_feature_coverage,
        "buyout_realization_field_audit": buyout_realization_field_audit,
        "deal_fund_link_audit": deal_fund_link_audit,
        "route_pooling_fallback_summary": route_pooling_fallback_summary,
        "universe_map": universe_map,
        "source_file_inventory": sources.get("source_file_inventory", pd.DataFrame()),
        "target_definition_main": build_target_definition_main(),
        "target_definition_sensitivity": build_target_definition_sensitivity(),
        "label_confidence_audit": build_label_confidence_audit(chosen_exits_main, chosen_exits_sensitivity),
    }


def build_sample_dataset(config: dict) -> dict:
    sample_inputs = load_sample_inputs(config)
    company_master = build_sample_company_master(sample_inputs)
    round_events = build_sample_round_events(sample_inputs)
    chosen_exits = build_sample_exits(sample_inputs, quarter_idx_from_label(config["analysis_end_quarter"]))
    density_by_entry_year = build_density_windows(round_events, chosen_exits, config)
    selected_min_entry_year = select_min_entry_year(density_by_entry_year, config)
    company_master, round_events, exit_candidates, chosen_exits, _ = filter_modeled_universe(
        company_master,
        round_events,
        chosen_exits,
        chosen_exits,
        pd.DataFrame(columns=["portfolio_company_id", "company_uuid"]),
        selected_min_entry_year,
    )
    panel = build_company_quarter_panel(company_master, round_events, chosen_exits, config)
    macro_panel = build_sample_macro_panel(sample_inputs)
    panel = attach_macro(panel, macro_panel)
    panel, company_master, universe_map = attach_universe_labels(panel, company_master, round_events)
    panel = add_bucket_feature_columns(panel)
    panel = add_interaction_candidate_columns(panel)
    panel = add_sector_conditional_patent_features(panel)
    panel = split_panel(panel, config)
    panel = add_realized_exit_within_horizon(panel, chosen_exits, int(config["holdout_horizon_quarters"]))
    panel = add_redesigned_targets(panel, chosen_exits.copy(), int(config["holdout_horizon_quarters"]))
    route_audit = build_sample_route_audit(chosen_exits)
    coverage_by_year, partition_summary = build_coverage_tables(panel)
    window_selection = density_by_entry_year[
        density_by_entry_year["min_entry_year"] == selected_min_entry_year
    ].reset_index(drop=True)
    return {
        "sources": {},
        "panel": panel,
        "company_master": company_master,
        "round_events": round_events,
        "chosen_exits": chosen_exits,
        "chosen_exits_main": chosen_exits,
        "chosen_exits_sensitivity": chosen_exits.copy(),
        "macro_panel": macro_panel,
        "route_audit": route_audit,
        "route_audit_main": route_audit,
        "route_audit_sensitivity": pd.DataFrame(
            columns=["route_label", "confidence_tier", "route_source", "candidate_count", "chosen_exit_count"]
        ),
        "route_confidence_summary": pd.DataFrame(
            columns=["mapping_scope", "confidence_tier", "route_source", "candidate_count", "chosen_exit_count"]
        ),
        "route_mapping_comparison": build_route_mapping_comparison(chosen_exits, chosen_exits.copy()),
        "coverage_by_year": coverage_by_year,
        "partition_summary": partition_summary,
        "route_support_by_split": build_route_support_by_split(panel),
        "crosswalk": pd.DataFrame(),
        "density_by_entry_year": density_by_entry_year,
        "window_selection_grid": density_by_entry_year.assign(
            train_end_quarter=str(config["train_end_quarter"]),
            validation_end_quarter=str(config["validation_end_quarter"]),
            test_end_quarter=str(config["test_end_quarter"]),
            selected=1,
            used_route_pooling_fallback=0,
            fallback_reason="Sample mode uses the fixed synthetic split.",
        ),
        "selected_min_entry_year": selected_min_entry_year,
        "window_selection": window_selection,
        "patent_matches": pd.DataFrame(),
        "patent_matches_baseline": pd.DataFrame(),
        "patent_match_audit": pd.DataFrame(
            columns=[
                "alias_source",
                "confidence_tier",
                "match_method",
                "candidate_patent_rows",
                "candidate_patents",
                "used_patent_rows",
                "used_patents",
                "matched_companies",
                "ambiguous_patent_rows",
            ]
        ),
        "patent_feature_coverage": build_patent_feature_coverage(company_master, panel, pd.DataFrame()),
        "patent_coverage_comparison": build_patent_coverage_comparison(
            company_master,
            panel,
            pd.DataFrame(),
            pd.DataFrame(),
        ),
        "sponsor_fund_feature_coverage": build_sponsor_fund_feature_coverage(panel),
        "buyout_realization_field_audit": pd.DataFrame(
            columns=["audit_area", "source_table", "field_names", "dated_field_present", "non_null_rows", "candidate_supported", "note"]
        ),
        "deal_fund_link_audit": pd.DataFrame(
            columns=["link_layer", "source_table", "join_keys", "keys_present", "dated_link_present", "pit_safe_supported", "active_status", "note"]
        ),
        "route_pooling_fallback_summary": build_route_pooling_fallback_summary(pd.Series({"min_entry_year": selected_min_entry_year}), False),
        "universe_map": universe_map,
        "source_file_inventory": pd.DataFrame(),
        "target_definition_main": build_target_definition_main(),
        "target_definition_sensitivity": build_target_definition_sensitivity(),
        "label_confidence_audit": build_label_confidence_audit(chosen_exits, chosen_exits.copy()),
    }


def build_dataset(config: dict) -> dict:
    resolved_config = DEFAULT_CONFIG.copy()
    resolved_config.update(config)
    data_mode = str(resolved_config.get("data_mode", "sample")).strip().lower()
    if data_mode == "actual":
        if config.get("use_quarter_fixed_effects") is None:
            resolved_config["use_quarter_fixed_effects"] = True
        return build_live_dataset(resolved_config)
    if data_mode == "sample":
        return build_sample_dataset(resolved_config)
    raise ValueError(f"Unsupported data_mode: {data_mode}")


def build_buyout_entry_override(bridge_company_deal: pd.DataFrame) -> pd.DataFrame:
    columns = ["company_id", "entry_date", "entry_quarter_idx", "entry_year"]
    if bridge_company_deal.empty:
        return pd.DataFrame(columns=columns)
    entry = bridge_company_deal.dropna(subset=["company_id", "deal_date", "deal_quarter_idx"]).copy()
    if entry.empty:
        return pd.DataFrame(columns=columns)
    entry["deal_date"] = pd.to_datetime(entry["deal_date"], errors="coerce")
    entry["deal_quarter_idx"] = pd.to_numeric(entry["deal_quarter_idx"], errors="coerce")
    entry = entry.dropna(subset=["deal_date", "deal_quarter_idx"]).copy()
    if entry.empty:
        return pd.DataFrame(columns=columns)
    entry = (
        entry.sort_values(["company_id", "deal_date", "deal_id"])
        .groupby("company_id", as_index=False)
        .first()
        .rename(columns={"deal_date": "entry_date", "deal_quarter_idx": "entry_quarter_idx"})
    )
    entry["entry_quarter_idx"] = entry["entry_quarter_idx"].astype(int)
    entry["entry_year"] = (entry["entry_quarter_idx"] // 4).astype(int)
    return entry[columns].copy()


def build_buyout_selection_exit_candidates(fact_buyout_realization_event: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "company_id",
        "event_date",
        "event_value_usd",
        "route_label",
        "confidence_tier",
        "route_source",
        "priority",
        "quarter_idx",
    ]
    if fact_buyout_realization_event.empty:
        return pd.DataFrame(columns=columns)
    candidates = fact_buyout_realization_event.copy()
    candidates["event_date"] = pd.to_datetime(candidates["event_date"], errors="coerce")
    candidates = candidates.loc[
        candidates["event_date"].notna()
        & candidates["pit_safe_flag"].astype(int).eq(1)
        & candidates["directness_class"].astype(str).eq("direct_dated")
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    candidates["event_value_usd"] = pd.to_numeric(candidates["value_amount"], errors="coerce")
    candidates["route_label"] = candidates["headline_route_family"].astype(str)
    candidates["confidence_tier"] = candidates["exit_label_confidence"].astype(str)
    candidates["priority"] = candidates["confidence_tier"].map({"high": 0, "medium": 1, "low": 2}).fillna(3).astype(int)
    candidates["quarter_idx"] = quarter_idx_from_dates(candidates["event_date"])
    return candidates[
        ["company_id", "event_date", "event_value_usd", "route_label", "confidence_tier", "route_source", "priority", "quarter_idx"]
    ].copy()


def build_live_buyout_only_dataset(config: dict) -> dict:
    sources = load_actual_inputs(config)
    preqin_master = build_preqin_company_master(sources["preqin_vc"], sources["preqin_buyout"])
    crosswalk = build_crosswalk(preqin_master, sources["cb_companies"])
    company_master = build_company_master(
        preqin_master,
        sources["cb_companies"],
        sources["cb_rounds"],
        crosswalk,
    )
    round_events = build_round_events(
        company_master,
        sources["preqin_vc"],
        sources["preqin_buyout"],
        sources["cb_rounds"],
    )
    universe_map_full = build_company_universe_map(round_events)
    buyout_company_ids = set(
        universe_map_full.loc[universe_map_full["universe"].astype(str).eq("buyout_pe"), "company_id"].astype(str)
    )
    if not buyout_company_ids:
        raise ValueError("No buyout_pe companies were identified in the staged actual extracts.")
    company_master_buyout = company_master.loc[
        company_master["company_id"].astype(str).isin(buyout_company_ids)
    ].copy()
    round_events_buyout = round_events.loc[
        round_events["company_id"].astype(str).isin(buyout_company_ids)
    ].copy()
    universe_map = pd.DataFrame({"company_id": sorted(buyout_company_ids), "universe": "buyout_pe"})
    bridge_company_deal = build_bridge_company_deal(
        company_master_buyout,
        sources.get("preqin_buyout", pd.DataFrame()),
        universe_map,
    )
    buyout_entry_override = build_buyout_entry_override(bridge_company_deal)
    if buyout_entry_override.empty:
        raise ValueError("No dated buyout company-to-deal rows were available for the buyout-only rebuild.")
    bridge_deal_fund = build_bridge_deal_fund(bridge_company_deal)
    bridge_deal_firm = build_bridge_deal_firm(bridge_company_deal)
    analysis_end_idx = quarter_idx_from_label(str(config["analysis_end_quarter"]))
    bridge_company_fund_active_window = build_bridge_company_fund_active_window(
        bridge_company_deal,
        analysis_end_quarter_idx=analysis_end_idx,
    )
    fact_buyout_realization_event = build_fact_buyout_realization_event(
        company_master_buyout,
        sources,
        bridge_company_deal,
        int(config.get("buyout_realization_min_gap_quarters", 4)),
    )
    selection_exit_candidates = build_buyout_selection_exit_candidates(fact_buyout_realization_event)
    legacy_direct_exit_candidates = build_direct_exit_candidates(
        company_master_buyout,
        round_events_buyout,
        sources["preqin_vc"],
        sources["preqin_buyout"],
        sources["cb_acquisitions"],
        sources["cb_ipos"],
    )
    legacy_direct_exit_candidates = legacy_direct_exit_candidates.loc[
        legacy_direct_exit_candidates["company_id"].astype(str).isin(buyout_company_ids)
    ].copy()
    selection_candidates = selection_exit_candidates if not selection_exit_candidates.empty else legacy_direct_exit_candidates.copy()
    selection_exits = choose_first_exit(selection_candidates, analysis_end_idx) if not selection_candidates.empty else pd.DataFrame(
        columns=["company_id", "exit_date", "exit_quarter_idx", "route_label", "confidence_tier", "route_source", "event_value_usd"]
    )
    legacy_chosen_exits = choose_first_exit(legacy_direct_exit_candidates, analysis_end_idx) if not legacy_direct_exit_candidates.empty else pd.DataFrame(
        columns=["company_id", "exit_date", "exit_quarter_idx", "route_label", "confidence_tier", "route_source", "event_value_usd"]
    )
    selection_round_events = buyout_entry_override.rename(
        columns={"entry_date": "round_date", "entry_quarter_idx": "quarter_idx"}
    )[["company_id", "round_date", "quarter_idx"]].copy()
    window_selection_grid = build_window_selection_grid(selection_round_events, selection_exits, config)
    selected_window, window_selection_grid = select_actual_window(window_selection_grid, config)
    selected_min_entry_year = int(selected_window["min_entry_year"])
    config["train_end_quarter"] = str(selected_window["train_end_quarter"])
    config["validation_end_quarter"] = str(selected_window["validation_end_quarter"])
    config["test_end_quarter"] = str(selected_window["test_end_quarter"])
    config["panel_end_quarter"] = str(selected_window["test_end_quarter"])
    keep_company_ids = set(
        buyout_entry_override.loc[
            buyout_entry_override["entry_year"].astype(int).ge(selected_min_entry_year),
            "company_id",
        ].astype(str)
    )
    if not keep_company_ids:
        raise ValueError(f"No buyout companies remain after applying min_entry_year={selected_min_entry_year}.")
    company_master_buyout = company_master_buyout.loc[
        company_master_buyout["company_id"].astype(str).isin(keep_company_ids)
    ].copy()
    round_events_buyout = round_events_buyout.loc[
        round_events_buyout["company_id"].astype(str).isin(keep_company_ids)
    ].copy()
    universe_map = universe_map.loc[universe_map["company_id"].astype(str).isin(keep_company_ids)].copy()
    buyout_entry_override = buyout_entry_override.loc[
        buyout_entry_override["company_id"].astype(str).isin(keep_company_ids)
    ].copy()
    bridge_company_deal = bridge_company_deal.loc[
        bridge_company_deal["company_id"].astype(str).isin(keep_company_ids)
    ].copy()
    bridge_deal_fund = bridge_deal_fund.loc[
        bridge_deal_fund["deal_id"].astype(str).isin(set(bridge_company_deal["deal_id"].astype(str)))
    ].copy()
    bridge_deal_firm = bridge_deal_firm.loc[
        bridge_deal_firm["deal_id"].astype(str).isin(set(bridge_company_deal["deal_id"].astype(str)))
    ].copy()
    bridge_company_fund_active_window = bridge_company_fund_active_window.loc[
        bridge_company_fund_active_window["company_id"].astype(str).isin(keep_company_ids)
    ].copy()
    fact_buyout_realization_event = fact_buyout_realization_event.loc[
        fact_buyout_realization_event["company_id"].astype(str).isin(keep_company_ids)
    ].copy()
    legacy_direct_exit_candidates = legacy_direct_exit_candidates.loc[
        legacy_direct_exit_candidates["company_id"].astype(str).isin(keep_company_ids)
    ].copy()
    legacy_chosen_exits = legacy_chosen_exits.loc[
        legacy_chosen_exits["company_id"].astype(str).isin(keep_company_ids)
    ].copy()
    buyout_market_panel = build_buyout_sponsor_fund_market_panel(sources)
    panel = build_company_quarter_panel(
        company_master_buyout,
        round_events_buyout,
        legacy_chosen_exits,
        config,
        patent_event_lookup=None,
        entry_override=buyout_entry_override[["company_id", "entry_date", "entry_quarter_idx"]].copy(),
    )
    macro_panel = build_macro_panel(panel)
    panel = attach_macro(panel, macro_panel)
    panel["universe"] = "buyout_pe"
    company_master_buyout["universe"] = "buyout_pe"
    panel = attach_buyout_realization_features(
        panel,
        bridge_company_deal,
        fact_buyout_realization_event,
        buyout_market_panel,
    )
    panel = add_bucket_feature_columns(panel)
    panel = split_panel(panel, config)
    panel = add_realized_exit_within_horizon(panel, legacy_chosen_exits, int(config["holdout_horizon_quarters"]))
    panel = add_redesigned_targets(panel, legacy_chosen_exits.copy(), int(config["holdout_horizon_quarters"]))
    coverage_by_year, partition_summary = build_coverage_tables(panel)
    route_support_by_split = build_route_support_by_split(panel)
    density_by_entry_year = window_selection_grid[
        (window_selection_grid["train_end_quarter"] == str(selected_window["train_end_quarter"]))
        & (window_selection_grid["validation_end_quarter"] == str(selected_window["validation_end_quarter"]))
        & (window_selection_grid["test_end_quarter"] == str(selected_window["test_end_quarter"]))
    ].copy()
    window_selection = window_selection_grid[window_selection_grid["selected"] == 1].reset_index(drop=True)
    route_pooling_fallback_summary = build_route_pooling_fallback_summary(
        selected_window,
        bool(window_selection["used_route_pooling_fallback"].iloc[0]) if not window_selection.empty else False,
    )
    return {
        "sources": sources,
        "panel": panel,
        "company_master": company_master_buyout,
        "round_events": round_events_buyout,
        "chosen_exits": legacy_chosen_exits,
        "chosen_exits_main": legacy_chosen_exits,
        "chosen_exits_sensitivity": legacy_chosen_exits.copy(),
        "macro_panel": macro_panel,
        "buyout_market_panel": buyout_market_panel,
        "crosswalk": crosswalk.loc[crosswalk["portfolio_company_id"].isin(set(company_master_buyout["portfolio_company_id"].dropna()))].copy(),
        "coverage_by_year": coverage_by_year,
        "partition_summary": partition_summary,
        "route_support_by_split": route_support_by_split,
        "density_by_entry_year": density_by_entry_year,
        "window_selection_grid": window_selection_grid,
        "selected_min_entry_year": selected_min_entry_year,
        "window_selection": window_selection,
        "route_pooling_fallback_summary": route_pooling_fallback_summary,
        "universe_map": universe_map,
        "source_file_inventory": sources.get("source_file_inventory", pd.DataFrame()),
        "buyout_realization_field_audit": build_buyout_realization_field_audit(sources, round_events_buyout),
        "deal_fund_link_audit": build_deal_fund_link_audit(sources, round_events_buyout),
        "bridge_company_deal": bridge_company_deal,
        "bridge_deal_fund": bridge_deal_fund,
        "bridge_deal_firm": bridge_deal_firm,
        "bridge_company_fund_active_window": bridge_company_fund_active_window,
        "fact_buyout_realization_event": fact_buyout_realization_event,
    }


def buyout_target_feature_columns(panel: pd.DataFrame) -> list[str]:
    columns = [
        "age_q",
        "time_since_last_round_q",
        "log_last_round_usd",
        "time_since_acquisition_q",
        "historical_hold_window_lagged",
        "fund_dry_powder_proxy_lagged",
        "lp_demand_index_lagged",
        "buyout_sponsor_raise_10y_lagged",
        "buyout_sponsor_coinvest_share_lagged",
        "buyout_returning_lp_pct_lagged",
        "buyout_fund_months_to_final_close_lagged",
        *[column for column in sector_dummy_columns() if column in panel.columns],
        *[column for column in stage_dummy_columns() if column in panel.columns],
    ]
    return list(dict.fromkeys([column for column in columns if column in panel.columns]))


def write_run_manifest_markdown(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Run Manifest",
        "",
        f"- Git hash: `{manifest.get('git_hash', '')}`",
        f"- Command line: `{manifest.get('command_line', '')}`",
        f"- Output directory: `{manifest.get('output_dir', '')}`",
        f"- Random seed: `{manifest.get('random_seed', '')}`",
        f"- Selected minimum entry year: `{manifest.get('selected_min_entry_year', '')}`",
        f"- Train / validation / test quarters: `{manifest.get('train_end_quarter', '')}` / `{manifest.get('validation_end_quarter', '')}` / `{manifest.get('test_end_quarter', '')}`",
        f"- Candidate targets: `{manifest.get('candidate_targets', '')}`",
        f"- Target-selection protocol file: `{manifest.get('target_protocol_file', '')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_join_confidence_rules(path: Path) -> None:
    lines = [
        "# Join Confidence Rules",
        "",
        "- `deterministic`: exact identifier join on staged vendor keys already present in the local extracts.",
        "- `fuzzy`: text-based join requiring explicit evidence and audit. This pass does not use fuzzy ownership joins in production outputs.",
        "- `unsupported`: the necessary join keys are absent in the staged local export, so the bridge is emitted as empty and the limitation is documented.",
        "- Company-to-deal uses exact `portfolio_company_id` alignment through the reconstructed company master.",
        "- Deal-to-fund and deal-to-firm remain unsupported unless the staged buyout extract contains populated `fund_id` or `firm_id` keys.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_buyout_gate_definitions(path: Path, config: dict) -> None:
    lines = [
        "# Buyout Gate Definitions",
        "",
        f"- Minimum train positives: `{int(config.get('min_train_exits', 100))}`.",
        f"- Minimum validation positives: `{int(config.get('buyout_min_validation_positives', 25))}`.",
        f"- Minimum test positives: `{int(config.get('buyout_min_test_positives', 25))}`.",
        f"- Minimum direct-dated share: `{float(config.get('target_selection_min_direct_dated_share', 0.25)):.2f}`.",
        f"- Maximum inferred-transition share: `{float(config.get('buyout_max_inferred_transition_share', 0.75)):.2f}`.",
        f"- Maximum validation calibration gap: `{float(config.get('promotion_gate_calibration_gap_max', 0.05)):.2f}`.",
        f"- Maximum high-confidence calibration gap: `{float(config.get('promotion_gate_high_conf_gap_max', 0.08)):.2f}`.",
        f"- Policy acceptance band: `{float(config.get('promotion_gate_min_policy_acceptance', 0.005)):.3f}` to `{float(config.get('target_selection_max_policy_acceptance', 0.50)):.2f}`.",
        "- Fallback proxy candidates are never headline-eligible even if their diagnostic metrics look better.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_buyout_route_support_gate_audit(
    target_route_support_by_split: pd.DataFrame,
    target_selection_gates: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    columns = [
        "target_key",
        "target_name",
        "universe",
        "requested_stage2_route_set",
        "actual_stage2_route_set",
        "train_positive_events_raw",
        "train_positive_events_stage2",
        "min_train_route_support_raw",
        "min_train_route_support_stage2",
        "min_train_route_support_used",
        "route_support_scope_used",
        "stage2_min_route_support_required",
        "acceptable_route_support_pass",
        "route_support_gate_repaired",
    ]
    if target_route_support_by_split.empty:
        return pd.DataFrame(columns=columns)
    train = target_route_support_by_split.loc[
        target_route_support_by_split["split"].astype(str).eq("train")
    ].copy()
    raw = train.loc[
        train.get("support_scope", pd.Series("raw_target_routes", index=train.index)).astype(str).eq("raw_target_routes")
    ].groupby(["target_key", "target_name", "universe"], as_index=False).agg(
        train_positive_events_raw=("positive_event_count", "sum"),
        min_train_route_support_raw=("positive_event_count", "min"),
    )
    stage2 = train.loc[
        train.get("support_scope", pd.Series(index=train.index, dtype=object)).astype(str).eq("stage2_actual_view")
    ].groupby(["target_key", "target_name", "universe"], as_index=False).agg(
        train_positive_events_stage2=("positive_event_count", "sum"),
        min_train_route_support_stage2=("positive_event_count", "min"),
        requested_stage2_route_set=("requested_stage2_route_set", "max"),
        actual_stage2_route_set=("actual_stage2_route_set", "max"),
    )
    audit = raw.merge(stage2, on=["target_key", "target_name", "universe"], how="outer")
    if not target_selection_gates.empty:
        audit = audit.merge(
            target_selection_gates[
                [
                    "target_key",
                    "target_name",
                    "route_support_scope_used",
                    "min_train_route_support",
                    "acceptable_route_support_pass",
                ]
            ].rename(columns={"min_train_route_support": "min_train_route_support_used"}),
            on=["target_key", "target_name"],
            how="left",
        )
    audit["stage2_min_route_support_required"] = int(config.get("stage2_min_route_support", 5))
    audit["train_positive_events_raw"] = pd.to_numeric(audit["train_positive_events_raw"], errors="coerce").fillna(0).astype(int)
    audit["train_positive_events_stage2"] = pd.to_numeric(audit["train_positive_events_stage2"], errors="coerce").fillna(0).astype(int)
    audit["min_train_route_support_raw"] = pd.to_numeric(audit["min_train_route_support_raw"], errors="coerce").fillna(0).astype(int)
    audit["min_train_route_support_stage2"] = pd.to_numeric(audit["min_train_route_support_stage2"], errors="coerce").fillna(0).astype(int)
    audit["min_train_route_support_used"] = pd.to_numeric(audit["min_train_route_support_used"], errors="coerce").fillna(0).astype(int)
    audit["acceptable_route_support_pass"] = pd.to_numeric(audit["acceptable_route_support_pass"], errors="coerce").fillna(0).astype(int)
    audit["route_support_scope_used"] = audit.get("route_support_scope_used", pd.Series(index=audit.index, dtype=object)).fillna("raw_target_routes")
    audit["route_support_gate_repaired"] = (
        audit["route_support_scope_used"].astype(str).eq("stage2_actual_view")
        & audit["min_train_route_support_raw"].lt(int(config.get("stage2_min_route_support", 5)))
        & audit["min_train_route_support_stage2"].ge(int(config.get("stage2_min_route_support", 5)))
    ).astype(int)
    return audit[columns].sort_values(["universe", "target_name"]).reset_index(drop=True)


def write_buyout_route_support_gate_note(path: Path, audit: pd.DataFrame, config: dict) -> None:
    required = int(config.get("stage2_min_route_support", 5))
    repaired = audit.loc[audit["route_support_gate_repaired"].astype(int).eq(1)].copy() if not audit.empty else pd.DataFrame()
    lines = [
        "# Buyout Route-Support Gate Note",
        "",
        f"- The active support gate now evaluates the actual pooled stage-2 route view used in estimation, with minimum train support `{required}`.",
        "- Raw candidate-route rows are still emitted for audit, but they no longer force a failure when the model reports a pooled stage-2 route set instead of those raw subroutes.",
        f"- Targets repaired by this alignment in the current run: `{int(len(repaired))}`.",
        "",
    ]
    if not repaired.empty:
        lines.extend(
            [
                "## Repaired Targets",
                "",
                dataframe_to_markdown(
                    repaired[
                        [
                            "target_name",
                            "requested_stage2_route_set",
                            "actual_stage2_route_set",
                            "min_train_route_support_raw",
                            "min_train_route_support_stage2",
                            "route_support_scope_used",
                        ]
                    ]
                ),
                "",
            ]
        )
    else:
        lines.append("- No target required route-support repair in this run.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_buyout_policy_selection_protocol(path: Path, config: dict) -> None:
    threshold_labels = ", ".join(f"`{format_threshold_label(value)}`" for value in BUYOUT_POLICY_PROBABILITY_THRESHOLDS)
    quantile_labels = ", ".join(f"`{int(round(100 * value))}%`" for value in BUYOUT_POLICY_TOP_QUANTILES)
    lines = [
        "# Buyout Policy Selection Protocol",
        "",
        "- Target selection and policy selection are separate.",
        "- Target choice remains validation-first and calibration-first at the target level.",
        "- Policy choice is performed on the validation split only after target probabilities are estimated.",
        "- The locked test split is used exactly once for confirmation of the already-selected validation policy.",
        "",
        "## Evaluated Policy Families",
        "",
        f"- Fixed probability thresholds on target probability: {threshold_labels}.",
        f"- Top-quantile screens on target probability: {quantile_labels}.",
        "- Dual screens: target probability threshold plus predicted NPV > 0.",
        "- Dual screens: target probability threshold plus certainty equivalent > 0.",
        "- Dual screens: target probability threshold plus predicted exit-by-horizon threshold.",
        "",
        "## Acceptance-Band Selector",
        "",
        f"- Default acceptance band: `{float(config.get('buyout_policy_acceptance_min', 0.005)):.3f}` to `{float(config.get('buyout_policy_acceptance_max', 0.50)):.2f}`.",
        f"- One-time fallback band if no feasible validation policy exists: `{float(config.get('buyout_policy_acceptance_min', 0.005)):.3f}` to `{float(config.get('buyout_policy_acceptance_max_fallback', 0.60)):.2f}`.",
        "- Ranking inside the feasible band is: validation precision, lift over prevalence, balanced accuracy, then distance to the band midpoint.",
        "- If no feasible validation policy exists even after fallback, the best non-degenerate validation rule is still reported, but the activation gate remains failed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_buyout_policy_milestone_note(path: Path, baseline_dir: Path) -> None:
    lines = [
        "# Buyout Policy Milestone Note",
        "",
        f"- Frozen baseline bundle: `{baseline_dir}`.",
        "- The prior milestone repaired the buyout realization spine by restricting buyout targets to direct-dated IPO, M&A, sponsor-sale, and related dated realization events.",
        "- Directness and calibration are no longer the primary buyout blocker in this pass.",
        "- This pass is policy-focused because the remaining open questions are whether a validation-selected buyout policy can stay inside the acceptance band and whether the route-support gate matches the pooled stage-2 route view actually reported by the model.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_buyout_status_summary_lines(
    reporting_status: str,
    blocking_target: str,
    limiting_factor: str,
    caveat: str,
) -> list[str]:
    status = str(reporting_status).strip().lower()
    if status == "promoted":
        return [
            f"- The promoted buyout target is `{blocking_target}`.",
            "- No hard promotion gate is currently failing in this bounded buyout-only bundle.",
            f"- Residual limitation: `{limiting_factor}` is `none`; the remaining caveats are about ownership-link completeness and external generalization, not the current gate state.",
        ]
    detail = caveat if str(caveat).strip() else "the validation-selected buyout target still fails at least one hard gate."
    return [
        f"- The current blocking validation target is `{blocking_target}` with limiting factor `{limiting_factor}`.",
        f"- Immediate blocker detail: {detail}",
    ]


def run_buyout_realization_pipeline(user_config: dict | None = None) -> dict:
    config = DEFAULT_CONFIG.copy()
    if user_config:
        config.update(user_config)
    if str(config.get("data_mode", "actual")).strip().lower() != "actual":
        raise ValueError("buyout_only mode requires actual data mode.")
    output_dir = Path(
        config.get("output_dir")
        or (Path(__file__).resolve().parent / f"rendered-live-buyout-realization-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_live_buyout_only_dataset(config)
    sources = dataset.get("sources", {})
    buyout_panel = dataset["panel"].copy()
    bridge_company_deal = dataset.get("bridge_company_deal", pd.DataFrame())
    bridge_deal_fund = dataset.get("bridge_deal_fund", empty_bridge_deal_fund())
    bridge_deal_firm = dataset.get("bridge_deal_firm", empty_bridge_deal_firm())
    bridge_company_fund_active_window = dataset.get(
        "bridge_company_fund_active_window",
        empty_bridge_company_fund_active_window(),
    )
    fact_buyout_realization_event = dataset.get("fact_buyout_realization_event", pd.DataFrame())
    buyout_realization_event_audit = build_buyout_realization_event_audit(fact_buyout_realization_event)
    feature_dictionary = build_buyout_feature_dictionary()
    buyout_feature_coverage = build_buyout_feature_coverage(buyout_panel, feature_dictionary)
    buyout_feature_availability_by_quarter = build_buyout_feature_availability_by_quarter(
        buyout_panel,
        feature_dictionary,
    )
    sponsor_fund_join_audit = build_sponsor_fund_join_audit(
        dataset["round_events"],
        sources=sources,
        buyout_market_panel=dataset.get("buyout_market_panel"),
    )
    lp_demand_join_audit = build_lp_demand_join_audit()
    join_confidence_summary = build_join_confidence_summary(
        bridge_company_deal,
        bridge_deal_fund,
        bridge_deal_firm,
        bridge_company_fund_active_window,
    )
    buyout_field_inventory = build_buyout_field_inventory(sources)
    buyout_missing_field_manifest = build_buyout_missing_field_manifest(
        dataset.get("buyout_realization_field_audit", pd.DataFrame()),
        dataset.get("deal_fund_link_audit", pd.DataFrame()),
        sponsor_fund_join_audit,
        lp_demand_join_audit,
    )
    buyout_target_registry = build_buyout_target_registry(fact_buyout_realization_event, config)

    target_definition_frames: list[pd.DataFrame] = []
    target_prevalence_frames: list[pd.DataFrame] = []
    target_route_support_frames: list[pd.DataFrame] = []
    target_source_mix_frames: list[pd.DataFrame] = []
    target_label_audit_frames: list[pd.DataFrame] = []
    target_time_distribution_frames: list[pd.DataFrame] = []
    target_calibration_summary_frames: list[pd.DataFrame] = []
    target_evaluation_frames: list[pd.DataFrame] = []
    target_decision_backtest_frames: list[pd.DataFrame] = []
    target_policy_search_validation_frames: list[pd.DataFrame] = []
    target_policy_search_validation_feasible_frames: list[pd.DataFrame] = []
    target_policy_confirmation_test_frames: list[pd.DataFrame] = []

    for spec_row in buyout_target_registry.to_dict(orient="records"):
        if str(spec_row.get("target_family", "")) == "buyout_event_spine":
            candidate_panel, target_col, realized_prefix = build_buyout_target_candidate_panel(
                buyout_panel,
                spec_row,
                fact_buyout_realization_event,
            )
        else:
            candidate_panel, target_col, realized_prefix = build_target_candidate_panel(buyout_panel, spec_row)
        result = evaluate_target_candidate(
            candidate_panel,
            spec_row,
            target_col,
            realized_prefix,
            dataset["company_master"],
            config,
            feature_columns_override=buyout_target_feature_columns(candidate_panel),
            feature_backbone="buyout_realization_core",
        )
        target_definition_frames.append(result["definition"])
        target_prevalence_frames.append(result["prevalence"])
        target_route_support_frames.append(result["route_support"])
        target_source_mix_frames.append(result["source_mix"])
        target_label_audit_frames.append(result["label_audit"])
        target_time_distribution_frames.append(result["time_distribution"])
        if isinstance(result["calibration_summary"], pd.DataFrame) and not result["calibration_summary"].empty:
            target_calibration_summary_frames.append(result["calibration_summary"])
        if isinstance(result["evaluation_metrics"], pd.DataFrame) and not result["evaluation_metrics"].empty:
            target_evaluation_frames.append(result["evaluation_metrics"])
        if isinstance(result["decision_backtest"], pd.DataFrame) and not result["decision_backtest"].empty:
            target_decision_backtest_frames.append(result["decision_backtest"])
        if isinstance(result.get("policy_search_validation"), pd.DataFrame) and not result["policy_search_validation"].empty:
            target_policy_search_validation_frames.append(result["policy_search_validation"])
        if isinstance(result.get("policy_search_validation_feasible"), pd.DataFrame) and not result["policy_search_validation_feasible"].empty:
            target_policy_search_validation_feasible_frames.append(result["policy_search_validation_feasible"])
        if isinstance(result.get("policy_confirmation_test"), pd.DataFrame) and not result["policy_confirmation_test"].empty:
            target_policy_confirmation_test_frames.append(result["policy_confirmation_test"])

    target_definitions = pd.concat(target_definition_frames, ignore_index=True) if target_definition_frames else pd.DataFrame()
    target_prevalence_by_split = pd.concat(target_prevalence_frames, ignore_index=True) if target_prevalence_frames else pd.DataFrame()
    target_route_support_by_split = pd.concat(target_route_support_frames, ignore_index=True) if target_route_support_frames else pd.DataFrame()
    target_source_mix = pd.concat(target_source_mix_frames, ignore_index=True) if target_source_mix_frames else pd.DataFrame()
    target_label_confidence_audit = pd.concat(target_label_audit_frames, ignore_index=True) if target_label_audit_frames else pd.DataFrame()
    target_time_distribution = pd.concat(target_time_distribution_frames, ignore_index=True) if target_time_distribution_frames else pd.DataFrame()
    calibration_targets_summary = pd.concat(target_calibration_summary_frames, ignore_index=True) if target_calibration_summary_frames else pd.DataFrame()
    evaluation_metrics_targets = pd.concat(target_evaluation_frames, ignore_index=True) if target_evaluation_frames else pd.DataFrame()
    decision_backtest_targets = pd.concat(target_decision_backtest_frames, ignore_index=True) if target_decision_backtest_frames else pd.DataFrame()
    buyout_policy_search_validation = pd.concat(target_policy_search_validation_frames, ignore_index=True) if target_policy_search_validation_frames else pd.DataFrame()
    buyout_policy_search_validation_feasible = pd.concat(target_policy_search_validation_feasible_frames, ignore_index=True) if target_policy_search_validation_feasible_frames else pd.DataFrame()
    buyout_policy_confirmation_test = pd.concat(target_policy_confirmation_test_frames, ignore_index=True) if target_policy_confirmation_test_frames else pd.DataFrame()
    validation_target_gap = evaluation_metrics_targets.loc[
        evaluation_metrics_targets["evaluation_view"].astype(str).eq("validation_selection"),
        ["target_key", "feature_backbone", "mean_abs_calibration_gap"],
    ].rename(columns={"mean_abs_calibration_gap": "target_validation_mean_abs_calibration_gap"})

    def attach_validation_gap(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        merged = frame.merge(
            validation_target_gap,
            on=["target_key", "feature_backbone"],
            how="left",
        )
        merged["target_validation_calibration_gate_pass"] = pd.to_numeric(
            merged["target_validation_mean_abs_calibration_gap"],
            errors="coerce",
        ).fillna(np.inf).le(float(config.get("promotion_gate_calibration_gap_max", 0.05))).astype(int)
        return merged

    buyout_policy_search_validation = attach_validation_gap(buyout_policy_search_validation)
    buyout_policy_search_validation_feasible = attach_validation_gap(buyout_policy_search_validation_feasible)
    buyout_policy_confirmation_test = attach_validation_gap(buyout_policy_confirmation_test)
    directness_by_target = build_target_source_summary(target_source_mix)
    directness_by_universe = build_directness_by_universe(directness_by_target)
    selected_target_feature_backbones = select_target_feature_backbones(
        buyout_target_registry,
        evaluation_metrics_targets,
        decision_backtest_targets,
        config,
    )
    buyout_target_selection_gates = build_buyout_target_selection_gates(
        buyout_target_registry,
        selected_target_feature_backbones,
        evaluation_metrics_targets,
        decision_backtest_targets,
        target_route_support_by_split,
        directness_by_target,
        target_prevalence_by_split,
        config,
    )
    buyout_target_leaderboard_validation = build_target_leaderboard_validation(buyout_target_selection_gates)
    buyout_target_confirmation_test = build_target_confirmation_test(
        buyout_target_leaderboard_validation,
        evaluation_metrics_targets,
        decision_backtest_targets,
    )
    buyout_decision_usefulness_by_target = build_decision_usefulness_by_target(
        buyout_target_leaderboard_validation,
        buyout_target_confirmation_test,
        target_prevalence_by_split,
        config,
    )
    buyout_target_recommendation_validation = build_target_recommendation_table_v2(
        buyout_target_leaderboard_validation,
        buyout_target_confirmation_test,
        config,
    )
    buyout_promotion_gate = build_buyout_promotion_gate(
        buyout_target_leaderboard_validation,
        buyout_target_confirmation_test,
        buyout_target_selection_gates,
        config,
    )
    buyout_claim_matrix = build_buyout_claim_matrix(
        buyout_target_recommendation_validation,
        buyout_promotion_gate,
    )
    buyout_stage2_route_view = target_route_support_by_split.loc[
        target_route_support_by_split.get("support_scope", pd.Series(index=target_route_support_by_split.index, dtype=object)).astype(str).eq("stage2_actual_view")
    ].copy()
    buyout_route_support_gate_audit = build_buyout_route_support_gate_audit(
        target_route_support_by_split,
        buyout_target_selection_gates,
        config,
    )
    buyout_recommendation = buyout_target_recommendation_validation.loc[
        buyout_target_recommendation_validation["recommended_for_universe"].astype(int).eq(1)
    ].copy()
    baseline_bundle_dir = Path(__file__).resolve().parent / "rendered-live-buyout-realization-20260403-empirical"

    run_manifest = {
        "git_hash": safe_git_hash(Path(__file__).resolve().parents[5]),
        "command_line": str(config.get("command_line", "")),
        "output_dir": str(output_dir),
        "random_seed": int(config.get("random_seed", 42)),
        "selected_min_entry_year": int(dataset["selected_min_entry_year"]),
        "train_end_quarter": str(config["train_end_quarter"]),
        "validation_end_quarter": str(config["validation_end_quarter"]),
        "test_end_quarter": str(config["test_end_quarter"]),
        "candidate_targets": "|".join(buyout_target_registry["target_name"].astype(str).tolist()),
        "target_protocol_file": "buyout_policy_selection_protocol.md",
        "frozen_buyout_baseline_bundle": str(baseline_bundle_dir),
    }

    generated_files: list[Path] = []
    for path, frame in [
        (output_dir / "buyout_field_inventory.csv", buyout_field_inventory),
        (output_dir / "source_file_inventory.csv", dataset.get("source_file_inventory", pd.DataFrame())),
        (output_dir / "buyout_missing_field_manifest.csv", buyout_missing_field_manifest),
        (output_dir / "fact_buyout_realization_event.csv", fact_buyout_realization_event),
        (output_dir / "buyout_realization_event_audit.csv", buyout_realization_event_audit),
        (output_dir / "bridge_company_deal.csv", bridge_company_deal),
        (output_dir / "bridge_deal_fund.csv", bridge_deal_fund),
        (output_dir / "bridge_deal_firm.csv", bridge_deal_firm),
        (output_dir / "bridge_company_fund_active_window.csv", bridge_company_fund_active_window),
        (output_dir / "sponsor_fund_join_audit.csv", sponsor_fund_join_audit),
        (output_dir / "lp_demand_join_audit.csv", lp_demand_join_audit),
        (output_dir / "join_confidence_summary.csv", join_confidence_summary),
        (output_dir / "buyout_feature_dictionary.csv", feature_dictionary),
        (output_dir / "buyout_feature_coverage.csv", buyout_feature_coverage),
        (output_dir / "buyout_feature_availability_by_quarter.csv", buyout_feature_availability_by_quarter),
        (output_dir / "buyout_target_registry.csv", buyout_target_registry),
        (output_dir / "buyout_target_support_by_split.csv", target_route_support_by_split),
        (output_dir / "buyout_stage2_route_view.csv", buyout_stage2_route_view),
        (output_dir / "buyout_route_support_gate_audit.csv", buyout_route_support_gate_audit),
        (output_dir / "buyout_target_prevalence.csv", target_prevalence_by_split),
        (output_dir / "buyout_target_directness.csv", directness_by_target),
        (output_dir / "buyout_target_leaderboard_validation.csv", buyout_target_leaderboard_validation),
        (output_dir / "buyout_target_confirmation_test.csv", buyout_target_confirmation_test),
        (output_dir / "buyout_policy_search_validation.csv", buyout_policy_search_validation),
        (output_dir / "buyout_policy_search_validation_feasible.csv", buyout_policy_search_validation_feasible),
        (output_dir / "buyout_policy_confirmation_test.csv", buyout_policy_confirmation_test),
        (output_dir / "buyout_decision_usefulness_by_target.csv", buyout_decision_usefulness_by_target),
        (output_dir / "buyout_target_recommendation_validation.csv", buyout_target_recommendation_validation),
        (output_dir / "buyout_promotion_gate.csv", buyout_promotion_gate),
        (output_dir / "buyout_claim_matrix.csv", buyout_claim_matrix),
        (output_dir / "directness_by_target.csv", directness_by_target),
        (output_dir / "directness_by_universe.csv", directness_by_universe),
        (output_dir / "evaluation_metrics_targets.csv", evaluation_metrics_targets),
        (output_dir / "decision_backtest_targets.csv", decision_backtest_targets),
        (output_dir / "calibration_targets_summary.csv", calibration_targets_summary),
        (output_dir / "target_definitions.csv", target_definitions),
        (output_dir / "label_confidence_audit_targets.csv", target_label_confidence_audit),
        (output_dir / "target_time_distribution_all.csv", target_time_distribution),
        (output_dir / "selected_target_feature_backbones.csv", selected_target_feature_backbones),
        (output_dir / "target_selection_gates.csv", buyout_target_selection_gates),
        (output_dir / "run_manifest_helper.csv", pd.DataFrame([run_manifest])),
    ]:
        frame.to_csv(path, index=False)
        generated_files.append(path)

    write_buyout_field_inventory(output_dir / "buyout_field_inventory.md", buyout_field_inventory)
    generated_files.append(output_dir / "buyout_field_inventory.md")
    write_buyout_missing_field_manifest(output_dir / "buyout_missing_field_manifest.md", buyout_missing_field_manifest)
    generated_files.append(output_dir / "buyout_missing_field_manifest.md")
    write_buyout_realization_event_audit(output_dir / "buyout_realization_event_audit.md", buyout_realization_event_audit)
    generated_files.append(output_dir / "buyout_realization_event_audit.md")
    write_buyout_target_registry(output_dir / "buyout_target_registry.md", buyout_target_registry)
    generated_files.append(output_dir / "buyout_target_registry.md")
    write_buyout_feature_pit_rules(output_dir / "buyout_feature_pit_rules.md", feature_dictionary)
    generated_files.append(output_dir / "buyout_feature_pit_rules.md")
    write_join_confidence_rules(output_dir / "join_confidence_rules.md")
    generated_files.append(output_dir / "join_confidence_rules.md")
    write_buyout_gate_definitions(output_dir / "buyout_gate_definitions.md", config)
    generated_files.append(output_dir / "buyout_gate_definitions.md")
    write_json(output_dir / "run_manifest.json", run_manifest)
    generated_files.append(output_dir / "run_manifest.json")
    write_run_manifest_markdown(output_dir / "run_manifest.md", run_manifest)
    generated_files.append(output_dir / "run_manifest.md")
    write_buyout_policy_milestone_note(output_dir / "buyout_policy_milestone_note.md", baseline_bundle_dir)
    generated_files.append(output_dir / "buyout_policy_milestone_note.md")
    write_buyout_policy_selection_protocol(output_dir / "buyout_policy_selection_protocol.md", config)
    generated_files.append(output_dir / "buyout_policy_selection_protocol.md")
    write_buyout_route_support_gate_note(output_dir / "buyout_route_support_gate_note.md", buyout_route_support_gate_audit, config)
    generated_files.append(output_dir / "buyout_route_support_gate_note.md")
    (output_dir / "buyout_calibration_metric_dictionary.md").write_text(
        "\n".join(
            [
                "# Buyout Calibration Metric Dictionary",
                "",
                "- `mean_abs_calibration_gap`: canonical validation ranking metric.",
                "- `max_abs_calibration_gap`: worst decile gap; diagnostic only.",
                "- `brier_score`: supporting diagnostic only.",
                "- `pr_auc` and `roc_auc`: supporting ranking diagnostics only.",
                "- `high_confidence_mean_abs_calibration_gap`: robustness slice built from exit-label confidence, not entity-match confidence.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generated_files.append(output_dir / "buyout_calibration_metric_dictionary.md")
    (output_dir / "buyout_source_robustness_report.md").write_text(
        "\n".join(
            [
                "# Buyout Source Robustness Report",
                "",
                "## Directness By Target",
                "",
                dataframe_to_markdown(directness_by_target),
                "",
                "## High-Confidence Exit-Label Slice",
                "",
                dataframe_to_markdown(
                    selected_target_view_frame(
                        buyout_target_leaderboard_validation,
                        evaluation_metrics_targets,
                        "high_confidence_exit_label_only",
                    )
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    generated_files.append(output_dir / "buyout_source_robustness_report.md")
    (output_dir / "label_provenance_dictionary.md").write_text(
        "\n".join(
            [
                "# Label Provenance Dictionary",
                "",
                "- `direct_dated`: directly dated realization event from Crunchbase or a later completed Preqin buyout row on the same company.",
                "- `direct_undated`: direct event without a safe event date. Not used in headline-eligible targets in this pass.",
                "- `inferred_transition`: dated transition inferred from weaker provider status logic.",
                "- `proxy_only`: fallback proxy evidence only.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generated_files.append(output_dir / "label_provenance_dictionary.md")
    (output_dir / "buyout_panel_design_note.md").write_text(
        "\n".join(
            [
                "# Buyout Panel Design Note",
                "",
                "- This pass keeps the chapter’s company-quarter surface but restricts buyout rows to quarters after the first dated buyout acquisition.",
                "- The reason is pragmatic: the staged local export supports deterministic company-to-deal timing, but not company-to-fund ownership windows.",
                "- The resulting risk set is a post-acquisition buyout company-quarter panel, not a pure venture-style company life-cycle panel.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generated_files.append(output_dir / "buyout_panel_design_note.md")
    (output_dir / "buyout_risk_set_definition.md").write_text(
        "\n".join(
            [
                "# Buyout Risk Set Definition",
                "",
                "- Universe: `buyout_pe` company-quarter rows only.",
                "- Start of risk set: first quarter strictly after the first dated buyout deal observed for the company.",
                "- End of risk set: earliest qualifying target event quarter, or the configured panel end quarter if no qualifying event occurs.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generated_files.append(output_dir / "buyout_risk_set_definition.md")
    selected_row = buyout_recommendation.head(1)
    validation_selected_row = buyout_target_recommendation_validation.loc[
        buyout_target_recommendation_validation["selected_by_validation"].astype(int).eq(1)
    ].head(1)
    selected_target_name = str(selected_row["target_name"].iloc[0]) if not selected_row.empty else "none"
    selected_target_key = str(selected_row["target_key"].iloc[0]) if not selected_row.empty else ""
    validation_selected_target_name = str(validation_selected_row["target_name"].iloc[0]) if not validation_selected_row.empty else selected_target_name
    validation_selected_target_key = str(validation_selected_row["target_key"].iloc[0]) if not validation_selected_row.empty else selected_target_key
    selected_status = (
        str(buyout_claim_matrix["reporting_status"].iloc[0])
        if not buyout_claim_matrix.empty
        else "provisional"
    )
    selected_caveat = (
        str(buyout_claim_matrix["unresolved_caveat"].iloc[0])
        if not buyout_claim_matrix.empty and "unresolved_caveat" in buyout_claim_matrix.columns
        else ""
    )
    selected_limiting_factor = (
        str(buyout_claim_matrix["main_limiting_factor"].iloc[0])
        if not buyout_claim_matrix.empty and "main_limiting_factor" in buyout_claim_matrix.columns
        else "buyout_realization_support_still_incomplete"
    )
    blocking_target = (
        str(buyout_claim_matrix["blocking_validation_target_name"].iloc[0])
        if not buyout_claim_matrix.empty and "blocking_validation_target_name" in buyout_claim_matrix.columns
        else selected_target_name
    )
    selected_policy_validation_row = buyout_policy_search_validation.loc[
        buyout_policy_search_validation["target_key"].astype(str).eq(validation_selected_target_key)
        & buyout_policy_search_validation["selected_on_validation"].astype(int).eq(1)
    ].head(1) if not buyout_policy_search_validation.empty else pd.DataFrame()
    selected_policy_test_row = buyout_policy_confirmation_test.loc[
        buyout_policy_confirmation_test["target_key"].astype(str).eq(validation_selected_target_key)
        & buyout_policy_confirmation_test["selected_on_validation"].astype(int).eq(1)
    ].head(1) if not buyout_policy_confirmation_test.empty else pd.DataFrame()
    selected_route_support_row = buyout_route_support_gate_audit.loc[
        buyout_route_support_gate_audit["target_key"].astype(str).eq(validation_selected_target_key)
    ].head(1)
    selected_policy_key = str(selected_policy_validation_row["policy_key"].iloc[0]) if not selected_policy_validation_row.empty else ""
    selected_policy_scope = str(selected_route_support_row["route_support_scope_used"].iloc[0]) if not selected_route_support_row.empty else "raw_target_routes"
    route_support_repaired = int(selected_route_support_row["route_support_gate_repaired"].iloc[0]) if not selected_route_support_row.empty else 0
    validation_policy_table = selected_policy_validation_row[
        [
            "policy_key",
            "policy_family",
            "acceptance_rate",
            "precision",
            "recall",
            "lift_over_prevalence",
            "balanced_accuracy",
            "selection_status",
            "selection_fallback_band_used",
        ]
    ].copy() if not selected_policy_validation_row.empty else pd.DataFrame()
    test_policy_table = selected_policy_test_row[
        [
            "policy_key",
            "policy_family",
            "acceptance_rate",
            "precision",
            "recall",
            "lift_over_prevalence",
            "balanced_accuracy",
        ]
    ].copy() if not selected_policy_test_row.empty else pd.DataFrame()
    policy_status_line = (
        f"Buyout promoted to chapter-headline empirical target with `{selected_target_name}` and policy `{selected_policy_key}`."
        if selected_status == "promoted"
        else f"Buyout remains provisional because `{selected_limiting_factor}`."
    )
    (output_dir / "chapter_target_doctrine.md").write_text(
        "\n".join(
            [
                "# Chapter Target Doctrine",
                "",
                "- Venture/growth remains `hard_timely_liquidity_by_8q` with status `doctrinal_baseline`.",
                f"- Buyout/PE selected target in this rebuild: `{selected_target_name}`.",
                f"- Validation-selected buyout target: `{validation_selected_target_name}`.",
                (f"- Validation-selected buyout policy: `{selected_policy_key}`." if selected_policy_key else "- Validation-selected buyout policy: `none`."),
                f"- Buyout/PE reporting status: `{selected_status}`.",
                "- Buyout is promoted only if directness, calibration, policy-usefulness, and support gates all pass on the rebuilt realization spine.",
                f"- Route-support gate scope used: `{selected_policy_scope}`.",
                f"- Current blocking validation target: `{blocking_target}`.",
                (f"- Current caveat: {selected_caveat}" if selected_caveat else "- Current caveat: the validation-selected buyout target still fails at least one hard gate."),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generated_files.append(output_dir / "chapter_target_doctrine.md")
    (output_dir / "chapter_target_selection_limitations.md").write_text(
        "\n".join(
            [
                "# Chapter Target Selection Limitations",
                "",
                "- Preqin and Crunchbase are commercial, selective, and partly self-reported sources.",
                "- The staged local buyout extract still lacks direct deal-to-fund and deal-to-firm ownership keys.",
                "- Company-linked LP-demand joins remain unsupported in the staged graph.",
                f"- Route-support gate evaluation now uses `{selected_policy_scope}` for the validation-selected buyout target.",
                *build_buyout_status_summary_lines(
                    selected_status,
                    blocking_target,
                    selected_limiting_factor,
                    selected_caveat,
                ),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generated_files.append(output_dir / "chapter_target_selection_limitations.md")
    (output_dir / "buyout_realization_mechanics_note.md").write_text(
        "\n".join(
            [
                "# Buyout Realization Mechanics Note",
                "",
                "- Newly direct-dated mechanics in this pass come from later completed Preqin buyout rows on the same company plus direct Crunchbase IPO/M&A events.",
                "- Explicit recapitalization, secondary-like, continuation, and partial-realization wording is used when it appears on those later dated rows.",
                "- Direct deal-to-fund ownership linkage is still unavailable, so the realization spine is stronger on event timing than on owner attribution.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generated_files.append(output_dir / "buyout_realization_mechanics_note.md")
    (output_dir / "buyout_target_doctrine.md").write_text(
        "\n".join(
            [
                "# Buyout Target Doctrine",
                "",
                f"- Final buyout status: `{selected_status}`.",
                policy_status_line,
                f"- Reported buyout target: `{selected_target_name}`.",
                f"- Validation-selected buyout target: `{validation_selected_target_name}`.",
                (f"- Validation-selected policy: `{selected_policy_key}`." if selected_policy_key else "- Validation-selected policy: `none`."),
                f"- Route-support scope used for gating: `{selected_policy_scope}`.",
                f"- Route-support gate repaired by pooled-stage2 alignment: `{route_support_repaired}`.",
                f"- Main limiting factor: `{selected_limiting_factor}`.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generated_files.append(output_dir / "buyout_target_doctrine.md")
    (output_dir / "chapter_buyout_policy_findings.md").write_text(
        "\n".join(
            [
                "# Chapter Buyout Policy Findings",
                "",
                "- Directness is repaired: the buyout candidate targets in this pass are direct-dated only.",
                f"- Validation-selected buyout target: `{validation_selected_target_name}`.",
                (f"- Validation-selected buyout policy: `{selected_policy_key}`." if selected_policy_key else "- Validation-selected buyout policy: `none`."),
                (f"- Validation policy metrics: acceptance `{float(selected_policy_validation_row['acceptance_rate'].iloc[0]):.4f}`, precision `{float(selected_policy_validation_row['precision'].iloc[0]):.4f}`, recall `{float(selected_policy_validation_row['recall'].iloc[0]):.4f}`, lift `{float(selected_policy_validation_row['lift_over_prevalence'].iloc[0]):.4f}`, balanced accuracy `{float(selected_policy_validation_row['balanced_accuracy'].iloc[0]):.4f}`." if not selected_policy_validation_row.empty else "- Validation policy metrics are unavailable."),
                (f"- Locked-test confirmation for that policy: acceptance `{float(selected_policy_test_row['acceptance_rate'].iloc[0]):.4f}`, precision `{float(selected_policy_test_row['precision'].iloc[0]):.4f}`, recall `{float(selected_policy_test_row['recall'].iloc[0]):.4f}`, lift `{float(selected_policy_test_row['lift_over_prevalence'].iloc[0]):.4f}`, balanced accuracy `{float(selected_policy_test_row['balanced_accuracy'].iloc[0]):.4f}`." if not selected_policy_test_row.empty else "- Locked-test policy confirmation metrics are unavailable."),
                f"- Route-support gate scope used: `{selected_policy_scope}`; repaired by pooled-stage2 alignment: `{route_support_repaired}`.",
                f"- Final buyout status: `{selected_status}`.",
                f"- Final blocker: `{selected_limiting_factor}`.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generated_files.append(output_dir / "chapter_buyout_policy_findings.md")
    (output_dir / "chapter_buyout_policy_tables.md").write_text(
        "\n".join(
            [
                "# Chapter Buyout Policy Tables",
                "",
                "## Validation-Selected Policy",
                "",
                (dataframe_to_markdown(validation_policy_table) if not validation_policy_table.empty else "_No selected validation policy row._"),
                "",
                "## Locked-Test Confirmation",
                "",
                (dataframe_to_markdown(test_policy_table) if not test_policy_table.empty else "_No selected locked-test policy row._"),
                "",
                "## Promotion Gate",
                "",
                dataframe_to_markdown(buyout_promotion_gate),
                "",
                "## Route-Support Gate Audit",
                "",
                dataframe_to_markdown(
                    selected_route_support_row if not selected_route_support_row.empty else buyout_route_support_gate_audit.head(0)
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    generated_files.append(output_dir / "chapter_buyout_policy_tables.md")
    (output_dir / "chapter_buyout_policy_limitations.md").write_text(
        "\n".join(
            [
                "# Chapter Buyout Policy Limitations",
                "",
                "- The staged local buyout graph is materially better on dated realization timing than on owner attribution.",
                "- Direct company-to-fund and company-to-firm ownership joins remain incomplete in the local staged extracts.",
                "- LP-demand signals remain market-quarter aggregates, not company-linked owner signals.",
                f"- The validation-selected buyout target is `{validation_selected_target_name}`, but the final reported status is `{selected_status}` because `{selected_limiting_factor}` remains binding.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generated_files.append(output_dir / "chapter_buyout_policy_limitations.md")
    (output_dir / "buyout_policy_run_manifest.md").write_text(
        "\n".join(
            [
                "# Buyout Policy Run Manifest",
                "",
                f"- Frozen baseline bundle: `{baseline_bundle_dir}`.",
                f"- Output directory: `{output_dir}`.",
                f"- Command line: `{config.get('command_line', '')}`.",
                f"- Selected minimum entry year: `{int(dataset['selected_min_entry_year'])}`.",
                f"- Train / validation / test quarters: `{config['train_end_quarter']}` / `{config['validation_end_quarter']}` / `{config['test_end_quarter']}`.",
                f"- Validation-selected buyout target: `{validation_selected_target_name}`.",
                (f"- Validation-selected buyout policy: `{selected_policy_key}`." if selected_policy_key else "- Validation-selected buyout policy: `none`."),
                f"- Route-support gate scope used: `{selected_policy_scope}`.",
                f"- Final buyout status: `{selected_status}`.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generated_files.append(output_dir / "buyout_policy_run_manifest.md")

    return {
        "config": config,
        "dataset": dataset,
        "output_dir": output_dir,
        "buyout_recommendation": buyout_recommendation,
        "buyout_promotion_gate": buyout_promotion_gate,
        "generated_files": generated_files,
        "fact_buyout_realization_event": fact_buyout_realization_event,
        "buyout_claim_matrix": buyout_claim_matrix,
    }


def run_live_pipeline(user_config: dict | None = None) -> dict:
    config = DEFAULT_CONFIG.copy()
    if user_config:
        config.update(user_config)
    if bool(config.get("buyout_only")):
        return run_buyout_realization_pipeline(config)
    data_mode = str(config.get("data_mode", "sample")).strip().lower()
    if data_mode == "live":
        data_mode = "actual"
        config["data_mode"] = data_mode
    if data_mode == "actual" and (not user_config or "skip_feature_search" not in user_config):
        config["skip_feature_search"] = True
    if data_mode == "actual" and config.get("use_quarter_fixed_effects") is None:
        config["use_quarter_fixed_effects"] = True
    default_dir_name = "rendered-sample" if data_mode == "sample" else "rendered-live"
    output_dir = Path(config.get("output_dir") or (Path(__file__).resolve().parent / default_dir_name)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(config)
    if not dataset["window_selection"].empty:
        selected_window = dataset["window_selection"].iloc[0]
        for key in ["train_end_quarter", "validation_end_quarter", "test_end_quarter"]:
            if key in selected_window.index:
                config[key] = str(selected_window[key])
        config["panel_end_quarter"] = str(config.get("panel_end_quarter") or config["test_end_quarter"])
    train_panel = dataset["panel"][dataset["panel"]["split"] == "train"].copy()
    fitted = fit_multinomial_hazard(train_panel, config)
    horizon = int(config["holdout_horizon_quarters"])
    mode_suffix = "sample" if data_mode == "sample" else "actual"
    route_pooling_used = bool(
        dataset["route_pooling_fallback_summary"]["used_route_pooling_fallback"].iloc[0]
    ) if not dataset["route_pooling_fallback_summary"].empty else False
    probability_thresholds = resolve_probability_thresholds(config)
    primary_threshold = float(config.get("primary_confusion_threshold", 0.02))
    if round(primary_threshold, 4) not in {round(value, 4) for value in probability_thresholds}:
        probability_thresholds = sorted(set(probability_thresholds + [primary_threshold]))
    primary_policy_key = f"dual_prob_ge_{format_threshold_label(primary_threshold)}_ce_gt_0"

    holdout_panel = dataset["panel"][dataset["panel"]["split"] == "test"].copy()
    validation_panel = dataset["panel"][dataset["panel"]["split"] == "validation"].copy()
    stage1_base_feature_columns = stage1_feature_columns(dataset["panel"])
    stage1_models, universe_metrics = fit_stage1_models_by_universe(
        train_panel,
        config,
        "realized_hard_timely_liquidity_by_horizon",
        stage1_base_feature_columns,
    )
    stage1_scored_validation, _ = score_stage1_holdout_panel(
        validation_panel,
        stage1_models,
        horizon,
        config,
        dataset["company_master"],
    )
    stage1_recalibrators = fit_probability_recalibrators(
        stage1_scored_validation,
        "pred_hard_timely_liquidity_by_horizon",
        "realized_hard_timely_liquidity_by_horizon",
    )
    stage1_scored_validation = apply_probability_recalibrators(
        stage1_scored_validation,
        stage1_recalibrators,
        "pred_hard_timely_liquidity_by_horizon",
    )
    stage1_scored_holdout, stage1_point_matrix = score_stage1_holdout_panel(
        holdout_panel,
        stage1_models,
        horizon,
        config,
        dataset["company_master"],
    )
    stage1_scored_holdout = apply_probability_recalibrators(
        stage1_scored_holdout,
        stage1_recalibrators,
        "pred_hard_timely_liquidity_by_horizon",
    )
    evaluation_metrics_main, stage1_calibration, stage1_calibration_high_confidence, stage1_stress_mask = build_evaluation_metrics_main(
        stage1_scored_holdout,
        config,
    )
    stage1_calibration_stress = calibration_by_decile(
        stage1_scored_holdout.loc[stage1_stress_mask].copy(),
        prediction_col="pred_hard_timely_liquidity_by_horizon",
        realized_col="realized_hard_timely_liquidity_by_horizon",
    )
    evaluation_metrics_by_universe = build_evaluation_metrics_by_universe(stage1_scored_holdout, config)
    sector_stage_metrics = build_sector_stage_metrics(stage1_scored_holdout, config)
    stage1_metrics = evaluation_metrics_main.copy()
    stage1_metrics["target_name"] = HARD_TIMELY_LIQUIDITY_TARGET
    stage2_classes, stage2_train_support = select_stage2_route_classes(train_panel, config)
    stage2_model = build_stage2_probability_tables(train_panel, stage2_classes, config)
    stage2_route_support = build_stage2_route_support(dataset["panel"], stage2_classes)
    stage2_route_support = pd.concat([stage2_route_support, stage2_train_support.assign(universe="all", companies=np.nan)], ignore_index=True, sort=False)
    stage2_route_probs_holdout = predict_stage2_route_probs(stage1_scored_holdout, stage2_model)
    stage1_scored_holdout = stage1_scored_holdout.merge(
        stage2_route_probs_holdout.drop(columns=["universe", "sector_bucket", "stage_bucket"]),
        on=["company_id", "quarter_idx"],
        how="left",
    )
    stage1_scored_holdout["stage2_route"] = stage1_scored_holdout["company_exit_route"].map(
        lambda value: map_stage2_route_label(value, stage2_classes)
    )
    stage2_route_metrics = build_stage2_route_metrics(stage1_scored_holdout, stage2_classes)
    universe_support = build_universe_support(dataset["panel"])
    feature_coverage_by_block = build_feature_coverage_by_block(dataset["panel"], dataset["company_master"], dataset["patent_matches"])
    sponsor_fund_join_audit = build_sponsor_fund_join_audit(
        dataset["round_events"],
        buyout_market_panel=dataset.get("buyout_market_panel"),
    )
    lp_demand_join_audit = build_lp_demand_join_audit()
    patent_sector_model_comparison = build_patent_sector_model_comparison(dataset, config, stage1_base_feature_columns)
    patent_crosswalk_confidence = build_patent_crosswalk_confidence(dataset["patent_match_audit"])
    decision_backtest_screening, decision_backtest_economic, policy_activation_summary = build_redesigned_policy_backtests(
        validation_panel,
        holdout_panel,
        stage1_models,
        stage2_model,
        config,
        stage1_recalibrators=stage1_recalibrators,
    )
    promotion_gate_v2 = build_promotion_gate_v2(
        evaluation_metrics_main,
        dataset["label_confidence_audit"],
        stage2_route_support,
        policy_activation_summary,
        config,
    )
    target_registry = build_target_registry()
    target_candidate_results: dict[str, dict[str, pd.DataFrame | str]] = {}
    target_definition_frames: list[pd.DataFrame] = []
    target_prevalence_frames: list[pd.DataFrame] = []
    target_route_support_frames: list[pd.DataFrame] = []
    target_source_mix_frames: list[pd.DataFrame] = []
    target_label_audit_frames: list[pd.DataFrame] = []
    target_time_distribution_frames: list[pd.DataFrame] = []
    target_calibration_summary_frames: list[pd.DataFrame] = []
    target_evaluation_frames: list[pd.DataFrame] = []
    target_decision_backtest_frames: list[pd.DataFrame] = []
    buyout_sponsor_fund_active = bool(
        dataset.get("sponsor_fund_feature_coverage") is not None
        and not dataset["sponsor_fund_feature_coverage"].empty
        and (
            dataset["sponsor_fund_feature_coverage"]["universe"].astype(str).eq("buyout_pe")
            & pd.to_numeric(dataset["sponsor_fund_feature_coverage"]["coverage_share"], errors="coerce").fillna(0.0).gt(0.0)
        ).any()
    )
    for spec_row in target_registry.to_dict(orient="records"):
        candidate_panel, target_col, realized_prefix = build_target_candidate_panel(dataset["panel"], spec_row)
        base_feature_columns = target_exploration_feature_columns(candidate_panel, include_sponsor_fund=False)
        result = evaluate_target_candidate(
            candidate_panel,
            spec_row,
            target_col,
            realized_prefix,
            dataset["company_master"],
            config,
            feature_columns_override=base_feature_columns,
            feature_backbone=TARGET_BASE_FEATURE_BACKBONE,
        )
        target_key = str(spec_row["target_key"])
        target_candidate_results[target_key] = result
        target_definition_frames.append(result["definition"])
        target_prevalence_frames.append(result["prevalence"])
        target_route_support_frames.append(result["route_support"])
        target_source_mix_frames.append(result["source_mix"])
        target_label_audit_frames.append(result["label_audit"])
        target_time_distribution_frames.append(result["time_distribution"])
        if isinstance(result["calibration_summary"], pd.DataFrame) and not result["calibration_summary"].empty:
            target_calibration_summary_frames.append(result["calibration_summary"])
        if isinstance(result["evaluation_metrics"], pd.DataFrame) and not result["evaluation_metrics"].empty:
            target_evaluation_frames.append(result["evaluation_metrics"])
        if isinstance(result["decision_backtest"], pd.DataFrame) and not result["decision_backtest"].empty:
            target_decision_backtest_frames.append(result["decision_backtest"])
        if (
            buyout_sponsor_fund_active
            and str(spec_row["universe"]) == "buyout_pe"
            and int(spec_row["data_supported"]) == 1
        ):
            sponsor_feature_columns = target_exploration_feature_columns(candidate_panel, include_sponsor_fund=True)
            if sponsor_feature_columns != base_feature_columns:
                sponsor_result = evaluate_target_candidate(
                    candidate_panel,
                    spec_row,
                    target_col,
                    realized_prefix,
                    dataset["company_master"],
                    config,
                    feature_columns_override=sponsor_feature_columns,
                    feature_backbone=TARGET_SPONSOR_FUND_FEATURE_BACKBONE,
                )
                if isinstance(sponsor_result["calibration_summary"], pd.DataFrame) and not sponsor_result["calibration_summary"].empty:
                    target_calibration_summary_frames.append(sponsor_result["calibration_summary"])
                if isinstance(sponsor_result["evaluation_metrics"], pd.DataFrame) and not sponsor_result["evaluation_metrics"].empty:
                    target_evaluation_frames.append(sponsor_result["evaluation_metrics"])
                if isinstance(sponsor_result["decision_backtest"], pd.DataFrame) and not sponsor_result["decision_backtest"].empty:
                    target_decision_backtest_frames.append(sponsor_result["decision_backtest"])
    target_definitions = pd.concat(target_definition_frames, ignore_index=True) if target_definition_frames else pd.DataFrame()
    target_prevalence_by_split = pd.concat(
        [frame for frame in target_prevalence_frames if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in target_prevalence_frames) else pd.DataFrame()
    target_route_support_by_split = pd.concat(target_route_support_frames, ignore_index=True) if target_route_support_frames else pd.DataFrame()
    target_source_mix = pd.concat(target_source_mix_frames, ignore_index=True) if target_source_mix_frames else pd.DataFrame()
    target_label_confidence_audit = pd.concat(
        [frame for frame in target_label_audit_frames if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in target_label_audit_frames) else pd.DataFrame()
    target_time_distribution = pd.concat(
        [frame for frame in target_time_distribution_frames if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in target_time_distribution_frames) else pd.DataFrame()
    target_calibration_summary = pd.concat(target_calibration_summary_frames, ignore_index=True) if target_calibration_summary_frames else pd.DataFrame()
    evaluation_metrics_targets = pd.concat(target_evaluation_frames, ignore_index=True) if target_evaluation_frames else pd.DataFrame()
    decision_backtest_targets = pd.concat(target_decision_backtest_frames, ignore_index=True) if target_decision_backtest_frames else pd.DataFrame()
    target_source_summary = build_target_source_summary(target_source_mix)
    selected_target_feature_backbones = select_target_feature_backbones(
        target_registry,
        evaluation_metrics_targets,
        decision_backtest_targets,
        config,
    )
    target_selection_gates = build_target_selection_gates(
        target_registry,
        selected_target_feature_backbones,
        evaluation_metrics_targets,
        decision_backtest_targets,
        target_route_support_by_split,
        target_source_summary,
        config,
    )
    target_leaderboard_validation = build_target_leaderboard_validation(target_selection_gates)
    target_confirmation_test = build_target_confirmation_test(
        target_leaderboard_validation,
        evaluation_metrics_targets,
        decision_backtest_targets,
    )
    decision_usefulness_by_target = build_decision_usefulness_by_target(
        target_leaderboard_validation,
        target_confirmation_test,
        target_prevalence_by_split,
        config,
    )
    buyout_target_with_without_sponsor_fund = build_buyout_target_with_without_sponsor_fund(
        evaluation_metrics_targets,
        decision_backtest_targets,
    )
    target_recommendation_table = build_target_recommendation_table_v2(
        target_leaderboard_validation,
        target_confirmation_test,
        config,
    )
    universe_claim_matrix = build_universe_claim_matrix(
        target_recommendation_table,
        target_selection_gates,
        target_confirmation_test,
    )
    scored_holdout, _ = probability_path_summary_vectorized(
        holdout_panel,
        fitted,
        horizon,
        config,
        "baseline",
    )
    scored_holdout = scored_holdout.merge(
        holdout_panel[
            [
                "company_id",
                "quarter_idx",
                "route_label",
                "company_name",
                "realized_exit_by_horizon",
                "company_exit_route",
                "company_exit_value_usd",
                "company_exit_confidence_tier",
                "company_exit_route_source",
                "exit_quarter_idx",
                "log_last_round_usd",
            ]
        ],
        on=["company_id", "quarter_idx"],
        how="left",
    )
    scored_holdout = scored_holdout.merge(
        realized_exit_paths(holdout_panel, horizon),
        on=["company_id", "quarter_idx"],
        how="left",
    )
    scored_holdout = scored_holdout.merge(
        dataset["chosen_exits_main"][["company_id", "confidence_tier", "route_source"]].rename(
            columns={
                "confidence_tier": "main_exit_confidence_tier",
                "route_source": "main_exit_route_source",
            }
        ),
        on="company_id",
        how="left",
    )
    scored_holdout = scored_holdout.merge(
        dataset["company_master"][["company_id", "match_confidence", "company_source"]].rename(
            columns={"match_confidence": "entity_match_confidence"}
        ),
        on="company_id",
        how="left",
    )
    scored_holdout["pred_direct_exit_by_horizon"] = (
        pd.to_numeric(scored_holdout["cum_ipo"], errors="coerce").fillna(0.0)
        + pd.to_numeric(scored_holdout["cum_mna"], errors="coerce").fillna(0.0)
        + pd.to_numeric(scored_holdout["cum_sponsor_sale"], errors="coerce").fillna(0.0)
    )
    scored_holdout["pred_pooled_strategic_exit"] = (
        pd.to_numeric(scored_holdout["cum_ipo"], errors="coerce").fillna(0.0)
        + pd.to_numeric(scored_holdout["cum_mna"], errors="coerce").fillna(0.0)
    )
    scored_holdout["realized_direct_exit_by_horizon"] = scored_holdout["realized_exit_by_horizon"]
    scored_holdout["realized_pooled_strategic_exit"] = (
        scored_holdout["company_exit_route"].astype(str).isin(["ipo", "mna"])
        & pd.to_numeric(scored_holdout["realized_exit_by_horizon"], errors="coerce").fillna(0).astype(int).eq(1)
    ).astype(int)
    exit_confusion_long, exit_confusion_summary = build_binary_confusion_exports(
        scored_holdout,
        prediction_col="pred_exit_by_horizon",
        actual_col="realized_exit_by_horizon",
        thresholds=probability_thresholds,
        data_mode=data_mode,
        evaluation_view="any_exit_aggregate",
        target_label="realized_exit_by_8q",
        prediction_label="predicted_prob_exit_by_8q",
    )
    calibration = calibration_by_decile(scored_holdout)
    high_confidence_mask = target_positive_subset_mask(
        scored_holdout,
        "realized_exit_by_horizon",
        "exit_label_confidence_high",
    )
    high_confidence_subset_definition = "exit_label_confidence_high_or_negative"
    calibration_high_confidence = calibration.copy()
    if high_confidence_mask.sum() > 0:
        calibration_high_confidence = calibration_by_decile(scored_holdout.loc[high_confidence_mask].copy())
    entity_match_high_confidence_mask = target_positive_subset_mask(
        scored_holdout,
        "realized_exit_by_horizon",
        "entity_match_confidence_high",
    )
    stress_start_idx = quarter_idx_from_label(str(config.get("stress_slice_start_quarter", "2020Q1")))
    stress_end_idx = quarter_idx_from_label(str(config.get("stress_slice_end_quarter", "2020Q4")))
    stress_mask = (
        (pd.to_numeric(scored_holdout["quarter_idx"], errors="coerce") >= stress_start_idx)
        & (pd.to_numeric(scored_holdout["quarter_idx"], errors="coerce") <= stress_end_idx)
    )
    calibration_stress_slice = calibration_by_decile(scored_holdout.loc[stress_mask].copy())
    calibration_summary = pd.concat(
        [
            summarize_calibration(calibration, "main"),
            summarize_calibration(calibration_high_confidence, "high_confidence_subset"),
            summarize_calibration(calibration_stress_slice, "stress_slice"),
        ],
        ignore_index=True,
    )
    route_multiple_params = calibrate_route_multiples(dataset["round_events"], dataset["chosen_exits"])
    evaluation_views = {
        "any_exit_aggregate": scored_holdout,
        "competing_risk_aggregate": scored_holdout,
        "high_confidence_subset": scored_holdout.loc[high_confidence_mask].copy(),
        "high_confidence_entity_match": scored_holdout.loc[entity_match_high_confidence_mask].copy(),
        "stress_regime_subperiod": scored_holdout.loc[stress_mask].copy(),
    }
    evaluation_metrics = [
        summarize_evaluation_view(scored_holdout, "any_exit_aggregate", "pred_exit_by_horizon", "realized_exit_by_horizon", horizon),
        summarize_evaluation_view(scored_holdout, "competing_risk_aggregate", "pred_direct_exit_by_horizon", "realized_direct_exit_by_horizon", horizon),
            summarize_evaluation_view(
            evaluation_views["high_confidence_subset"],
            "high_confidence_subset",
            "pred_exit_by_horizon",
            "realized_exit_by_horizon",
            horizon,
        ),
        summarize_evaluation_view(
            evaluation_views["high_confidence_entity_match"],
            "high_confidence_entity_match",
            "pred_exit_by_horizon",
            "realized_exit_by_horizon",
            horizon,
        ),
        summarize_evaluation_view(
            evaluation_views["stress_regime_subperiod"],
            "stress_regime_subperiod",
            "pred_exit_by_horizon",
            "realized_exit_by_horizon",
            horizon,
        ),
    ]
    if route_pooling_used:
        evaluation_metrics.append(
            summarize_evaluation_view(
                scored_holdout,
                "pooled_strategic_fallback",
                "pred_pooled_strategic_exit",
                "realized_pooled_strategic_exit",
                horizon,
            )
        )
    evaluation_metrics = pd.concat(evaluation_metrics, ignore_index=True)
    evaluation_view_definitions = build_evaluation_view_definitions(data_mode, route_pooling_used)
    route_competing_risks_summary = build_route_competing_risks_summary(
        {
            "any_exit_aggregate": scored_holdout,
            "high_confidence_subset": evaluation_views["high_confidence_subset"],
            "high_confidence_entity_match": evaluation_views["high_confidence_entity_match"],
            "stress_regime_subperiod": evaluation_views["stress_regime_subperiod"],
        },
        route_pooling_used,
    )

    decision_quarter = int(holdout_panel.loc[holdout_panel["route_label"] == "no_exit", "quarter_idx"].max())
    decision_panel_raw = holdout_panel[
        (holdout_panel["quarter_idx"] == decision_quarter)
        & (holdout_panel["route_label"] == "no_exit")
    ].copy()
    decision_summary, decision_points = probability_path_summary_vectorized(
        decision_panel_raw,
        fitted,
        horizon,
        config,
        "baseline",
    )
    decision_panel = decision_panel_raw.merge(decision_summary, on=["company_id", "quarter_idx"], how="left")
    decision_panel = decision_panel.merge(
        realized_exit_paths(decision_panel_raw, horizon),
        on=["company_id", "quarter_idx"],
        how="left",
    )
    decision_panel = decision_panel.merge(
        simulate_panel_value_summary(
            decision_panel_raw,
            decision_points,
            route_multiple_params,
            config,
            "baseline",
            n_paths=int(config.get("decision_eval_paths", 64)),
        ),
        on=["company_id", "quarter_idx"],
        how="left",
    )
    decision_panel = decision_panel.merge(
        build_realized_value_proxy(decision_panel_raw, route_multiple_params, config),
        on=["company_id", "quarter_idx", "realized_exit_by_horizon", "company_exit_route"],
        how="left",
    )
    decision_backtest = build_decision_backtest(decision_panel, config)
    decision_policy_confusion_long, decision_policy_confusion_summary = build_decision_policy_confusion_exports(
        decision_panel,
        config,
    )
    route_multiclass, route_multiclass_status = build_route_multiclass_diagnostics(
        scored_holdout,
        data_mode,
        route_pooling_used,
    )
    feature_registry = build_feature_registry(dataset["panel"], data_mode)
    sector_bucket_mapping = build_sector_bucket_mapping(dataset["company_master"])
    sector_stage_support = build_sector_stage_support(dataset["panel"], config)
    feature_importance_target_definition = build_feature_importance_target_definition(route_pooling_used)
    route_support_for_importance = build_route_support_for_importance(dataset["route_support_by_split"], route_pooling_used)
    feature_search_skipped = bool(config.get("skip_feature_search", False))
    if feature_search_skipped:
        skipped_exports = empty_feature_search_exports()
        dataset["feature_analysis_panels"] = skipped_exports["analysis_panels"]
        ablation_exports = skipped_exports["ablation_exports"]
        feature_importance_permutation = skipped_exports["feature_importance_permutation"]
        feature_group_importance_permutation = skipped_exports["feature_group_importance_permutation"]
        search_group_columns = skipped_exports["search_group_columns"]
        full_search_feature_columns = skipped_exports["full_search_feature_columns"]
        feature_combo_validation_leaderboard = skipped_exports["feature_combo_validation_leaderboard"]
        feature_combo_test_leaderboard = skipped_exports["feature_combo_test_leaderboard"]
        feature_combo_pareto_frontier = skipped_exports["feature_combo_pareto_frontier"]
        chosen_feature_combo_summary = skipped_exports["chosen_feature_combo_summary"]
        combo_cache = skipped_exports["combo_cache"]
        sector_feature_importance = skipped_exports["sector_feature_importance"]
        patent_value_by_sector = skipped_exports["patent_value_by_sector"]
        sector_combo_challengers = skipped_exports["sector_combo_challengers"]
        interaction_screen_results = skipped_exports["interaction_screen_results"]
        interaction_keep_drop_summary = skipped_exports["interaction_keep_drop_summary"]
    else:
        dataset["feature_analysis_panels"] = build_feature_analysis_panels(dataset, config)
        ablation_exports, ablation_results_by_group, search_group_columns, full_search_feature_columns = run_feature_group_ablation(
            dataset,
            config,
        )
        full_search_result = ablation_results_by_group["full_active"]
        feature_importance_permutation, feature_group_importance_permutation = build_permutation_importance_exports(
            dataset,
            config,
            full_search_result,
        )
        combo_precomputed_cache: dict[str, dict[str, object]] = {}
        active_optional_groups = [group for group in OPTIONAL_FEATURE_GROUPS if bool(search_group_columns.get(group))]
        full_search_groups = [*BASELINE_FEATURE_GROUPS, *active_optional_groups]
        combo_precomputed_cache[combo_key_from_groups(full_search_groups)] = {
            **full_search_result,
            "feature_columns": full_search_feature_columns,
            "feature_groups": full_search_groups,
        }
        for group_name in active_optional_groups:
            if group_name not in ablation_results_by_group:
                continue
            reduced_groups = [group for group in full_search_groups if group != group_name]
            combo_precomputed_cache[combo_key_from_groups(reduced_groups)] = {
                **ablation_results_by_group[group_name],
                "feature_columns": [column for column in full_search_feature_columns if column not in search_group_columns[group_name]],
                "feature_groups": reduced_groups,
            }
        (
            feature_combo_validation_leaderboard,
            feature_combo_test_leaderboard,
            feature_combo_pareto_frontier,
            chosen_feature_combo_summary,
            combo_cache,
        ) = run_feature_combo_search(dataset, config, search_group_columns, combo_precomputed_cache)
        sector_feature_importance = build_sector_feature_importance(
            dataset,
            config,
            full_search_result,
            search_group_columns,
            sector_stage_support,
        )
        patent_value_by_sector = build_patent_value_by_sector(sector_feature_importance)
        sector_combo_challengers = build_sector_combo_challengers(combo_cache, sector_stage_support)
        interaction_screen_results, interaction_keep_drop_summary = run_interaction_screen(
            dataset,
            config,
            chosen_feature_combo_summary,
            combo_cache,
        )

    display_selection = choose_display_view(dataset["panel"], fitted, route_multiple_params, config)
    stylized = display_selection["selected_row"]
    incidence_map, npv_map, summary_metrics = build_display_outputs(
        display_selection,
        fitted,
        route_multiple_params,
        config,
    )
    summary_metrics["display_mode"] = str(display_selection["display_mode"])
    summary_metrics["display_label"] = str(display_selection["display_label"])
    summary_metrics["selected_min_entry_year"] = int(dataset["selected_min_entry_year"])
    base_future_probs = predict_route_probs(
        build_future_states(
            stylized,
            horizon,
            config,
            "baseline",
            feature_columns=fitted.get("model_state", {}).get("feature_columns"),
        ).assign(route_label="no_exit"),
        fitted,
    )
    higher_price_config = config.copy()
    higher_price_config["purchase_price_fraction_of_v0"] = float(config["purchase_price_fraction_of_v0"]) * 1.20
    higher_price_npv = simulate_npv(stylized, base_future_probs, route_multiple_params, higher_price_config, "baseline")
    summary_metrics.loc[summary_metrics["scenario"] == "baseline", "higher_purchase_price_mean_npv"] = float(
        higher_price_npv["npv"].mean()
    )
    stage2_incidence_map: dict[str, pd.DataFrame] = {}
    for scenario_name in ("baseline", "exit_freeze"):
        stage1_future_states = build_future_states(
            stylized,
            horizon,
            config,
            scenario_name,
            feature_columns=stage1_base_feature_columns,
        )
        for static_column in ["universe", "sector_bucket", "stage_bucket", "company_name"]:
            if static_column in stylized.index:
                stage1_future_states[static_column] = stylized[static_column]
        _, stage1_future_points = binary_probability_path_summary_vectorized(
            stage1_future_states.assign(realized_hard_timely_liquidity_by_horizon=0),
            stage1_models.get(str(stylized.get("universe", "venture_growth")), stage1_models["_overall"]),
            horizon,
            config,
            scenario_name=scenario_name,
            prediction_label="pred_hard_timely_liquidity_by_horizon",
        )
        incidence = build_stage2_cumulative_incidence(stage1_future_states, stage1_future_points[:1], stage2_model)
        if "cum_ipo" not in incidence.columns:
            incidence["cum_ipo"] = 0.0
        if "cum_mna" not in incidence.columns:
            incidence["cum_mna"] = 0.0
        if "cum_sponsor_sale" not in incidence.columns:
            incidence["cum_sponsor_sale"] = 0.0
        if "cum_pooled_strategic" not in incidence.columns:
            incidence["cum_pooled_strategic"] = pd.to_numeric(incidence.get("cum_ipo"), errors="coerce").fillna(0.0) + pd.to_numeric(
                incidence.get("cum_mna"), errors="coerce"
            ).fillna(0.0)
        stage2_incidence_map[scenario_name] = incidence
    if feature_search_skipped:
        top_combo_confusion_summary = pd.DataFrame(columns=["combo_key", "combo_rank"])
        top_combo_decision_backtest = pd.DataFrame(columns=["combo_key", "combo_rank"])
        top_combo_summary_metrics = pd.DataFrame(columns=["combo_key", "combo_rank"])
    else:
        top_combo_keys = feature_combo_validation_leaderboard["combo_key"].head(3).astype(str).tolist()
        (
            top_combo_confusion_summary,
            top_combo_decision_backtest,
            top_combo_summary_metrics,
        ) = build_top_combo_economic_diagnostics(
            dataset,
            config,
            top_combo_keys,
            combo_cache,
            display_selection,
        )
    placeholder_status = build_feature_placeholder_status()
    run_metadata = build_run_metadata(config, dataset, fitted, stylized)
    run_metadata["selected_train_end_quarter"] = str(config["train_end_quarter"])
    run_metadata["selected_validation_end_quarter"] = str(config["validation_end_quarter"])
    run_metadata["selected_test_end_quarter"] = str(config["test_end_quarter"])
    run_metadata["panel_end_quarter"] = str(config.get("panel_end_quarter", config["test_end_quarter"]))
    run_metadata["high_confidence_subset_definition"] = high_confidence_subset_definition
    run_metadata["used_route_pooling_fallback"] = int(route_pooling_used)
    run_metadata["display_mode"] = str(display_selection["display_mode"])
    run_metadata["display_label"] = str(display_selection["display_label"])
    run_metadata["stress_slice_rows"] = int(stress_mask.sum())
    run_metadata["primary_confusion_threshold"] = primary_threshold
    run_metadata["primary_policy_key"] = primary_policy_key
    selected_screening_policy = decision_backtest_screening[
        decision_backtest_screening["selected_on_validation"].astype(int).eq(1)
    ].copy()
    selected_economic_policy = decision_backtest_economic[
        decision_backtest_economic["selected_on_validation"].astype(int).eq(1)
    ].copy()
    selected_screening_policy_key = (
        str(selected_screening_policy["policy_key"].iloc[0]) if not selected_screening_policy.empty else ""
    )
    selected_economic_policy_key = (
        str(selected_economic_policy["policy_key"].iloc[0]) if not selected_economic_policy.empty else ""
    )
    run_metadata["appendix_primary_policy_key"] = primary_policy_key
    run_metadata["selected_screening_policy_key"] = selected_screening_policy_key
    run_metadata["selected_economic_policy_key"] = selected_economic_policy_key
    run_metadata["certainty_equivalent_threshold"] = float(config.get("certainty_equivalent_threshold", 0.0))
    run_metadata["skip_feature_search"] = int(feature_search_skipped)
    run_metadata["search_full_feature_columns"] = "|".join(full_search_feature_columns)
    run_metadata["chosen_feature_combo_key"] = (
        str(chosen_feature_combo_summary["combo_key"].iloc[0]) if not chosen_feature_combo_summary.empty else ""
    )
    run_metadata["chosen_feature_groups"] = (
        str(chosen_feature_combo_summary["feature_groups"].iloc[0]) if not chosen_feature_combo_summary.empty else ""
    )
    promotion_gate = build_promotion_gate(
        summary_metrics,
        calibration,
        calibration_high_confidence,
        dataset["route_mapping_comparison"],
    )
    for column in promotion_gate.columns:
        run_metadata[column] = promotion_gate.iloc[0][column]
    for column in promotion_gate_v2.columns:
        run_metadata[f"v2_{column}"] = promotion_gate_v2.iloc[0][column]
    run_metadata["headline_target_name"] = HARD_TIMELY_LIQUIDITY_TARGET
    run_metadata["stage2_route_class_set"] = "|".join(stage2_classes)
    recommended_targets = selected_universe_recommendations(target_recommendation_table)
    venture_recommended = recommended_targets.loc[
        recommended_targets["universe"].astype(str).eq("venture_growth")
    ].head(1)
    buyout_recommended = recommended_targets.loc[
        recommended_targets["universe"].astype(str).eq("buyout_pe")
    ].head(1)
    run_metadata["recommended_target_venture_growth"] = (
        str(venture_recommended["target_name"].iloc[0]) if not venture_recommended.empty else ""
    )
    run_metadata["recommended_target_venture_growth_feature_backbone"] = (
        str(venture_recommended["selected_feature_backbone"].iloc[0]) if not venture_recommended.empty else ""
    )
    run_metadata["recommended_target_buyout_pe"] = (
        str(buyout_recommended["target_name"].iloc[0]) if not buyout_recommended.empty else ""
    )
    run_metadata["recommended_target_buyout_pe_feature_backbone"] = (
        str(buyout_recommended["selected_feature_backbone"].iloc[0]) if not buyout_recommended.empty else ""
    )
    run_metadata["buyout_target_reporting_status"] = (
        str(buyout_recommended["chapter_reporting_status"].iloc[0]) if not buyout_recommended.empty else ""
    )
    run_metadata["venture_target_reporting_status"] = (
        str(venture_recommended["chapter_reporting_status"].iloc[0]) if not venture_recommended.empty else ""
    )
    run_metadata["target_selection_protocol"] = "validation_only_with_locked_test_confirmation"
    target_recommendation_output = target_recommendation_table[
        [column for column in target_recommendation_table.columns if not str(column).startswith("_")]
    ].copy()
    source_mix_by_target = target_source_summary.copy()
    source_mix_by_universe = build_source_mix_by_universe(target_source_summary)
    directness_by_target = target_source_summary[
        [
            "target_key",
            "target_name",
            "universe",
            "positive_event_total",
            "direct_dated_events",
            "inferred_transition_events",
            "synthetic_dated_events",
            "sensitivity_proxy_events",
            "label_confidence_events",
            "direct_dated_share",
            "inferred_transition_share",
            "synthetic_dated_share",
            "label_confidence_share",
        ]
    ].copy()
    directness_by_universe = build_directness_by_universe(target_source_summary)
    confidence_mask_definitions = build_confidence_mask_definitions()
    target_high_confidence_exit_label_only = selected_target_view_frame(
        target_leaderboard_validation,
        evaluation_metrics_targets,
        "high_confidence_exit_label_only",
    )
    target_high_confidence_entity_match_only = selected_target_view_frame(
        target_leaderboard_validation,
        evaluation_metrics_targets,
        "high_confidence_entity_match_only",
    )
    target_high_confidence_overlap = selected_target_view_frame(
        target_leaderboard_validation,
        evaluation_metrics_targets,
        "high_confidence_overlap",
    )
    buyout_target_registry = target_registry.loc[target_registry["universe"].astype(str).eq("buyout_pe")].copy()
    buyout_target_recommendation_validation = target_leaderboard_validation.loc[
        target_leaderboard_validation["universe"].astype(str).eq("buyout_pe")
    ].copy()
    buyout_target_confirmation_test = target_confirmation_test.loc[
        target_confirmation_test["universe"].astype(str).eq("buyout_pe")
    ].copy()
    buyout_missing_field_manifest = build_buyout_missing_field_manifest(
        dataset["buyout_realization_field_audit"],
        dataset["deal_fund_link_audit"],
        sponsor_fund_join_audit,
        lp_demand_join_audit,
    )

    primary_exit_confusion_row = exit_confusion_summary.loc[
        exit_confusion_summary["threshold"].round(4).eq(round(primary_threshold, 4))
    ].iloc[0]
    primary_policy_confusion_row = decision_policy_confusion_summary.loc[
        decision_policy_confusion_summary["policy_key"].astype(str).eq(primary_policy_key)
        & decision_policy_confusion_summary["target_label"].astype(str).eq("realized_npv_proxy_positive")
    ].iloc[0]
    route_multiclass_classes = route_multiclass_status["class_set"].iloc[0].split("|") if not route_multiclass_status.empty else []

    generated_files = [
        plot_calibration_deciles(stage1_calibration, stage1_calibration_stress, output_dir),
        plot_stage2_cumulative_incidence(stage2_incidence_map, output_dir, str(display_selection["display_label"])),
        plot_npv_distribution(npv_map, output_dir, str(display_selection["display_label"])),
        plot_route_waterfall(incidence_map, output_dir),
        plot_policy_backtest(decision_backtest_screening, decision_backtest_economic, output_dir),
        plot_binary_confusion_heatmap(
            primary_exit_confusion_row,
            f"{mode_suffix.title()} exit-by-8Q confusion at {primary_threshold:.2f}",
            output_dir / f"confusion_matrix_exit_by_8q_{mode_suffix}_heatmap.png",
        ),
        plot_multiclass_confusion_heatmap(
            route_multiclass,
            route_multiclass_classes,
            f"{mode_suffix.title()} route confusion diagnostic",
            output_dir / f"confusion_matrix_route_multiclass_{mode_suffix}_heatmap.png",
        ),
    ]
    if not feature_search_skipped:
        generated_files.extend(
            [
                plot_feature_importance_groups(feature_group_importance_permutation, output_dir),
                plot_feature_group_ablation(ablation_exports["test"], output_dir),
                plot_feature_combo_heatmap(feature_combo_validation_leaderboard, output_dir),
                plot_sector_feature_importance(sector_feature_importance, output_dir),
                plot_patent_value_by_sector(patent_value_by_sector, output_dir),
            ]
        )
    if data_mode == "actual":
        generated_files.append(
            plot_binary_confusion_heatmap(
                primary_policy_confusion_row,
                f"Actual dual-rule NPV confusion for {primary_policy_key}",
                output_dir / "confusion_matrix_dual_rule_actual_npv_heatmap.png",
            )
        )

    calibration.to_csv(output_dir / "summary_calibration.csv", index=False)
    calibration_high_confidence.to_csv(output_dir / "summary_calibration_high_confidence.csv", index=False)
    calibration_summary.to_csv(output_dir / "summary_calibration_overview.csv", index=False)
    calibration_stress_slice.to_csv(output_dir / "calibration_stress_slice.csv", index=False)
    stage1_metrics.to_csv(output_dir / "stage1_metrics.csv", index=False)
    stage2_route_metrics.to_csv(output_dir / "stage2_route_metrics.csv", index=False)
    stage2_route_support.to_csv(output_dir / "stage2_route_support.csv", index=False)
    target_registry.to_csv(output_dir / "target_registry.csv", index=False)
    target_definitions.to_csv(output_dir / "target_definitions.csv", index=False)
    target_prevalence_by_split.to_csv(output_dir / "target_prevalence_by_split_all.csv", index=False)
    target_route_support_by_split.to_csv(output_dir / "route_support_by_split_targets.csv", index=False)
    target_source_mix.to_csv(output_dir / "target_source_mix_all.csv", index=False)
    target_label_confidence_audit.to_csv(output_dir / "label_confidence_audit_targets.csv", index=False)
    target_time_distribution.to_csv(output_dir / "target_time_distribution_all.csv", index=False)
    evaluation_metrics_targets.to_csv(output_dir / "evaluation_metrics_targets.csv", index=False)
    target_calibration_summary.to_csv(output_dir / "calibration_targets_summary.csv", index=False)
    decision_backtest_targets.to_csv(output_dir / "decision_backtest_targets.csv", index=False)
    target_leaderboard_validation.to_csv(output_dir / "target_leaderboard_validation.csv", index=False)
    target_confirmation_test.to_csv(output_dir / "target_confirmation_test.csv", index=False)
    target_selection_gates.to_csv(output_dir / "target_selection_gates.csv", index=False)
    selected_target_feature_backbones.to_csv(output_dir / "selected_target_feature_backbones.csv", index=False)
    source_mix_by_target.to_csv(output_dir / "source_mix_by_target.csv", index=False)
    source_mix_by_universe.to_csv(output_dir / "source_mix_by_universe.csv", index=False)
    directness_by_target.to_csv(output_dir / "directness_by_target.csv", index=False)
    directness_by_universe.to_csv(output_dir / "directness_by_universe.csv", index=False)
    confidence_mask_definitions.to_csv(output_dir / "confidence_mask_definitions.csv", index=False)
    target_high_confidence_exit_label_only.to_csv(output_dir / "target_high_confidence_exit_label_only.csv", index=False)
    target_high_confidence_entity_match_only.to_csv(output_dir / "target_high_confidence_entity_match_only.csv", index=False)
    target_high_confidence_overlap.to_csv(output_dir / "target_high_confidence_overlap.csv", index=False)
    decision_usefulness_by_target.to_csv(output_dir / "decision_usefulness_by_target.csv", index=False)
    buyout_target_registry.to_csv(output_dir / "buyout_target_registry.csv", index=False)
    buyout_target_recommendation_validation.to_csv(output_dir / "buyout_target_recommendation_validation.csv", index=False)
    buyout_target_confirmation_test.to_csv(output_dir / "buyout_target_confirmation_test.csv", index=False)
    buyout_target_with_without_sponsor_fund.to_csv(output_dir / "buyout_target_with_without_sponsor_fund.csv", index=False)
    universe_claim_matrix.to_csv(output_dir / "universe_claim_matrix.csv", index=False)
    target_recommendation_output.to_csv(output_dir / "target_recommendation_table.csv", index=False)
    dataset["target_definition_main"].to_csv(output_dir / "target_definition_main.csv", index=False)
    dataset["target_definition_sensitivity"].to_csv(output_dir / "target_definition_sensitivity.csv", index=False)
    dataset["label_confidence_audit"].to_csv(output_dir / "label_confidence_audit.csv", index=False)
    universe_support.to_csv(output_dir / "universe_support.csv", index=False)
    evaluation_metrics_main.to_csv(output_dir / "evaluation_metrics_main.csv", index=False)
    evaluation_metrics_by_universe.to_csv(output_dir / "evaluation_metrics_by_universe.csv", index=False)
    evaluation_metrics_by_universe.to_csv(output_dir / "universe_metrics.csv", index=False)
    sector_stage_metrics.to_csv(output_dir / "sector_stage_metrics.csv", index=False)
    feature_coverage_by_block.to_csv(output_dir / "feature_coverage_by_block.csv", index=False)
    sponsor_fund_join_audit.to_csv(output_dir / "sponsor_fund_join_audit.csv", index=False)
    dataset["sponsor_fund_feature_coverage"].to_csv(output_dir / "sponsor_fund_feature_coverage.csv", index=False)
    lp_demand_join_audit.to_csv(output_dir / "lp_demand_join_audit.csv", index=False)
    dataset["buyout_realization_field_audit"].to_csv(output_dir / "buyout_realization_field_audit.csv", index=False)
    dataset["deal_fund_link_audit"].to_csv(output_dir / "deal_fund_link_audit.csv", index=False)
    buyout_missing_field_manifest.to_csv(output_dir / "buyout_missing_field_manifest.csv", index=False)
    patent_sector_model_comparison.to_csv(output_dir / "patent_sector_model_comparison.csv", index=False)
    patent_crosswalk_confidence.to_csv(output_dir / "patent_crosswalk_confidence.csv", index=False)
    decision_backtest_screening.to_csv(output_dir / "decision_backtest_screening.csv", index=False)
    decision_backtest_economic.to_csv(output_dir / "decision_backtest_economic.csv", index=False)
    policy_activation_summary.to_csv(output_dir / "policy_activation_summary.csv", index=False)
    promotion_gate_v2.to_csv(output_dir / "promotion_gate_v2.csv", index=False)
    evaluation_metrics.to_csv(output_dir / "evaluation_metrics.csv", index=False)
    decision_backtest.to_csv(output_dir / "decision_backtest.csv", index=False)
    route_competing_risks_summary.to_csv(output_dir / "route_competing_risks_summary.csv", index=False)
    feature_registry.to_csv(output_dir / "feature_registry.csv", index=False)
    feature_importance_permutation.to_csv(output_dir / "feature_importance_permutation.csv", index=False)
    feature_group_importance_permutation.to_csv(output_dir / "feature_group_importance_permutation.csv", index=False)
    ablation_exports["validation"].to_csv(output_dir / "feature_group_ablation_validation.csv", index=False)
    ablation_exports["test"].to_csv(output_dir / "feature_group_ablation_test.csv", index=False)
    ablation_exports["stress"].to_csv(output_dir / "feature_group_ablation_stress.csv", index=False)
    ablation_exports["high_confidence"].to_csv(output_dir / "feature_group_ablation_high_confidence.csv", index=False)
    feature_combo_validation_leaderboard.to_csv(output_dir / "feature_combo_validation_leaderboard.csv", index=False)
    feature_combo_test_leaderboard.to_csv(output_dir / "feature_combo_test_leaderboard.csv", index=False)
    feature_combo_pareto_frontier.to_csv(output_dir / "feature_combo_pareto_frontier.csv", index=False)
    chosen_feature_combo_summary.to_csv(output_dir / "chosen_feature_combo_summary.csv", index=False)
    sector_bucket_mapping.to_csv(output_dir / "sector_bucket_mapping.csv", index=False)
    sector_stage_support.to_csv(output_dir / "sector_stage_support.csv", index=False)
    sector_feature_importance.to_csv(output_dir / "sector_feature_importance.csv", index=False)
    patent_value_by_sector.to_csv(output_dir / "patent_value_by_sector.csv", index=False)
    sector_combo_challengers.to_csv(output_dir / "sector_combo_challengers.csv", index=False)
    interaction_screen_results.to_csv(output_dir / "interaction_screen_results.csv", index=False)
    interaction_keep_drop_summary.to_csv(output_dir / "interaction_keep_drop_summary.csv", index=False)
    feature_importance_target_definition.to_csv(output_dir / "feature_importance_target_definition.csv", index=False)
    route_support_for_importance.to_csv(output_dir / "route_support_for_importance.csv", index=False)
    top_combo_confusion_summary.to_csv(output_dir / "top_combo_confusion_summary.csv", index=False)
    top_combo_decision_backtest.to_csv(output_dir / "top_combo_decision_backtest.csv", index=False)
    top_combo_summary_metrics.to_csv(output_dir / "top_combo_summary_metrics.csv", index=False)
    for rank in range(1, 4):
        if "combo_rank" in top_combo_summary_metrics.columns:
            subset = top_combo_summary_metrics[top_combo_summary_metrics["combo_rank"].astype(int).eq(rank)].copy()
        else:
            subset = pd.DataFrame()
        subset.to_csv(output_dir / f"summary_metrics_best_combo_{rank}.csv", index=False)
    exit_confusion_long.to_csv(output_dir / "confusion_matrix_exit_by_8q.csv", index=False)
    exit_confusion_summary.to_csv(output_dir / "confusion_matrix_exit_by_8q_summary.csv", index=False)
    decision_policy_confusion_long.to_csv(output_dir / "confusion_matrix_decision_policy.csv", index=False)
    decision_policy_confusion_summary.to_csv(output_dir / "confusion_matrix_decision_policy_summary.csv", index=False)
    route_multiclass.to_csv(output_dir / "confusion_matrix_route_multiclass.csv", index=False)
    route_multiclass_status.to_csv(output_dir / "confusion_matrix_route_multiclass_status.csv", index=False)
    evaluation_view_definitions.to_csv(output_dir / "evaluation_view_definitions.csv", index=False)
    summary_metrics.to_csv(output_dir / "summary_metrics.csv", index=False)
    dataset["route_audit"].to_csv(output_dir / "route_audit.csv", index=False)
    dataset["route_audit_main"].to_csv(output_dir / "route_audit_main.csv", index=False)
    dataset["route_audit_sensitivity"].to_csv(output_dir / "route_audit_sensitivity.csv", index=False)
    dataset["route_confidence_summary"].to_csv(output_dir / "route_confidence_summary.csv", index=False)
    dataset["route_mapping_comparison"].to_csv(output_dir / "route_mapping_comparison.csv", index=False)
    dataset["coverage_by_year"].to_csv(output_dir / "coverage_by_year.csv", index=False)
    dataset["partition_summary"].to_csv(output_dir / "partition_summary.csv", index=False)
    dataset["route_support_by_split"].to_csv(output_dir / "route_support_by_split.csv", index=False)
    dataset["density_by_entry_year"].to_csv(output_dir / "density_by_entry_year.csv", index=False)
    dataset["window_selection_grid"].to_csv(output_dir / "window_selection_grid.csv", index=False)
    dataset["window_selection"].to_csv(output_dir / "model_window_selection.csv", index=False)
    dataset["window_selection_grid"].to_csv(output_dir / "model_window_selection_validation_only.csv", index=False)
    dataset["route_pooling_fallback_summary"].to_csv(output_dir / "route_pooling_fallback_summary.csv", index=False)
    dataset["patent_match_audit"].to_csv(output_dir / "patent_match_audit.csv", index=False)
    dataset["patent_feature_coverage"].to_csv(output_dir / "patent_feature_coverage.csv", index=False)
    dataset["patent_coverage_comparison"].to_csv(output_dir / "patent_coverage_comparison.csv", index=False)
    display_selection["audit"].to_csv(output_dir / "stylized_selection_audit.csv", index=False)
    placeholder_status.to_csv(output_dir / "feature_placeholder_status.csv", index=False)
    run_metadata.to_csv(output_dir / "run_metadata.csv", index=False)
    promotion_gate.to_csv(output_dir / "promotion_gate.csv", index=False)
    for spec_row in target_registry.to_dict(orient="records"):
        target_key = str(spec_row["target_key"])
        file_stub = target_key
        candidate_result = target_candidate_results[target_key]
        definition = candidate_result["definition"]
        definition.to_csv(output_dir / f"target_definition_{file_stub}.csv", index=False)
        write_target_definition_markdown(output_dir / f"target_definition_{file_stub}.md", definition)
        candidate_result["label_audit"].to_csv(output_dir / f"label_confidence_audit_{file_stub}.csv", index=False)
        candidate_result["route_support"].to_csv(output_dir / f"route_support_by_split_{file_stub}.csv", index=False)
        candidate_result["prevalence"].to_csv(output_dir / f"target_prevalence_by_split_{file_stub}.csv", index=False)
        candidate_result["source_mix"].to_csv(output_dir / f"target_source_mix_{file_stub}.csv", index=False)
        candidate_result["time_distribution"].to_csv(output_dir / f"target_time_distribution_{file_stub}.csv", index=False)
    write_markdown_table_report(
        output_dir / "evaluation_metrics_targets.md",
        "Evaluation Metrics Targets",
        ["Canonical full-test and high-confidence evaluation views for each candidate target."],
        evaluation_metrics_targets,
    )
    write_markdown_table_report(
        output_dir / "decision_backtest_targets.md",
        "Decision Backtest Targets",
        ["Validation selects the policy rule; test rows show the out-of-time policy comparison."],
        decision_backtest_targets,
    )
    write_target_recommendation_summary(
        output_dir / "target_recommendation_summary.md",
        target_recommendation_output,
    )
    write_chapter_target_doctrine(
        output_dir / "chapter_target_doctrine.md",
        target_recommendation_output,
    )
    write_universe_claim_matrix(
        output_dir / "universe_claim_matrix.md",
        universe_claim_matrix,
    )
    write_chapter_objective_definition_findings(
        output_dir / "chapter_objective_definition_findings.md",
        target_recommendation_output,
        evaluation_metrics_targets,
    )
    write_chapter_target_tables(
        output_dir / "chapter_target_tables.md",
        target_recommendation_output,
        evaluation_metrics_targets,
        decision_backtest_targets,
    )
    write_chapter_target_journey_update(
        output_dir / "chapter_target_journey_update.md",
        target_recommendation_output,
    )
    write_markdown_table_report(
        output_dir / "target_leaderboard_validation.md",
        "Target Leaderboard Validation",
        ["Candidates ranked only on locked validation data after feature-backbone selection."],
        target_leaderboard_validation,
    )
    write_markdown_table_report(
        output_dir / "target_confirmation_test.md",
        "Target Confirmation Test",
        ["Locked test confirmation after validation-only target selection."],
        target_confirmation_test,
    )
    (output_dir / "calibration_metric_dictionary.md").write_text(
        "\n".join(
            [
                "# Calibration Metric Dictionary",
                "",
                f"- Canonical target-selection metric: validation-split decile `{CANONICAL_TARGET_CALIBRATION_METRIC}`.",
                "- `mean_abs_calibration_gap`: mean absolute difference between decile-level predicted and realized rates; this is the canonical selection and promotion-gate calibration metric.",
                "- `max_abs_calibration_gap`: worst decile gap; diagnostic only.",
                "- `brier_score`: probability loss metric; supporting diagnostic only.",
                "- `integrated_brier_score`: multi-horizon loss summary; supporting diagnostic only.",
                "- `pr_auc` and `roc_auc`: ranking diagnostics; reported for context but not used as the canonical target-selection metric.",
                "- `calibration_slope` and `calibration_intercept`: logistic recalibration diagnostics; reported but not the headline gate.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "target_selection_protocol.md").write_text(
        "\n".join(
            [
                "# Target Selection Protocol",
                "",
                "- Candidate models are fit on the training split.",
                "- Candidate targets are ranked on validation-only metrics after selecting the best available feature backbone per target.",
                f"- The canonical ranking metric is validation decile `{CANONICAL_TARGET_CALIBRATION_METRIC}` after the required support, directness, confidence, and policy-activation gates.",
                "- The locked test slice is used only for post-selection confirmation; it is never used to rank or rescue targets.",
                "- Exit-label confidence and entity-match confidence are tracked separately. High-confidence target views are defined by exit-label provenance, not by entity-match confidence alone.",
                "- Venture/growth remains a doctrinal baseline unless multiple viable empirical candidates exist with sufficient support.",
                "- Buyout/PE remains provisional unless the validation gates, locked-test confirmation, and source/directness diagnostics all support promotion.",
                "- The canonical reduced-load actual path skips generic feature search by default to preserve protocol stability.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "window_selection_protocol.md").write_text(
        "\n".join(
            [
                "# Window Selection Protocol",
                "",
                "- The current window grid reports train, validation, and test support for transparency.",
                "- Locked selection now sorts on memory budget plus train-and-validation support only.",
                "- Test support is reported, but not used for window ranking.",
                "- The exploratory objective score remains in the grid for auditability and backward comparison only.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "label_provenance_dictionary.md").write_text(
        "\n".join(
            [
                "# Label Provenance Dictionary",
                "",
                "- `direct_dated_event`: dated IPO, acquisition, or sponsor-sale evidence from direct event records. This is the strongest label class.",
                "- `inferred_transition`: dated buyout transition evidence inferred from staged Preqin deal-status transitions. This is weaker than direct-dated evidence and is reported separately.",
                "- `synthetic_dated_event`: scenario-pack event available only in sample mode.",
                "- `sensitivity_proxy`: low-confidence proxy retained only for sensitivity work.",
                "- `source_preqin_only`: label evidence sourced only from Preqin extracts.",
                "- `source_crunchbase_only`: label evidence sourced only from Crunchbase extracts.",
                "- `source_both`: combined or explicitly dual-sourced evidence.",
                "- `source_manual_override`: manually overridden evidence if present.",
                "- `source_unknown`: any residual source bucket not covered above.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "source_robustness_report.md").write_text(
        "\n".join(
            [
                "# Source Robustness Report",
                "",
                "## Selected-Backbone Robustness Views",
                "",
                dataframe_to_markdown(target_high_confidence_exit_label_only),
                "",
                dataframe_to_markdown(target_high_confidence_entity_match_only),
                "",
                dataframe_to_markdown(target_high_confidence_overlap),
                "",
                "Direct-dated and direct-plus-high-confidence-inferred views remain in `evaluation_metrics_targets.csv` for every candidate/backbone pair.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "venture_target_doctrine.md").write_text(
        "\n".join(
            [
                "# Venture Target Doctrine",
                "",
                "- Venture/growth target status: retained by doctrine.",
                "- Current target: `hard_timely_liquidity_by_8q`.",
                "- Reason: the current event count does not yet support a broader empirical retargeting exercise with multiple viable candidates.",
                "- Reporting rule: doctrinal baseline only, not a fresh validation-selected target search winner.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "buyout_realization_mechanics_note.md").write_text(
        "\n".join(
            [
                "# Buyout Realization Mechanics Note",
                "",
                "- Buyout/PE is treated as a realization-window problem rather than a copy of the venture milestone target.",
                "- Buyout candidates are rebuilt around sponsor-sale and M&A realization routes, with hard-liquidity benchmarks retained for comparison.",
                "- Partial-realization and recap/secondary candidates remain unsupported when dated fields are absent in the staged local extracts; see `buyout_missing_field_manifest.csv`.",
                "- The sponsor/fund challenger uses PIT-lagged market-quarter fund, manager, cash-flow, and LP-demand aggregates rather than direct deal-to-fund joins.",
                "- The current bundle remains diagnostic; buyout/PE is still provisional until the direct-dated realization spine improves materially.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "decision_usefulness_note.md").write_text(
        "\n".join(
            [
                "# Decision Usefulness Note",
                "",
                "- Decision usefulness is evaluated on validation and then confirmed on the locked test split.",
                "- Policies that accept almost everything or almost nothing are flagged through `acceptance_band_pass` and the degenerate-rule indicators.",
                "- A target is not promotable if policy usefulness is degenerate, even when one calibration statistic looks good.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "chapter_target_selection_limitations.md").write_text(
        "\n".join(
            [
                "# Chapter Target Selection Limitations",
                "",
                "- Preqin and Crunchbase are commercial sources with partial and selectively observed coverage.",
                "- Direct-dated labels are stronger than inferred transitions; both are now reported separately.",
                "- Entity-match confidence and exit-label confidence are distinct and should not be conflated.",
                "- Venture/growth remains a doctrinal baseline rather than a newly promoted empirical headline winner in this pass.",
                "- Buyout/PE is more naturally framed around realization windows, hold period, market conditions, and sponsor/fund state, but the staged local data still lacks enough direct-dated realization support for headline promotion.",
                "- Unsupported buyout mechanics and join layers are recorded in `buyout_missing_field_manifest.csv` rather than being approximated silently.",
                "- This canonical bundle should be treated as a diagnostic milestone for buyout/PE, not final empirical evidence.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "chapter_integration_status.md").write_text(
        "\n".join(
            [
                "# Chapter Integration Status",
                "",
                f"- Venture/growth headline status: `{str(venture_recommended['chapter_reporting_status'].iloc[0]) if not venture_recommended.empty else 'n/a'}`.",
                f"- Buyout/PE headline status: `{str(buyout_recommended['chapter_reporting_status'].iloc[0]) if not buyout_recommended.empty else 'n/a'}`.",
                "- Use `universe_claim_matrix.csv` as the canonical claim-control surface instead of a single global promotion gate.",
                "- Instruction: do not promote buyout/PE target findings into headline empirical prose unless the provisional status clears in a later pass.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "sponsor_fund_feature_pit_rules.md").write_text(
        "\n".join(
            [
                "# Sponsor Fund Feature PIT Rules",
                "",
                "- Fund-launch and close aggregates become available only after the dated launch or close quarter and are merged using `quarter_idx - 1` lookup.",
                "- Reported DPI, RVPI, and multiple fields are keyed by `date_reported` and lagged by the same quarter-offset merge.",
                "- Manager-level raise and co-invest fields are keyed by `lastupdated` and lagged before use.",
                "- Fund-term fields without explicit report dates are observed conservatively no earlier than the available close-date chain.",
                "- No direct deal-to-fund join is available in the staged company deal graph, so these are buyout market-quarter features rather than sponsor-specific company joins.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_buyout_missing_field_manifest(output_dir / "buyout_missing_field_manifest.md", buyout_missing_field_manifest)
    pd.DataFrame(
        [
            {
                "company_id": stylized["company_id"],
                "company_name": stylized["company_name"],
                "quarter_label": quarter_label_from_idx(int(stylized["quarter_idx"])),
                "display_mode": str(display_selection["display_mode"]),
                "display_label": str(display_selection["display_label"]),
                "age_q": stylized["age_q"],
                "time_since_last_round_q": stylized["time_since_last_round_q"],
                "log_last_round_usd": stylized["log_last_round_usd"],
                "patent_apps_visible_l4q": stylized["patent_apps_visible_l4q"],
                "patent_stock_visible": stylized["patent_stock_visible"],
                "patent_grants_l4q": stylized["patent_grants_l4q"],
                "sponsor_score": stylized["sponsor_score"],
                "market_regime": stylized["market_regime"],
            }
        ]
    ).to_csv(output_dir / "stylized_company_snapshot.csv", index=False)
    write_json(
        output_dir / "run_manifest.json",
        {
            "data_mode": str(config["data_mode"]),
            "output_dir": str(output_dir),
            "pack_label": str(config["pack_label"]),
            "selected_min_entry_year": int(dataset["selected_min_entry_year"]),
            "panel_rows": int(len(dataset["panel"])),
            "company_chunk_size": int(config["company_chunk_size"]),
            "max_train_rows": int(config["max_train_rows"]),
            "n_simulations": int(config["n_simulations"]),
            "skip_feature_search": bool(feature_search_skipped),
            "use_quarter_fixed_effects": bool(config.get("use_quarter_fixed_effects", False)),
            "selected_train_end_quarter": str(config["train_end_quarter"]),
            "selected_validation_end_quarter": str(config["validation_end_quarter"]),
            "selected_test_end_quarter": str(config["test_end_quarter"]),
            "used_route_pooling_fallback": bool(route_pooling_used),
            "primary_confusion_threshold": primary_threshold,
            "primary_policy_key": primary_policy_key,
            "appendix_primary_policy_key": primary_policy_key,
            "selected_screening_policy_key": selected_screening_policy_key,
            "selected_economic_policy_key": selected_economic_policy_key,
            "headline_target_name": HARD_TIMELY_LIQUIDITY_TARGET,
            "recommended_target_venture_growth": str(run_metadata["recommended_target_venture_growth"].iloc[0]),
            "recommended_target_buyout_pe": str(run_metadata["recommended_target_buyout_pe"].iloc[0]),
            "venture_target_reporting_status": str(run_metadata["venture_target_reporting_status"].iloc[0]),
            "buyout_target_reporting_status": str(run_metadata["buyout_target_reporting_status"].iloc[0]),
            "stage2_route_class_set": "|".join(stage2_classes),
            "chosen_feature_combo_key": str(chosen_feature_combo_summary["combo_key"].iloc[0]) if not chosen_feature_combo_summary.empty else "",
            "chapter_evidence_ready": bool(promotion_gate["chapter_evidence_ready"].iloc[0]),
            "chapter_evidence_ready_v2": bool(promotion_gate_v2["chapter_evidence_ready"].iloc[0]),
            "stylized_company_id": str(stylized["company_id"]),
            "stylized_company_name": str(stylized["company_name"]),
            "display_mode": str(display_selection["display_mode"]),
            "display_label": str(display_selection["display_label"]),
            "stylized_quarter": quarter_label_from_idx(int(stylized["quarter_idx"])),
            "optimization_iterations": int(fitted["optimization_iterations"]),
            "optimization_message": str(fitted["optimization_message"]),
        },
    )
    generated_files.append(
        write_feature_importance_notes(output_dir)
    )
    generated_files.append(
        write_confusion_matrix_notes(
            output_dir,
            data_mode,
            primary_threshold,
            primary_policy_key,
        )
    )
    generated_files.append(
        write_promotion_gate_explanation(
            output_dir,
            promotion_gate,
            route_pooling_used,
        )
    )
    generated_files.append(
        write_chapter_summary_v2(
            output_dir,
            run_metadata,
            dataset["target_definition_main"],
            dataset["label_confidence_audit"],
            promotion_gate_v2,
        )
    )
    generated_files.append(
        write_chapter_tables_v2(
            output_dir,
            universe_support,
            dataset["label_confidence_audit"],
            evaluation_metrics_main,
            sector_stage_metrics,
            decision_backtest_screening,
            decision_backtest_economic,
        )
    )
    generated_files.append(
        write_chapter_findings_v2(
            output_dir,
            evaluation_metrics_main,
            evaluation_metrics_by_universe,
            stage2_route_metrics,
            policy_activation_summary,
            promotion_gate_v2,
            patent_sector_model_comparison,
        )
    )
    calibration_status_path = output_dir / "calibration_status_notes.md"
    calibration_status_path.write_text(build_calibration_status_notes(evaluation_metrics_main), encoding="utf-8")
    generated_files.append(calibration_status_path)
    generated_files.append(output_dir / "evaluation_metrics_targets.md")
    generated_files.append(output_dir / "decision_backtest_targets.md")
    generated_files.append(output_dir / "target_recommendation_summary.md")
    generated_files.append(output_dir / "universe_claim_matrix.md")
    generated_files.append(output_dir / "chapter_target_doctrine.md")
    generated_files.append(output_dir / "chapter_objective_definition_findings.md")
    generated_files.append(output_dir / "chapter_target_tables.md")
    generated_files.append(output_dir / "chapter_target_journey_update.md")
    generated_files.append(output_dir / "buyout_missing_field_manifest.md")
    appendix_confusion_notes = output_dir / "appendix_confusion_notes.md"
    write_appendix_confusion_notes(appendix_confusion_notes)
    generated_files.append(appendix_confusion_notes)
    promotion_gate_v2_path = output_dir / "promotion_gate_v2_explanation.md"
    write_promotion_gate_v2_explanation(promotion_gate_v2_path, promotion_gate_v2)
    generated_files.append(promotion_gate_v2_path)

    return {
        "config": config,
        "dataset": dataset,
        "fitted": fitted,
        "calibration": calibration,
        "calibration_high_confidence": calibration_high_confidence,
        "calibration_stress_slice": calibration_stress_slice,
        "calibration_summary": calibration_summary,
        "evaluation_metrics": evaluation_metrics,
        "decision_backtest": decision_backtest,
        "confusion_matrix_exit_by_8q_summary": exit_confusion_summary,
        "confusion_matrix_decision_policy_summary": decision_policy_confusion_summary,
        "feature_registry": feature_registry,
        "feature_combo_validation_leaderboard": feature_combo_validation_leaderboard,
        "summary_metrics": summary_metrics,
        "target_registry": target_registry,
        "evaluation_metrics_targets": evaluation_metrics_targets,
        "decision_backtest_targets": decision_backtest_targets,
        "target_recommendation_table": target_recommendation_output,
        "universe_claim_matrix": universe_claim_matrix,
        "route_audit": dataset["route_audit"],
        "placeholder_status": placeholder_status,
        "run_metadata": run_metadata,
        "promotion_gate": promotion_gate,
        "promotion_gate_v2": promotion_gate_v2,
        "display_selection": display_selection,
        "generated_files": generated_files,
    }

