# (c) 2027, Michael Robbins
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from chapter_figure_exports import build_chapter_figure_suite
from path_helpers import LIVE_DATA_ENV_VAR, resolve_live_data_dir
from vc_pe_exit_prediction import run_live_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Chapter 9 VC and PE exit prediction workflow."
    )
    parser.add_argument(
        "--data-mode",
        default="sample",
        choices=["sample", "live", "actual"],
        help=(
            "Data mode. `sample` reads the bundled scenario pack; `live` reads "
            "local proprietary inputs outside the repo. `actual` remains as a "
            "deprecated compatibility alias."
        ),
    )
    parser.add_argument(
        "--live-data-dir",
        default="",
        help=(
            f"Optional proprietary live data root. Defaults to the "
            f"{LIVE_DATA_ENV_VAR} environment variable when set."
        ),
    )
    parser.add_argument(
        "--paths-file",
        default="",
        help=(
            "Optional path to a local paths YAML. When omitted, live mode falls "
            "back to the environment variable or the repo local paths files."
        ),
    )
    parser.add_argument(
        "--sample-pack-dir",
        default="",
        help=(
            "Optional path to the bundled sample pack root. Defaults to "
            "auto-discovery in either the repo or standalone layout."
        ),
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Optional output directory. Defaults to a rendered-* directory beside this script.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=900000,
        help="Maximum weighted training rows retained after sampling no-exit rows.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Number of pathwise valuation simulations for the stylized company.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling and valuation.",
    )
    parser.add_argument(
        "--min-entry-year",
        type=int,
        default=0,
        help="Optional explicit minimum entry year for the modeled company universe. Defaults to automatic density-based selection.",
    )
    parser.add_argument(
        "--target-panel-rows",
        type=int,
        default=1500000,
        help="Approximate panel-row budget used when selecting the live cohort automatically.",
    )
    parser.add_argument(
        "--company-chunk-size",
        type=int,
        default=1000,
        help="Number of companies to expand per panel-building chunk.",
    )
    parser.add_argument(
        "--decision-eval-paths",
        type=int,
        default=48,
        help="Number of decision-path draws used in the redesigned policy backtests.",
    )
    parser.add_argument(
        "--feature-search-max-train-rows",
        type=int,
        default=30000,
        help="Maximum training rows used by repeated feature-search retrains.",
    )
    parser.add_argument(
        "--feature-search-validation-max-rows",
        type=int,
        default=120000,
        help="Maximum validation rows used by feature-search diagnostics in live mode.",
    )
    parser.add_argument(
        "--feature-search-test-max-rows",
        type=int,
        default=160000,
        help="Maximum test rows used by feature-search diagnostics in live mode.",
    )
    parser.add_argument(
        "--feature-search-decision-eval-paths",
        type=int,
        default=16,
        help="Number of decision-path draws used by feature-search challenger diagnostics.",
    )
    feature_search_group = parser.add_mutually_exclusive_group()
    feature_search_group.add_argument(
        "--skip-feature-search",
        action="store_true",
        help="Skip the generic feature-search stack and write placeholder diagnostics instead.",
    )
    feature_search_group.add_argument(
        "--run-feature-search",
        action="store_true",
        help="Force the full feature-search stack. Live mode skips it by default unless this flag is supplied.",
    )
    parser.add_argument(
        "--buyout-only",
        action="store_true",
        help="Run the Chapter 9 buyout realization-spine rebuild instead of the full chapter pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    public_mode = "live" if args.data_mode == "actual" else args.data_mode
    internal_mode = "actual" if public_mode == "live" else public_mode
    live_data_dir = resolve_live_data_dir(args.live_data_dir or None) if public_mode == "live" else None
    if args.buyout_only:
        if public_mode == "live":
            default_dir = f"rendered-live-buyout-policy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        else:
            default_dir = "rendered-sample-buyout-policy"
    else:
        default_dir = "rendered-sample" if public_mode == "sample" else "rendered-live"
    outdir = Path(args.outdir).resolve() if args.outdir else script_dir / default_dir
    skip_feature_search = bool(args.skip_feature_search)
    if args.run_feature_search:
        skip_feature_search = False
    elif public_mode == "live":
        skip_feature_search = True
    config = {
        "data_mode": internal_mode,
        "pack_label": "scenario" if public_mode == "sample" else "licensed",
        "paths_file": args.paths_file or None,
        "sample_pack_dir": args.sample_pack_dir or None,
        "live_data_dir": live_data_dir,
        "output_dir": outdir,
        "max_train_rows": int(args.max_train_rows),
        "n_simulations": int(args.simulations),
        "random_seed": int(args.seed),
        "min_entry_year": int(args.min_entry_year) if args.min_entry_year else None,
        "target_panel_rows": int(args.target_panel_rows),
        "company_chunk_size": int(args.company_chunk_size),
        "decision_eval_paths": int(args.decision_eval_paths),
        "feature_search_max_train_rows": int(args.feature_search_max_train_rows),
        "feature_search_validation_max_rows": int(args.feature_search_validation_max_rows),
        "feature_search_test_max_rows": int(args.feature_search_test_max_rows),
        "feature_search_decision_eval_paths": int(args.feature_search_decision_eval_paths),
        "skip_feature_search": skip_feature_search,
        "buyout_only": bool(args.buyout_only),
        "command_line": " ".join(__import__("sys").argv),
    }

    outputs = run_live_pipeline(config)
    doctrine_dir = None
    if public_mode == "live":
        doctrine_candidate = script_dir / "rendered-live-final-doctrine-20260404"
        if doctrine_candidate.exists():
            doctrine_dir = doctrine_candidate
    exported_figures = build_chapter_figure_suite(
        outdir,
        outdir,
        doctrine_dir=doctrine_dir,
        mode_label=public_mode,
    )
    if args.buyout_only:
        print("\nBuyout policy rebuild")
        print(f"Output directory: {outputs['output_dir']}")
        print(f"Selected minimum entry year: {outputs['dataset']['selected_min_entry_year']}")
        if "buyout_recommendation" in outputs and not outputs["buyout_recommendation"].empty:
            print("\nSelected buyout target")
            print(outputs["buyout_recommendation"].round(4).to_string(index=False))
        if "buyout_promotion_gate" in outputs and not outputs["buyout_promotion_gate"].empty:
            print("\nBuyout promotion gate")
            print(outputs["buyout_promotion_gate"].round(4).to_string(index=False))
    else:
        print("\nKey summary")
        print(outputs["summary_metrics"].round(4).to_string(index=False))
        print(f"\nSelected minimum entry year: {outputs['dataset']['selected_min_entry_year']}")
        print("\nRoute audit")
        print(outputs["route_audit"].round(2).to_string(index=False))
    print("\nRendered files")
    for path in outputs["generated_files"]:
        print(path.name)
    print("\nChapter-facing SVG figures")
    for path in exported_figures:
        print(path.name)


if __name__ == "__main__":
    main()
