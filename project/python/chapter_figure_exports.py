# (c) 2027, Michael Robbins
from __future__ import annotations

import html
import math
import shutil
from pathlib import Path

import pandas as pd


FIGURE_FILENAMES = {
    "fig1": "ch09_vcpe_fig1_sample_construction_and_label_confidence.svg",
    "fig2": "ch09_vcpe_fig2_main_evaluation_table.svg",
    "fig3": "ch09_vcpe_fig3_universe_sector_stage_heterogeneity.svg",
    "fig4": "ch09_vcpe_fig4_hard_timely_liquidity_calibration.svg",
    "fig5": "ch09_vcpe_fig5_route_cumulative_incidence.svg",
    "fig6": "ch09_vcpe_fig6_policy_backtest.svg",
    "fig7": "ch09_vcpe_fig7_chapter_dashboard.svg",
}


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:,.3f}".rstrip("0").rstrip(".")
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _svg_table(title: str, subtitle: str, frame: pd.DataFrame, destination: Path) -> None:
    if frame.empty:
        frame = pd.DataFrame([{"status": "missing"}])
    columns = list(frame.columns)
    body_rows = frame.astype(object).where(pd.notna(frame), "").values.tolist()
    width = max(1200, 230 * max(1, len(columns)))
    header_height = 118
    row_height = 34
    footer_height = 24
    height = header_height + row_height * (len(body_rows) + 1) + footer_height
    col_width = (width - 80) / max(1, len(columns))
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        '.bg { fill: #fbf8f1; }',
        '.title { font: 700 28px Georgia, serif; fill: #1d2624; }',
        '.subtitle { font: 400 15px Georgia, serif; fill: #56615d; }',
        '.head { font: 700 14px Consolas, monospace; fill: #fbf8f1; }',
        '.cell { font: 400 13px Consolas, monospace; fill: #1d2624; }',
        '.stripe { fill: #f2ede2; }',
        '.header { fill: #2f4b43; }',
        '.grid { stroke: #d5cdc0; stroke-width: 1; }',
        '</style>',
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" />',
        f'<text class="title" x="40" y="46">{html.escape(title)}</text>',
        f'<text class="subtitle" x="40" y="74">{html.escape(subtitle)}</text>',
        f'<rect class="header" x="40" y="{header_height - 10}" width="{width - 80}" height="{row_height}" rx="8" ry="8" />',
    ]
    for idx, column in enumerate(columns):
        x = 40 + idx * col_width + 10
        y = header_height + 12
        lines.append(f'<text class="head" x="{x:.1f}" y="{y}">{html.escape(str(column))}</text>')
    base_y = header_height + row_height
    for row_idx, row in enumerate(body_rows):
        y_top = base_y + row_idx * row_height
        if row_idx % 2 == 0:
            lines.append(
                f'<rect class="stripe" x="40" y="{y_top - 10}" width="{width - 80}" height="{row_height}" rx="4" ry="4" />'
            )
        for col_idx, value in enumerate(row):
            x = 40 + col_idx * col_width + 10
            y = y_top + 12
            text = html.escape(_format_value(value))
            if len(text) > 28:
                text = f"{html.escape(text[:25])}..."
            lines.append(f'<text class="cell" x="{x:.1f}" y="{y}">{text}</text>')
    lines.append("</svg>")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_dashboard(
    mode_label: str,
    doctrine_rows: list[tuple[str, str]],
    blocker_rows: list[tuple[str, str]],
    destination: Path,
) -> None:
    width = 1320
    height = 820
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        '.bg { fill: #f6f3eb; }',
        '.card { fill: #ffffff; stroke: #d2cabb; stroke-width: 1.5; }',
        '.title { font: 700 30px Georgia, serif; fill: #1d2624; }',
        '.subtitle { font: 400 16px Georgia, serif; fill: #56615d; }',
        '.cardtitle { font: 700 18px Georgia, serif; fill: #234137; }',
        '.body { font: 400 14px Consolas, monospace; fill: #1d2624; }',
        '.statusok { fill: #2f7d59; }',
        '.statuswarn { fill: #b86d1b; }',
        '.statusbad { fill: #a83d2f; }',
        '</style>',
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" />',
        '<text class="title" x="40" y="54">Chapter 9 Current-State Dashboard</text>',
        f'<text class="subtitle" x="40" y="84">Mode: {html.escape(mode_label)}. The public package ships the current empirical draft, not an evidence-ready claim.</text>',
    ]
    cards = [
        (40, 120, 390, 170, "Doctrine", doctrine_rows[:3]),
        (455, 120, 390, 170, "Promotion Gate", doctrine_rows[3:6]),
        (870, 120, 410, 170, "Claim Boundary", doctrine_rows[6:9]),
        (40, 320, 600, 430, "Current Blockers", blocker_rows[:7]),
        (670, 320, 610, 430, "Publication Stance", blocker_rows[7:]),
    ]
    for x, y, w, h, title, rows in cards:
        lines.append(f'<rect class="card" x="{x}" y="{y}" width="{w}" height="{h}" rx="16" ry="16" />')
        lines.append(f'<text class="cardtitle" x="{x + 22}" y="{y + 34}">{html.escape(title)}</text>')
        for idx, (label, value) in enumerate(rows):
            yy = y + 68 + idx * 24
            lines.append(f'<text class="body" x="{x + 22}" y="{yy}">{html.escape(label)}: {html.escape(value)}</text>')
    lines.append("</svg>")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_chapter_figure_suite(
    source_dir: str | Path,
    outdir: str | Path,
    doctrine_dir: str | Path | None = None,
    mode_label: str = "live",
) -> list[Path]:
    source_root = Path(source_dir).resolve()
    out_root = Path(outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    doctrine_root = Path(doctrine_dir).resolve() if doctrine_dir else None

    partition_summary = _read_csv(source_root / "partition_summary.csv")
    label_confidence = _read_csv(source_root / "label_confidence_audit.csv")
    evaluation_metrics = _read_csv(source_root / "evaluation_metrics_main.csv")
    heterogeneity = _read_csv(source_root / "sector_stage_metrics.csv")
    promotion_gate = _read_csv(source_root / "promotion_gate_v2.csv")

    fig1_rows: list[dict[str, object]] = []
    if not partition_summary.empty:
        for row in partition_summary.to_dict("records"):
            fig1_rows.append(
                {
                    "section": "sample_construction",
                    "item": str(row.get("split", "")),
                    "rows": row.get("rows", ""),
                    "companies": row.get("companies", ""),
                    "hard_exits": row.get("exits", ""),
                }
            )
    if not label_confidence.empty:
        for row in label_confidence.to_dict("records"):
            fig1_rows.append(
                {
                    "section": "label_confidence",
                    "item": str(row.get("route_label", "")),
                    "rows": row.get("chosen_exit_count", ""),
                    "companies": row.get("confidence_tier", ""),
                    "hard_exits": row.get("route_source", ""),
                }
            )
    _svg_table(
        "Sample Construction And Label Confidence",
        "Frozen canonical bundle summary for the current Chapter 9 live objective.",
        pd.DataFrame(fig1_rows),
        out_root / FIGURE_FILENAMES["fig1"],
    )

    if not evaluation_metrics.empty:
        keep = [
            "evaluation_view",
            "rows",
            "brier_score",
            "integrated_brier_score",
            "pr_auc",
            "roc_auc",
            "calibration_slope_status",
        ]
        fig2 = evaluation_metrics[[column for column in keep if column in evaluation_metrics.columns]].copy()
    else:
        fig2 = pd.DataFrame()
    _svg_table(
        "Main Evaluation Table",
        "Live evaluation views for hard_timely_liquidity_by_8q.",
        fig2,
        out_root / FIGURE_FILENAMES["fig2"],
    )

    if not heterogeneity.empty:
        fig3 = heterogeneity.copy()
        if "hard_timely_liquidity_events" in fig3.columns:
            fig3 = fig3[pd.to_numeric(fig3["hard_timely_liquidity_events"], errors="coerce").fillna(0) > 0]
        if "rows" in fig3.columns:
            fig3 = fig3.sort_values(["universe", "rows"], ascending=[True, False])
        fig3 = fig3[[column for column in [
            "universe",
            "sector_bucket",
            "stage_bucket",
            "rows",
            "hard_timely_liquidity_events",
            "pr_auc",
            "roc_auc",
        ] if column in fig3.columns]].head(14)
    else:
        fig3 = pd.DataFrame()
    _svg_table(
        "Universe, Sector, And Stage Heterogeneity",
        "Highest-support sector-stage buckets in the frozen current-state bundle.",
        fig3,
        out_root / FIGURE_FILENAMES["fig3"],
    )

    copy_pairs = [
        ("vcpe-calibration-deciles.svg", FIGURE_FILENAMES["fig4"]),
        ("vcpe-cumulative-incidence.svg", FIGURE_FILENAMES["fig5"]),
        ("vcpe-policy-backtest.svg", FIGURE_FILENAMES["fig6"]),
    ]
    generated: list[Path] = [
        out_root / FIGURE_FILENAMES["fig1"],
        out_root / FIGURE_FILENAMES["fig2"],
        out_root / FIGURE_FILENAMES["fig3"],
    ]
    for source_name, output_name in copy_pairs:
        source_path = source_root / source_name
        destination = out_root / output_name
        shutil.copyfile(source_path, destination)
        generated.append(destination)

    doctrine_rows: list[tuple[str, str]] = [
        ("venture doctrine", "hard_timely_liquidity_by_8q"),
        ("buyout extension", "any_direct_realization_by_16q"),
        ("reporting split", "venture main, buyout provisional"),
    ]
    if not promotion_gate.empty:
        gate_row = promotion_gate.iloc[0].to_dict()
        doctrine_rows.extend(
            [
                ("route support", str(gate_row.get("enough_route_support", ""))),
                ("policy activation", str(gate_row.get("enough_policy_activation", ""))),
                ("chapter evidence ready", str(gate_row.get("chapter_evidence_ready", ""))),
            ]
        )
    else:
        doctrine_rows.extend(
            [
                ("route support", "unavailable"),
                ("policy activation", "unavailable"),
                ("chapter evidence ready", "unavailable"),
            ]
        )
    if doctrine_root:
        claim_matrix = _read_csv(doctrine_root / "universe_claim_matrix.csv")
        if not claim_matrix.empty:
            for row in claim_matrix.to_dict("records")[:3]:
                doctrine_rows.append(
                    (str(row.get("universe", "")), str(row.get("reporting_status", "")))
                )
    else:
        doctrine_rows.extend(
            [
                ("venture claim", "sample teaching bundle"),
                ("buyout claim", "sample teaching bundle"),
                ("claim boundary", "static figure package only"),
            ]
        )
    blocker_rows = [
        ("1", "Full-test calibration still fails the promotion gate."),
        ("2", "Decision policies activate, but screening precision remains limited."),
        ("3", "Buyout/PE remains an appendix-level extension, not a headline claim."),
        ("4", "Sponsor/fund joins are still coverage-constrained in local staged files."),
        ("5", "The public repo ships static live figures only, never live vendor tables."),
        ("6", "Sample mode is the default public teaching path."),
        ("7", "Live mode remains a proprietary-data rerender path."),
        ("publication", "Describe the chapter as an empirical draft / current-state package."),
        ("manuscript", "Use venture_growth as the worked doctrine and buyout as the honest non-promotion case study."),
        ("repo", "Keep sample/live terminology and keep proprietary paths outside tracked files."),
    ]
    dashboard_path = out_root / FIGURE_FILENAMES["fig7"]
    _svg_dashboard(mode_label, doctrine_rows, blocker_rows, dashboard_path)
    generated.append(dashboard_path)
    return generated
