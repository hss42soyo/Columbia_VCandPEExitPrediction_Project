# (c) 2027, Michael Robbins
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


CODE_ROOT = Path(__file__).resolve().parents[1]
BOOK2_ROOT = Path(r"D:\dev\ECR Capital Management\Book2")
DATA_ROOT = (
    BOOK2_ROOT
    / "data"
    / "eamples"
    / "Chapter 9"
    / "VC & PE Exit Prediction"
    / "third_party_reproduction_20260521"
)

CRUNCHBASE_ROOT = Path(r"D:\data\Crunchbase")
PREQIN_LINKAGE_ROOT = Path(r"D:\data\VendorLinkage\preqin_crunchbase_v1_20260514_run4")
COMMON_CRAWL_ROOT = Path(r"D:\data\Common Crawl")
PAPER7_ROOT = BOOK2_ROOT / "Private" / "Cruchbase+Preqin" / "Paper 7"

FROZEN_OUTPUT_ROOTS = [
    PAPER7_ROOT / "outputs" / "paper7-final-policy-synthesis-20260512",
    PAPER7_ROOT / "outputs" / "paper7-final-evidence-memo-20260510",
    PAPER7_ROOT / "outputs" / "paper7-final-evidence-acceptance-20260510",
    PAPER7_ROOT / "outputs" / "paper7-commoncrawl-freeze-post-cleanup-20260510",
    PAPER7_ROOT / "outputs" / "paper7-author-package-vs-canonical-commoncrawl-20260510",
    PAPER7_ROOT / "outputs" / "paper7-author-homepage-commoncrawl-comparison-20260510-final",
    PAPER7_ROOT / "experimental_models" / "company_specific_v2_20260513" / "outputs" / "company_specific_v2_20260513-20260513-222709",
    PAPER7_ROOT / "experimental_models" / "company_specific_v2_20260513" / "outputs" / "company_specific_v2_20260513-20260513-225031",
    PAPER7_ROOT / "experimental_models" / "company_specific_v2_20260513" / "outputs" / "exit_model_signal_diagnostics_20260513-20260513-234652",
    PAPER7_ROOT / "docs" / "paper7_reproduction_final_report.md",
]

DATA_COMPONENTS = [
    ("crunchbase", CRUNCHBASE_ROOT, DATA_ROOT / "data" / "crunchbase"),
    ("preqin_linkage", PREQIN_LINKAGE_ROOT, DATA_ROOT / "data" / "preqin_linkage"),
    ("common_crawl", COMMON_CRAWL_ROOT, DATA_ROOT / "data" / "common_crawl"),
]


@dataclass(frozen=True)
class FileRecord:
    component: str
    source_root: Path
    source_path: Path
    dest_root: Path
    dest_path: Path
    relpath: str
    source_bytes: int
    source_mtime_utc: str
    source_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
        if path.is_file():
            yield path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_one(record: FileRecord) -> dict:
    ensure_dir(record.dest_path.parent)
    action = "copied"
    if record.dest_path.exists():
        if record.dest_path.stat().st_size == record.source_bytes:
            dest_hash = sha256_file(record.dest_path)
            if dest_hash == record.source_sha256:
                action = "skipped_existing_match"
                return ledger_row(record, action, record.dest_path.stat().st_size, dest_hash, "ok")
        action = "overwrote_existing_mismatch"

    shutil.copy2(record.source_path, record.dest_path)
    dest_bytes = record.dest_path.stat().st_size
    dest_hash = sha256_file(record.dest_path)
    status = "ok" if dest_bytes == record.source_bytes and dest_hash == record.source_sha256 else "hash_or_size_mismatch"
    return ledger_row(record, action, dest_bytes, dest_hash, status)


def ledger_row(record: FileRecord, action: str, dest_bytes: int, dest_sha256: str, status: str) -> dict:
    return {
        "component": record.component,
        "source_root": str(record.source_root),
        "source_relpath": record.relpath,
        "source_path": str(record.source_path),
        "dest_root": str(record.dest_root),
        "dest_path": str(record.dest_path),
        "source_bytes": record.source_bytes,
        "dest_bytes": dest_bytes,
        "source_mtime_utc": record.source_mtime_utc,
        "source_sha256": record.source_sha256,
        "dest_sha256": dest_sha256,
        "action": action,
        "status": status,
        "packaged_at_utc": utc_now(),
    }


def inventory_component(component: str, source_root: Path, dest_root: Path) -> list[FileRecord]:
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source root for {component}: {source_root}")
    records: list[FileRecord] = []
    for source_path in iter_files(source_root):
        relpath = source_path.name if source_root.is_file() else str(source_path.relative_to(source_root))
        dest_path = dest_root / relpath
        records.append(
            FileRecord(
                component=component,
                source_root=source_root,
                source_path=source_path,
                dest_root=dest_root,
                dest_path=dest_path,
                relpath=relpath,
                source_bytes=source_path.stat().st_size,
                source_mtime_utc=iso_mtime(source_path),
                source_sha256=sha256_file(source_path),
            )
        )
    return records


def frozen_output_components() -> list[tuple[str, Path, Path]]:
    components: list[tuple[str, Path, Path]] = []
    base_dest = DATA_ROOT / "data" / "frozen_outputs"
    for source in FROZEN_OUTPUT_ROOTS:
        if not source.exists():
            print(f"WARNING: missing frozen output source, skipping: {source}", file=sys.stderr)
            continue
        name = source.stem if source.is_file() else source.name
        components.append((f"frozen_outputs/{name}", source, base_dest / name))
    return components


def parquet_table_record(component: str, source_root: Path, parquet_path: Path) -> dict:
    relpath = parquet_path.name if source_root.is_file() else str(parquet_path.relative_to(source_root))
    metadata = pq.ParquetFile(parquet_path)
    schema = metadata.schema_arrow
    columns = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]
    column_names = [column["name"] for column in columns]
    candidate_keys = [
        name
        for name in column_names
        if name.lower() in {"uuid", "id", "permalink", "cb_url", "domain", "company_id", "fund_id", "manager_id"}
    ]
    candidate_foreign_keys = [
        name
        for name in column_names
        if name.lower().endswith("_uuid") or name.lower().endswith("_id")
    ]
    return {
        "component": component,
        "table_name": parquet_path.stem,
        "source_relpath": relpath,
        "source_path": str(parquet_path),
        "file_bytes": parquet_path.stat().st_size,
        "row_count": metadata.metadata.num_rows,
        "column_count": len(columns),
        "columns": columns,
        "candidate_primary_keys": candidate_keys,
        "candidate_foreign_keys": candidate_foreign_keys,
    }


def write_schema_catalog(component: str, source_root: Path, json_path: Path, md_path: Path, rel_path: Path) -> None:
    tables = [parquet_table_record(component, source_root, path) for path in iter_files(source_root) if path.suffix.lower() == ".parquet"]
    payload = {
        "component": component,
        "generated_at_utc": utc_now(),
        "source_root": str(source_root),
        "tables": tables,
    }
    ensure_dir(json_path.parent)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# {component.replace('_', ' ').title()} Schema Catalog",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "| Table | Rows | Columns | Bytes | Relative Path |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for table in tables:
        lines.append(
            f"| `{table['table_name']}` | {table['row_count']} | {table['column_count']} | {table['file_bytes']} | `{table['source_relpath']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rel_lines = [
        f"component: {component}",
        f"generated_at_utc: {payload['generated_at_utc']}",
        "relationships:",
    ]
    for table in tables:
        rel_lines.append(f"  - table: {table['table_name']}")
        rel_lines.append(f"    candidate_primary_keys: {json.dumps(table['candidate_primary_keys'])}")
        rel_lines.append(f"    candidate_foreign_keys: {json.dumps(table['candidate_foreign_keys'])}")
    rel_path.write_text("\n".join(rel_lines) + "\n", encoding="utf-8")


def write_commoncrawl_rebuild_manifest(path: Path) -> None:
    lane_rows = []
    for manifest in sorted(COMMON_CRAWL_ROOT.rglob("manifest.json"), key=lambda p: str(p).lower()):
        lane_root = manifest.parent
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        lane_rows.append(
            {
                "lane": str(lane_root.relative_to(COMMON_CRAWL_ROOT)),
                "manifest": str(manifest),
                "files": len(list(iter_files(lane_root))),
                "manifest_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
            }
        )

    lines = [
        "component: common_crawl",
        f"generated_at_utc: {utc_now()}",
        "classification: private_public_derived_linked_to_licensed_surfaces",
        "rebuild_note: retained lane artifacts and manifests validate/rebuild the homepage proxy without obsolete bulk recovery caches",
        "lanes:",
    ]
    for row in lane_rows:
        lines.append(f"  - lane: {row['lane']}")
        lines.append(f"    manifest: {row['manifest']}")
        lines.append(f"    files: {row['files']}")
        lines.append(f"    manifest_keys: {json.dumps(row['manifest_keys'])}")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_package_docs(source_rows: list[dict], ledger_rows: list[dict]) -> None:
    total_bytes = sum(int(row["dest_bytes"]) for row in ledger_rows if row["status"] == "ok")
    docs = {
        "PACKAGE_CONTENTS.md": [
            "# Package Contents",
            "",
            f"Generated: {utc_now()}",
            "",
            f"Files packaged: {len(ledger_rows)}",
            f"Verified bytes: {total_bytes}",
            "",
            "Components:",
        ]
        + [f"- {component}" for component, _, _ in DATA_COMPONENTS + frozen_output_components()],
        "DATA_LICENSE_NOTES.md": [
            "# Data License Notes",
            "",
            "This package is licensed/private. It is intended only for a third party that is authorized to receive the underlying Crunchbase and Preqin-derived data.",
            "",
            "Common Crawl source material is public-derived, but the retained proxy artifacts are linked to licensed company/domain surfaces and should travel with the same private handling.",
            "",
            "Do not redistribute this package as a public synthetic-data artifact.",
        ],
    }
    for name, lines in docs.items():
        (DATA_ROOT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    status_counts: dict[str, int] = {}
    for row in ledger_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    report_lines = [
        "# Package Validation Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Source files inventoried: {len(source_rows)}",
        f"Copied or verified files: {len(ledger_rows)}",
        "",
        "| Status | Files |",
        "| --- | ---: |",
    ]
    for status, count in sorted(status_counts.items()):
        report_lines.append(f"| {status} | {count} |")
    (DATA_ROOT / "manifests" / "package_validation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dir(DATA_ROOT)
    ensure_dir(CODE_ROOT / "schemas")
    ensure_dir(DATA_ROOT / "manifests")

    print(f"Code root: {CODE_ROOT}")
    print(f"Data root: {DATA_ROOT}")

    all_components = DATA_COMPONENTS + frozen_output_components()
    all_records: list[FileRecord] = []
    for component, source_root, dest_root in all_components:
        print(f"Inventorying {component}: {source_root}")
        records = inventory_component(component, source_root, dest_root)
        print(f"  files={len(records)} bytes={sum(record.source_bytes for record in records)}")
        all_records.extend(records)

    source_rows = [
        {
            "component": record.component,
            "source_root": str(record.source_root),
            "source_relpath": record.relpath,
            "source_path": str(record.source_path),
            "source_bytes": record.source_bytes,
            "source_mtime_utc": record.source_mtime_utc,
            "source_sha256": record.source_sha256,
            "dest_path": str(record.dest_path),
        }
        for record in all_records
    ]
    source_fields = [
        "component",
        "source_root",
        "source_relpath",
        "source_path",
        "source_bytes",
        "source_mtime_utc",
        "source_sha256",
        "dest_path",
    ]
    write_csv(DATA_ROOT / "manifests" / "source_manifest.csv", source_rows, source_fields)

    ledger_rows = []
    for index, record in enumerate(all_records, start=1):
        print(f"[{index}/{len(all_records)}] {record.component}: {record.relpath}")
        ledger_rows.append(copy_one(record))

    ledger_fields = [
        "component",
        "source_root",
        "source_relpath",
        "source_path",
        "dest_root",
        "dest_path",
        "source_bytes",
        "dest_bytes",
        "source_mtime_utc",
        "source_sha256",
        "dest_sha256",
        "action",
        "status",
        "packaged_at_utc",
    ]
    write_csv(DATA_ROOT / "manifests" / "copy_ledger.csv", ledger_rows, ledger_fields)
    write_csv(DATA_ROOT / "manifests" / "hash_manifest.csv", ledger_rows, ledger_fields)

    print("Writing schema catalogs")
    write_schema_catalog(
        "crunchbase",
        CRUNCHBASE_ROOT / "parquet",
        CODE_ROOT / "schemas" / "crunchbase_schema.json",
        CODE_ROOT / "schemas" / "crunchbase_schema.md",
        CODE_ROOT / "schemas" / "crunchbase_relationships.yml",
    )
    write_schema_catalog(
        "preqin",
        PREQIN_LINKAGE_ROOT,
        CODE_ROOT / "schemas" / "preqin_schema.json",
        CODE_ROOT / "schemas" / "preqin_schema.md",
        CODE_ROOT / "schemas" / "preqin_relationships.yml",
    )
    write_schema_catalog(
        "commoncrawl",
        COMMON_CRAWL_ROOT,
        CODE_ROOT / "schemas" / "commoncrawl_schema.json",
        CODE_ROOT / "schemas" / "commoncrawl_schema.md",
        CODE_ROOT / "schemas" / "commoncrawl_relationships.yml",
    )
    write_commoncrawl_rebuild_manifest(CODE_ROOT / "schemas" / "commoncrawl_rebuild_manifest.yml")
    write_package_docs(source_rows, ledger_rows)

    failures = [row for row in ledger_rows if row["status"] != "ok"]
    if failures:
        print(f"Package completed with {len(failures)} validation failures.", file=sys.stderr)
        return 2
    print("Package completed and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
