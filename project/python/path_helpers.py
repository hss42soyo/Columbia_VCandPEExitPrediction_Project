# (c) 2027, Michael Robbins
from __future__ import annotations

import os
from pathlib import Path


SAMPLE_PACK_SUFFIX = Path("packs") / "scenario" / "pe-vc-hazard"
LIVE_DATA_ENV_VAR = "BOOK2_PE_VC_HAZARD_LIVE_DATA_DIR"
REQUIRED_SAMPLE_PACK_FILES = (
    "dim_private_company.csv",
    "dim_fund.csv",
    "fact_private_round.csv",
    "fact_private_investor_participation.csv",
    "fact_private_exit.csv",
    "event_market_regime.csv",
)


def _anchor_dir(anchor: str | Path) -> Path:
    path = Path(anchor).resolve()
    return path if path.is_dir() else path.parent


def has_sample_pack_files(pack_dir: str | Path) -> bool:
    root = Path(pack_dir).resolve()
    return all((root / filename).exists() for filename in REQUIRED_SAMPLE_PACK_FILES)


def candidate_sample_pack_dirs(anchor: str | Path) -> list[Path]:
    anchor_dir = _anchor_dir(anchor)
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in [anchor_dir, *anchor_dir.parents]:
        for candidate in (root / SAMPLE_PACK_SUFFIX, root / "repo" / SAMPLE_PACK_SUFFIX):
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)
    return candidates


def resolve_sample_pack_dir(anchor: str | Path, explicit: str | Path | None = None) -> Path:
    if explicit:
        explicit_path = Path(explicit).resolve()
        if not has_sample_pack_files(explicit_path):
            raise FileNotFoundError(
                "Explicit sample pack directory is missing required files: "
                f"{explicit_path}"
            )
        return explicit_path

    for candidate in candidate_sample_pack_dirs(anchor):
        if has_sample_pack_files(candidate):
            return candidate

    searched = "\n".join(f"- {path}" for path in candidate_sample_pack_dirs(anchor))
    raise FileNotFoundError(
        "Could not locate the bundled sample pack. Checked:\n"
        f"{searched}\n"
        "Pass --sample-pack-dir to point at a valid pack root."
    )


def default_sample_pack_dir(anchor: str | Path) -> Path:
    anchor_dir = _anchor_dir(anchor)
    for candidate in candidate_sample_pack_dirs(anchor_dir):
        if has_sample_pack_files(candidate):
            return candidate
    for root in [anchor_dir, *anchor_dir.parents]:
        if root.name.lower() == "repo":
            return (root / SAMPLE_PACK_SUFFIX).resolve()
    return (anchor_dir / SAMPLE_PACK_SUFFIX).resolve()


def resolve_live_data_dir(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        live_root = Path(explicit).resolve()
    else:
        env_value = os.environ.get(LIVE_DATA_ENV_VAR, "").strip()
        if not env_value:
            return None
        live_root = Path(env_value).resolve()
    if not live_root.exists():
        raise FileNotFoundError(
            "Live data root does not exist. Set "
            f"{LIVE_DATA_ENV_VAR} or pass --live-data-dir to a valid local root: {live_root}"
        )
    return live_root
