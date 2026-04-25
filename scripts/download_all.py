"""One-shot orchestrator: download every dataset needed to train.

Usage:
    python scripts/download_all.py                  # full pipeline (~overnight, includes HLS chips)
    python scripts/download_all.py --tier mini      # NASS + USDM only (~5 min, no Earth Engine)
    python scripts/download_all.py --tier quick     # enough to train an XGBoost model (~2 hr)
    python scripts/download_all.py --tier full      # everything incl. HLS chips for Prithvi (~overnight)
    python scripts/download_all.py --force          # re-run even if outputs exist
    python scripts/download_all.py --skip hls       # skip specific step(s)
    python scripts/download_all.py --only nass      # run only specific step(s)
    python scripts/download_all.py --dry-run        # print plan, run nothing

Tiers:
    mini   = nass, usdm                                          → labels + drought, no EE needed
    quick  = mini + gridmet, landsat, labels, features, check    → trains an analog-feature XGBoost
    full   = quick + gnatsgo, hls_ndvi, hls                      → enables Prithvi+LoRA fine-tune

Each step writes to a known parquet/zarr path; if that path exists and --force is
not passed, the step is skipped (idempotent).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass
class Step:
    key: str
    label: str
    script: str                          # path relative to ROOT
    output: Path                         # canonical output to test for completeness
    needs_env: tuple[str, ...] = ()      # required env vars
    est: str = "?"                       # rough wall-clock estimate

STEPS: list[Step] = [
    Step("nass",     "NASS QuickStats yield + crop progress",
         "scripts/data/fetch_nass.py",
         ROOT / "data/raw/nass/yield_county.parquet",
         needs_env=("NASS_API_KEY",), est="~3 min"),
    Step("usdm",     "US Drought Monitor weekly county stats",
         "scripts/data/fetch_usdm.py",
         ROOT / "data/raw/usdm/county_weekly.parquet",
         est="~2 min"),
    Step("gnatsgo",  "gNATSGO soil county aggregates (GEE)",
         "scripts/data/fetch_gnatsgo.py",
         ROOT / "data/raw/gnatsgo/county_soil.parquet",
         needs_env=("EE_PROJECT",), est="~10 min"),
    Step("gridmet",  "gridMET county-day weather (GEE)",
         "scripts/data/export_gridmet.py",
         ROOT / "data/raw/gridmet/county_daily.parquet",
         needs_env=("EE_PROJECT",), est="~30 min"),
    Step("landsat",  "Landsat C2 NDVI backfill 2005-2014 (GEE)",
         "scripts/data/export_landsat_c2_backfill.py",
         ROOT / "data/raw/landsat/county_month_vi.parquet",
         needs_env=("EE_PROJECT",), est="~1 hr"),
    Step("hls_ndvi", "HLS county-month NDVI 2015-2025 (GEE)",
         "scripts/data/export_hls_ndvi.py",
         ROOT / "data/processed/features/hls_county_month_ndvi.parquet",
         needs_env=("EE_PROJECT",), est="~1 hr"),
    Step("hls",      "HLS Prithvi chips 2015-2025 (GEE, HEAVY)",
         "scripts/data/export_hls_chips.py",
         ROOT / "data/processed/chips",          # directory; idempotent inside script
         needs_env=("EE_PROJECT",), est="~overnight"),
    Step("labels",   "Build chip metadata + label tables",
         "scripts/training/build_labels_metadata.py",
         ROOT / "data/processed/labels/chip_metadata.parquet",
         est="~2 min"),
    Step("features", "Build state-checkpoint analog-matching features",
         "scripts/training/build_features.py",
         ROOT / "data/processed/features/state_checkpoint_features.parquet",
         est="~5 min"),
    Step("check",    "Sanity-check all data joins",
         "notebooks/01_sanity_check.py",
         ROOT / "data/processed/features/state_checkpoint_features.parquet",  # re-uses
         est="~1 min"),
]

# Tier presets: which step keys belong to each tier.
# mini  = absolute minimum to have yield labels + drought signal (no Earth Engine).
# quick = enough to train an analog-features XGBoost model end-to-end.
# full  = everything including the overnight HLS chip export for Prithvi fine-tuning.
TIERS: dict[str, list[str]] = {
    "mini":  ["nass", "usdm"],
    "quick": ["nass", "usdm", "gridmet", "landsat", "labels", "features", "check"],
    "full":  [s.key for s in STEPS],
}


# ---------- pretty printing ----------
def hr(s: str) -> None: print(f"\n{'═' * 8} {s} {'═' * 8}")
def fmt(t: float) -> str: return f"{int(t // 60)}m{int(t % 60):02d}s"


def output_present(step: Step) -> bool:
    if step.output.is_dir():
        # For chip dir, "present" = at least one zarr exists
        return any(step.output.rglob("*.zarr"))
    return step.output.exists()


def check_env(step: Step) -> list[str]:
    return [v for v in step.needs_env if not os.getenv(v)]


def run_step(step: Step, dry: bool) -> bool:
    hr(f"{step.key}  {step.label}  ({step.est})")
    missing = check_env(step)
    if missing:
        print(f"  SKIP — missing env: {missing}. See .env.example")
        return False
    cmd = [sys.executable, str(ROOT / step.script)]
    print(f"  $ {' '.join(cmd)}")
    if dry:
        print("  (dry-run, not executing)")
        return True
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    if rc != 0:
        print(f"  FAILED in {fmt(dt)} (exit {rc})")
        return False
    print(f"  done in {fmt(dt)}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), default="full",
                    help="preset bundle of steps (default: full). mini=NASS+USDM only; "
                         "quick=enough to train an analog XGBoost; full=everything incl. HLS chips")
    ap.add_argument("--force", action="store_true", help="re-run steps even if outputs exist")
    ap.add_argument("--skip", action="append", default=[], help="step key(s) to skip")
    ap.add_argument("--only", action="append", default=[], help="run only these step key(s)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tier_keys = set(TIERS[args.tier])

    # Pre-flight: print plan
    hr(f"PLAN  (tier={args.tier})")
    selected = []
    for s in STEPS:
        if args.only and s.key not in args.only:    status = "skip (--only)"
        elif s.key in args.skip:                    status = "skip (--skip)"
        elif s.key not in tier_keys:                status = f"skip (not in tier={args.tier})"
        elif not args.force and output_present(s):  status = "skip (output exists)"
        else:
            status = "run"
            selected.append(s)
        print(f"  [{status:28s}] {s.key:9s} {s.label}")

    if not selected:
        print("\nNothing to do.")
        return

    # Pre-flight: verify GEE auth once
    if any("EE_PROJECT" in s.needs_env for s in selected):
        try:
            import ee  # noqa: F401
            ee.Initialize(project=os.getenv("EE_PROJECT"))
            print(f"\nGEE OK (project={os.getenv('EE_PROJECT')})")
        except Exception as e:
            print(f"\nGEE init failed: {e}")
            print("Run: earthengine authenticate")
            sys.exit(2)

    # Execute
    overall_t0 = time.time()
    failed: list[str] = []
    for s in selected:
        if not run_step(s, args.dry_run):
            failed.append(s.key)
    hr("SUMMARY")
    print(f"  total wall: {fmt(time.time() - overall_t0)}")
    print(f"  ran: {len(selected) - len(failed)}/{len(selected)}")
    if failed:
        print(f"  failed: {failed}")
        sys.exit(1)
    print("  All steps complete.")


if __name__ == "__main__":
    main()
