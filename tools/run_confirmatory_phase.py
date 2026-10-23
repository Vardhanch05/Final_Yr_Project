"""
Confirmatory Phase Execution Runner for TabM Post-Hoc Aggregations Study.

Pre-Registration: v2.2 (October 18, 2026)
Benchmark: TabReD (8 datasets x 5 seeds = 40 training runs)
Protocol:
- Aborts on config hash mismatch (verifies against pre-registered canonical hashes).
- Retrains all 40 runs from scratch.
- Fully resumable and idempotent on disk.
- Zero inspection of intermediate test metrics during execution.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import tomli
import tomli_w

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABM_PAPER_DIR = PROJECT_ROOT / "tabm" / "paper"

DATASETS = [
    "cooking-time",
    "delivery-eta",
    "ecom-offers",
    "homecredit-default",
    "homesite-insurance",
    "maps-routing",
    "sberbank-housing",
    "weather",
]

SEEDS = [0, 1, 2, 3, 4]

# Frozen configuration SHA256 hashes (both canonical LF and Windows CRLF)
EXPECTED_HASHES = {
    "cooking-time": {
        "lf": "901bbf02fcf312d864d6f41596bbdc3dfe10ae9634dfbd16c8ddf40ba87225f3",
        "crlf": "8bd53da73578a697438fcd26f6b4c9f20507385601aed80ba9774b2e3dd08c5c",
    },
    "delivery-eta": {
        "lf": "fa83fb18969a192fb8f98e1fc5c1630a3efc66c41c6b8a9aade41b927c377214",
        "crlf": "1699a254827332ef3eec89a050f50dbfa70ebdb3967eec182f65ea9fac9734cc",
    },
    "ecom-offers": {
        "lf": "742f82218b45212976fa93eca91e03152c9d2fa9b01fc25c8a8182e47abaf2d2",
        "crlf": "2031af829016ef835175b28882cfca59d9a33e8e15a16e1fe92e6d418b01eef8",
    },
    "homecredit-default": {
        "lf": "f8c0cbbc07c3d474974b31fb143476ff7244f15b86943670c3658b1ad3df255a",
        "crlf": "54c5da07e78a69b3d1c0d129c3a4779da8ebcb3ce7c9081364b59ce21b4e6b16",
    },
    "homesite-insurance": {
        "lf": "163d767c4e5ee4e9471de1572dfbaf523d8b2a27e7c6034f2051cada8efbbe12",
        "crlf": "e37118ff0b52c9867b99d43b1d1f6b23e3aaf75c8b65f9580e05ffd8a2170866",
    },
    "maps-routing": {
        "lf": "21c176e01af048c31d8a8726aff45b90f77288c4a012e7cd61dea733cf303e9b",
        "crlf": "c05460274c476e88ca44ccc73e1f0a5e498c2f434a91705ab761a387e75692e9",
    },
    "sberbank-housing": {
        "lf": "7323132ff67b95987cd9a3ee726cfe0444627b5d3004ea97703a0a28c0441038",
        "crlf": "cb4bb523573b072d0a568913eaa06b6fda109625abcb863d8a3ea98f2fb4b3db",
    },
    "weather": {
        "lf": "c59da405b8fb311eb93e02d3a82a75e88c9864e933bb166517c24d9a646fc38e",
        "crlf": "7e99d0ddd0bf2cfe2731a5e5979fb30ed027a2fb0bdd2ad4b3ae33a62268544a",
    },
}


def verify_config_hashes() -> None:
    print("=" * 80)
    print("[Pre-Flight Check] Verifying released config SHA256 hashes...")
    print("=" * 80)
    for dataset in DATASETS:
        config_path = (
            TABM_PAPER_DIR
            / "exp"
            / "tabm"
            / "tabred"
            / dataset
            / "0-evaluation"
            / "0.toml"
        )
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file missing: {config_path}")

        raw_bytes = config_path.read_bytes()
        actual_crlf = hashlib.sha256(raw_bytes).hexdigest()
        actual_lf = hashlib.sha256(raw_bytes.replace(b"\r\n", b"\n")).hexdigest()

        expected = EXPECTED_HASHES[dataset]
        if actual_lf != expected["lf"] and actual_crlf != expected["crlf"]:
            raise ValueError(
                f"Config hash mismatch for {dataset}!\n"
                f"  Expected LF:   {expected['lf']}\n"
                f"  Actual LF:     {actual_lf}\n"
                f"  Expected CRLF: {expected['crlf']}\n"
                f"  Actual CRLF:   {actual_crlf}"
            )
        print(f"  [PASS] {dataset:<20} SHA256: {actual_lf[:16]}... verified.")
    print("=" * 80)
    print("All released configuration hashes successfully verified.\n")


def is_run_complete(output_dir: Path) -> bool:
    done_file = output_dir / "DONE"
    report_file = output_dir / "report.json"
    heads_file = output_dir / "head_predictions.npz"
    if done_file.exists() and report_file.exists() and heads_file.exists():
        try:
            with open(report_file, "r") as f:
                rep = json.load(f)
            if "time" in rep and "metrics" in rep:
                return True
        except Exception:
            return False
    return False


def run_confirmatory_phase(force_all: bool = False) -> None:
    verify_config_hashes()

    total_runs = len(DATASETS) * len(SEEDS)
    completed_runs = 0
    start_time_all = time.time()

    print(f"Starting Confirmatory Phase: {len(DATASETS)} datasets x {len(SEEDS)} seeds = {total_runs} runs.")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    for d_idx, dataset in enumerate(DATASETS, 1):
        base_config_path = (
            TABM_PAPER_DIR
            / "exp"
            / "tabm"
            / "tabred"
            / dataset
            / "0-evaluation"
            / "0.toml"
        )
        base_config = tomli.loads(base_config_path.read_text(encoding="utf-8"))

        dataset_out_base = (
            TABM_PAPER_DIR / "exp" / "tabm" / "tabred" / dataset / "confirmatory"
        )
        dataset_out_base.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            run_output_dir = dataset_out_base / str(seed)
            run_config_path = dataset_out_base / f"{seed}.toml"

            if not force_all and is_run_complete(run_output_dir):
                completed_runs += 1
                print(
                    f"[{completed_runs}/{total_runs}] [SKIP] {dataset} seed={seed} already completed."
                )
                continue

            # Write seed config
            seed_config = dict(base_config)
            seed_config["seed"] = seed
            run_config_path.write_text(tomli_w.dumps(seed_config), encoding="utf-8")

            print(
                f"\n>>> [{completed_runs + 1}/{total_runs}] Training {dataset} (seed {seed}) | "
                f"Elapsed: {(time.time() - start_time_all)/60:.1f}m"
            )

            t0 = time.time()
            rel_config_path = run_config_path.relative_to(TABM_PAPER_DIR)

            cmd = [
                sys.executable,
                "bin/model.py",
                str(rel_config_path),
                "--force",
            ]

            env = os.environ.copy()
            # Ensure Python uses utf-8 encoding for stdout/stderr
            env["PYTHONIOENCODING"] = "utf-8"

            res = subprocess.run(cmd, cwd=str(TABM_PAPER_DIR), env=env)
            if res.returncode != 0:
                raise RuntimeError(
                    f"Training run failed for {dataset} seed {seed} with return code {res.returncode}"
                )

            # Assert outputs exist
            if not is_run_complete(run_output_dir):
                raise RuntimeError(
                    f"Run completed with code 0 but required artifacts are missing in {run_output_dir}"
                )

            duration = time.time() - t0
            completed_runs += 1

            # Load report to check memory only (blind to test metrics per pre-registration protocol)
            with open(run_output_dir / "report.json", "r") as f:
                rep = json.load(f)
            peak_alloc = rep.get("peak_gpu_memory_allocated_mb", 0.0)
            peak_res = rep.get("peak_gpu_memory_reserved_mb", 0.0)

            print(
                f"[DONE] {dataset} seed={seed} finished in {duration:.1f}s ({duration/60:.2f}m). "
                f"Peak GPU: {peak_alloc:.1f} MB alloc, {peak_res:.1f} MB res."
            )

    total_time = (time.time() - start_time_all) / 60.0
    print("\n" + "=" * 80)
    print(f"CONFIRMATORY TRAINING COMPLETE: All {total_runs} runs finished successfully.")
    print(f"Total Wall-Clock Time: {total_time:.2f} minutes ({total_time/60:.2f} hours).")
    print("=" * 80)


if __name__ == "__main__":
    force_flag = "--force-all" in sys.argv
    run_confirmatory_phase(force_all=force_flag)
