"""Small CPU demonstration with synthetic market inputs; no workbook required."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autocallable_direct import equicorrelation, parse_research_contract, simulate_direct_replication
from autocallable_bb import simulate_conditioned_replication, simulate_direct_rqmc_replication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=512)
    parser.add_argument("--replications", type=int, default=4)
    parser.add_argument("--methods", nargs="+", choices=["M0", "M1", "M2", "M3"], default=["M0", "M1", "M2", "M3"])
    args = parser.parse_args()
    if args.paths < 2 or args.paths & (args.paths - 1):
        parser.error("--paths must be a power of two, at least 2")
    if args.replications < 2:
        parser.error("--replications must be at least 2 to estimate replication uncertainty")
    config = json.loads((ROOT / "config/core_project_config.json").read_text(encoding="utf-8"))
    contract = parse_research_contract(config, "RC-A")
    common = dict(contract=contract, risk_free_rate=0.03,
                  dividend_yields=np.array([0.01, 0.005, 0.012]),
                  volatilities=np.array([0.18, 0.26, 0.22]),
                  correlation=equicorrelation(0.4), annual_coupon=0.0875,
                  n_paths=args.paths)
    rows = []
    for method in dict.fromkeys(args.methods):
        for replication in range(args.replications):
            seed = 20260920 + replication
            if method == "M0":
                result = simulate_direct_replication(**common, steps_per_year=52, seed=seed)
            elif method == "M1":
                result = simulate_direct_rqmc_replication(**common, steps_per_year=52, seed=seed)
            else:
                result = simulate_conditioned_replication(
                    **common, outer_method="mc" if method == "M2" else "rqmc",
                    seed=seed, bridge_bank_seed=30260920 + replication,
                    inner_paths=64, bridge_substeps=8, paired_bernoulli=False)
            for key in ["component_identity_error", "probability_mass_error", "fair_coupon_residual"]:
                if not np.isfinite(result[key]) or abs(result[key]) > 1e-10:
                    raise RuntimeError(f"{method} failed {key}: {result[key]}")
            if not np.isfinite(result["total_value"]):
                raise RuntimeError(f"{method} produced a non-finite value")
            rows.append(dict(method=method, replication=replication,
                             total_value=result["total_value"], fair_coupon=result["fair_coupon"],
                             runtime_seconds=result["runtime_seconds"]))
    frame = pd.DataFrame(rows)
    summary = frame.groupby("method").agg(
        value_per_100=("total_value", "mean"), replication_sd=("total_value", "std"),
        fair_coupon=("fair_coupon", "mean"), total_seconds=("runtime_seconds", "sum"))
    summary["standard_error"] = summary["replication_sd"] / np.sqrt(args.replications)
    out = ROOT / "outputs/demo"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "replications.csv", index=False)
    summary.to_csv(out / "summary.csv")
    print("SYNTHETIC INPUTS - smoke demonstration, not the report's market calibration.")
    print("M0/M1 use a finite monitoring grid; M2/M3 use a small nested bridge budget.")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
