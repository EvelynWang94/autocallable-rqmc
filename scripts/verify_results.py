"""Check the archived Day 9 inventory and recompute its reference statistics."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from day9_analysis import reference_summary


def main():
    folder = ROOT / "results/day9_main_experiments"
    expected = {"baseline_replications.csv": 256, "reference_replications.csv": 24,
                "regime_replications.csv": 352, "trigger_smoothing_replications.csv": 480}
    for filename, count in expected.items():
        actual = len(pd.read_csv(folder / filename))
        if actual != count:
            raise ValueError(f"{filename}: expected {count} rows, found {actual}")
    recomputed = reference_summary(pd.read_csv(folder / "reference_replications.csv")).set_index("metric")
    saved = pd.read_csv(folder / "reference_summary.csv").set_index("metric")
    for column in ["reference_mean", "reference_sd", "reference_se"]:
        np.testing.assert_allclose(recomputed.loc[saved.index, column], saved[column], rtol=1e-9, atol=1e-12)
    print("Archived inventory: 1,112 replications; saved reference mean, SD and SE reproduced.")
    print("This checks saved evidence; it does not rerun the full simulation or GPU timings.")


if __name__ == "__main__":
    main()
