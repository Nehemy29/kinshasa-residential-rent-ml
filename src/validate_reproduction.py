from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    difference = abs(actual - expected)
    if difference > tolerance:
        raise AssertionError(
            f"{label}: expected {expected:.4f} ± {tolerance:.4f}, "
            f"obtained {actual:.4f}."
        )
    print(f"PASS {label}: {actual:.4f}")


summary = pd.read_csv(RESULTS / "model_and_baseline_summary.csv")
paired = pd.read_csv(RESULTS / "paired_tests.csv")
lono = pd.read_csv(RESULTS / "leave_one_neighborhood_out.csv")

extra = summary.loc[summary["Method"] == "Extra Trees"].iloc[0]
close(extra["Mean_MAE_USD"], 11.87, 0.03, "Extra Trees repeated-CV MAE")
close(extra["Mean_R2"], 0.993, 0.002, "Extra Trees repeated-CV R2")

baseline = paired.loc[
    paired["Comparator"] == "Neighborhood-by-housing-type median"
].iloc[0]
close(baseline["Extra_Trees_MAE"], 10.56, 0.03, "Extra Trees fixed OOF MAE")
close(baseline["Comparator_MAE"], 44.00, 0.03, "Strongest baseline fixed OOF MAE")
close(baseline["Relative_gain_pct"], 76.0, 0.2, "Relative MAE reduction")

extra_lono = lono.loc[lono["Method"] == "Extra Trees"]
expected = {
    "Kingabwa": 94.44,
    "Mombele": 82.99,
    "Résidentiel": 261.55,
}
for neighborhood, expected_mae in expected.items():
    row = extra_lono.loc[extra_lono["Held_out_neighborhood"] == neighborhood].iloc[0]
    close(row["MAE_USD"], expected_mae, 0.05, f"LONO MAE {neighborhood}")

print("All principal manuscript results were reproduced successfully.")

