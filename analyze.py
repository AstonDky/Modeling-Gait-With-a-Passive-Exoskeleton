"""Reproduce the model comparisons and figures from the English project report."""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from dynamics import baseline_transient, condition_angles
from figures import plot_angles_and_posture, plot_force, plot_muscle_and_torque, plot_transient


def analyze(data, output):
    profiles = pd.read_csv(data / "phase_profiles.csv")
    summary = pd.read_csv(data / "condition_summary.csv")
    periods = summary.set_index("condition")["period_mean_s"]
    transient = baseline_transient(profiles, periods)
    angles = condition_angles(profiles, periods)

    output.mkdir(parents=True, exist_ok=True)
    transient.to_csv(output / "baseline_transient.csv", index=False)
    angles.to_csv(output / "joint_angles.csv", index=False)
    plot_force(profiles, output)
    plot_muscle_and_torque(profiles, summary, output)
    plot_transient(transient, output)
    plot_angles_and_posture(angles, output)

    print(f'{int(summary["cycles_used"].sum())} strides')
    for condition in ("pre", "exo", "post"):
        print(f"  {condition}: {periods.loc[condition]:.2f} s per stride")
    bare = transient["ankle_no_exo_deg"].max()
    worn = transient["ankle_with_exo_deg"].max()
    print(f"Baseline torque: peak model ankle angle {bare:.2f} deg without the device, "
          f"{worn:.2f} deg with it.")
    print(f"Results written to {output.resolve()}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data", help="Condition summary CSVs")
    parser.add_argument("--output", type=Path, default=root / "results", help="Figure and table directory")
    args = parser.parse_args()
    analyze(args.data, args.output)
