"""Build condition summaries from paired muscle and plantar-force recordings."""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from dynamics import JOINTS

FS_MUSCLE = 100.0
POINTS = 101
channels = [f"u{i}" for i in range(1, 7)]
conditions = ["pre", "exo", "post"]


def trial_info(path):
    m = re.search(r"(pre|exo|post)(\d+)", path.stem)
    return f"{m.group(1)}{m.group(2)}", m.group(1), int(m.group(2))


def read_muscle(path):
    with path.open(encoding="utf-8-sig") as source:
        header = next(i for i, line in enumerate(source) if line.startswith("Index,"))
    raw = pd.read_csv(path, skiprows=header, encoding="utf-8-sig")
    raw = raw.rename(columns={str(i): ch for i, ch in enumerate(channels, start=1)})
    label, cond, num = trial_info(path)
    df = raw[["Index"] + channels].copy()
    df[channels] = df[channels].astype(float)
    df["time_s"] = df["Index"] / FS_MUSCLE
    df["trial"] = label
    df["condition"] = cond
    df["number"] = num
    return df


def read_force(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        items = line.split()
        if len(items) == 55:
            rows.append([float(item) for item in items])

    data = np.array(rows, dtype=float)
    time_s = (data[:, -1] - data[0, -1]) / 1000.0
    sensor = data[:, :-1].reshape(len(data), 18, 3)

    base_n = max(10, min(len(sensor) // 10, 40))
    base = np.median(sensor[:base_n], axis=0)
    sensor = sensor - base
    load = np.linalg.norm(sensor, axis=2).sum(axis=1)
    load = load - load.min()

    dt = np.median(np.diff(time_s))
    win = max(3, int(round(0.16 / dt)))
    win += 1 - win % 2
    load = np.convolve(load, np.ones(win) / win, mode="same")
    return time_s, load


def find_cycles(time_s, load, limit_time):
    dt = np.median(np.diff(time_s))
    distance = max(1, int(0.55 / dt))
    peaks = find_peaks(load, distance=distance, prominence=0.03)[0]
    cycles = []
    for i in range(len(peaks) - 1):
        start = time_s[peaks[i]]
        stop = time_s[peaks[i + 1]]
        period = stop - start
        if 0.75 <= period <= 1.80 and stop <= limit_time:
            cycles.append((start, stop, period))
    return cycles


def resample(time_s, y, start, stop):
    x = np.linspace(start, stop, POINTS)
    return np.interp(x, time_s, y)


def sort_trial(label):
    m = re.search(r"(pre|exo|post)(\d+)", label)
    return conditions.index(m.group(1)), int(m.group(2))


def process(source, output):
    muscle = {trial_info(p)[0]: read_muscle(p) for p in sorted((source / "肌肉形变").glob("*.csv"))}
    if not muscle:
        raise ValueError(f"No muscle CSV files found in {source / '肌肉形变'}")
    mins = pd.concat([df[channels] for df in muscle.values()]).min()
    maxs = pd.concat([df[channels] for df in muscle.values()]).max()
    for df in muscle.values():
        df.loc[:, channels] = (df[channels] - mins) / (maxs - mins)

    force = {}
    for path in sorted((source / "足底力").glob("*-[LR].txt")):
        label, _, _ = trial_info(path)
        side = path.stem.rsplit("-", 1)[1]
        force.setdefault(label, {})[side] = read_force(path)

    curve_rows = []
    trial_rows = []
    percent = np.linspace(0, 100, POINTS)
    for label in sorted(muscle, key=sort_trial):
        if label not in force or set(force[label]) != {"L", "R"}:
            raise ValueError(f"Both foot recordings are required for {label}")

        cond = muscle[label]["condition"].iloc[0]
        number = int(muscle[label]["number"].iloc[0])
        t_l, load_l_raw = force[label]["L"]
        t_r, load_r_raw = force[label]["R"]
        scale = max(load_l_raw.max(), load_r_raw.max())
        load_l = load_l_raw / scale
        load_r = load_r_raw / scale

        t_m = muscle[label]["time_s"].to_numpy()
        limit_time = min(t_l[-1], t_r[-1], t_m[-1])
        cycles = find_cycles(t_r, load_r, limit_time)

        trial_periods = []
        trial_left = []
        trial_right = []
        for cycle_id, (start, stop, period) in enumerate(cycles, start=1):
            trial_periods.append(period)
            left_cycle = resample(t_l, load_l, start, stop)
            right_cycle = resample(t_r, load_r, start, stop)
            trial_left.append(left_cycle.mean())
            trial_right.append(right_cycle.mean())

            row_base = {
                "trial": label,
                "condition": cond,
                "number": number,
                "cycle": cycle_id,
                "period_s": period,
            }
            muscle_cycle = {}
            for ch in channels:
                muscle_cycle[ch] = resample(t_m, muscle[label][ch].to_numpy(), start, stop)

            for j, p in enumerate(percent):
                row = row_base.copy()
                row["percent"] = p
                row["load_left"] = left_cycle[j]
                row["load_right"] = right_cycle[j]
                row["load_total"] = left_cycle[j] + right_cycle[j]
                for ch in channels:
                    row[ch] = muscle_cycle[ch][j]
                curve_rows.append(row)

        if not cycles:
            raise ValueError(f"No valid strides detected for {label}")
        left_mean = np.mean(trial_left)
        right_mean = np.mean(trial_right)
        trial_rows.append(
            {
                "trial": label,
                "condition": cond,
                "number": number,
                "cycles_used": len(cycles),
                "period_mean_s": np.mean(trial_periods),
                "period_sd_s": np.std(trial_periods, ddof=1) if len(trial_periods) > 1 else 0.0,
                "left_load_mean": left_mean,
                "right_load_mean": right_mean,
                "right_minus_left_percent": 100 * (right_mean - left_mean) / (right_mean + left_mean),
                "load_asymmetry_percent": 100 * abs(right_mean - left_mean) / (right_mean + left_mean),
            }
        )

    cycle_curves = pd.DataFrame(curve_rows)
    trial_summary = pd.DataFrame(trial_rows)

    for muscle, pair in {"GM": ["u1", "u2"], "RF": ["u3", "u4"], "MG": ["u5", "u6"]}.items():
        cycle_curves[muscle] = cycle_curves[pair].mean(axis=1)
    signals = ["load_left", "load_right", "GM", "RF", "MG"]
    phase = cycle_curves.groupby(["condition", "percent"])[signals].agg(["mean", "std"])
    phase.columns = ["_".join(column) for column in phase.columns]
    summary = trial_summary.groupby("condition").agg(
        cycles_used=("cycles_used", "sum"),
        period_mean_s=("period_mean_s", "mean"),
        left_load_mean=("left_load_mean", "mean"),
        right_load_mean=("right_load_mean", "mean"),
        load_asymmetry_percent=("load_asymmetry_percent", "mean"),
    )
    # The report weights strides equally for torque and trials equally for stride period.
    stride_means = cycle_curves.groupby(["condition", "trial", "cycle"])[["GM", "RF", "MG"]].mean()
    for joint, params in JOINTS.items():
        torque = stride_means[params.muscle] * params.torque_gain
        grouped = torque.groupby(level="condition")
        summary[f"{joint}_tau_mean_Nm"] = grouped.mean()
        summary[f"{joint}_tau_sd_Nm"] = grouped.std()

    output.mkdir(parents=True, exist_ok=True)
    phase.to_csv(output / "phase_profiles.csv")
    summary.loc[conditions].to_csv(output / "condition_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path, help="Folder containing 肌肉形变 and 足底力")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the two summary CSVs")
    args = parser.parse_args()
    process(args.recordings, args.output)
