"""Plot the paper's condition averages, model comparisons and gait postures."""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "lines.linewidth": 1.15,
        "axes.unicode_minus": False,
    }
)

CONDITIONS = ["pre", "exo", "post"]
CONDITION_LABELS = {
    "pre": "Baseline",
    "exo": "With exoskeleton",
    "post": "Post-removal",
}
CONDITION_COLORS = {"pre": "#1f5f99", "exo": "#c43c39", "post": "#2f8f5b"}


def save(fig, output, name):
    fig.savefig(output / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transient(signals, output):
    fig, axes = plt.subplots(3, 3, figsize=(6.85, 3.35), dpi=300, sharex=True)
    time = signals["time_s"]
    for row, joint in enumerate(("hip", "knee", "ankle")):
        axes[row, 0].plot(time, signals[f"{joint}_torque_Nm"], color="#384b52")
        axes[row, 0].set_ylabel(joint.title())
        bare = signals[f"{joint}_no_exo_deg"]
        worn = signals[f"{joint}_with_exo_deg"]
        axes[row, 1].plot(time, bare, color="#1f5f99", label="No exoskeleton")
        axes[row, 1].plot(time, worn, color="#c43c39", linestyle="--", label="With exoskeleton")
        axes[row, 2].axhline(0, color="#999999", linewidth=0.6)
        axes[row, 2].plot(time, worn - bare, color="#555555")
        for ax in axes[row]:
            ax.margins(x=0)
            ax.grid(alpha=0.22)
    axes[0, 0].set_title("Baseline torque index (N m)")
    axes[0, 1].set_title("Model angle response (deg)")
    axes[0, 2].set_title("With minus without (deg)")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2)
    fig.subplots_adjust(left=0.075, right=0.987, top=0.93, bottom=0.20, wspace=0.33, hspace=0.23)
    save(fig, output, "transient_response.png")


def plot_force(curves, output):
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 2.55), dpi=300, sharex=True, constrained_layout=True)
    for ax, condition in zip(axes, CONDITIONS):
        subset = curves[curves["condition"] == condition]
        for side, color, label in [("load_left", "#4e79a7", "Left foot"), ("load_right", "#e15759", "Right foot")]:
            mean = subset[f"{side}_mean"].to_numpy()
            sd = subset[f"{side}_std"].to_numpy()
            x = subset["percent"].to_numpy()
            ax.plot(x, mean, color=color, label=label)
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
        ax.text(0.02, 0.82, CONDITION_LABELS[condition], transform=ax.transAxes, fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[1].set_ylabel("Normalized plantar-load index", labelpad=4)
    axes[-1].set_xlabel("Gait cycle (%)")
    save(fig, output, "force_cycle.png")


def plot_muscle_and_torque(curves, summary, output):

    fig, axes = plt.subplots(1, 3, figsize=(4.2, 2.35), dpi=300, sharey=True, constrained_layout=True)
    muscle_colors = {"GM": "#1f5f99", "RF": "#d28b26", "MG": "#2f8f5b"}
    for ax, condition in zip(axes, CONDITIONS):
        subset = curves[curves["condition"] == condition]
        for muscle in ["GM", "RF", "MG"]:
            mean = subset[f"{muscle}_mean"].to_numpy()
            sd = subset[f"{muscle}_std"].to_numpy()
            x = subset["percent"].to_numpy()
            ax.plot(x, mean, color=muscle_colors[muscle], label=muscle)
            ax.fill_between(x, mean - sd, mean + sd, color=muscle_colors[muscle], alpha=0.14, linewidth=0)
        ax.set_title(CONDITION_LABELS[condition])
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Normalized deformation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    save(fig, output, "muscle_normalized.png")

    fig, ax = plt.subplots(figsize=(3.5, 2.65), dpi=300, constrained_layout=True)
    joints = [("hip_tau", "Hip"), ("knee_tau", "Knee"), ("ankle_tau", "Ankle")]
    x = np.arange(len(joints))
    width = 0.24
    for index, condition in enumerate(CONDITIONS):
        row = summary[summary["condition"] == condition].iloc[0]
        y = np.array([row[f"{joint}_mean_Nm"] for joint, _ in joints])
        error = np.array([row[f"{joint}_sd_Nm"] for joint, _ in joints])
        ax.bar(x + (index - 1) * width, y, width, yerr=error, capsize=2.2, color=CONDITION_COLORS[condition], label=CONDITION_LABELS[condition])
    ax.set_xticks(x, [label for _, label in joints])
    ax.set_ylabel("Mean equivalent torque (N m)")
    ax.grid(axis="y", alpha=0.25)
    upper = max((summary[f"{joint}_mean_Nm"] + summary[f"{joint}_sd_Nm"]).max() for joint, _ in joints)
    ax.set_ylim(0, upper + 8)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, output, "torque_comparison.png")


def plot_angles_and_posture(angles, output):
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 2.55), dpi=300, sharex=True, constrained_layout=True)
    for ax, joint, title in zip(axes, ["hip", "knee", "ankle"], ["Hip response", "Knee response", "Ankle response"]):
        for condition in CONDITIONS:
            subset = angles[angles["condition"] == condition]
            ax.plot(subset["percent"], subset[f"{joint}_angle_deg"], color=CONDITION_COLORS[condition], label=CONDITION_LABELS[condition])
        ax.set_title(title, loc="left", pad=1)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    axes[1].set_ylabel("Equivalent angle response (deg)", labelpad=4)
    axes[-1].set_xlabel("Gait cycle (%)")
    save(fig, output, "joint_angle_curves.png")

    fig, ax = plt.subplots(figsize=(5.2, 2.25), dpi=300)
    phases = [0, 25, 50, 75, 100]
    lengths = np.array([0.40, 0.37, 0.17])
    depth_by_condition = {"pre": -0.06, "exo": 0.0, "post": 0.06}
    depth_x, depth_z = 0.75, 0.32
    for phase_index, phase in enumerate(phases):
        x_offset = phase_index * 0.70
        for condition in CONDITIONS:
            subset = angles[angles["condition"] == condition].set_index("percent")
            row = subset.iloc[np.abs(subset.index.to_numpy() - phase).argmin()]
            thigh = np.radians(-90 + row["hip_angle_deg"])
            shank = thigh - np.radians(row["knee_angle_deg"])
            foot = shank + np.radians(90 + row["ankle_angle_deg"])
            hip = np.array([0.0, 0.0])
            knee = hip + lengths[0] * np.array([np.cos(thigh), np.sin(thigh)])
            ankle = knee + lengths[1] * np.array([np.cos(shank), np.sin(shank)])
            toe = ankle + lengths[2] * np.array([np.cos(foot), np.sin(foot)])
            points = np.vstack([hip, knee, ankle, toe])
            points[:, 0] += x_offset
            depth = depth_by_condition[condition]
            points[:, 0] += depth_x * depth
            points[:, 1] += depth_z * depth
            ax.plot(points[:, 0], points[:, 1], "-o", color=CONDITION_COLORS[condition], ms=2.5, linewidth=1.35, label=CONDITION_LABELS[condition] if phase_index == 0 else None)
        ground_x = np.array([x_offset - 0.22, x_offset + 0.34])
        ground_y = np.array([-0.84, -0.84])
        ax.plot(ground_x, ground_y, color="#b8b8b8", linewidth=0.7)
        ax.plot(ground_x + depth_x * 0.08, ground_y + depth_z * 0.08, color="#d0d0d0", linewidth=0.6)
        ax.text(x_offset + 0.02, 0.09, f"{phase}%", ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-0.24, 3.35)
    ax.set_ylim(-0.90, 0.16)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.00))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.15)
    save(fig, output, "gait_posture.png")
