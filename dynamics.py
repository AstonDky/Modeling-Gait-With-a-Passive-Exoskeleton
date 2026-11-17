"""Sagittal-plane joint dynamics driven by deformation-derived torque indices."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Joint:
    muscle: str
    mass: float
    length: float
    frame_mass: float
    damping: float
    stiffness: float
    torque_gain: float


# Both sides have identical mechanical parameters. The measured left and right
# muscle channels for each joint are averaged before driving the model.
JOINTS = {
    "hip": Joint("GM", 7.02, 0.40, 0.44, 35.0, 63.66, 63.66),
    "knee": Joint("RF", 2.44, 0.37, 0.59, 15.0, 79.57, 79.57),
    "ankle": Joint("MG", 1.18, 0.17, 0.00, 3.0, 33.66, 33.66),
}
HEEL_SPRING_N_PER_M = 7500.0


def solve_response(joint, with_device, phase, torque, period, times, max_step=np.inf):
    params = JOINTS[joint]
    mass = params.mass + (params.frame_mass if with_device else 0.0)
    inertia = mass * params.length**2 / 3.0
    stiffness = params.stiffness
    if with_device and joint == "ankle":
        stiffness += HEEL_SPRING_N_PER_M * params.length**2

    def derivative(t, state):
        theta, omega = state
        drive = np.interp((t % period) / period * 100.0, phase, torque)
        return omega, (drive - params.damping * omega - stiffness * theta) / inertia

    sol = solve_ivp(derivative, (0.0, times[-1]), (0.0, 0.0), t_eval=times,
                    rtol=1e-8, atol=1e-10, max_step=max_step)
    if not sol.success:
        raise RuntimeError(f"{joint}: {sol.message}")
    return np.rad2deg(sol.y[0])


def baseline_transient(profiles, periods):
    baseline = profiles.loc[profiles["condition"] == "pre"].sort_values("percent")
    period = periods.loc["pre"]
    phase = baseline["percent"].to_numpy()
    times = np.linspace(0.0, 2.0, 900)
    transient = {"time_s": times}
    for joint, params in JOINTS.items():
        torque = baseline[f"{params.muscle}_mean"].to_numpy() * params.torque_gain
        transient[f"{joint}_torque_Nm"] = np.interp(times % period / period * 100, phase, torque)
        for name, with_device in (("no_exo", False), ("with_exo", True)):
            transient[f"{joint}_{name}_deg"] = solve_response(
                joint, with_device, phase, torque, period, times, max_step=period / 100,
            )
    return pd.DataFrame(transient)


def condition_angles(profiles, periods):
    frames = []
    for condition, part in profiles.groupby("condition", sort=False):
        part = part.sort_values("percent")
        period = periods.loc[condition]
        phase = part["percent"].to_numpy()
        times = np.linspace(4 * period, 5 * period, len(phase))
        angles = {"condition": condition, "percent": phase}
        for joint, params in JOINTS.items():
            torque = part[f"{params.muscle}_mean"].to_numpy() * params.torque_gain
            angles[f"{joint}_angle_deg"] = solve_response(
                joint, condition == "exo", phase, torque, period, times,
            )
        frames.append(pd.DataFrame(angles))
    return pd.concat(frames, ignore_index=True)
