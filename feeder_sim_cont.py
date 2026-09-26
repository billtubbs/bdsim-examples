"""Continuous-time feeder models for the SAG mill feeder system.

Reference:
    Sbarbaro, D., Barriga, J., Valenzuela, H., and Cortes, G.,
    "A Multi-Input-Single-Output Smith Predictor for Feeders Control in
    SAG Grinding Plants," IEEE Transactions on Control Systems Technology,
    vol. 13, no. 6, pp. 1069-1078, Nov. 2005.

This script builds a continuous-time approximation of the feeder dynamics
using a first-order lag and a Pade approximation of the belt transport delay.
The delay is computed from the feeder-to-weight-meter distance and the nominal
feeder speed, which is inferred from the published discrete-time delay and 5 s
sampling time.
"""

from __future__ import annotations

import os

import control as ct
import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Table II: feeder model parameters from the paper
K = np.array([0.3892, 0.4148, 0.4823], dtype=float)
T = np.array([-0.5919, -0.5058, -0.5048], dtype=float)
d = np.array([16, 14, 11], dtype=int)

# Table I: feeder-to-weight-meter distance (m)
distance = np.array([116.0, 99.0, 82.4], dtype=float)

sample_period = 5.0
n_samples = 120
time = sample_period * np.arange(n_samples)


def discrete_feeder_response(
    Ki: float, Ti: float, delay_steps: int, n_steps: int
) -> np.ndarray:
    """Step response of the discrete feeder model used in the paper."""
    y = np.zeros(n_steps, dtype=float)
    u = np.ones(n_steps, dtype=float)

    for k in range(n_steps):
        delayed_input = u[k - delay_steps] if k >= delay_steps else 0.0
        if k == 0:
            y[k] = Ki * delayed_input
        else:
            y[k] = -Ti * y[k - 1] + Ki * delayed_input
    return y


def nominal_feeder_speed(
    distance_m: float, delay_steps: int, sample_time: float
) -> float:
    """Infer nominal feeder speed from the delay and the sample period."""
    return distance_m / (delay_steps * sample_time)


def continuous_time_constant(Ti: float) -> float:
    """Approximate a discrete pole with the equivalent continuous-time pole."""
    return sample_period / (-np.log(-Ti))


def feeder_continuous_model(
    Ki: float,
    Ti: float,
    distance_m: float,
    speed_mps: float,
):
    """Build a CT feeder model that matches the discrete DC gain."""
    transport_delay = distance_m / speed_mps
    tau = continuous_time_constant(Ti)
    # Match the discrete step gain at steady state:
    # G(z) -> Ki / (1 + Ti) as z -> 1.
    gain = Ki / (1.0 + Ti)
    plant = ct.TransferFunction([gain], [tau, 1])
    delay_tf = ct.TransferFunction(*ct.pade(transport_delay, 2))
    return plant * delay_tf


def step_response(
    tf,
    t_end: float = 600.0,
    n_points: int = 1201,
):
    """Compute a unit-step response for a continuous-time SISO system."""
    t = np.linspace(0.0, t_end, n_points)
    t_out, y_out = ct.step_response(tf, T=t)
    return t_out, np.asarray(y_out).ravel()


nominal_speeds = np.array(
    [
        nominal_feeder_speed(dist, int(delay), sample_period)
        for dist, delay in zip(distance, d)
    ],
    dtype=float,
)

# Step responses for the discrete reference and continuous approximation.
discrete_outputs = [
    discrete_feeder_response(Ki, Ti, int(delay), n_samples)
    for Ki, Ti, delay in zip(K, T, d)
]
continuous_outputs = []
continuous_t = []
for Ki, Ti, dist, speed in zip(K, T, distance, nominal_speeds):
    model = feeder_continuous_model(Ki, Ti, dist, speed)
    t_ct, y_ct = step_response(model, t_end=600.0, n_points=1201)
    continuous_t.append(t_ct)
    continuous_outputs.append(y_ct)

# Interpolate CT responses to the discrete sampling grid for comparison.
continuous_samples = [
    np.interp(time, t_ct, y_ct)
    for t_ct, y_ct in zip(continuous_t, continuous_outputs)
]
combined_discrete = np.sum(discrete_outputs, axis=0)
combined_continuous = np.sum(continuous_samples, axis=0)

print("Nominal feeder speeds (m/s):")
for i, speed in enumerate(nominal_speeds, start=1):
    print(f"  Feeder {i}: {speed:.3f}")

print("\nMax absolute differences between CT and discrete step responses:")
for i, (y_ct, y_disc) in enumerate(
    zip(continuous_samples, discrete_outputs), start=1
):
    print(f"  Feeder {i}: {np.max(np.abs(y_ct - y_disc)):.6e}")
print(
    "  Combined: "
    f"{np.max(np.abs(combined_continuous - combined_discrete)):.6e}"
)

n = len(discrete_outputs) + 1
fig, axes = plt.subplots(n, 1, figsize=(7.5, 1 + 1.5 * n), sharex=True)
fig.suptitle(
    "Continuous-time feeder models compared with discrete-time models",
    fontsize=16,
)

for ax, y_ct, y_disc, idx in zip(
    axes[:-1], continuous_samples, discrete_outputs, range(1, 4)
):
    ax.plot(time, y_ct, linewidth=2, label="continuous")
    ax.plot(
        time,
        y_disc,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="discrete",
    )
    ax.set_ylabel("Output")
    ax.set_title(f"Feeder {idx}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

axes[-1].plot(time, combined_continuous, linewidth=2, label="continuous")
axes[-1].plot(
    time,
    combined_discrete,
    linestyle="--",
    linewidth=1.5,
    alpha=0.8,
    label="discrete",
)
axes[-1].set_title("Combined feed to SAG mill")
axes[-1].set_xlabel("Time (s)")
axes[-1].set_ylabel("Feed")
axes[-1].grid(True, alpha=0.3)
axes[-1].legend(loc="upper left")

for ax in axes:
    ax.set_xlim(0, time[-1])

plt.tight_layout(rect=[0, 0, 1, 0.97])
output_path = os.path.join(PLOT_DIR, "feeder_system_outputs_continuous.png")
plt.savefig(output_path, dpi=200)
print(f"\nSaved continuous-time comparison plot to {output_path}")
plt.show()
