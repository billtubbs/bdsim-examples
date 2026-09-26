"""Simulation of the three ore feeder models feeding a SAG mill.

Reference:
    Sbarbaro, D., Barriga, J., Valenzuela, H., and Cortes, G.,
    "A Multi-Input-Single-Output Smith Predictor for Feeders Control in
    SAG Grinding Plants," IEEE Transactions on Control Systems Technology,
    vol. 13, no. 6, pp. 1069-1078, Nov. 2005.

Section A describes the feeder system as three variable-speed feeders feeding a
constant-speed conveyor that delivers ore to the SAG mill. The belt travel time
from each feeder discharge to the weight meter is reflected by the delay term
z^-d_i in the feeder model.

The normalized feeder output for each feeder is modeled as

    G_i(z) = K_i z^-d_i / (1 + T_i z^-1),  i = 1, 2, 3

with a unit step input and a 5 s sample period. The total feed to the SAG mill is
the sum of the three normalized feeder outputs.
"""

# I generally do not use type annotations in my work.

import os

import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Table II: feeder model parameters
K = np.array([0.3892, 0.4148, 0.4823], dtype=float)
T = np.array([-0.5919, -0.5058, -0.5048], dtype=float)
d = np.array([16, 14, 11], dtype=int)

# Table I: feeder-to-weight-meter distance (converted from meters to delay steps at
# 5 s sampling time; the transport delays are represented by z^-d_i in the model)
distance = np.array([116.0, 99.0, 82.4], dtype=float)

sample_period = 5.0
n_samples = 120
time = sample_period * np.arange(n_samples)


def feeder_response(Ki, Ti, delay_steps, n_steps):
    """Step response of one discrete feeder model."""
    y = np.zeros(n_steps, dtype=float)
    u = np.ones(n_steps, dtype=float)

    for k in range(n_steps):
        delayed_input = u[k - delay_steps] if k >= delay_steps else 0.0
        if k == 0:
            y[k] = Ki * delayed_input
        else:
            y[k] = -Ti * y[k - 1] + Ki * delayed_input
    return y


outputs = [
    feeder_response(Ki, Ti, int(delay), n_samples)
    for Ki, Ti, delay in zip(K, T, d)
]
combined = np.sum(outputs, axis=0)

n = len(outputs) + 1
fig, axes = plt.subplots(n, 1, figsize=(7.5, 1 + 1.5 * n), sharex=True)
fig.suptitle("Feeder system outputs and SAG mill feed", fontsize=16)

for ax, y, idx in zip(axes[:-1], outputs, range(1, 4)):
    ax.plot(time, y, linewidth=2)
    ax.set_ylabel("Output")
    ax.set_title(f"Feeder {idx}")
    ax.grid(True, alpha=0.3)

axes[-1].plot(time, combined, color="k", linewidth=2.5)
axes[-1].set_title("Combined feed to SAG mill")
axes[-1].set_xlabel("Time (s)")
axes[-1].set_ylabel("Feed")
axes[-1].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlim(0, time[-1])

plt.tight_layout(rect=[0, 0, 1, 0.97])
output_path = os.path.join(PLOT_DIR, "feeder_system_outputs.png")
plt.savefig(output_path, dpi=200)
print(f"Saved plot to {output_path}")
plt.show()

# --------------------------------------------------------------------
# Exact-delay speed scenarios.
# For each feeder, use a slow (0.5x), medium (1.0x), and fast (2.0x)
# nominal speed. Since delay scales as 1 / speed at fixed distance,
# these give delays of 2d, d, and d/2 samples.
# --------------------------------------------------------------------
from itertools import product

slow_delay = 2 * d
medium_delay = d
fast_delay = np.rint(d / 2).astype(int)

mode_delay = {
    "slow": slow_delay,
    "medium": medium_delay,
    "fast": fast_delay,
}

speed_by_mode = {
    mode: distance / (delay * sample_period)
    for mode, delay in mode_delay.items()
}

# Top three panels: show the three speed scenarios for each feeder.
# Bottom panel: show all combinatorial combinations with a single low-alpha
# color to reveal the spread of possible combined outputs.
fig, axes = plt.subplots(4, 1, figsize=(7.5, 1 + 1.5 * 4), sharex=True)
fig.suptitle("Feeder speed scenario sweep", fontsize=16)

for feeder_idx in range(3):
    for mode in ("slow", "medium", "fast"):
        delay_steps = mode_delay[mode][feeder_idx]
        y = feeder_response(K[feeder_idx], T[feeder_idx], int(delay_steps), n_samples)
        axes[feeder_idx].plot(time, y, linewidth=2, alpha=0.85)

    axes[feeder_idx].set_ylabel("Output")
    axes[feeder_idx].set_title(f"Feeder {feeder_idx + 1}")
    axes[feeder_idx].grid(True, alpha=0.3)

combined_scenarios = []
for mode_tuple in product(["slow", "medium", "fast"], repeat=3):
    delay_steps = np.array(
        [mode_delay[mode][idx] for idx, mode in enumerate(mode_tuple)],
        dtype=int,
    )
    feeder_outputs = [
        feeder_response(K[i], T[i], int(delay), n_samples)
        for i, delay in enumerate(delay_steps)
    ]
    combined_scenarios.append(np.sum(feeder_outputs, axis=0))

for y in combined_scenarios:
    axes[-1].plot(time, y, color="C0", linewidth=1.2, alpha=0.12)

axes[-1].set_title("Combined feeder output spread")
axes[-1].set_xlabel("Time (s)")
axes[-1].set_ylabel("Feed")
axes[-1].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlim(0, time[-1])

scenario_path = os.path.join(PLOT_DIR, "feeder_speed_sweep.png")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(scenario_path, dpi=200)
print(f"Saved scenario sweep to {scenario_path}")
plt.show()
