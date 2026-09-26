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

from __future__ import annotations

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


def feeder_response(
    Ki: float, Ti: float, delay_steps: int, n_steps: int
) -> np.ndarray:
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
