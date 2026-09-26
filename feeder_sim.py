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

import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Table II: feeder model parameters
K = np.array([0.3892, 0.4148, 0.4823], dtype=float)
T = np.array([-0.5919, -0.5058, -0.5048], dtype=float)
d = np.array([16, 14, 11], dtype=int)

# Table I: feeder-to-weight-meter distance (converted from meters to delay
# steps at 5 s sampling time; the transport delays are represented by z^-d_i
# in the model)
distance = np.array([116.0, 99.0, 82.4], dtype=float)

sample_period = 5.0
total_duration = 150.0
step_time = 0.0
n_samples = int(total_duration / sample_period)
time = sample_period * np.arange(n_samples)
step_index = int(step_time / sample_period)


def feeder_response(Ki, Ti, delay_steps, n_steps):
    """Step response of one discrete feeder model."""
    y = np.zeros(n_steps, dtype=float)
    u = np.zeros(n_steps, dtype=float)
    u[step_index:] = 1.0

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
# Mass-flow speed scenarios.
# The transport delay is fixed by the downstreamconveyor and does not change
# with feeder speed. Instead, different feeder speeds change the ore flow rate,
# which scales the feeder gains. We therefore compare slow/medium/fast flow
# conditions by scaling the gain K_i while keeping d_i fixed.
# --------------------------------------------------------------------

flow_factors = {
    "slow": 0.5,
    "medium": 1.0,
    "fast": 2.0,
}

# Top three panels: show one gain-scaled scenario for each feeder.
# Bottom panel: show all combinatorial scenarios with a single low-alpha color
# to reveal the spread of possible combined outputs.
fig, axes = plt.subplots(4, 1, figsize=(7.5, 1 + 1.5 * 4), sharex=True)
fig.suptitle("Feeder flow-rate scenario sweep", fontsize=16)

for feeder_idx in range(3):
    for mode in ("slow", "medium", "fast"):
        gain_scale = flow_factors[mode]
        y = feeder_response(
            K[feeder_idx] * gain_scale,
            T[feeder_idx],
            int(d[feeder_idx]),
            n_samples,
        )
        axes[feeder_idx].plot(time, y, linewidth=2, alpha=0.85)

    axes[feeder_idx].set_ylabel("Output")
    axes[feeder_idx].set_title(f"Feeder {feeder_idx + 1}")
    axes[feeder_idx].grid(True, alpha=0.3)

combined_scenarios = []
for mode_tuple in product(["slow", "medium", "fast"], repeat=3):
    gain_vector = np.array(
        [flow_factors[mode] for mode in mode_tuple],
        dtype=float,
    )
    feeder_outputs = [
        feeder_response(K[i] * gain_vector[i], T[i], int(d[i]), n_samples)
        for i in range(3)
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
