"""
Step response of a second-order system.
"""

import bdsim

sim = bdsim.BDSim()
bd = sim.blockdiagram()

# Define blocks
zeta = 0.7
wn = 1.0
step = bd.STEP(T=1, pos=1)
plant = bd.LTI_SISO([wn**2], [1, 2 * zeta * wn, wn**2])
scope = bd.SCOPE()

# Connect blocks
bd.connect(step, plant)
bd.connect(plant, scope)

bd.compile()
out = sim.run(bd, T=10)
