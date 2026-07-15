import bdsim

sim = bdsim.BDSim()
bd = sim.blockdiagram()

# Define blocks
step = bd.STEP(T=1, pos=1)
plant = bd.LTI_SISO(1, [1, 1]) # 1/(s+1)
scope = bd.SCOPE()

# Connect blocks
bd.connect(step, plant)
bd.connect(plant, scope)

bd.compile()
out = sim.run(bd, T=5)
