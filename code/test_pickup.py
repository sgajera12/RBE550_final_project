import sys
import numpy as np
import genesis as gs

from scenes import create_scene_6blocks
from motion_primitives import MotionPrimitiveExecutor

# 1) Init Genesis (same pattern as demo.py)
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level="Warning", logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

# 2) Build scene
scene, franka, blocks_state = create_scene_6blocks()

# 3) Set control gains (copy from demo.py)
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

# 4) Create executor
executor = MotionPrimitiveExecutor(scene, franka, blocks_state)

# 5) Try to pick up the red block ('r')
executor.pick_up("r")

# Keep sim running so you can see the result
while True:
    scene.step()
