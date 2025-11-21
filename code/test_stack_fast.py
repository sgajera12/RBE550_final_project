"""
test_stack_fast.py

Fast, clean test without unnecessary monitoring.
No grip checks, no waiting - just pick and stack immediately.

Usage:
    python test_stack_fast.py [gpu]
"""

import sys
import numpy as np
import genesis as gs

from scenes import create_scene_6blocks
from motion_primitives import MotionPrimitiveExecutor

# Initialize
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level="Warning", logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

scene, franka, blocks_state = create_scene_6blocks()

# Strong gripper control
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 2000, 2000]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 200, 200]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -200, -200]),
    np.array([87, 87, 87, 87, 12, 12, 12, 200, 200]),
)

print("="*70)
print("FAST PICK AND STACK TEST")
print("="*70)

# Move to home quickly
safe_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=float)

print("\nMoving to home...")
current = franka.get_qpos()
if hasattr(current, "cpu"):
    current = current.cpu().numpy()

for i in range(200):  # Faster: 2 seconds instead of 3
    alpha = (i + 1) / 200.0
    q = (1 - alpha) * current + alpha * safe_home
    franka.control_dofs_position(q)
    scene.step()

print("✅ At home\n")

# Minimal settling
for _ in range(50):
    franka.control_dofs_position(safe_home)
    scene.step()

# Show positions
print("Initial positions:")
for key, block in blocks_state.items():
    pos = np.array(block.get_pos())
    print(f"  {key}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

# Create executor
executor = MotionPrimitiveExecutor(scene, franka, blocks_state)

# ============================================================================
# PICK RED
# ============================================================================
print("\n" + "="*70)
print("STEP 1: PICK RED")
print("="*70)

success_pick = executor.pick_up("r")

if not success_pick:
    print("❌ Failed to pick")
    sys.exit(1)

print("\n✅ Red picked!")

# NO MONITORING - Go straight to stack!

# ============================================================================
# STACK ON GREEN
# ============================================================================
print("\n" + "="*70)
print("STEP 2: STACK ON GREEN")
print("="*70)

green_pos = np.array(blocks_state['g'].get_pos())
print(f"Green at: [{green_pos[0]:.3f}, {green_pos[1]:.3f}, {green_pos[2]:.3f}]")

success_stack = executor.stack_on("g")

if not success_stack:
    print("❌ Failed to stack")
else:
    print("\n✅ Stacking completed!")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

# Let physics settle briefly
print("\nLetting physics settle...")
for _ in range(100):  # 1 second
    scene.step()

# Check final positions
print("\nFinal positions:")
red_final = np.array(blocks_state['r'].get_pos())
green_final = np.array(blocks_state['g'].get_pos())

for key, block in blocks_state.items():
    pos = np.array(block.get_pos())
    print(f"  {key}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

height_diff = red_final[2] - green_final[2]
horiz_dist = np.linalg.norm(red_final[:2] - green_final[:2])

print(f"\nStack analysis:")
print(f"  Vertical separation: {height_diff:.4f}m (expect ~0.04m)")
print(f"  Horizontal alignment: {horiz_dist:.4f}m (expect <0.03m)")

if 0.03 < height_diff < 0.05 and horiz_dist < 0.05:
    print("\n" + "="*70)
    print("✅✅✅ SUCCESS! RED IS STACKED ON GREEN! ✅✅✅")
    print("="*70)
else:
    print("\n" + "="*70)
    print("⚠️  Stack may not be perfect")
    print("="*70)

print("\nSimulation running. Press Ctrl+C to exit...")
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")