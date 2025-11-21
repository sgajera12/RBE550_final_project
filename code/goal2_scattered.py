"""
goal2_scattered.py

Goal 2: Build 5-block tower from scattered blocks
Order (top to bottom): MAGENTA-YELLOW-BLUE-RED-GREEN
= GREEN (base) → RED → BLUE → YELLOW → MAGENTA (top)

Starting scene: All blocks scattered on floor

Usage:
    python goal2_scattered.py [gpu]
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

print("="*80)
print("GOAL 2: BUILD 5-BLOCK TOWER (SCATTERED START)")
print("Order (top to bottom): MAGENTA-YELLOW-BLUE-RED-GREEN")
print("Building: GREEN (base) → RED → BLUE → YELLOW → MAGENTA (top)")
print("="*80)

# Move to home
safe_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=float)

print("\nMoving to home...")
current = franka.get_qpos()
if hasattr(current, "cpu"):
    current = current.cpu().numpy()

for i in range(200):
    alpha = (i + 1) / 200.0
    q = (1 - alpha) * current + alpha * safe_home
    franka.control_dofs_position(q)
    scene.step()

print("✅ At home\n")

for _ in range(50):
    franka.control_dofs_position(safe_home)
    scene.step()

# Show initial positions
print("INITIAL POSITIONS:")
for key, block in blocks_state.items():
    pos = np.array(block.get_pos())
    print(f"  {key.upper()}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

executor = MotionPrimitiveExecutor(scene, franka, blocks_state)

# ============================================================================
# BUILD 5-BLOCK TOWER: GREEN → RED → BLUE → YELLOW → MAGENTA
# ============================================================================
print("\n" + "="*80)
print("BUILDING 5-BLOCK TOWER")
print("="*80)

print("\n[1/8] Pick RED")
if not executor.pick_up("r"):
    print("❌ Failed")
    sys.exit(1)

print("\n[2/8] Stack RED on GREEN")
if not executor.stack_on("g"):
    print("❌ Failed")
    sys.exit(1)

print("\n[3/8] Pick BLUE")
if not executor.pick_up("b"):
    print("❌ Failed")
    sys.exit(1)

print("\n[4/8] Stack BLUE on RED")
if not executor.stack_on("r"):
    print("❌ Failed")
    sys.exit(1)

print("\n[5/8] Pick YELLOW")
if not executor.pick_up("y"):
    print("❌ Failed")
    sys.exit(1)

print("\n[6/8] Stack YELLOW on BLUE")
if not executor.stack_on("b"):
    print("❌ Failed")
    sys.exit(1)

print("\n[7/8] Pick MAGENTA")
if not executor.pick_up("m"):
    print("❌ Failed")
    sys.exit(1)

print("\n[8/8] Stack MAGENTA on YELLOW")
if not executor.stack_on("y"):
    print("❌ Failed")
    sys.exit(1)

print("\n✅ 5-BLOCK TOWER COMPLETE!")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("FINAL VERIFICATION")
print("="*80)

for _ in range(30):
    scene.step()

print("\nFINAL POSITIONS:")
final_pos = {}
for key, block in blocks_state.items():
    pos = np.array(block.get_pos())
    final_pos[key] = pos
    print(f"  {key.upper()}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

# Analyze tower
g_pos = final_pos['g']
r_pos = final_pos['r']
b_pos = final_pos['b']
y_pos = final_pos['y']
m_pos = final_pos['m']

print("\n" + "="*80)
print("5-BLOCK TOWER ANALYSIS:")
print("="*80)

# Check each interface
interfaces = [
    ("GREEN-RED", g_pos, r_pos),
    ("RED-BLUE", r_pos, b_pos),
    ("BLUE-YELLOW", b_pos, y_pos),
    ("YELLOW-MAGENTA", y_pos, m_pos),
]

all_good = True
for name, bottom, top in interfaces:
    v_dist = top[2] - bottom[2]
    h_dist = np.sqrt((top[0] - bottom[0])**2 + (top[1] - bottom[1])**2)
    is_good = (0.03 < v_dist < 0.05 and h_dist < 0.03)
    all_good = all_good and is_good
    print(f"{name}: v={v_dist:.3f}m, h={h_dist:.3f}m → {'✅' if is_good else '❌'}")

print(f"\nTotal tower height: {m_pos[2] - g_pos[2]:.3f}m (expect ~0.16m)")

print("\n" + "="*80)
if all_good:
    print("🎉🎉🎉 SUCCESS! GOAL 2 COMPLETE! 🎉🎉🎉")
    print("\n✅ 5-block tower: MAGENTA-YELLOW-BLUE-RED-GREEN (top to bottom)")
else:
    print("⚠️  TOWER MAY HAVE ISSUES")
print("="*80)

print("\nPress Ctrl+C to exit...")
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")