"""
goal2_stacked.py

Goal 2: Build 5-block tower from pre-stacked 6-block tower
Order (top to bottom): MAGENTA-YELLOW-BLUE-RED-GREEN
= GREEN (base) → RED → BLUE → YELLOW → MAGENTA (top)

Starting scene: All 6 blocks in one tower
Strategy: Remove CYAN, keep other 5 in tower (they're already correct!)

Usage:
    python goal2_stacked.py [gpu]
"""

import sys
import numpy as np
import genesis as gs

from scenes import create_scene_stacked
from motion_primitives import MotionPrimitiveExecutor

# Initialize
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level="Warning", logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

scene, franka, blocks_state = create_scene_stacked()

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
print("GOAL 2: BUILD 5-BLOCK TOWER (STACKED START)")
print("Starting: 6-block tower (R-G-B-Y-M-C from bottom)")
print("Target: 5-block tower MAGENTA-YELLOW-BLUE-RED-GREEN (top to bottom)")
print("Strategy: Remove CYAN only - other 5 blocks already in correct order!")
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
print("INITIAL POSITIONS (6-block tower):")
for key, block in blocks_state.items():
    pos = np.array(block.get_pos())
    print(f"  {key.upper()}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

executor = MotionPrimitiveExecutor(scene, franka, blocks_state)

# ============================================================================
# STRATEGY: Just remove CYAN!
# Initial tower: R-G-B-Y-M-C (bottom to top)
# We need: R-G-B-Y-M (which is already there!)
# So: pick CYAN and place it aside
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: UNSTACK ALL BLOCKS")
print("Initial tower: R-G-B-Y-M-C (wrong order)")
print("Need: G-R-B-Y-M (GREEN base)")
print("="*80)

# Unstack all 6 blocks and place aside
print("\n[1/11] Pick CYAN (top)")
if not executor.pick_up("c"):
    sys.exit(1)
print("[2/11] Place CYAN aside")
if not executor.put_down(x=0.35, y=0.3):
    sys.exit(1)

print("\n[3/11] Pick MAGENTA")
if not executor.pick_up("m"):
    sys.exit(1)
print("[4/11] Place MAGENTA aside")
if not executor.put_down(x=0.35, y=0.1):
    sys.exit(1)

print("\n[5/11] Pick YELLOW")
if not executor.pick_up("y"):
    sys.exit(1)
print("[6/11] Place YELLOW aside")
if not executor.put_down(x=0.35, y=-0.1):
    sys.exit(1)

print("\n[7/11] Pick BLUE")
if not executor.pick_up("b"):
    sys.exit(1)
print("[8/11] Place BLUE aside")
if not executor.put_down(x=0.35, y=-0.3):
    sys.exit(1)

print("\n[9/11] Pick GREEN")
if not executor.pick_up("g"):
    sys.exit(1)
print("[10/11] Place GREEN as base")
if not executor.put_down(x=0.55, y=0.0):
    sys.exit(1)

# RED stays at bottom - that's fine, we'll leave it

print("\n✅ All blocks unstacked")

# ============================================================================
# PHASE 2: BUILD 5-BLOCK TOWER: GREEN → RED → BLUE → YELLOW → MAGENTA
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: BUILD 5-BLOCK TOWER")
print("="*80)

print("\n[11/16] Pick RED")
if not executor.pick_up("r"):
    sys.exit(1)

print("\n[12/16] Stack RED on GREEN")
if not executor.stack_on("g"):
    sys.exit(1)

print("\n[13/16] Pick BLUE")
if not executor.pick_up("b"):
    sys.exit(1)

print("\n[14/16] Stack BLUE on RED")
if not executor.stack_on("r"):
    sys.exit(1)

print("\n[15/16] Pick YELLOW")
if not executor.pick_up("y"):
    sys.exit(1)

print("\n[16/16] Stack YELLOW on BLUE")
if not executor.stack_on("b"):
    sys.exit(1)

print("\n[17/16] Pick MAGENTA")
if not executor.pick_up("m"):
    sys.exit(1)

print("\n[18/16] Stack MAGENTA on YELLOW")
if not executor.stack_on("y"):
    sys.exit(1)

print("\n✅ 5-BLOCK TOWER COMPLETE!")

# Wait for stabilization
print("\nLetting tower stabilize...")
for _ in range(50):
    scene.step()

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

# Analyze tower (G-R-B-Y-M)
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
    print("🎉🎉🎉 SUCCESS! GOAL 2 COMPLETE (FROM STACKED)! 🎉🎉🎉")
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