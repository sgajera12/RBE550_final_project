"""
goal1_stacked.py

Goal 1: Build two 3-block towers from pre-stacked tower
Tower 1: BLUE (base) → GREEN → RED (top)
Tower 2: CYAN (base) → MAGENTA → YELLOW (top)

Starting scene: All 6 blocks in one tower (need to unstack and rebuild)

Usage:
    python goal1_stacked.py [gpu]
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
print("GOAL 1: BUILD TWO TOWERS (STACKED START)")
print("Starting: All blocks in one 6-block tower")
print("Target Tower 1: BLUE (base) → GREEN → RED (top)")
print("Target Tower 2: CYAN (base) → MAGENTA → YELLOW (top)")
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

print("\n" + "="*80)
print("PHASE 1: UNSTACK TOP BLOCKS")
print("Initial tower: R(base)-G-B-Y-M-C(top)")
print("="*80)

# Unstack CYAN (top) and place aside
print("\n[1/14] Pick CYAN (top of tower)")
if not executor.pick_up("c"):
    sys.exit(1)

print("\n[2/14] Place CYAN aside (future Tower 2 base)")
if not executor.put_down(x=0.35, y=0.1):  # Place to the side
    sys.exit(1)

# Unstack MAGENTA
print("\n[3/14] Pick MAGENTA")
if not executor.pick_up("m"):
    sys.exit(1)

print("\n[4/14] Place MAGENTA aside")
if not executor.put_down(x=0.35, y=-0.1):
    sys.exit(1)

# Unstack YELLOW
print("\n[5/14] Pick YELLOW")
if not executor.pick_up("y"):
    sys.exit(1)

print("\n[6/14] Place YELLOW aside")
if not executor.put_down(x=0.35, y=-0.3):
    sys.exit(1)

print("\n✅ Phase 1 complete. Remaining tower: R-G-B")

# ============================================================================
# PHASE 2: BUILD TOWER 1 (BLUE-GREEN-RED)
# Current state: R-G-B tower, need B-G-R
# Strategy: B is already at good spot, just need to verify
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: BUILD TOWER 1 (BLUE-GREEN-RED)")
print("="*80)

# Check if R-G-B is already in right place, or need to rebuild
# For simplicity, let's unstack and rebuild

print("\n[7/14] Pick BLUE (top)")
if not executor.pick_up("b"):
    sys.exit(1)

print("\n[8/14] Place BLUE as Tower 1 base")
if not executor.put_down(x=0.55, y=0.2):
    sys.exit(1)

print("\n[9/14] Pick GREEN")
if not executor.pick_up("g"):
    sys.exit(1)

print("\n[10/14] Stack GREEN on BLUE")
if not executor.stack_on("b"):
    sys.exit(1)

print("\n[11/14] Pick RED")
if not executor.pick_up("r"):
    sys.exit(1)

print("\n[12/14] Stack RED on GREEN")
if not executor.stack_on("g"):
    sys.exit(1)

print("\n✅ TOWER 1 COMPLETE: BLUE-GREEN-RED")

# ============================================================================
# PHASE 3: BUILD TOWER 2 (CYAN-MAGENTA-YELLOW)
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: BUILD TOWER 2 (CYAN-MAGENTA-YELLOW)")
print("="*80)

print("\n[13/14] Pick MAGENTA")
if not executor.pick_up("m"):
    sys.exit(1)

print("\n[14/14] Stack MAGENTA on CYAN")
if not executor.stack_on("c"):
    sys.exit(1)

print("\n[15/14] Pick YELLOW")
if not executor.pick_up("y"):
    sys.exit(1)

print("\n[16/14] Stack YELLOW on MAGENTA")
if not executor.stack_on("m"):
    sys.exit(1)

print("\n✅ TOWER 2 COMPLETE: CYAN-MAGENTA-YELLOW")

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

# Analyze towers
b_pos, g_pos, r_pos = final_pos['b'], final_pos['g'], final_pos['r']
c_pos, m_pos, y_pos = final_pos['c'], final_pos['m'], final_pos['y']

print("\n" + "="*80)
print("TOWER 1 (BLUE-GREEN-RED):")
bg_v = g_pos[2] - b_pos[2]
bg_h = np.sqrt((g_pos[0] - b_pos[0])**2 + (g_pos[1] - b_pos[1])**2)
gr_v = r_pos[2] - g_pos[2]
gr_h = np.sqrt((r_pos[0] - g_pos[0])**2 + (r_pos[1] - g_pos[1])**2)

print(f"BLUE-GREEN: v={bg_v:.3f}m, h={bg_h:.3f}m → {'✅' if (0.03 < bg_v < 0.05 and bg_h < 0.03) else '❌'}")
print(f"GREEN-RED:  v={gr_v:.3f}m, h={gr_h:.3f}m → {'✅' if (0.03 < gr_v < 0.05 and gr_h < 0.03) else '❌'}")

print("\n" + "="*80)
print("TOWER 2 (CYAN-MAGENTA-YELLOW):")
cm_v = m_pos[2] - c_pos[2]
cm_h = np.sqrt((m_pos[0] - c_pos[0])**2 + (m_pos[1] - c_pos[1])**2)
my_v = y_pos[2] - m_pos[2]
my_h = np.sqrt((y_pos[0] - m_pos[0])**2 + (y_pos[1] - m_pos[1])**2)

print(f"CYAN-MAGENTA: v={cm_v:.3f}m, h={cm_h:.3f}m → {'✅' if (0.03 < cm_v < 0.05 and cm_h < 0.03) else '❌'}")
print(f"MAGENTA-YELLOW: v={my_v:.3f}m, h={my_h:.3f}m → {'✅' if (0.03 < my_v < 0.05 and my_h < 0.03) else '❌'}")

tower1_ok = (0.03 < bg_v < 0.05 and bg_h < 0.03 and 0.03 < gr_v < 0.05 and gr_h < 0.03)
tower2_ok = (0.03 < cm_v < 0.05 and cm_h < 0.03 and 0.03 < my_v < 0.05 and my_h < 0.03)

print("\n" + "="*80)
if tower1_ok and tower2_ok:
    print("🎉 SUCCESS! GOAL 1 COMPLETE (FROM STACKED)! 🎉")
    print("\n✅ Tower 1: RED-GREEN-BLUE (top to bottom)")
    print("✅ Tower 2: YELLOW-MAGENTA-CYAN (top to bottom)")
else:
    print("⚠️  INCOMPLETE")
print("="*80)

print("\nPress Ctrl+C to exit...")
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")