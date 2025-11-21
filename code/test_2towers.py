"""
test_2towers.py

Build TWO 3-block towers:
- Tower 1: GREEN (base) → RED → BLUE
- Tower 2: YELLOW (base) → MAGENTA → CYAN

Usage:
    python test_2towers.py [gpu]
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
print("2-TOWER STACKING TEST")
print("Tower 1: GREEN → RED → BLUE")
print("Tower 2: YELLOW → MAGENTA → CYAN")
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
print("="*80)
print("INITIAL POSITIONS:")
print("="*80)
for key, block in blocks_state.items():
    pos = np.array(block.get_pos())
    print(f"  {key.upper()}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

executor = MotionPrimitiveExecutor(scene, franka, blocks_state)

# ============================================================================
# TOWER 1: GREEN → RED → BLUE
# ============================================================================
print("\n" + "="*80)
print("TOWER 1: GREEN → RED → BLUE")
print("="*80)

# Pick RED
print("\n[1/6] Pick RED")
if not executor.pick_up("r"):
    print("❌ Failed")
    sys.exit(1)

# Stack RED on GREEN
print("\n[2/6] Stack RED on GREEN")
if not executor.stack_on("g"):
    print("❌ Failed")
    sys.exit(1)

# Pick BLUE
print("\n[3/6] Pick BLUE")
if not executor.pick_up("b"):
    print("❌ Failed")
    sys.exit(1)

# Stack BLUE on RED
print("\n[4/6] Stack BLUE on RED")
if not executor.stack_on("r"):
    print("❌ Failed")
    sys.exit(1)

print("\n✅ TOWER 1 COMPLETE: GREEN → RED → BLUE")

# ============================================================================
# TOWER 2: YELLOW → MAGENTA → CYAN
# ============================================================================
print("\n" + "="*80)
print("TOWER 2: YELLOW → MAGENTA → CYAN")
print("="*80)

# Pick MAGENTA
print("\n[5/6] Pick MAGENTA")
if not executor.pick_up("m"):
    print("❌ Failed")
    sys.exit(1)

# Stack MAGENTA on YELLOW
print("\n[6/6] Stack MAGENTA on YELLOW")
if not executor.stack_on("y"):
    print("❌ Failed")
    sys.exit(1)

# Pick CYAN
print("\n[7/6] Pick CYAN")
if not executor.pick_up("c"):
    print("❌ Failed")
    sys.exit(1)

# Stack CYAN on MAGENTA
print("\n[8/6] Stack CYAN on MAGENTA")
if not executor.stack_on("m"):
    print("❌ Failed")
    sys.exit(1)

print("\n✅ TOWER 2 COMPLETE: YELLOW → MAGENTA → CYAN")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("FINAL VERIFICATION")
print("="*80)

# Brief settling
for _ in range(30):
    scene.step()

# Get final positions
print("\nFINAL POSITIONS:")
final_pos = {}
for key, block in blocks_state.items():
    pos = np.array(block.get_pos())
    final_pos[key] = pos
    print(f"  {key.upper()}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

# Analyze Tower 1
print("\n" + "="*80)
print("TOWER 1 ANALYSIS (GREEN-RED-BLUE):")
print("="*80)

g_pos, r_pos, b_pos = final_pos['g'], final_pos['r'], final_pos['b']

gr_vdist = r_pos[2] - g_pos[2]
gr_hdist = np.sqrt((r_pos[0] - g_pos[0])**2 + (r_pos[1] - g_pos[1])**2)
rb_vdist = b_pos[2] - r_pos[2]
rb_hdist = np.sqrt((b_pos[0] - r_pos[0])**2 + (b_pos[1] - r_pos[1])**2)

print(f"GREEN-RED: v={gr_vdist:.3f}m, h={gr_hdist:.3f}m → {'✅' if (0.03 < gr_vdist < 0.05 and gr_hdist < 0.03) else '❌'}")
print(f"RED-BLUE:  v={rb_vdist:.3f}m, h={rb_hdist:.3f}m → {'✅' if (0.03 < rb_vdist < 0.05 and rb_hdist < 0.03) else '❌'}")

# Analyze Tower 2
print("\n" + "="*80)
print("TOWER 2 ANALYSIS (YELLOW-MAGENTA-CYAN):")
print("="*80)

y_pos, m_pos, c_pos = final_pos['y'], final_pos['m'], final_pos['c']

ym_vdist = m_pos[2] - y_pos[2]
ym_hdist = np.sqrt((m_pos[0] - y_pos[0])**2 + (m_pos[1] - y_pos[1])**2)
mc_vdist = c_pos[2] - m_pos[2]
mc_hdist = np.sqrt((c_pos[0] - m_pos[0])**2 + (c_pos[1] - m_pos[1])**2)

print(f"YELLOW-MAGENTA: v={ym_vdist:.3f}m, h={ym_hdist:.3f}m → {'✅' if (0.03 < ym_vdist < 0.05 and ym_hdist < 0.03) else '❌'}")
print(f"MAGENTA-CYAN:   v={mc_vdist:.3f}m, h={mc_hdist:.3f}m → {'✅' if (0.03 < mc_vdist < 0.05 and mc_hdist < 0.03) else '❌'}")

# Overall check
tower1_ok = (0.03 < gr_vdist < 0.05 and gr_hdist < 0.03 and 
             0.03 < rb_vdist < 0.05 and rb_hdist < 0.03)
tower2_ok = (0.03 < ym_vdist < 0.05 and ym_hdist < 0.03 and 
             0.03 < mc_vdist < 0.05 and mc_hdist < 0.03)

print("\n" + "="*80)
if tower1_ok and tower2_ok:
    print("🎉🎉🎉 SUCCESS! BOTH TOWERS ARE STABLE! 🎉🎉🎉")
    print("\n✅ Tower 1: GREEN → RED → BLUE")
    print("✅ Tower 2: YELLOW → MAGENTA → CYAN")
else:
    print("⚠️  ONE OR MORE TOWERS MAY HAVE ISSUES")
    if not tower1_ok:
        print("  Issue: Tower 1")
    if not tower2_ok:
        print("  Issue: Tower 2")
print("="*80)

print("\nSimulation running. Press Ctrl+C to exit...")

try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")