"""
test_stack_3blocks.py

Stack 3 blocks: GREEN (base) → RED (middle) → BLUE (top)

Sequence:
1. Pick up RED and stack on GREEN (2-block tower)
2. Pick up BLUE and stack on RED (3-block tower)

Usage:
    python test_stack_3blocks.py [gpu]
"""

import sys
import numpy as np
import genesis as gs

from scenes import create_scene_6blocks
from motion_primitives import MotionPrimitiveExecutor

# Initialize Genesis
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level="Warning", logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

scene, franka, blocks_state = create_scene_6blocks()

# Strong gripper control gains
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
print("3-BLOCK STACKING TEST")
print("Goal: Build tower GREEN (base) → RED (middle) → BLUE (top)")
print("="*80)

# Move to home position
safe_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=float)

print("\nMoving to home position...")
current = franka.get_qpos()
if hasattr(current, "cpu"):
    current = current.cpu().numpy()

for i in range(200):
    alpha = (i + 1) / 200.0
    q = (1 - alpha) * current + alpha * safe_home
    franka.control_dofs_position(q)
    scene.step()

print("✅ At home position\n")

# Let physics settle
for _ in range(50):
    franka.control_dofs_position(safe_home)
    scene.step()

# Show initial block positions
print("="*80)
print("INITIAL BLOCK POSITIONS:")
print("="*80)
for key, block in blocks_state.items():
    pos = np.array(block.get_pos())
    print(f"  {key.upper()}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

# Create motion primitive executor
executor = MotionPrimitiveExecutor(scene, franka, blocks_state)

# ============================================================================
# STEP 1: Pick up RED
# ============================================================================
print("\n" + "="*80)
print("STEP 1 OF 4: PICK UP RED BLOCK")
print("="*80)

success_pick_red = executor.pick_up("r")

if not success_pick_red:
    print("\n❌ Failed to pick up RED block. Aborting.")
    sys.exit(1)

print("\n✅ RED block picked up successfully!")

# ============================================================================
# STEP 2: Stack RED on GREEN (build 2-block tower)
# ============================================================================
print("\n" + "="*80)
print("STEP 2 OF 4: STACK RED ON GREEN")
print("="*80)

green_pos = np.array(blocks_state['g'].get_pos())
print(f"Green position: [{green_pos[0]:.3f}, {green_pos[1]:.3f}, {green_pos[2]:.3f}]")

success_stack_red = executor.stack_on("g")

if not success_stack_red:
    print("\n❌ Failed to stack RED on GREEN. Aborting.")
    sys.exit(1)

print("\n✅ 2-block tower complete! (GREEN-RED)")

# Show positions after first stack
print("\n" + "="*80)
print("POSITIONS AFTER FIRST STACK:")
print("="*80)
for key, block in blocks_state.items():
    pos = np.array(block.get_pos())
    print(f"  {key.upper()}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

# Verify 2-block stack quality
red_pos = np.array(blocks_state['r'].get_pos())
green_pos = np.array(blocks_state['g'].get_pos())
height_diff_1 = red_pos[2] - green_pos[2]
horiz_dist_1 = np.sqrt((red_pos[0] - green_pos[0])**2 + (red_pos[1] - green_pos[1])**2)

print(f"\n2-Block Tower Analysis:")
print(f"  Vertical separation: {height_diff_1:.4f}m (expect ~0.04m)")
print(f"  Horizontal alignment: {horiz_dist_1:.4f}m (expect <0.03m)")

if not (0.03 < height_diff_1 < 0.05 and horiz_dist_1 < 0.05):
    print("\n⚠️  WARNING: 2-block tower may not be stable!")
    print("Continuing anyway...")

# ============================================================================
# STEP 3: Pick up BLUE
# ============================================================================
print("\n" + "="*80)
print("STEP 3 OF 4: PICK UP BLUE BLOCK")
print("="*80)

success_pick_blue = executor.pick_up("b")

if not success_pick_blue:
    print("\n❌ Failed to pick up BLUE block. Aborting.")
    sys.exit(1)

print("\n✅ BLUE block picked up successfully!")

# ============================================================================
# STEP 4: Stack BLUE on RED (complete 3-block tower)
# ============================================================================
print("\n" + "="*80)
print("STEP 4 OF 4: STACK BLUE ON RED")
print("="*80)

# Get FRESH red position (it's now on top of green)
red_pos = np.array(blocks_state['r'].get_pos())
print(f"Red position (on green): [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")

success_stack_blue = executor.stack_on("r")

if not success_stack_blue:
    print("\n❌ Failed to stack BLUE on RED.")
    sys.exit(1)

print("\n✅ 3-block tower complete! (GREEN-RED-BLUE)")

# ============================================================================
# FINAL VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("FINAL VERIFICATION")
print("="*80)

# Brief settling
for _ in range(30):
    scene.step()

# Get final positions
print("\n" + "="*80)
print("FINAL BLOCK POSITIONS:")
print("="*80)
final_positions = {}
for key, block in blocks_state.items():
    pos = np.array(block.get_pos())
    final_positions[key] = pos
    print(f"  {key.upper()}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

# Analyze the 3-block tower
green_final = final_positions['g']
red_final = final_positions['r']
blue_final = final_positions['b']

# Check GREEN-RED stack
height_diff_gr = red_final[2] - green_final[2]
horiz_dist_gr = np.sqrt((red_final[0] - green_final[0])**2 + (red_final[1] - green_final[1])**2)

# Check RED-BLUE stack
height_diff_rb = blue_final[2] - red_final[2]
horiz_dist_rb = np.sqrt((blue_final[0] - red_final[0])**2 + (blue_final[1] - red_final[1])**2)

print("\n" + "="*80)
print("TOWER ANALYSIS:")
print("="*80)
print("\nGREEN-RED interface:")
print(f"  Vertical separation: {height_diff_gr:.4f}m (expect ~0.04m)")
print(f"  Horizontal alignment: {horiz_dist_gr:.4f}m (expect <0.03m)")
print(f"  Status: {'✅ GOOD' if (0.03 < height_diff_gr < 0.05 and horiz_dist_gr < 0.05) else '❌ POOR'}")

print("\nRED-BLUE interface:")
print(f"  Vertical separation: {height_diff_rb:.4f}m (expect ~0.04m)")
print(f"  Horizontal alignment: {horiz_dist_rb:.4f}m (expect <0.03m)")
print(f"  Status: {'✅ GOOD' if (0.03 < height_diff_rb < 0.05 and horiz_dist_rb < 0.05) else '❌ POOR'}")

print("\nTotal tower height:")
print(f"  Green base: {green_final[2]:.4f}m")
print(f"  Red middle: {red_final[2]:.4f}m (+{height_diff_gr:.4f}m)")
print(f"  Blue top: {blue_final[2]:.4f}m (+{height_diff_rb:.4f}m)")
print(f"  Total stack height: {blue_final[2] - green_final[2]:.4f}m (expect ~0.08m)")

# Overall success check
gr_good = (0.03 < height_diff_gr < 0.05 and horiz_dist_gr < 0.05)
rb_good = (0.03 < height_diff_rb < 0.05 and horiz_dist_rb < 0.05)

if gr_good and rb_good:
    print("\n" + "="*80)
    print("🎉🎉🎉 SUCCESS! 3-BLOCK TOWER IS STABLE! 🎉🎉🎉")
    print("="*80)
    print("\n✅ GREEN (base)")
    print("✅ RED (middle)")
    print("✅ BLUE (top)")
    print("\nTower configuration: GREEN → RED → BLUE")
else:
    print("\n" + "="*80)
    print("⚠️  3-BLOCK TOWER MAY NOT BE PERFECT")
    print("="*80)
    if not gr_good:
        print("  Issue: GREEN-RED interface")
    if not rb_good:
        print("  Issue: RED-BLUE interface")

print("\n" + "="*80)
print("Simulation running. Press Ctrl+C to exit...")
print("="*80)

try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")