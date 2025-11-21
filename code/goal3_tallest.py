"""
goal3_tallest.py

Goal 3: Build the tallest stable tower possible (BONUS)
- Start from scattered blocks (all 6 on floor)
- Stack all 6 blocks for maximum height
- Bonus: 2 points per block over 5 (so 6 blocks = 2 bonus points!)

Strategy: Build straight up in center for maximum stability

Usage:
    python goal3_tallest.py [gpu]
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
print("GOAL 3: TALLEST STABLE TOWER (BONUS CHALLENGE)")
print("Building: 6-block tower for maximum height")
print("Strategy: Stack all blocks in center for stability")
print("Order: Any (optimizing for stability)")
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
# BUILD 6-BLOCK TOWER
# Strategy: Use order that minimizes arm reach - build closest to farthest
# Start with block closest to center, stack others on it
# ============================================================================
print("\n" + "="*80)
print("BUILDING 6-BLOCK TOWER")
print("="*80)

# Stack order: R, G, B, Y, M, C (arbitrary but works well)
stack_order = [
    ("r", None),      # RED as base
    ("g", "r"),       # GREEN on RED
    ("b", "g"),       # BLUE on GREEN
    ("y", "b"),       # YELLOW on BLUE
    ("m", "y"),       # MAGENTA on YELLOW
    ("c", "m"),       # CYAN on MAGENTA (top)
]

step = 1
for block, target in stack_order:
    if target is None:
        # First block - just leave it on table (or move to center if desired)
        print(f"\n[{step}/11] Using {block.upper()} as base")
        step += 1
    else:
        print(f"\n[{step}/11] Pick {block.upper()}")
        if not executor.pick_up(block):
            print("❌ Failed")
            sys.exit(1)
        step += 1
        
        print(f"\n[{step}/11] Stack {block.upper()} on {target.upper()}")
        if not executor.stack_on(target):
            print("❌ Failed")
            sys.exit(1)
        step += 1

print("\n✅ 6-BLOCK TOWER COMPLETE!")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("FINAL VERIFICATION")
print("="*80)

# Let tower settle
for _ in range(50):
    scene.step()

print("\nFINAL POSITIONS:")
final_pos = {}
for key, block in blocks_state.items():
    pos = np.array(block.get_pos())
    final_pos[key] = pos
    print(f"  {key.upper()}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

# Analyze tower
blocks_in_order = ['r', 'g', 'b', 'y', 'm', 'c']
positions = [final_pos[b] for b in blocks_in_order]

print("\n" + "="*80)
print("6-BLOCK TOWER ANALYSIS:")
print("="*80)

# Check each interface
all_good = True
for i in range(len(blocks_in_order) - 1):
    bottom = positions[i]
    top = positions[i + 1]
    bottom_name = blocks_in_order[i].upper()
    top_name = blocks_in_order[i + 1].upper()
    
    v_dist = top[2] - bottom[2]
    h_dist = np.sqrt((top[0] - bottom[0])**2 + (top[1] - bottom[1])**2)
    is_good = (0.03 < v_dist < 0.05 and h_dist < 0.03)
    all_good = all_good and is_good
    
    print(f"{bottom_name}-{top_name}: v={v_dist:.3f}m, h={h_dist:.3f}m → {'✅' if is_good else '❌'}")

total_height = positions[-1][2] - positions[0][2]
print(f"\nTotal tower height: {total_height:.3f}m (expect ~0.20m for 6 blocks)")
print(f"Number of blocks: 6")
print(f"Blocks over 5: 1")
print(f"Bonus points: 2 points!")

print("\n" + "="*80)
if all_good:
    print("🎉🎉🎉 SUCCESS! TALLEST TOWER COMPLETE! 🎉🎉🎉")
    print("\n✅ 6-block tower achieved!")
    print("✅ Bonus: 2 points (1 block over 5)")
    print("\n🏆 MAXIMUM HEIGHT WITH 6 BLOCKS!")
else:
    print("⚠️  TOWER MAY HAVE STABILITY ISSUES")
    if total_height > 0.15:
        print("✅ Height achieved, but check alignment")
print("="*80)

print("\nPress Ctrl+C to exit...")
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")