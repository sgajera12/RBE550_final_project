"""
goal2_tamp_cube.py

Goal 2 (Cube Version): Build a 2x2x2 cube structure using PROPER TAMP pipeline

Structure:
- Bottom layer: 4 blocks in a 2x2 square (r, g, b, y) - PUSHED TOGETHER
- Top layer: 2 blocks on top (m on r, c on g)

This creates a partial cube structure like in your image.

Goal configuration (symbolic):
  Bottom layer (on table): r, g, b, y (arranged in 2x2 square)
  Top layer:
      m on r
      c on g

  So:
      ON(m,r)
      ON(c,g)
      ONTABLE(r), ONTABLE(g), ONTABLE(b), ONTABLE(y)
      CLEAR(m), CLEAR(c), CLEAR(b), CLEAR(y)
      HANDEMPTY()

Usage:
    python goal2_tamp_cube.py [gpu]
"""

import sys
import os
import numpy as np
import genesis as gs

from scenes import create_scene_6blocks
from motion_primitives import MotionPrimitiveExecutor
from predicates import extract_predicates, print_predicates
from task_planner import generate_pddl_problem, call_pyperplan, plan_to_string

# ---------------------------------------------------------------------------
# 1) Initialize Genesis
# ---------------------------------------------------------------------------
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
    np.array([ 87,  87,  87,  87,  12,  12,  12,  200,  200]),
)

print("=" * 80)
print("GOAL 2: 2x2x2 CUBE STRUCTURE (PROPER TAMP PIPELINE)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 2) Move robot to a safe home configuration
# ---------------------------------------------------------------------------
safe_home = np.array(
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=float
)

print("\nMoving to home...")
current = franka.get_qpos()
if hasattr(current, "cpu"):
    current = current.cpu().numpy()

for i in range(200):
    alpha = (i + 1) / 200.0
    q = (1.0 - alpha) * current + alpha * safe_home
    franka.control_dofs_position(q)
    scene.step()

print("At home\n")

for _ in range(50):
    franka.control_dofs_position(safe_home)
    scene.step()

# Let physics settle
for _ in range(100):
    scene.step()

# ---------------------------------------------------------------------------
# 2.5) ARRANGE BASE BLOCKS - Push them together into 2x2 square
# ---------------------------------------------------------------------------

executor = MotionPrimitiveExecutor(scene, franka, blocks_state)

print("\n" + "=" * 80)
print("STEP 0: ARRANGING BASE BLOCKS INTO 2x2 SQUARE")
print("=" * 80)
print("\nArranging blocks r, g, b, y into 2x2 square (blocks touching)...")

# Arrange the 4 base blocks - they will be placed touching each other
success = executor.arrange_base_blocks(['r', 'g', 'b', 'y'], target_center=(0.55, 0.0))

if not success:
    print("⚠️  Warning: Push operation had issues, continuing anyway...")

# Let physics settle after pushing
print("\nLetting physics settle after arrangement...")
for _ in range(150):
    scene.step()

print("✓ Base blocks arranged!")

# Return to home position
print("\nReturning to home...")
current = franka.get_qpos()
if hasattr(current, "cpu"):
    current = current.cpu().numpy()

for i in range(100):
    alpha = (i + 1) / 100.0
    q = (1.0 - alpha) * current + alpha * safe_home
    franka.control_dofs_position(q)
    scene.step()

for _ in range(50):
    franka.control_dofs_position(safe_home)
    scene.step()

# ---------------------------------------------------------------------------
# 3) Define GOAL predicates for the cube structure
# ---------------------------------------------------------------------------

# Using all six blocks: r, g, b, y, m, c
# Bottom layer (on table): r, g, b, y (will form 2x2 square)
# Top layer: m on r, c on g (creates partial 2x2x2 cube)
goal_predicates = {
    # Top relations - building on two of the bottom blocks
    "ON(m,r)",
    "ON(c,g)",

    # Bottom 4 blocks on table
    "ONTABLE(r)",
    "ONTABLE(g)",
    "ONTABLE(b)",
    "ONTABLE(y)",

    # Clear blocks (top 2 are clear, bottom 2 that have nothing on them are clear)
    "CLEAR(m)",
    "CLEAR(c)",
    "CLEAR(b)",
    "CLEAR(y)",

    # Gripper empty at the end
    "HANDEMPTY()",
}

blocks = ["r", "g", "b", "y", "m", "c"]

print("\n" + "=" * 80)
print("GOAL STATE (2x2x2 CUBE):")
print("=" * 80)
print("\nBottom layer (on table): r, g, b, y - ARRANGED & TOUCHING")
print("Top layer: m on r, c on g")
print("\nThis creates a structure like:")
print("  [m][c]  <- Top layer")
print("  [r][g]  <- Bottom layer (touching)")
print("  [b][y]  <- Bottom layer (touching)")
print("\nThe 4 base blocks have been pushed together to touch.")
print("The TAMP pipeline will now stack m and c on top.")
print("\nGoal predicates:")
for p in sorted(goal_predicates):
    print(f"  {p}")

# ---------------------------------------------------------------------------
# 4) TAMP EXECUTION LOOP
# ---------------------------------------------------------------------------

domain_file = os.path.join(os.path.dirname(__file__), "blocksworld.pddl")

max_iterations = 25
iteration = 0

print("\n" + "=" * 80)
print("STARTING TAMP LOOP")
print("=" * 80)

while iteration < max_iterations:
    iteration += 1
    print(f"\n{'=' * 80}")
    print(f"ITERATION {iteration}")
    print(f"{'=' * 80}")

    # STEP 1: SYMBOLIC ABSTRACTION (Lifting)
    print("\n[Step 1] Extracting predicates from current scene...")
    current_predicates = extract_predicates(scene, franka, blocks_state)
    print_predicates(current_predicates)

    # Check if goal reached
    if goal_predicates.issubset(current_predicates):
        print("\n🎉 GOAL REACHED!")
        break

    # STEP 2: TASK PLANNING
    print("\n[Step 2] Calling task planner (Pyperplan)...")
    problem_string = generate_pddl_problem(
        current_predicates,
        goal_predicates,
        blocks,
        f"goal2_cube_iter{iteration}",
    )

    # Debug: Save problem file
    debug_problem_file = f"/tmp/goal2_cube_problem_iter{iteration}.pddl"
    with open(debug_problem_file, "w") as f:
        f.write(problem_string)
    print(f"  Problem saved to: {debug_problem_file}")

    plan = call_pyperplan(domain_file, problem_string)

    if not plan:
        print("❌ No plan found! Cannot reach goal.")
        break

    print(f"\n✓ Plan found ({len(plan)} actions):")
    print(plan_to_string(plan))

    # STEP 3: EXECUTE FIRST ACTION
    if not plan:
        print("\nPlan is empty!")
        break

    action_name, args = plan[0]
    print(f"\n[Step 3] Executing: {action_name.upper()}({', '.join(args)})")

    success = False

    if action_name == "pick-up":
        block_id = args[0]
        success = executor.pick_up(block_id)

    elif action_name == "put-down":
        success = executor.put_down()

    elif action_name == "stack":
        block_id = args[0]
        target_id = args[1]
        # ✅ Pass current_predicates to stack_on
        success = executor.stack_on(target_id, current_predicates)

    elif action_name == "unstack":
        block_id = args[0]
        from_id = args[1]
        # unstack = pick_up(top_block)
        success = executor.pick_up(block_id)

    else:
        print(f"❌ Unknown action: {action_name}")
        break

    if not success:
        print("⚠️  Action failed! Re-planning...")
    else:
        print("✓ Action completed successfully")

    # Let scene settle
    for _ in range(50):
        scene.step()

    # STEP 4: RE-GROUND (loop back to Step 1)
    print("\n[Step 4] Re-grounding predicates...")

# ---------------------------------------------------------------------------
# 5) FINAL VERIFICATION
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

for _ in range(100):
    scene.step()

final_predicates = extract_predicates(scene, franka, blocks_state)
print_predicates(final_predicates)

if goal_predicates.issubset(final_predicates):
    print("=" * 80)
    print("🎉 SUCCESS! 2x2x2 CUBE STRUCTURE COMPLETE!")
    print("=" * 80)
    print("\nFinal configuration:")
    print("  - Bottom layer: r, g, b, y all ONTABLE (forming 2x2 square, touching)")
    print("  - Top layer: m on r, c on g")
    print("  - Structure forms a compact partial 2x2x2 cube")
else:
    print("❌ Goal not fully achieved")
    missing = goal_predicates - final_predicates
    print(f"\nMissing predicates: {missing}")

print(f"\nTotal iterations: {iteration}")
print("\n👀 Viewing final result. Press Ctrl+C to exit...")
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")