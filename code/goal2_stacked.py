"""
Task 2: Single 5-block tower - STACKED SCENARIO
Starting with: All 6 blocks in one tower (R-G-B-Y-M-C from bottom to top)

Goal: Build one 5-block tower
Tower: GREEN-RED-BLUE-YELLOW-MAGENTA (g-r-b-y-m)
Cyan (c) remains on table
"""

import sys
import numpy as np
import genesis as gs

from scenes import create_scene_stacked
from motion_primitives import MotionPrimitiveExecutor
from predicates import extract_predicates, print_predicates
from task_planner import generate_pddl_problem, call_pyperplan, plan_to_string

# Initialize Genesis
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level="Warning", logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

scene, franka, blocks_state = create_scene_stacked()

# For Strong gripper control
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 2000, 2000]),)
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 200, 200]),)
franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -200, -200]),np.array([87, 87, 87, 87, 12, 12, 12, 200, 200]),)

print("\nScene and Franka robot initialized.")

# Our Home position (safe for 6-block setup)
safe_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=float)

print("\nMoving to home position")
current = franka.get_qpos()
if hasattr(current, "cpu"):
    current = current.cpu().numpy()

for i in range(200):
    alpha = (i + 1) / 200.0
    q = (1 - alpha) * current + alpha * safe_home
    franka.control_dofs_position(q)
    scene.step()

print("At home\n")

for _ in range(50):
    franka.control_dofs_position(safe_home)
    scene.step()

# Let physics settle
for _ in range(100):
    scene.step()

# DEFINE GOAL - Same as scattered scenario
goal_predicates = {
    "ONTABLE(g)",
    "ON(r,g)",
    "ON(b,r)",
    "ON(y,b)",
    "ON(m,y)",
    # top block clear
    "CLEAR(m)",
    # Cyan remains on table
    "ONTABLE(c)",
    "CLEAR(c)",
    # Gripper
    "HANDEMPTY()"
}

blocks = ['r', 'g', 'b', 'y', 'm', 'c']

print("Goal state")
print("\nTower: GREEN > RED > BLUE > YELLOW > MAGENTA")
print("Cyan remains on table")
print("\nGoal predicates:")

for p in sorted(goal_predicates):
    print(f"{p}")

# TAMP EXECUTION LOOP
executor = MotionPrimitiveExecutor(scene, franka, blocks_state)
domain_file = "/home/pinaka/RBE550Final/RBE550_final_project/code/blocksworld.pddl"
max_iterations = 30
iteration = 0

print("\nStarting TAMP loop")
while iteration < max_iterations:
    iteration += 1
    print(f"iteration {iteration}")

    # STEP 1: SYMBOLIC ABSTRACTION
    print("\n1.Extracting predicates from current scene")
    current_predicates = extract_predicates(scene, franka, blocks_state)
    print_predicates(current_predicates)

    # Check goal reached
    if goal_predicates.issubset(current_predicates):
        print("\nGOAL REACHED!")
        break

    # STEP 2: TASK PLANNING
    print("\n2.Calling task planner (Pyperplan)")
    problem_string = generate_pddl_problem(current_predicates,goal_predicates,blocks,f"goal2_stacked_iter{iteration}")

    # Debug: Save problem to file
    debug_problem_file = f"/tmp/problem_goal2_stacked_iter{iteration}.pddl"
    with open(debug_problem_file, "w") as f:
        f.write(problem_string)
    print(f"  Problem saved to: {debug_problem_file}")

    plan = call_pyperplan(domain_file, problem_string)

    if not plan:
        print("no plan found Cannot reach goal.")
        break

    print(f"\nPlan found ({len(plan)} actions):")
    print(plan_to_string(plan))

    # STEP 3: Execute FIRST action
    action_name, args = plan[0]
    print(f"\n3.Executing: {action_name.upper()}({', '.join(args)})")
    success = False

    if action_name == "pick-up":
        block_id = args[0]
        success = executor.pick_up(block_id)
    elif action_name == "put-down":
        block_id = args[0]
        # Same safe positions as Goal 1 stacked
        safe_positions = {
            'r': (0.45, -0.45),
            'g': (0.45,  -0.20),
            'b': (0.45,  0.15),
            'y': (0.65, -0.15),
            'm': (0.65,  0.00),
            'c': (0.65,  0.15),
        }
        if block_id in safe_positions:
            x, y = safe_positions[block_id]
            success = executor.put_down(x, y)
        else:
            success = executor.put_down()
    elif action_name == "stack":
        block_id = args[0]
        target_id = args[1]
        success = executor.stack_on(target_id, current_predicates)
    elif action_name == "unstack":
        block_id = args[0]
        success = executor.pick_up(block_id)
    else:
        print(f"Unknown action: {action_name}")
        break

    if not success:
        print("Action failed! Re-planning")
    else:
        print("Action completed successfully")

    for _ in range(50):
        scene.step()

    # STEP 4: re-ground 
    print("\n4.Re-grounding predicates")

# FINAL VERIFICATION
for _ in range(100):
    scene.step()

final_predicates = extract_predicates(scene, franka, blocks_state)
print_predicates(final_predicates)

if goal_predicates.issubset(final_predicates):
    print("\nGoal fully achieved!")
else:
    print("Goal not fully achieved")
    missing = goal_predicates - final_predicates
    print(f"\nMissing predicates: {missing}")

print(f"\nTotal iterations: {iteration}")
print("\nCtrl+C to exit")
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting simulation.`")
