"""
Task 1: Two 3-block towers - SCATTERED SCENARIO
Starting with: All 6 blocks in grounded scattered positions

Our Pipeline:
1. Symbolic Abstraction: Extract predicates from scene
2. Task Planning: Call Pyperplan to get action sequence
3. Execute first action: Use motion primitives
4. Re-ground: Extract new predicates
5. Re-plan if needed: Loop until goal reached

Goal:
Tower 1: GREEN-RED-BLUE (g on table, r on g, b on r)
Tower 2: MAGENTA-YELLOW-CYAN (m on table, y on m, c on y)

"""

import sys
import numpy as np
import genesis as gs

from scenes import create_scene_6blocks
from motion_primitives import MotionPrimitiveExecutor
from predicates import extract_predicates, print_predicates
from task_planner import generate_pddl_problem, call_pyperplan, plan_to_string

# Initialize
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level="Warning", logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

scene, franka, blocks_state = create_scene_6blocks()

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

# DEFINE GOAL
goal_predicates = {
    #Tower 1: g-r-b
    "ON(r,g)",
    "ON(b,r)",
    "ONTABLE(g)",
    "CLEAR(b)",
    #Tower 2: m-y-c
    "ON(y,m)",
    "ON(c,y)",
    "ONTABLE(m)",
    "CLEAR(c)",
    #Gripper
    "HANDEMPTY()"
}

blocks = ['r', 'g', 'b', 'y', 'm', 'c']
print("Goal state")
print("\nTower 1: GREEN > RED > BLUE | Tower 2: MAGENTA > YELLOW > CYAN")
print("\nGoal predicates:")

for p in sorted(goal_predicates):
    print(f" {p}")

# Tamp execution setup
executor = MotionPrimitiveExecutor(scene, franka, blocks_state)
domain_file = "/home/pinaka/RBE550Final/RBE550_final_project/code/blocksworld.pddl"
max_iterations = 20
iteration = 0

print("\nStarting TAMP loop")
while iteration < max_iterations:
    iteration += 1
    print(f"iteration {iteration}")
    
    # STEP 1: SYMBOLIC ABSTRACTION
    print("\n1.Extracting predicates from current scene")
    current_predicates = extract_predicates(scene, franka, blocks_state)
    print_predicates(current_predicates)
    
    # Check if goal reached
    if goal_predicates.issubset(current_predicates):
        print("\nGOAL REACHED!")
        break
    
    # STEP 2: TASK PLANNING
    print("\n2.Calling task planner (Pyperplan)")
    problem_string = generate_pddl_problem(current_predicates,goal_predicates,blocks,f"goal1_iter{iteration}")
    
    # Debug: Save problem file
    debug_problem_file = f"/tmp/problem_iter{iteration}.pddl"
    with open(debug_problem_file, 'w') as f:
        f.write(problem_string)
    print(f"Problem saved to: {debug_problem_file}")
    plan = call_pyperplan(domain_file, problem_string)
    
    if not plan:
        print("no plan found. Cannot reach goal.")
        break
    
    print(f"\nPlan found ({len(plan)} actions):")
    print(plan_to_string(plan))
    
    # STEP 3: EXECUTE FIRST ACTION
    if not plan:
        print("\nPlan is empty.")
        break
    
    action_name, args = plan[0]
    print(f"\n3.Executing: {action_name.upper()}({', '.join(args)})")
    success = False
    
    if action_name == "pick-up":
        block_id = args[0]
        success = executor.pick_up(block_id)
    elif action_name == "put-down":
        block_id = args[0]
        success = executor.put_down()
    elif action_name == "stack":
        block_id = args[0]
        target_id = args[1]
        success = executor.stack_on(target_id,current_predicates)
    elif action_name == "unstack":
        block_id = args[0]
        from_id = args[1]
        success = executor.pick_up(block_id)
    else:
        print(f"Unknown action: {action_name}")
        break
    
    if not success:
        print(f"Action failed! Re-planning")
    else:
        print(f"Action completed successfully")
    
    # To let scene settle
    for _ in range(50):
        scene.step()
    
    # STEP 4: RE-GROUND (loop back to Step 1)
    print("\n4.Re-grounding predicates")

# FINAL VERIFICATION
for _ in range(100):
    scene.step()

final_predicates = extract_predicates(scene, franka, blocks_state)
print_predicates(final_predicates)

# Check goal
if goal_predicates.issubset(final_predicates):
    print("Goal fully achieved!")
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
    print("\nExiting simulation.")