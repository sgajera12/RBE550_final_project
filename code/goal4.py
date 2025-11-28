"""
goal4_pentagon_tamp.py

Goal 4: Pentagon Structure - Proper TAMP Implementation

Builds 5-block pentagon in layer 1 using full TAMP pipeline

Usage:
    python goal4_pentagon_tamp.py [gpu]
"""

import sys
import numpy as np
import genesis as gs

from scenes import create_scene_10blocks
from motion_primitives import MotionPrimitiveExecutor
from pentagon_geometry import PENTAGON_EDGES, PENTAGON_CENTER, print_pentagon_geometry
from pentagon_predicates import extract_pentagon_predicates, print_pentagon_predicates
from pentagon_predicates import count_pentagon_layer_blocks
from pentagon_task_planner import generate_pentagon_pddl_problem, call_pyperplan_pentagon, plan_to_string
from pentagon_motion_primitives import add_pentagon_support

# Initialize Genesis
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level="Warning", logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

print("="*80)
print("GOAL 4: PENTAGON STRUCTURE - PROPER TAMP")
print("="*80)

print_pentagon_geometry()

# Create scene
scene, franka, blocks_state = create_scene_10blocks()

franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 2000, 2000]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 200, 200]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -200, -200]),
    np.array([87, 87, 87, 87, 12, 12, 12, 200, 200]),
)

# Move to home
safe_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=float)
current = franka.get_qpos()
if hasattr(current, "cpu"):
    current = current.cpu().numpy()

for i in range(200):
    alpha = (i + 1) / 200.0
    q = (1 - alpha) * current + alpha * safe_home
    franka.control_dofs_position(q)
    scene.step()

for _ in range(100):
    scene.step()

# Select 5 closest blocks
CENTER = np.array([PENTAGON_CENTER[0], PENTAGON_CENTER[1]])
block_distances = []
for block_id in blocks_state.keys():
    pos = np.array(blocks_state[block_id].get_pos())
    dist = np.linalg.norm(pos[:2] - CENTER)
    block_distances.append((block_id, dist))

block_distances.sort(key=lambda x: x[1])
selected_blocks = [x[0] for x in block_distances[:5]]

print("\nSelected blocks (closest to center):")
for i, block_id in enumerate(selected_blocks):
    print(f"  {i+1}. {block_id.upper()}")

# Assign to edges
edge_assignments = {}
for i, block_id in enumerate(selected_blocks):
    edge_name = f"edge{i+1}"
    edge_assignments[block_id] = edge_name

# Define goal
goal_predicates = set()
for block_id, edge_name in edge_assignments.items():
    goal_predicates.add(f"AT-EDGE({block_id},{edge_name},layer1)")
    goal_predicates.add(f"EDGE-OCCUPIED({edge_name},layer1)")
goal_predicates.add("HANDEMPTY()")

# Setup executor with pentagon support
MotionPrimitiveExecutor = add_pentagon_support(MotionPrimitiveExecutor)
executor = MotionPrimitiveExecutor(scene, franka, blocks_state)

# TAMP loop
domain_file = "/home/pinaka/RBE550Final/RBE550_final_project/code/pentagon_domain.pddl"
blocks = list(blocks_state.keys())
edges = ["edge1", "edge2", "edge3", "edge4", "edge5"]
layers = ["layer1", "layer2"]

iteration = 0
max_iterations = 30

while iteration < max_iterations:
    iteration += 1
    print(f"\n{'='*80}\nITERATION {iteration}\n{'='*80}")
    
    # STEP 1: Extract predicates
    current_predicates = extract_pentagon_predicates(scene, franka, blocks_state)
    print_pentagon_predicates(current_predicates)
    
    if goal_predicates.issubset(current_predicates):
        print("\nSUCCESS! GOAL REACHED!")
        break
    
    # STEP 2: Plan
    problem_string = generate_pentagon_pddl_problem(
        current_predicates, goal_predicates, blocks, edges, layers, f"pentagon_iter{iteration}"
    )
    
    plan = call_pyperplan_pentagon(domain_file, problem_string)
    if not plan:
        print("ERROR: No plan found!")
        break
    
    print(f"\nPlan: {len(plan)} actions")
    print(plan_to_string(plan))
    
    # STEP 3: Execute first action
    action_name, args = plan[0]
    print(f"\nExecuting: {action_name.upper()}({', '.join(args)})")
    
    success = False
    if action_name == "pick-up":
        success = executor.pick_up(args[0])
    elif action_name == "place-at-edge":
        block_id, edge_name, layer_name = args[0], args[1], args[2]
        layer_num = 1 if layer_name == "layer1" else 2
        success = executor.place_at_pentagon_edge(edge_name, layer_num)
    elif action_name == "pickup-from-edge":
        success = executor.pick_up(args[0])
    
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    
    for _ in range(50):
        scene.step()

# Final verification
for _ in range(150):
    scene.step()

final_predicates = extract_pentagon_predicates(scene, franka, blocks_state)
print("\n" + "="*80)
print("FINAL STATE")
print("="*80)
print_pentagon_predicates(final_predicates)

if goal_predicates.issubset(final_predicates):
    print("\nGOAL 4 PENTAGON COMPLETE!")
    layer1_count = count_pentagon_layer_blocks(final_predicates, 1)
    print(f"Pentagon layer 1: {layer1_count}/5 blocks")
else:
    print("\nGoal not achieved")

print(f"\nTotal iterations: {iteration}")
print("\nPress Ctrl+C to exit...")

try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")