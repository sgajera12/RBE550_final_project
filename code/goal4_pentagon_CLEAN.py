"""
goal4_pentagon_CLEAN.py

Clean TAMP implementation - extends put_down with rotation
"""

import sys
import numpy as np
import math
import genesis as gs

from scenes import create_scene_10blocks
from motion_primitives import MotionPrimitiveExecutor
from pentagon_geometry import PENTAGON_EDGES, PENTAGON_CENTER

sys.path.insert(0, '/mnt/user-data/outputs')
from pentagon_predicates_FIXED import extract_pentagon_predicates, count_pentagon_layer_blocks
from pentagon_task_planner import generate_pentagon_pddl_problem, call_pyperplan_pentagon

if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level="Warning", logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

print("="*80)
print("GOAL 4: PENTAGON - CLEAN IMPLEMENTATION")
print("="*80)

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

for i in range(100):
    alpha = (i + 1) / 200.0
    q = (1 - alpha) * current + alpha * safe_home
    franka.control_dofs_position(q)
    scene.step()

for _ in range(100):
    scene.step()

# Select blocks
CENTER = np.array([PENTAGON_CENTER[0], PENTAGON_CENTER[1]])

line1_blocks = ["r", "g", "b", "y", "o"]
line2_blocks = ["r2", "g2", "b2", "y2", "o2"]

def calc_avg_dist(block_list):
    total = 0
    count = 0
    for bid in block_list:
        if bid in blocks_state:
            pos = np.array(blocks_state[bid].get_pos())
            dist = np.linalg.norm(pos[:2] - CENTER)
            total += dist
            count += 1
    return total / count if count > 0 else 999

dist1 = calc_avg_dist(line1_blocks)
dist2 = calc_avg_dist(line2_blocks)

selected_blocks = line1_blocks if dist1 < dist2 else line2_blocks
print(f"\nUsing blocks: {selected_blocks}")

# Assign to edges
edge_assignments = {}
for i, block_id in enumerate(selected_blocks):
    edge_assignments[block_id] = f"edge{i+1}"

# Setup executor
executor = MotionPrimitiveExecutor(scene, franka, blocks_state)

def put_down_with_rotation(x, y, rotation_deg):
    """
    put_down with rotation - follows original logic exactly
    """
    if not executor.gripper_holding:
        return False
    
    print(f"  Placing at ({x:.3f}, {y:.3f}), rot={rotation_deg:.0f}°")
    
    target_j7 = math.radians(rotation_deg)
    TABLE_BLOCK_CENTER_Z = 0.02
    GRASP_OFFSET = 0.12
    
    # Find held block
    hand = executor.robot.get_link("hand")
    hand_pos = np.array(hand.get_pos())
    held_block = None
    for key, block in executor.blocks_state.items():
        block_pos = np.array(block.get_pos())
        if np.linalg.norm(block_pos - hand_pos) < 0.15:
            held_block = block
            break
    
    # Calculate positions
    place_center = np.array([x, y, TABLE_BLOCK_CENTER_Z])
    place_gripper = place_center.copy()
    place_gripper[2] = TABLE_BLOCK_CENTER_Z + GRASP_OFFSET
    
    approach_pos = place_gripper.copy()
    approach_pos[2] += 0.15
    
    # Move to approach (with motion planning)
    q_approach = executor._ik_for_pose(approach_pos, executor.grasp_quat)
    if q_approach is None or not executor._plan_and_execute(q_approach, attached_object=held_block):
        return False
    
    # Rotate wrist AT APPROACH
    current_q = executor.robot.get_qpos()
    if hasattr(current_q, "cpu"):
        q = current_q.cpu().numpy().copy()
    else:
        q = np.array(current_q, dtype=float, copy=True)
    
    start_j7 = q[6]
    for i in range(30):
        alpha = (i + 1) / 30.0
        q[6] = (1 - alpha) * start_j7 + alpha * target_j7
        q[-2:] = 0.0
        executor.robot.control_dofs_position(q)
        executor.scene.step()
    
    # Descend (direct interpolation with rotation maintained)
    current_q = executor.robot.get_qpos()
    if hasattr(current_q, "cpu"):
        start_q = current_q.cpu().numpy().copy()
    else:
        start_q = np.array(current_q, dtype=float, copy=True)
    
    q_place = executor._ik_for_pose(place_gripper, executor.grasp_quat)
    if q_place is None:
        return False
    
    q_place[6] = target_j7  # Override rotation
    
    for i in range(40):
        alpha = (i + 1) / 40.0
        q = (1 - alpha) * start_q + alpha * q_place
        q[6] = target_j7  # Force rotation
        q[-2:] = 0.0
        executor.robot.control_dofs_position(q)
        executor.scene.step()
    
    # Open gripper
    executor.open_gripper()
    
    # Lift up (SAME AS ORIGINAL put_down)
    current_pos = np.array(hand.get_pos())
    up_pos = current_pos.copy()
    up_pos[2] += 0.10
    
    q_up = executor._ik_for_pose(up_pos, executor.grasp_quat)
    if q_up is not None:
        current_q = executor.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        for i in range(30):
            alpha = (i + 1) / 30.0
            q = (1 - alpha) * start_q + alpha * q_up
            executor.robot.control_dofs_position(q)
            executor.scene.step()
    
    return True

def place_at_edge(edge_name):
    edge = PENTAGON_EDGES[edge_name]
    pos = edge.get_block_placement_position(layer=1)
    rot = edge.get_block_rotation(layer=1)
    return put_down_with_rotation(pos[0], pos[1], rot)

executor.place_at_pentagon_edge = place_at_edge

# TAMP LOOP
domain_file = "/home/pinaka/RBE550Final/RBE550_final_project/code/pentagon_domain.pddl"
blocks = list(blocks_state.keys())
edges = ["edge1", "edge2", "edge3", "edge4", "edge5"]
layers = ["layer1", "layer2"]

iteration = 0
max_iterations = 30
blocks_placed = 0

print("\n" + "="*80)
print("TAMP LOOP")
print("="*80)

while iteration < max_iterations and blocks_placed < 5:
    iteration += 1
    print(f"\n--- ITERATION {iteration} ---")
    
    current_predicates = extract_pentagon_predicates(scene, franka, blocks_state)
    layer1_count = count_pentagon_layer_blocks(current_predicates, 1)
    print(f"Progress: {layer1_count}/5")
    
    if layer1_count >= 5:
        print("DONE!")
        break
    
    # Find next block
    next_block = None
    next_edge = None
    for block_id, edge_name in edge_assignments.items():
        if f"AT-EDGE({block_id},{edge_name},layer1)" not in current_predicates:
            next_block = block_id
            next_edge = edge_name
            break
    
    if not next_block:
        break
    
    print(f"Goal: {next_block} -> {next_edge}")
    
    goal_predicates = {
        f"AT-EDGE({next_block},{next_edge},layer1)",
        "HANDEMPTY()"
    }
    
    problem_string = generate_pentagon_pddl_problem(
        current_predicates, goal_predicates, blocks, edges, layers, f"iter{iteration}"
    )
    
    plan = call_pyperplan_pentagon(domain_file, problem_string)
    
    if not plan:
        print("No plan!")
        break
    
    action_name, args = plan[0]
    print(f"Action: {action_name}({', '.join(args)})")
    
    success = False
    
    if action_name == "pick-up":
        success = executor.pick_up(args[0])
    elif action_name == "place-at-edge":
        success = executor.place_at_pentagon_edge(args[1])
        if success:
            blocks_placed += 1
    
    print(f"Result: {'✓' if success else '✗'}")
    
    for _ in range(50):
        scene.step()

print("\n" + "="*80)
print("FINAL")
print("="*80)

for _ in range(100):
    scene.step()

final_predicates = extract_pentagon_predicates(scene, franka, blocks_state)
layer1_count = count_pentagon_layer_blocks(final_predicates, 1)

print(f"\nBlocks: {layer1_count}/5")
print(f"Iterations: {iteration}")

if layer1_count >= 5:
    print("\nGOAL 4 COMPLETE!")

print("\nCtrl+C to exit...")
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("Done")
