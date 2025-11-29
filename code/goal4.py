"""
goal4_pentagon_tamp_with_rotation.py

FINAL VERSION - Combines:
- TAMP loop (incremental strategy) from goal4_pentagon_incremental.py
- Working rotation function from goal4_pentagon_correct.py
- Pentagon geometry with correct positions and angles
"""

import sys
import numpy as np
import math
import genesis as gs

from scenes import create_scene_10blocks
from motion_primitives import MotionPrimitiveExecutor
from pentagon_geometry import PENTAGON_EDGES, PENTAGON_CENTER

sys.path.insert(0, '/mnt/user-data/outputs')
from pentagon_predicates_FIXED import extract_pentagon_predicates, print_pentagon_predicates
from pentagon_predicates_FIXED import count_pentagon_layer_blocks

from pentagon_task_planner import generate_pentagon_pddl_problem, call_pyperplan_pentagon, plan_to_string

if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level="Warning", logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

print("="*80)
print("GOAL 4: PENTAGON - TAMP WITH ROTATION")
print("="*80)
print("\nCombining TAMP pipeline with working rotation function")

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

# Define CENTER first (needed for block selection)
CENTER = np.array([PENTAGON_CENTER[0], PENTAGON_CENTER[1]])

# Select blocks for layer 1 - use ONE complete set
# We have two sets: line1 (r, g, b, y, o) and line2 (r2, g2, b2, y2, o2)
# Choose the set that's closer to center on average

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

if dist1 < dist2:
    selected_blocks = line1_blocks
    print(f"\nUsing line 1 blocks (closer): {selected_blocks}")
else:
    selected_blocks = line2_blocks
    print(f"\nUsing line 2 blocks (closer): {selected_blocks}")

print("\nSelected 5 blocks for layer 1 (complete set):")
for i, block_id in enumerate(selected_blocks):
    if block_id in blocks_state:
        pos = blocks_state[block_id].get_pos()
        print(f"  {i+1}. {block_id} at ({pos[0]:.3f}, {pos[1]:.3f})")
    else:
        print(f"  {i+1}. {block_id} - NOT FOUND!")

# Assign to edges
edge_assignments = {}
for i, block_id in enumerate(selected_blocks):
    edge_assignments[block_id] = f"edge{i+1}"

print("\nEdge assignments with rotations:")
for block_id, edge_name in edge_assignments.items():
    edge = PENTAGON_EDGES[edge_name]
    pos = edge.get_block_placement_position(layer=1)
    rot = edge.get_block_rotation(layer=1)
    print(f"  {block_id} -> {edge_name} at ({pos[0]:.3f}, {pos[1]:.3f}) rot={rot:.0f}°")

# Setup executor
executor = MotionPrimitiveExecutor(scene, franka, blocks_state)

# ============================================================================
# WORKING ROTATION FUNCTION (from goal4_pentagon_correct.py)
# ============================================================================

def place_rotated_no_drift(x, y, joint7_deg):
    """Place block with rotation - no arm drift"""
    if not executor.gripper_holding:
        print("  ERROR: Not holding block")
        return False
    
    print(f"  Placing at ({x:.3f}, {y:.3f}), J7={joint7_deg:.0f}°")
    
    # Find held block
    hand = executor.robot.get_link("hand")
    hand_pos = np.array(hand.get_pos())
    held_block = None
    
    for key, block in executor.blocks_state.items():
        block_pos = np.array(block.get_pos())
        dist = np.linalg.norm(block_pos - hand_pos)
        if dist < 0.15:
            held_block = block
            break
    
    TABLE_Z = 0.02
    GRASP_OFFSET = 0.12
    
    place_gripper = np.array([x, y, TABLE_Z + GRASP_OFFSET])
    approach_pos = place_gripper.copy()
    approach_pos[2] += 0.15
    
    # Move to approach
    q_approach = executor._ik_for_pose(approach_pos, executor.grasp_quat)
    if q_approach is None or not executor._plan_and_execute(q_approach, attached_object=held_block):
        print("  ERROR: Approach failed")
        return False
    
    # Rotate Joint 7
    current_q = executor.robot.get_qpos()
    if hasattr(current_q, "cpu"):
        start_q = current_q.cpu().numpy().copy()
    else:
        start_q = np.array(current_q, dtype=float, copy=True)
    
    target_j7 = math.radians(joint7_deg)
    
    print(f"  Rotating J7 from {math.degrees(start_q[6]):.0f}° to {joint7_deg:.0f}°")
    
    for i in range(30):
        alpha = (i + 1) / 30.0
        q = start_q.copy()
        q[6] = (1 - alpha) * start_q[6] + alpha * target_j7
        q[-2:] = 0.0  # Keep gripper closed
        executor.robot.control_dofs_position(q)
        executor.scene.step()
    
    # Descend with rotation
    current_q = executor.robot.get_qpos()
    if hasattr(current_q, "cpu"):
        start_q = current_q.cpu().numpy().copy()
    else:
        start_q = np.array(current_q, dtype=float, copy=True)
    
    q_place = executor._ik_for_pose(place_gripper, executor.grasp_quat)
    if q_place is None:
        print("  ERROR: IK failed for placement")
        return False
    
    q_place[6] = target_j7  # Keep rotation
    
    print(f"  Descending to placement...")
    
    for i in range(40):
        alpha = (i + 1) / 40.0
        q = (1 - alpha) * start_q + alpha * q_place
        q[6] = target_j7  # Maintain rotation
        q[-2:] = 0.0
        executor.robot.control_dofs_position(q)
        executor.scene.step()
    
    # Hold position before opening gripper
    for _ in range(50):
        q_place[-2:] = 0.0
        executor.robot.control_dofs_position(q_place)
        executor.scene.step()
    
    # Open gripper slowly (command ALL joints to prevent drift)
    current_q = executor.robot.get_qpos()
    if hasattr(current_q, "cpu"):
        arm_q = current_q.cpu().numpy().copy()
    else:
        arm_q = np.array(current_q, dtype=float, copy=True)
    
    print(f"  Opening gripper...")
    
    for i in range(50):
        alpha = (i + 1) / 50.0
        arm_q[-2:] = (1 - alpha) * 0.0 + alpha * 0.04
        executor.robot.control_dofs_position(arm_q)
        executor.scene.step()
    
    executor.gripper_holding = False
    
    # Hold position
    for _ in range(30):
        executor.robot.control_dofs_position(arm_q)
        executor.scene.step()
    
    # Skip complex lift - go DIRECTLY to safe home
    print(f"  Returning to home...")
    safe_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=float)
    
    current_q = executor.robot.get_qpos()
    if hasattr(current_q, "cpu"):
        start_q = current_q.cpu().numpy()
    else:
        start_q = np.array(current_q, dtype=float)
    
    # Gradually move to home over 150 steps (slower for safety)
    for i in range(150):
        alpha = (i + 1) / 150.0
        q = (1 - alpha) * start_q + alpha * safe_home
        executor.robot.control_dofs_position(q)
        executor.scene.step()
    
    # Hold at home
    for _ in range(50):
        executor.robot.control_dofs_position(safe_home)
        executor.scene.step()
    
    print("  ✓ Placed and returned home!")
    return True

# Pentagon placement wrapper
def place_at_pentagon_edge_tamp(edge_name, layer=1):
    """Wrapper for TAMP - uses geometry + rotation function"""
    edge = PENTAGON_EDGES[edge_name]
    target_pos = edge.get_block_placement_position(layer=layer)
    rotation_deg = edge.get_block_rotation(layer=layer)
    
    print(f"  [PENTAGON] Edge {edge_name}, Layer {layer}")
    return place_rotated_no_drift(target_pos[0], target_pos[1], rotation_deg)

executor.place_at_pentagon_edge = place_at_pentagon_edge_tamp

# ============================================================================
# TAMP LOOP (incremental strategy)
# ============================================================================

domain_file = "/home/pinaka/RBE550Final/RBE550_final_project/code/pentagon_domain.pddl"
blocks = list(blocks_state.keys())
edges = ["edge1", "edge2", "edge3", "edge4", "edge5"]
layers = ["layer1", "layer2"]

iteration = 0
max_iterations = 30
blocks_placed = 0

print("\n" + "="*80)
print("TAMP LOOP WITH ROTATION")
print("="*80)

while iteration < max_iterations and blocks_placed < 5:
    iteration += 1
    print(f"\n{'='*80}\nITERATION {iteration}\n{'='*80}")
    
    # Extract predicates
    current_predicates = extract_pentagon_predicates(scene, franka, blocks_state)
    layer1_count = count_pentagon_layer_blocks(current_predicates, 1)
    print(f"Progress: {layer1_count}/5 blocks")
    
    if layer1_count >= 5:
        print("\n✓ All 5 blocks placed!")
        break
    
    # Find next block to place
    next_block = None
    next_edge = None
    for block_id, edge_name in edge_assignments.items():
        if f"AT-EDGE({block_id},{edge_name},layer1)" not in current_predicates:
            next_block = block_id
            next_edge = edge_name
            break
    
    if not next_block:
        break
    
    print(f"\nGoal: Place {next_block} at {next_edge}")
    
    goal_predicates = {
        f"AT-EDGE({next_block},{next_edge},layer1)",
        f"EDGE-OCCUPIED({next_edge},layer1)",
        "HANDEMPTY()"
    }
    
    # Plan
    problem_string = generate_pentagon_pddl_problem(
        current_predicates, goal_predicates, blocks, edges, layers, f"iter{iteration}"
    )
    
    plan = call_pyperplan_pentagon(domain_file, problem_string)
    
    if not plan:
        print("ERROR: No plan found!")
        break
    
    print(f"Plan: {len(plan)} actions")
    
    # Execute first action
    action_name, args = plan[0]
    print(f"\nExecuting: {action_name}({', '.join(args)})")
    
    success = False
    
    if action_name == "pick-up":
        success = executor.pick_up(args[0])
    
    elif action_name == "place-at-edge":
        block_id, edge_name, layer_name = args[0], args[1], args[2]
        layer_num = 1 if layer_name == "layer1" else 2
        success = executor.place_at_pentagon_edge(edge_name, layer_num)
        
        if success:
            for _ in range(50):
                scene.step()
            
            # DEBUG: Check if block was detected at edge
            check_predicates = extract_pentagon_predicates(scene, franka, blocks_state)
            expected_pred = f"AT-EDGE({block_id},{edge_name},layer1)"
            
            print(f"\n  [DEBUG] Checking placement...")
            print(f"    Expected: {expected_pred}")
            
            if expected_pred in check_predicates:
                blocks_placed += 1
                print(f"    ✓ DETECTED! Block {blocks_placed}/5 confirmed at {edge_name}")
            else:
                print(f"    ✗ NOT DETECTED - block may have moved!")
                
                # Check actual block position
                if block_id in blocks_state:
                    actual_pos = blocks_state[block_id].get_pos()
                    edge = PENTAGON_EDGES[edge_name]
                    expected_pos = edge.get_block_placement_position(layer=1)
                    distance = np.linalg.norm([actual_pos[0] - expected_pos[0], 
                                              actual_pos[1] - expected_pos[1]])
                    print(f"    Actual: ({actual_pos[0]:.3f}, {actual_pos[1]:.3f})")
                    print(f"    Expected: ({expected_pos[0]:.3f}, {expected_pos[1]:.3f})")
                    print(f"    Distance: {distance*100:.1f}cm (tolerance: 2.5cm)")
                    
                    if distance < 0.025:
                        print(f"    Within tolerance - might be predicate bug!")
                        blocks_placed += 1  # Count it anyway

    
    elif action_name == "put-down":
        success = executor.put_down(0.35, 0.0)
    
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    
    for _ in range(50):
        scene.step()

# Final verification
print("\n" + "="*80)
print("FINAL VERIFICATION")
print("="*80)

for _ in range(150):
    scene.step()

final_predicates = extract_pentagon_predicates(scene, franka, blocks_state)
print_pentagon_predicates(final_predicates)

layer1_count = count_pentagon_layer_blocks(final_predicates, 1)

if layer1_count >= 5:
    print(f"\n🎉 GOAL 4 COMPLETE WITH ROTATION!")
    print(f"   Pentagon layer 1: {layer1_count}/5 blocks")
    print(f"   Each block rotated to align with pentagon edges")
    print(f"   Total iterations: {iteration}")
else:
    print(f"\nPartial: {layer1_count}/5 blocks")

print("\n👀 Check top view - blocks should form pentagon with rotations!")
print("\nPress Ctrl+C to exit...")

try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")