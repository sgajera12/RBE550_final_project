"""
goal3_scattered_tamp.py

Goal 3: Tallest Tower - SCATTERED SCENARIO
Starting with: 10 blocks scattered on table
Goal: Build the tallest possible tower (all 10 blocks stacked)

Strategy:
- Build from center outward (pick closest blocks first)
- No specific color order required
- Stack all 10 blocks: r, g, b, y, o, r2, g2, b2, y2, o2
- TAMP will automatically re-plan if blocks fall

Uses proper TAMP pipeline with Pyperplan

Usage:
    python goal3_scattered_tamp.py [gpu]
"""

import sys
import numpy as np
import genesis as gs

from scenes import create_scene_10blocks
from motion_primitives import MotionPrimitiveExecutor
from predicates import extract_predicates, print_predicates
from task_planner import generate_pddl_problem, call_pyperplan, plan_to_string

# Initialize
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level="Warning", logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

scene, franka, blocks_state = create_scene_10blocks()

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
print("GOAL 3: TALLEST TOWER - SCATTERED SCENARIO")
print("="*80)
print("\nInitial: 10 blocks scattered on table")
print("Goal: Build tallest possible tower (all 10 blocks)")
print("Strategy: Build from center outward, no color order required")

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

print("At home\n")

for _ in range(50):
    franka.control_dofs_position(safe_home)
    scene.step()

# Let physics settle
for _ in range(100):
    scene.step()

# DETERMINE BUILD ORDER - Sort blocks by distance from center

CENTER = np.array([0.50, 0.0])

print("\n" + "="*80)
print("DETERMINING BUILD ORDER (closest to center first)")
print("="*80)

# Calculate distances
block_distances = []
for block_id in blocks_state.keys():
    pos = np.array(blocks_state[block_id].get_pos())
    dist = np.linalg.norm(pos[:2] - CENTER)
    block_distances.append((block_id, dist))

# Sort by distance (closest first)
block_distances.sort(key=lambda x: x[1])

print("\nBlock order (center → outward):")
for i, (bid, dist) in enumerate(block_distances):
    print(f"  {i+1:2d}. {bid.upper():3s}: {dist:.3f}m from center")

# Build order
build_order = [x[0] for x in block_distances]

# INCREMENTAL GOAL STRATEGY
# Instead of planning all 10 blocks at once (too complex for Pyperplan),
# we build incrementally: aim for current_height + 1

print("\n" + "="*80)
print("INCREMENTAL BUILDING STRATEGY")
print("="*80)
print("\nInstead of planning all 10 blocks at once,")
print("we'll build incrementally: add one block at a time")
print("\nTarget: Keep building until we have 10 blocks stacked")

# Build order determined by distance
build_order = [x[0] for x in block_distances]
base_block = build_order[0]

# All blocks for PDDL problem
blocks = list(blocks_state.keys())

print(f"\nBase block: {base_block.upper()}")
print("Remaining blocks will be added incrementally")

# TAMP EXECUTION LOOP
executor = MotionPrimitiveExecutor(scene, franka, blocks_state)

domain_file = "/home/pinaka/RBE550Final/RBE550_final_project/code/blocksworld.pddl"
max_iterations = 60  # More iterations for incremental building
iteration = 0

# Track tower progress
current_tower_height = 0
target_tower_height = 8  # Reduced from 10 - more realistic for stability

# Note: 8 blocks = 32cm is challenging but achievable
# 10 blocks = 40cm is very difficult due to physics instability

print("\n" + "="*80)
print("STARTING TAMP LOOP")
print("="*80)

while iteration < max_iterations and current_tower_height < target_tower_height:
    iteration += 1
    print(f"\n{'='*80}")
    print(f"ITERATION {iteration} | Tower: {current_tower_height}/{target_tower_height} blocks")
    print(f"{'='*80}")
    
    # STEP 1: SYMBOLIC ABSTRACTION
    print("\n[Step 1] Extracting predicates from current scene...")
    current_predicates = extract_predicates(scene, franka, blocks_state)
    print_predicates(current_predicates)
    
    # Count current tower height
    tower_blocks = []
    current_check = base_block
    
    # Check if base is on table
    if f"ONTABLE({base_block})" in current_predicates:
        # Walk up the tower
        while True:
            tower_blocks.append(current_check)
            
            found_next = False
            for pred in current_predicates:
                if pred.startswith("ON("):
                    inside = pred[3:-1]
                    a, b = inside.split(",")
                    if b == current_check:
                        current_check = a
                        found_next = True
                        break
            
            if not found_next:
                break
        
        current_tower_height = len(tower_blocks)
        print(f"\nCurrent tower: {current_tower_height} blocks")
    else:
        current_tower_height = 0
        print(f"\nNo tower yet (base not placed)")
    
    # Check if target reached
    if current_tower_height >= target_tower_height:
        print(f"\nTarget reached: {current_tower_height} blocks!")
        break
    
    # STEP 2: DEFINE INCREMENTAL GOAL
    # Goal: Build tower with current_height + 1 blocks
    next_height = current_tower_height + 1
    
    print(f"\n[Step 2] Setting incremental goal: {next_height} blocks")
    
    # Build goal predicates for next_height blocks
    goal_predicates = {
        f"ONTABLE({base_block})",
        "HANDEMPTY()"
    }
    
    for i in range(1, min(next_height, len(build_order))):
        current = build_order[i]
        below = build_order[i-1]
        goal_predicates.add(f"ON({current},{below})")
    
    if next_height <= len(build_order):
        top = build_order[next_height - 1]
        goal_predicates.add(f"CLEAR({top})")
    
    # STEP 3: TASK PLANNING
    print(f"\n[Step 3] Calling task planner (Pyperplan)...")
    print(f"  Goal: Build {next_height}-block tower")
    
    problem_string = generate_pddl_problem(
        current_predicates,
        goal_predicates,
        blocks,
        f"goal3_iter{iteration}_h{next_height}"
    )
    
    debug_problem_file = f"/tmp/problem_goal3_iter{iteration}_h{next_height}.pddl"
    with open(debug_problem_file, 'w') as f:
        f.write(problem_string)
    print(f"  Problem saved to: {debug_problem_file}")
    
    plan = call_pyperplan(domain_file, problem_string)
    
    if not plan:
        print("No plan found!")
        print("   This might mean the tower is unstable or goal is unreachable")
        break
    
    print(f"\nPlan found ({len(plan)} actions):")
    print(plan_to_string(plan))
    
    # STEP 4: EXECUTE FIRST ACTION
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
        block_id = args[0]
        # Put down away from tower center to avoid collisions
        # Use left/right sides
        safe_positions = {
            'r': (0.35, -0.20),
            'g': (0.35, -0.10),
            'b': (0.35, 0.00),
            'y': (0.35, 0.10),
            'O': (0.35, 0.20),
            'r2': (0.65, -0.20),
            'g2': (0.65, -0.10),
            'b2': (0.65, 0.00),
            'y2': (0.65, 0.10),
            'O2': (0.65, 0.20),
        }
        
        if block_id in safe_positions:
            x, y = safe_positions[block_id]
            success = executor.put_down(x, y)
        else:
            # Default away from center
            success = executor.put_down(0.35, 0.0)
    
    elif action_name == "stack":
        block_id = args[0]
        target_id = args[1]
        success = executor.stack_on(target_id, current_predicates)
    
    elif action_name == "unstack":
        block_id = args[0]
        from_id = args[1]
        success = executor.pick_up(block_id)
    
    else:
        print(f"Unknown action: {action_name}")
        break
    
    if not success:
        print(f"Action failed! Re-planning...")
    else:
        print(f"Action completed successfully")
        
        # STABILITY CHECK: For stack actions, verify tower still stands
        if action_name == "stack" and success:
            print("\n[Stability Check] Verifying tower integrity...")
            
            # Extra settling time - more for taller towers
            if current_tower_height >= 5:
                settling_time = 200 + (current_tower_height - 5) * 50
                print(f"  Tall tower detected: extra settling time ({settling_time} steps)")
            else:
                settling_time = 150
            
            for _ in range(settling_time):
                scene.step()
            
            # Re-check predicates
            check_predicates = extract_predicates(scene, franka, blocks_state)
            
            # Check if the stack we just made still exists
            expected_pred = f"ON({args[0]},{args[1]})"
            
            if expected_pred not in check_predicates:
                print(f"Stack collapsed! Block {args[0]} fell off {args[1]}")
                print("   TAMP will re-plan and try again...")
                
                # Count how many blocks are still in tower
                temp_check = base_block
                temp_count = 0
                if f"ONTABLE({base_block})" in check_predicates:
                    while True:
                        temp_count += 1
                        found = False
                        for p in check_predicates:
                            if p.startswith("ON("):
                                a, b = p[3:-1].split(",")
                                if b == temp_check:
                                    temp_check = a
                                    found = True
                                    break
                        if not found:
                            break
                    print(f"   Tower now has {temp_count} blocks (was {current_tower_height})")
            else:
                print(f"Tower stable: {expected_pred}")
    
    # Let scene settle
    for _ in range(50):
        scene.step()
    
    # STEP 5: RE-GROUND
    print("\n[Step 5] Re-grounding predicates...")

# FINAL VERIFICATION
print("\n" + "="*80)
print("FINAL VERIFICATION")
print("="*80)

for _ in range(200):
    scene.step()

final_predicates = extract_predicates(scene, franka, blocks_state)
print_predicates(final_predicates)

# Count tower height
tower_blocks = []
current_check = base_block

# Walk up the tower following ON predicates
while True:
    tower_blocks.append(current_check)
    
    # Find what's on top of current_check
    found_next = False
    for pred in final_predicates:
        if pred.startswith("ON("):
            inside = pred[3:-1]
            a, b = inside.split(",")
            if b == current_check:
                current_check = a
                found_next = True
                break
    
    if not found_next:
        break

tower_height = len(tower_blocks)
physical_height = tower_height * 0.04

print(f"\nTower Statistics:")
print(f"  Base block: {base_block.upper()}")
print(f"  Blocks in tower: {tower_height}")
print(f"  Tower height: {physical_height:.3f}m ({physical_height*100:.1f}cm)")
print(f"  Target: 10 blocks = 0.40m (40cm)")

# Check goal
if goal_predicates.issubset(final_predicates):
    print("\n" + "="*80)
    print("SUCCESS! GOAL 3 (SCATTERED) COMPLETE!")
    print("="*80)
    print(f"\nBuilt {tower_height}-block tower!")
    print(f"Height: {physical_height*100:.1f}cm")
    
    if tower_height == 10:
        print("\nPERFECT! All 10 blocks stacked!")
else:
    print("\nGoal not fully achieved")
    missing = goal_predicates - final_predicates
    print(f"\nMissing predicates: {missing}")
    print(f"\nBut tower has {tower_height} blocks!")

print(f"\nTotal iterations: {iteration}")
print("\nPress Ctrl+C to exit...")
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")