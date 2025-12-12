"""
Special Goal 4 Task 2: 3 Red and 3 Green Blocks with Directional Adjacency
"""

import sys
import os
import numpy as np
import genesis as gs

from scenes import create_scene_3red_3green
from motion_primitives_sp2 import MotionPrimitiveExecutor
from predicates_adjacent_sp2 import extract_predicates_directional as extract_predicates_with_adjacency, print_predicates
from task_planner_sp2 import generate_pddl_problem, call_pyperplan, plan_to_string


BLOCK_SIZE = 0.04  # 4cm blocks


def get_adjacent_blocks_info(block_id: str, blocks_state: dict) -> dict:
    """
    Determine which blocks are adjacent and in which directions.
    
    Returns:
        dict: {'+x': block_name, '-x': block_name, '+y': block_name, '-y': block_name}
    """
    if block_id not in blocks_state:
        return {}
    
    ref_pos = np.array(blocks_state[block_id].get_pos())
    adjacent = {}
    
    ADJACENCY_THRESHOLD = BLOCK_SIZE + 0.02  # 4cm + tolerance
    
    for other_id, other_obj in blocks_state.items():
        if other_id == block_id:
            continue
        
        other_pos = np.array(other_obj.get_pos())
        
        # Only consider blocks at similar Z height (same layer)
        if abs(other_pos[2] - ref_pos[2]) > 0.025:
            continue
        
        dx = other_pos[0] - ref_pos[0]
        dy = other_pos[1] - ref_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Check if blocks are adjacent (approximately one block width apart)
        if dist < ADJACENCY_THRESHOLD:
            # Determine direction
            if abs(dx) > abs(dy):  # Horizontal adjacency
                if dx > 0:
                    adjacent['+x'] = other_id
                else:
                    adjacent['-x'] = other_id
            else:  # Vertical adjacency
                if dy > 0:
                    adjacent['+y'] = other_id
                else:
                    adjacent['-y'] = other_id
    
    return adjacent


def calculate_gripper_rotation(adjacent_blocks: dict) -> float:
    """
    Calculate gripper wrist rotation to avoid collisions.
    
    LOGIC:
    - Default gripper (0°): Fingers extend in Y direction
    - If blocks adjacent in Y direction (+y or -y): Rotate 90° → fingers point in X
    - If blocks adjacent in X direction (+x or -x): Keep 0° → fingers stay in Y
    
    Returns:
        float: Wrist rotation in radians (0 or π/2)
    """
    if not adjacent_blocks:
        return 0.0  # Default orientation
    
    # If blocks in Y direction, rotate to avoid them
    if '+y' in adjacent_blocks or '-y' in adjacent_blocks:
        print(f"    [rotation] Blocks in Y direction → rotating 90° (fingers to X)")
        return np.pi / 2  # 90° rotation
    
    # If blocks only in X direction, keep default (fingers in Y)
    if '+x' in adjacent_blocks or '-x' in adjacent_blocks:
        print(f"    [rotation] Blocks in X direction → no rotation (fingers stay in Y)")
    
    return 0.0

if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level="Warning", logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

scene, franka, blocks_state = create_scene_3red_3green()

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
print("DIRECTIONAL ADJACENCY TAMP: Precise X/Y Placement")
print("=" * 80)

# Move to home
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

for _ in range(100):
    scene.step()

executor = MotionPrimitiveExecutor(scene, franka, blocks_state)
domain_file = os.path.join(os.path.dirname(__file__), "blocksworld_directional.pddl")

print("\n" + "=" * 80)
print("PHASE 1: BASE CONSTRUCTION (DIRECTIONAL ADJACENCY)")
print("=" * 80)
print("\nGoal: Arrange 4 blocks in 2x2 grid:")
print("  [r2][g2]  ← Back row")
print("  [r1][g1]  ← Front row")

phase1_goal = {
    # All blocks on table
    "ONTABLE(r1)", "ONTABLE(g1)", "ONTABLE(r2)", "ONTABLE(g2)",
    
    # X-direction adjacency (horizontal: left → right)
    "ADJACENT-X(g1,r1)",  # g1 is RIGHT of r1
    "ADJACENT-X(g2,r2)",  # g2 is RIGHT of r2
    
    # Y-direction adjacency (vertical: back → front)
    "ADJACENT-Y(r1,r2)",  # r1 is FRONT of r2
    "ADJACENT-Y(g1,g2)",  # g1 is FRONT of g2
    
    # All blocks clear and hand empty
    "CLEAR(r1)", "CLEAR(g1)", "CLEAR(r2)", "CLEAR(g2)",
    "HANDEMPTY()",
}

print("\nPhase 1 Goal Predicates:")
for p in sorted(phase1_goal):
    print(f"  {p}")

phase1_blocks = ["r1", "g1", "r2", "g2"]

# ============================================================================
# PHASE 1: PLAN ONCE AND EXECUTE
# ============================================================================

MAX_REPLAN_ATTEMPTS = 3
replan_attempt = 0

while replan_attempt < MAX_REPLAN_ATTEMPTS:
    replan_attempt += 1
    
    print(f"\n{'=' * 80}")
    print(f"PHASE 1 - ATTEMPT {replan_attempt}")
    print(f"{'=' * 80}")

    # Get current state
    current_predicates = extract_predicates_with_adjacency(scene, franka, blocks_state)
    
    relevant_predicates = {p for p in current_predicates 
                          if any(block in p for block in phase1_blocks) 
                          or p == "HANDEMPTY()"}
    
    print("\nInitial state:")
    print_predicates(relevant_predicates)

    # Check if goal already satisfied
    if phase1_goal.issubset(current_predicates):
        print("\n🎉 PHASE 1 ALREADY COMPLETE!")
        break

    # PLANNING
    print("\n[Planning] Generating complete plan for Phase 1...")
    
    problem_string = generate_pddl_problem(
        relevant_predicates,
        phase1_goal,
        phase1_blocks,
        f"phase1_attempt{replan_attempt}",
        domain_name="blocksworld-directional"
    )

    debug_problem_file = f"/tmp/phase1_problem_attempt{replan_attempt}.pddl"
    with open(debug_problem_file, "w") as f:
        f.write(problem_string)
    print(f"  Problem saved to: {debug_problem_file}")

    plan = call_pyperplan(domain_file, problem_string)

    if not plan:
        print("❌ No plan found for Phase 1!")
        break

    print(f"\n✓ Complete plan generated with {len(plan)} actions")
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE PLAN:")
    print("="*60)
    print(plan_to_string(plan))
    print("="*60)
    
    # EXECUTION
    print(f"\n[Execution] Executing complete plan ({len(plan)} actions)...")
    print("Note: No replanning between actions!")
    
    plan_success = True
    
    for action_idx, (action_name, args) in enumerate(plan, 1):
        print(f"\n{'─' * 60}")
        print(f"ACTION {action_idx}/{len(plan)}: {action_name.upper()}({', '.join(args)})")
        print(f"{'─' * 60}")

        success = False

        if action_name == "pick-up":
            block_id = args[0]
            print(f"  → Picking up block '{block_id}'")
            
            adjacent = get_adjacent_blocks_info(block_id, blocks_state)
            if adjacent:
                print(f"  ℹ️  Adjacent blocks: {adjacent}")
                wrist_rotation = calculate_gripper_rotation(adjacent)
                print(f"  ℹ️  Gripper rotation: {np.degrees(wrist_rotation):.0f}°")
                success = executor.pick_up(block_id, wrist_rotation=wrist_rotation)
            else:
                success = executor.pick_up(block_id)

        elif action_name == "put-down":
            print(f"  → Putting down held block on table")
            success = executor.put_down()

        elif action_name == "put-down-adjacent":
            # Generic adjacent (for compatibility)
            block_id = args[0]
            adjacent_to = args[1]
            print(f"  → Placing '{block_id}' adjacent to '{adjacent_to}'")
            success = executor.put_down_adjacent_to(adjacent_to, current_predicates)
        
        elif action_name == "put-down-adjacent-x":
            block_id = args[0]
            adjacent_to = args[1]
            print(f"  → Placing '{block_id}' to RIGHT of '{adjacent_to}' (+X direction)")
            success = executor.put_down_adjacent_x(adjacent_to, direction="+x")
        
        elif action_name == "put-down-adjacent-y":
            block_id = args[0]
            adjacent_to = args[1]
            print(f"  → Placing '{block_id}' in FRONT of '{adjacent_to}' (+Y direction)")
            success = executor.put_down_adjacent_y(adjacent_to, direction="+y")

        elif action_name == "unstack":
            block_id = args[0]
            from_id = args[1]
            print(f"  → Unstacking '{block_id}' from '{from_id}'")
            
            adjacent = get_adjacent_blocks_info(block_id, blocks_state)
            if adjacent:
                print(f"  ℹ️  Adjacent blocks: {adjacent}")
                wrist_rotation = calculate_gripper_rotation(adjacent)
                print(f"  ℹ️  Gripper rotation: {np.degrees(wrist_rotation):.0f}°")
                success = executor.pick_up(block_id, wrist_rotation=wrist_rotation)
            else:
                success = executor.pick_up(block_id)

        else:
            print(f"  ❌ Unknown action: {action_name}")
            plan_success = False
            break

        if not success:
            print(f"  ❌ Action FAILED!")
            plan_success = False
            break
        else:
            print(f"  ✓ Action completed successfully")

        # Settle physics
        for _ in range(200):
            scene.step()
        
        # **NEW: Check if goal already satisfied after this action**
        current_predicates = extract_predicates_with_adjacency(scene, franka, blocks_state)
        if phase1_goal.issubset(current_predicates):
            print(f"\n✅ GOAL ACHIEVED after action {action_idx}/{len(plan)}! Skipping remaining actions.")
            break

        for _ in range(200):
            scene.step()
    
    # CHECK GOAL
    print("\n" + "=" * 60)
    print("PHASE 1: Checking goal satisfaction...")
    print("=" * 60)
    
    for _ in range(100):
        scene.step()
    
    final_predicates = extract_predicates_with_adjacency(scene, franka, blocks_state)
    
    if phase1_goal.issubset(final_predicates):
        print("✅ PHASE 1 GOAL ACHIEVED!")
        break
    else:
        print("⚠️  Phase 1 goal not fully satisfied")
        missing = phase1_goal - final_predicates
        print(f"\nMissing predicates ({len(missing)}):")
        for p in sorted(missing):
            print(f"  {p}")
        
        if replan_attempt < MAX_REPLAN_ATTEMPTS:
            print(f"\n🔄 Replanning (attempt {replan_attempt + 1}/{MAX_REPLAN_ATTEMPTS})...")
        else:
            print("\n❌ Max replan attempts reached")
            break


# Final Phase 1 verification
print("\n" + "=" * 80)
print("PHASE 1 FINAL VERIFICATION")
print("=" * 80)

phase1_final = extract_predicates_with_adjacency(scene, franka, blocks_state)

if not phase1_goal.issubset(phase1_final):
    print("❌ Phase 1 FAILED - base not properly constructed")
    print("\nMissing predicates:")
    missing = phase1_goal - phase1_final
    for p in sorted(missing):
        print(f"  {p}")
    sys.exit(1)

print("✅ Phase 1 SUCCESS - Base is complete!")

print("\nBase block positions:")
for block in phase1_blocks:
    pos = blocks_state[block].get_pos()
    print(f"  {block}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")


# ============================================================================
# PHASE 2: STACKING
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 2: STACKING")
print("=" * 80)
print("\nGoal: Stack r3 on r1, g3 on g2")

phase2_goal = {
    # Base blocks still on table
    "ONTABLE(r1)", "ONTABLE(g1)", "ONTABLE(r2)", "ONTABLE(g2)",
    
    # Base adjacency preserved
    "ADJACENT-X(g1,r1)", "ADJACENT-X(g2,r2)",
    "ADJACENT-Y(r1,r2)", "ADJACENT-Y(g1,g2)",
    
    # Stacking goals
    "ON(r3,r1)", "ON(g3,g1)",
    
    # Only the TOP blocks need to be clear
    "CLEAR(r3)", "CLEAR(g3)",
    "CLEAR(r2)", "CLEAR(g2)",  # These aren't stacked on
    
    # Hand empty
    "HANDEMPTY()",
}
phase2_blocks = ["r1", "r2", "r3", "g1", "g2", "g3"]

MAX_REPLAN_ATTEMPTS = 3
replan_attempt = 0

while replan_attempt < MAX_REPLAN_ATTEMPTS:
    replan_attempt += 1
    
    print(f"\n{'=' * 80}")
    print(f"PHASE 2 - ATTEMPT {replan_attempt}")
    print(f"{'=' * 80}")

    current_predicates = extract_predicates_with_adjacency(scene, franka, blocks_state)
    
    print("\nInitial state:")
    print_predicates(current_predicates)

    if phase2_goal.issubset(current_predicates):
        print("\n🎉 PHASE 2 ALREADY COMPLETE!")
        break

    # PLANNING
    print("\n[Planning] Generating complete plan for Phase 2...")
    
    problem_string = generate_pddl_problem(
        current_predicates,
        phase2_goal,
        phase2_blocks,
        f"phase2_attempt{replan_attempt}",
        domain_name="blocksworld-directional"
    )

    debug_problem_file = f"/tmp/phase2_problem_attempt{replan_attempt}.pddl"
    with open(debug_problem_file, "w") as f:
        f.write(problem_string)
    print(f"  Problem saved to: {debug_problem_file}")

    plan = call_pyperplan(domain_file, problem_string)

    if not plan:
        print("❌ No plan found for Phase 2!")
        break

    print(f"\n✓ Complete plan generated with {len(plan)} actions")
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE PLAN:")
    print("="*60)
    print(plan_to_string(plan))
    print("="*60)
    
    # EXECUTION
    print(f"\n[Execution] Executing complete plan ({len(plan)} actions)...")
    print("Note: No replanning between actions!")
    
    plan_success = True
    
    for action_idx, (action_name, args) in enumerate(plan, 1):
        print(f"\n{'─' * 60}")
        print(f"ACTION {action_idx}/{len(plan)}: {action_name.upper()}({', '.join(args)})")
        print(f"{'─' * 60}")

        success = False

        if action_name == "pick-up":
            block_id = args[0]
            print(f"  → Picking up block '{block_id}'")
            
            adjacent = get_adjacent_blocks_info(block_id, blocks_state)
            if adjacent:
                print(f"  ℹ️  Adjacent blocks: {adjacent}")
                wrist_rotation = calculate_gripper_rotation(adjacent)
                print(f"  ℹ️  Gripper rotation: {np.degrees(wrist_rotation):.0f}°")
                success = executor.pick_up(block_id, wrist_rotation=wrist_rotation)
            else:
                success = executor.pick_up(block_id)

        elif action_name == "put-down":
            print(f"  → Putting down held block")
            success = executor.put_down()
        
        elif action_name == "put-down-adjacent-x":
            block_id = args[0]
            adjacent_to = args[1]
            print(f"  → Placing '{block_id}' to RIGHT of '{adjacent_to}' (+X direction)")
            success = executor.put_down_adjacent_x(adjacent_to, direction="+x")
        
        elif action_name == "put-down-adjacent-y":
            block_id = args[0]
            adjacent_to = args[1]
            print(f"  → Placing '{block_id}' in FRONT of '{adjacent_to}' (+Y direction)")
            success = executor.put_down_adjacent_y(adjacent_to, direction="+y")

        elif action_name == "stack":
            block_id = args[0]
            target_id = args[1]
            print(f"  → Stacking '{block_id}' on '{target_id}'")
            
            adjacent = get_adjacent_blocks_info(target_id, blocks_state)
            if adjacent:
                print(f"  ℹ️  Blocks around target '{target_id}': {adjacent}")
            
            success = executor.stack_on(target_id, current_predicates)

        elif action_name == "unstack":
            block_id = args[0]
            from_id = args[1]
            print(f"  → Unstacking '{block_id}' from '{from_id}'")
            
            adjacent = get_adjacent_blocks_info(block_id, blocks_state)
            if adjacent:
                print(f"  ℹ️  Adjacent blocks: {adjacent}")
                wrist_rotation = calculate_gripper_rotation(adjacent)
                print(f"  ℹ️  Gripper rotation: {np.degrees(wrist_rotation):.0f}°")
                success = executor.pick_up(block_id, wrist_rotation=wrist_rotation)
            else:
                success = executor.pick_up(block_id)

        else:
            print(f"  ❌ Unknown action: {action_name}")
            plan_success = False
            break

        if not success:
            print(f"  ❌ Action FAILED!")
            plan_success = False
            break
        else:
            print(f"  ✓ Action completed successfully")

        # Settle physics
        for _ in range(100):
            scene.step()
        
        # **NEW: Check if goal already satisfied after this action**
        current_predicates = extract_predicates_with_adjacency(scene, franka, blocks_state)
        if phase2_goal.issubset(current_predicates):
            print(f"\n✅ GOAL ACHIEVED after action {action_idx}/{len(plan)}! Skipping remaining actions.")
            break

        for _ in range(100):
            scene.step()
    
    # CHECK GOAL
    print("\n" + "=" * 60)
    print("PHASE 2: Checking goal satisfaction...")
    print("=" * 60)
    
    for _ in range(100):
        scene.step()
    
    final_predicates = extract_predicates_with_adjacency(scene, franka, blocks_state)
    
    if phase2_goal.issubset(final_predicates):
        print("✅ PHASE 2 GOAL ACHIEVED!")
        break
    else:
        print("⚠️  Phase 2 goal not fully satisfied")
        missing = phase2_goal - final_predicates
        print(f"\nMissing predicates ({len(missing)}):")
        for p in sorted(missing):
            print(f"  {p}")
        
        if replan_attempt < MAX_REPLAN_ATTEMPTS:
            print(f"\n🔄 Replanning (attempt {replan_attempt + 1}/{MAX_REPLAN_ATTEMPTS})...")
        else:
            print("\n❌ Max replan attempts reached")
            break


# ============================================================================
# FINAL VERIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

for _ in range(100):
    scene.step()

final_predicates = extract_predicates_with_adjacency(scene, franka, blocks_state)
print("\nFinal State:")
print_predicates(final_predicates)

if phase2_goal.issubset(final_predicates):
    print("=" * 80)
    print("🎉 SUCCESS! DIRECTIONAL ADJACENCY TAMP COMPLETE!")
    print("=" * 80)
    print("\nFinal structure:")
    print("  [r3] [g3]         ← Top layer (stacked)")
    print("  [r1][g1]          ← Bottom layer")
    print("  [r2][g2]          ← Bottom layer")
    print("\nAll blocks positioned correctly with precise directional placement!")
else:
    print("=" * 80)
    print("❌ GOAL NOT ACHIEVED")
    print("=" * 80)
    print("\nMissing predicates:")
    missing = phase2_goal - final_predicates
    for p in sorted(missing):
        print(f"  {p}")

print("\n👀 Viewing final result. Press Ctrl+C to exit...")
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")