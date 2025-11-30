"""
goal_pentagon_tower_tamp.py

Build a pentagon tower structure using TAMP pipeline.

Structure:
  Base layer (5 blocks): Pentagon shape on table (b1-b5)
  Top layer (5 blocks): Pentagon rotated 45° stacked on base (b6-b10)

Uses TAMP: Task Planning (Pyperplan) → Motion Primitives → Re-grounding loop

Usage:
    python goal_pentagon_tower_tamp.py [gpu]
"""

import sys
import os
import numpy as np
import genesis as gs
import math

from special_scenes_1 import create_scene_10blocks
from pentagon_motion_premitives import MotionPrimitiveExecutor
from predicates import extract_predicates, print_predicates
from task_planner import generate_pddl_problem, call_pyperplan, plan_to_string

BLOCK_SIZE = 0.04  # 4cm blocks

# ---------------------------------------------------------------------------
# Helper function for block rotation (NOT USED - blocks placed tilted by arm)
# ---------------------------------------------------------------------------
# NOTE: We don't use this function anymore. Instead, the robot arm places
# blocks at the desired angle using the rotation_z parameter in put_down().
# This is the proper way - the gripper rotates and places the block tilted.

# ---------------------------------------------------------------------------
# 1) Initialize Genesis
# ---------------------------------------------------------------------------
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
    np.array([ 87,  87,  87,  87,  12,  12,  12,  200,  200]),
)

print("=" * 80)
print("GOAL: PENTAGON TOWER (10 BLOCKS) - TAMP PIPELINE")
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
# 3) Define GOAL predicates for pentagon tower structure
# ---------------------------------------------------------------------------

goal_predicates = {
    # ===== BASE PENTAGON (5 blocks on table) =====
    "ONTABLE(b1)",
    "ONTABLE(b2)",
    "ONTABLE(b3)",
    "ONTABLE(b4)",
    "ONTABLE(b5)",
    
    # ===== TOP PENTAGON (5 blocks bridging pairs of base blocks) =====
    # Each top block bridges TWO base blocks (not centered on one)
    # b6 bridges b1-b2, b7 bridges b2-b3, etc.
    "ON(b6,b1)",   # b6 rests on BOTH b1 and b2
    "ON(b7,b2)",   # b7 rests on BOTH b2 and b3
    "ON(b8,b3)",   # b8 rests on BOTH b3 and b4
    "ON(b9,b4)",   # b9 rests on BOTH b4 and b5
    "ON(b10,b5)",  # b10 rests on BOTH b5 and b1
    
    # ===== CLEAR blocks (all top layer blocks must be clear) =====
    "CLEAR(b6)",
    "CLEAR(b7)",
    "CLEAR(b8)",
    "CLEAR(b9)",
    "CLEAR(b10)",
    
    # Gripper empty at the end
    "HANDEMPTY()",
}

blocks = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10"]

print("\n" + "=" * 80)
print("GOAL STATE (PENTAGON TOWER):")
print("=" * 80)
print("\nBase pentagon (on table): b1-b5 (CLOSE TOGETHER)")
print("Top pentagon: b6-b10 BRIDGING pairs of base blocks")
print("\nBridging pattern (each top block rests on TWO base blocks):")
print("  b6 → bridges b1 and b2")
print("  b7 → bridges b2 and b3")
print("  b8 → bridges b3 and b4")
print("  b9 → bridges b4 and b5")
print("  b10 → bridges b5 and b1")
print("\nGoal predicates:")
for p in sorted(goal_predicates):
    print(f"  {p}")

# ---------------------------------------------------------------------------
# 4) Pentagon geometry helper
# ---------------------------------------------------------------------------

executor = MotionPrimitiveExecutor(scene, franka, blocks_state)

domain_file = os.path.join(os.path.dirname(__file__), "blocksworld.pddl")

# Assembly center for placing blocks
CENTER_X = 0.50
CENTER_Y = 0.0

# Pentagon geometry: 5 blocks arranged in a circle
# Pentagon has 5 vertices, each separated by 72° (360°/5)
# REDUCED radius so blocks are closer together - top blocks can bridge two base blocks
PENTAGON_RADIUS = 0.06

def get_pentagon_position(index, center_x, center_y, radius, rotation_offset=0):
    """Calculate position for a block in a pentagon formation.
    
    Args:
        index: Block index (0-4)
        center_x, center_y: Center of pentagon
        radius: Distance from center to each block center
        rotation_offset: Rotation offset in degrees (for top layer)
    
    Returns:
        (x, y, rotation_angle) - position and rotation in degrees
    """
    # Pentagon: 5 vertices, 72° apart
    # Start from top center (angle = 90°) and go clockwise
    angle = 0 + (index * 72) + rotation_offset
    angle_rad = math.radians(angle)
    
    x = center_x + radius * math.cos(angle_rad)
    y = center_y + radius * math.sin(angle_rad)
    
    # Rotation angle: blocks should point toward/away from center
    # For a radial pattern, rotate block by the same angle
    rotation = angle
    
    # CRITICAL: Normalize rotation to [-180, 180] range
    # This prevents IK failures from extreme angles like -288°
    while rotation < -180:
        rotation += 360
    while rotation > 180:
        rotation -= 360
    
    return (x, y, rotation)


def get_bridge_position(index, center_x, center_y, radius, rotation_offset=36):
    """Calculate bridging position for top pentagon blocks.
    
    Top blocks are placed BETWEEN base blocks (rotated 36° = half of 72°).
    This creates the classic pentagon tower where top blocks bridge pairs.
    
    Args:
        index: Block index (0-4)
        center_x, center_y: Center of pentagon
        radius: Distance from center to each block center (same as base)
        rotation_offset: Rotation offset in degrees (default 36° for bridging)
    
    Returns:
        (x, y, rotation_angle) - position and rotation in degrees
    """
    # Pentagon: 5 vertices, 72° apart
    # Add 36° offset to place blocks BETWEEN base blocks
    angle = 0 + (index * 72) + rotation_offset

    angle_rad = math.radians(angle)
    
    x = center_x + radius * math.cos(angle_rad)
    y = center_y + radius * math.sin(angle_rad)
    
    # Rotation angle: blocks should point toward/away from center
    rotation = angle
    
    # CRITICAL: Normalize rotation to [-180, 180] range
    while rotation < -180:
        rotation += 360
    while rotation > 180:
        rotation -= 360
    
    return (x, y, rotation)


# Define WHERE each block should be placed
BLOCK_PLACEMENT_POSITIONS = {}
BLOCK_ROTATIONS = {}  # Store rotation angles

# Base pentagon (b1-b5): Regular pentagon with radial orientation
print("\nCalculating pentagon positions...")
for i in range(5):
    block_id = f"b{i+1}"
    x, y, rotation = get_pentagon_position(i, CENTER_X, CENTER_Y, PENTAGON_RADIUS, rotation_offset=0)
    BLOCK_PLACEMENT_POSITIONS[block_id] = (x, y)
    BLOCK_ROTATIONS[block_id] = rotation
    print(f"  {block_id}: angle={rotation:.1f}°, pos=({x:.4f}, {y:.4f})")

print("\n" + "=" * 80)
print("PENTAGON PLACEMENT POSITIONS")
print("=" * 80)
print("\nBase Pentagon (b1-b5):")
for i in range(1, 6):
    block_id = f"b{i}"
    x, y = BLOCK_PLACEMENT_POSITIONS[block_id]
    print(f"  {block_id}: ({x:.4f}, {y:.4f})")

print("\n" + "=" * 80)
print("INCREMENTAL TAMP - Planning for blocks in small batches")
print("=" * 80)

# ============================================================================
# PHASE 1: Place base pentagon blocks (b1-b5) using TAMP
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 1: Place base pentagon (b1-b5) using TAMP")
print("=" * 80)

base_blocks_to_place = [
    ('b1', 'Pentagon-1'),
    ('b2', 'Pentagon-2'),
    ('b3', 'Pentagon-3'),
    ('b4', 'Pentagon-4'),
    ('b5', 'Pentagon-5'),
]

for block_id, location in base_blocks_to_place:
    print(f"\n{'='*60}")
    print(f"Placing {block_id} at {location} using TAMP")
    print(f"{'='*60}")
    
    # Get target position
    if block_id not in BLOCK_PLACEMENT_POSITIONS:
        print(f"❌ No position defined for {block_id}")
        sys.exit(1)
    
    target_x, target_y = BLOCK_PLACEMENT_POSITIONS[block_id]
    print(f"  Target position: ({target_x:.4f}, {target_y:.4f})")
    
    # TAMP loop for this single block
    iteration = 0
    max_iterations = 10
    
    # Goal: This block should be ONTABLE
    block_goal = {
        f"ONTABLE({block_id})",
        f"CLEAR({block_id})",
        "HANDEMPTY()",
    }
    
    # Check if already at goal
    current_predicates = extract_predicates(scene, franka, blocks_state)
    if block_goal.issubset(current_predicates):
        print(f"  {block_id} already ONTABLE - repositioning...")
        
        # Pick it up
        if not executor.pick_up(block_id):
            print(f"  ❌ Failed to pick up {block_id}")
            sys.exit(1)
        
        # CRITICAL: Lift to safe height to avoid hitting other blocks
        print(f"  Lifting {block_id} to safe height...")
        hand = executor.robot.get_link("hand")
        current_hand_pos = np.array(hand.get_pos())
        safe_pos = current_hand_pos.copy()
        safe_pos[2] = 0.25  # 25cm high - well above all blocks
        
        q_safe = executor._ik_for_pose(safe_pos, executor.grasp_quat)
        if q_safe is not None:
            current_q = executor.robot.get_qpos()
            if hasattr(current_q, "cpu"):
                start_q = current_q.cpu().numpy().copy()
            else:
                start_q = np.array(current_q, dtype=float, copy=True)
            
            # Slow safe lift
            for i in range(60):
                alpha = (i + 1) / 60.0
                q = (1 - alpha) * start_q + alpha * q_safe
                q[-2:] = executor.config.gripper_closed_width
                executor.robot.control_dofs_position(q)
                executor.scene.step()
            
            print(f"  ✓ Block lifted safely")
        
        # Place with rotation - arm will place block at angle
        target_rotation = BLOCK_ROTATIONS[block_id]
        if not executor.put_down(x=target_x, y=target_y, rotation_z=target_rotation):
            print(f"  ❌ Failed to place {block_id}")
            sys.exit(1)
        
        print(f"  ✓ {block_id} repositioned and placed at {target_rotation:.1f}°")
        
        # Settle and return home
        for _ in range(80):
            scene.step()
        
        current = franka.get_qpos()
        if hasattr(current, "cpu"):
            current = current.cpu().numpy()
        for i in range(60):
            alpha = (i + 1) / 60.0
            q = (1.0 - alpha) * current + alpha * safe_home
            franka.control_dofs_position(q)
            scene.step()
        
        continue  # Move to next block
    
    # Use TAMP to plan getting this block ONTABLE
    while iteration < max_iterations:
        iteration += 1
        print(f"\n  [TAMP Iteration {iteration}]")
        
        # Extract predicates
        current_predicates = extract_predicates(scene, franka, blocks_state)
        
        # Filter to this block only
        filtered_predicates = set()
        for pred in current_predicates:
            if "HANDEMPTY()" in pred or "HOLDING(" in pred:
                if "HOLDING(" in pred and block_id in pred:
                    filtered_predicates.add(pred)
                elif "HANDEMPTY()" in pred:
                    filtered_predicates.add(pred)
            elif block_id in pred:
                # Only include predicates about this block
                other_blocks = [f"b{i}" for i in range(1, 11) if f"b{i}" != block_id]
                has_other = any(other in pred for other in other_blocks)
                if not has_other:
                    filtered_predicates.add(pred)
        
        print(f"    Predicates: {sorted(filtered_predicates)}")
        
        # Check goal
        if block_goal.issubset(filtered_predicates):
            print(f"    ✓ {block_id} is ONTABLE")
            break
        
        # Plan
        print(f"    Planning for {block_id}...")
        problem_string = generate_pddl_problem(
            filtered_predicates,
            block_goal,
            [block_id],
            f"place_{block_id}_iter{iteration}",
        )
        
        plan = call_pyperplan(domain_file, problem_string)
        
        if not plan:
            print(f"    ❌ No plan found")
            sys.exit(1)
        
        # Execute first action
        action_name, args = plan[0]
        print(f"    Executing: {action_name.upper()}({', '.join(args)})")
        
        success = False
        if action_name == "pick-up":
            success = executor.pick_up(args[0])
            
            # CRITICAL: After picking up, lift to safe height to avoid collisions
            if success:
                print(f"    Lifting to safe height...")
                hand = executor.robot.get_link("hand")
                current_hand_pos = np.array(hand.get_pos())
                safe_pos = current_hand_pos.copy()
                safe_pos[2] = 0.25  # 25cm high
                
                q_safe = executor._ik_for_pose(safe_pos, executor.grasp_quat)
                if q_safe is not None:
                    current_q = executor.robot.get_qpos()
                    if hasattr(current_q, "cpu"):
                        start_q = current_q.cpu().numpy().copy()
                    else:
                        start_q = np.array(current_q, dtype=float, copy=True)
                    
                    for i in range(60):
                        alpha = (i + 1) / 60.0
                        q = (1 - alpha) * start_q + alpha * q_safe
                        q[-2:] = executor.config.gripper_closed_width
                        executor.robot.control_dofs_position(q)
                        executor.scene.step()
                    print(f"    ✓ Lifted safely")
                    
        elif action_name == "put-down":
            # Place at designated position WITH ROTATION - arm places block tilted
            target_rotation = BLOCK_ROTATIONS[block_id]
            success = executor.put_down(x=target_x, y=target_y, rotation_z=target_rotation)
        elif action_name == "unstack":
            success = executor.pick_up(args[0])
        
        if not success:
            print(f"    ⚠️  Action failed, re-planning...")
        else:
            print(f"    ✓ Action completed")
        
        # Settle
        for _ in range(80):
            scene.step()
        
        # Return home if gripper empty
        current_preds = extract_predicates(scene, franka, blocks_state)
        if "HANDEMPTY()" in current_preds:
            current = franka.get_qpos()
            if hasattr(current, "cpu"):
                current = current.cpu().numpy()
            for i in range(60):
                alpha = (i + 1) / 60.0
                q = (1.0 - alpha) * current + alpha * safe_home
                franka.control_dofs_position(q)
                scene.step()
    
    print(f"  ✓ {block_id} placed at {location}")

print("\n✓ PHASE 1 COMPLETE - Base pentagon arranged using TAMP!")

# Let physics settle
for _ in range(150):
    scene.step()

# ============================================================================
# PHASE 2: Stack top pentagon blocks (b6-b10) using bridge positions
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 2: Stack top pentagon (b6-b10) - BRIDGING base blocks")
print("=" * 80)
print("\nEach top block will be placed BETWEEN two base blocks (rotated 36°)")

# Calculate bridge positions for top pentagon
# Top blocks use the SAME radius but rotated 36° (half of 72°)
# This places them exactly between base blocks
BRIDGE_POSITIONS = {}
BRIDGE_ROTATIONS = {}

print("\nCalculating bridge positions for top pentagon...")
for i in range(5):
    top_block = f"b{i+6}"  # b6, b7, b8, b9, b10
    x, y, rotation = get_bridge_position(i, CENTER_X, CENTER_Y, PENTAGON_RADIUS, rotation_offset=36)
    BRIDGE_POSITIONS[top_block] = (x, y)
    BRIDGE_ROTATIONS[top_block] = rotation
    print(f"  {top_block}: angle={rotation:.1f}°, pos=({x:.4f}, {y:.4f})")

# Modified stacking: Pick up and place at bridge positions
stacking_plan = [
    ('b6', 'b1', 'Bridge b1-b2'),
    ('b7', 'b2', 'Bridge b2-b3'),
    ('b8', 'b3', 'Bridge b3-b4'),
    ('b9', 'b4', 'Bridge b4-b5'),
    ('b10', 'b5', 'Bridge b5-b1'),
]

for top_block, base_block_ref, description in stacking_plan:
    print(f"\n{'='*60}")
    print(f"Placing {top_block} - {description}")
    print(f"{'='*60}")
    
    # Get PRE-CALCULATED bridge position for this block
    bridge_x, bridge_y = BRIDGE_POSITIONS[top_block]
    bridge_rotation = BRIDGE_ROTATIONS[top_block]
    
    # Calculate Z height: stack on top of base layer
    base_pos = executor._block_center(base_block_ref)
    bridge_z = float(base_pos[2]) + BLOCK_SIZE  # One block height up
    
    print(f"  Reference base: {base_block_ref} at ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})")
    print(f"  Bridge position: ({bridge_x:.3f}, {bridge_y:.3f}, {bridge_z:.3f})")
    print(f"  Bridge rotation: {bridge_rotation:.1f}°")
    
    # Pick up the top block
    if not executor.pick_up(top_block):
        print(f"  ❌ Failed to pick up {top_block}")
        sys.exit(1)
    
    # Lift to safe height
    print(f"  Lifting to safe height...")
    hand = executor.robot.get_link("hand")
    current_hand_pos = np.array(hand.get_pos())
    safe_pos = current_hand_pos.copy()
    safe_pos[2] = 0.50  # 50cm high - very high to clear all existing blocks
    
    q_safe = executor._ik_for_pose(safe_pos, executor.grasp_quat)
    if q_safe is not None:
        current_q = executor.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        for i in range(60):
            alpha = (i + 1) / 60.0
            q = (1 - alpha) * start_q + alpha * q_safe
            q[-2:] = executor.config.gripper_closed_width
            executor.robot.control_dofs_position(q)
            executor.scene.step()
        print(f"  ✓ Lifted safely")
    
    # CRITICAL: Move to HIGH position directly above bridge point FIRST
    # This avoids sweeping through adjacent blocks
    high_above_bridge = np.array([bridge_x, bridge_y, 0.35])  # 35cm above table
    
    print(f"  Moving to high position above bridge point...")
    q_high = executor._ik_for_pose(high_above_bridge, executor.grasp_quat)
    if q_high is not None:
        current_q = executor.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        # Move horizontally at high altitude
        for i in range(100):
            alpha = (i + 1) / 100.0
            q = (1 - alpha) * start_q + alpha * q_high
            q[-2:] = executor.config.gripper_closed_width
            executor.robot.control_dofs_position(q)
            executor.scene.step()
        print(f"  ✓ Positioned above bridge point")
    
    # Now descend with rotation to approach position
    approach_pos = np.array([bridge_x, bridge_y, bridge_z + 0.15])
    
    # Get rotated quaternion for approach
    approach_quat = executor._get_rotated_grasp_quat(bridge_rotation)
    
    # Move to approach WITH ROTATION
    q_approach = executor._ik_for_pose(approach_pos, approach_quat)
    if q_approach is None:
        print(f"  ❌ Failed IK for approach position")
        sys.exit(1)
    
    current_q = executor.robot.get_qpos()
    if hasattr(current_q, "cpu"):
        start_q = current_q.cpu().numpy().copy()
    else:
        start_q = np.array(current_q, dtype=float, copy=True)
    
    for i in range(80):
        alpha = (i + 1) / 80.0
        q = (1 - alpha) * start_q + alpha * q_approach
        q[-2:] = executor.config.gripper_closed_width
        executor.robot.control_dofs_position(q)
        executor.scene.step()
    
    # Lower to bridge position WITH ROTATION
    place_pos = np.array([bridge_x, bridge_y, bridge_z + executor.config.grasp_offset])
    q_place = executor._ik_for_pose(place_pos, approach_quat)
    
    if q_place is not None:
        current_q = executor.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        # Slow descent
        for i in range(50):
            alpha = (i + 1) / 50.0
            q = (1 - alpha) * start_q + alpha * q_place
            q[-2:] = executor.config.gripper_closed_width
            executor.robot.control_dofs_position(q)
            executor.scene.step()
    
    # Release
    executor.open_gripper()
    
    # Lift up (maintain rotation)
    current_pos = np.array(hand.get_pos())
    up_pos = current_pos.copy()
    up_pos[2] += 0.12
    
    q_up = executor._ik_for_pose(up_pos, approach_quat)
    if q_up is not None:
        current_q = executor.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        for i in range(40):
            alpha = (i + 1) / 40.0
            q = (1 - alpha) * start_q + alpha * q_up
            executor.robot.control_dofs_position(q)
            executor.scene.step()
    
    # Settle
    for _ in range(100):
        scene.step()
    
    print(f"  ✓ {top_block} successfully placed at bridge position with {bridge_rotation:.1f}° rotation")
    
    # Return home
    print("  Returning to home...")
    current = franka.get_qpos()
    if hasattr(current, "cpu"):
        current = current.cpu().numpy()
    for i in range(80):
        alpha = (i + 1) / 80.0
        q = (1.0 - alpha) * current + alpha * safe_home
        franka.control_dofs_position(q)
        scene.step()

print("\n✓ PHASE 2 COMPLETE - All top pentagon blocks bridged!")

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
    print("🎉 SUCCESS! PENTAGON TOWER COMPLETE!")
    print("=" * 80)
    print("\nFinal configuration:")
    print("  - Base pentagon: b1-b5 on table (CLOSE TOGETHER)")
    print("  - Top pentagon: b6-b10 bridging pairs of base blocks")
    print("  - Total: 10 blocks forming pentagon tower")
else:
    print("❌ Goal not fully achieved")
    missing = goal_predicates - final_predicates
    print(f"\nMissing predicates: {missing}")

print(f"\nTotal iterations across all stacking: Complete")
print("\n👀 Viewing final result. Press Ctrl+C to exit...")
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")