"""
goal_pentagon_tower_tamp_two_phase.py

Two-phase TAMP:

Phase 1 (TAMP): Build the base pentagon (b1..b5) on the table
                at base1..base5 slots.

Phase 2 (TAMP): Stack the top pentagon (b6..b10) at top1..top5,
                each ON its corresponding base block.

Geometry (positions + orientations) is fixed and identical to the
layout used for the screenshots – only the symbolic plan (order of
actions) is decided by Pyperplan.

Usage:
    python special_goal_1.py [gpu]
"""

import sys
import os
import math
import numpy as np
import genesis as gs

from pentagon_scene import create_scene_10blocks
from pentagon_motion_primitives import MotionPrimitiveExecutor
from pentagon_predicates import extract_predicates, print_predicates
from pentagon_task_planner import generate_pddl_problem, call_pyperplan, plan_to_string

BLOCK_SIZE = 0.04  # 4cm blocks

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
    np.array([87, 87, 87, 87, 12, 12, 12, 200, 200]),
)

print("=" * 80)
print("GOAL: PENTAGON TOWER (10 BLOCKS) - TWO-PHASE TAMP PLAN")
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
# 3) Geometry helpers – EXACT same pentagon / bridge layout as before
# ---------------------------------------------------------------------------

executor = MotionPrimitiveExecutor(scene, franka, blocks_state)
domain_file = os.path.join(os.path.dirname(__file__), "pentagon_blocksworld.pddl")

CENTER_X = 0.50
CENTER_Y = 0.0
PENTAGON_RADIUS = 0.06  # tuned to get your nice pentagon


def get_pentagon_position(index, center_x, center_y, radius, rotation_offset=0.0):
    """Position + rotation for regular pentagon (5 vertices, 72° apart)."""
    angle = 0.0 + (index * 72.0) + rotation_offset
    angle_rad = math.radians(angle)

    x = center_x + radius * math.cos(angle_rad)
    y = center_y + radius * math.sin(angle_rad)

    rotation = angle
    while rotation < -180:
        rotation += 360
    while rotation > 180:
        rotation -= 360

    return x, y, rotation


def get_bridge_position(index, center_x, center_y, radius, rotation_offset=36.0):
    """Position + rotation for top pentagon blocks in between base vertices."""
    angle = 0.0 + (index * 72.0) + rotation_offset
    angle_rad = math.radians(angle)

    x = center_x + radius * math.cos(angle_rad)
    y = center_y + radius * math.sin(angle_rad)

    rotation = angle
    while rotation < -180:
        rotation += 360
    while rotation > 180:
        rotation -= 360

    return x, y, rotation


BLOCK_IDS_BASE = ["b1", "b2", "b3", "b4", "b5"]
BLOCK_IDS_TOP = ["b6", "b7", "b8", "b9", "b10"]

print("\nCalculating base pentagon positions...")
BASE_SLOT_NAMES = ["base1", "base2", "base3", "base4", "base5"]
BASE_SLOT_GEOM = {}  # slot -> (x, y, rot)

for i, slot in enumerate(BASE_SLOT_NAMES):
    x, y, rot = get_pentagon_position(i, CENTER_X, CENTER_Y, PENTAGON_RADIUS, 0.0)
    BASE_SLOT_GEOM[slot] = (x, y, rot)
    print(f"  {slot}: angle={rot:.1f}°, pos=({x:.4f}, {y:.4f})")

print("\nCalculating bridged top pentagon positions...")
TOP_SLOT_NAMES = ["top1", "top2", "top3", "top4", "top5"]
TOP_SLOT_GEOM = {}
TOP_SLOT_SUPPORT = {}  # slot -> base block id

for i, slot in enumerate(TOP_SLOT_NAMES):
    x, y, rot = get_bridge_position(i, CENTER_X, CENTER_Y, PENTAGON_RADIUS, 36.0)
    
    x += 0.0045

    TOP_SLOT_GEOM[slot] = (x, y, rot)
    TOP_SLOT_SUPPORT[slot] = BLOCK_IDS_BASE[i]  # top1 over b1, etc.
    print(f"  {slot}: angle={rot:.1f}°, pos=({x:.4f}, {y:.4f}) over {BLOCK_IDS_BASE[i]}")

print("\n" + "=" * 80)
print("GEOMETRY SUMMARY")
print("=" * 80)
print("Base slots (base1..5) form the table pentagon.")
print("Top slots (top1..5) are the rotated bridged pentagon.")

# ---------------------------------------------------------------------------
# Small helpers: continuous execution for base + top actions
# ---------------------------------------------------------------------------

def place_held_block_on_base_slot(slot_name: str) -> bool:
    """Assumes the block is already held; place at base pentagon slot."""
    if slot_name not in BASE_SLOT_GEOM:
        print(f"[ERROR] Unknown base slot: {slot_name}")
        return False

    x, y, rot = BASE_SLOT_GEOM[slot_name]
    print(f"[EXEC] put-down-base at {slot_name} → ({x:.3f}, {y:.3f}), rot={rot:.1f}")
    return executor.put_down(x=x, y=y, rotation_z=rot)


def place_held_block_on_top_slot(slot_name: str) -> bool:
    """Assumes the block is already held; place at bridged top slot."""
    if slot_name not in TOP_SLOT_GEOM:
        print(f"[ERROR] Unknown top slot: {slot_name}")
        return False

    x, y, rot = TOP_SLOT_GEOM[slot_name]
    support_block = TOP_SLOT_SUPPORT[slot_name]

    print(
        f"[EXEC] put-down-top at {slot_name} over {support_block} "
        f"→ ({x:.3f}, {y:.3f}), rot={rot:.1f}"
    )

    base_pos = executor._block_center(support_block)
    bridge_z = float(base_pos[2]) + BLOCK_SIZE

    hand = executor.robot.get_link("hand")

    # Lift high for safety
    current_hand_pos = np.array(hand.get_pos())
    safe_pos = current_hand_pos.copy()
    safe_pos[2] = max(safe_pos[2], 0.50)
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
            scene.step()

    # High above bridge
    high_above_bridge = np.array([x, y, 0.35])
    q_high = executor._ik_for_pose(high_above_bridge, executor.grasp_quat)
    if q_high is not None:
        current_q = executor.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        for i in range(100):
            alpha = (i + 1) / 100.0
            q = (1 - alpha) * start_q + alpha * q_high
            q[-2:] = executor.config.gripper_closed_width
            executor.robot.control_dofs_position(q)
            scene.step()

    # Approach
    approach_pos = np.array([x, y, bridge_z + 0.15])
    approach_quat = executor._get_rotated_grasp_quat(rot)
    q_approach = executor._ik_for_pose(approach_pos, approach_quat)
    if q_approach is None:
        print("[ERROR] IK failed for top approach.")
        return False

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
        scene.step()

    # Place
    place_pos = np.array([x, y, bridge_z + executor.config.grasp_offset])
    q_place = executor._ik_for_pose(place_pos, approach_quat)
    if q_place is not None:
        current_q = executor.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        for i in range(50):
            alpha = (i + 1) / 50.0
            q = (1 - alpha) * start_q + alpha * q_place
            q[-2:] = executor.config.gripper_closed_width
            executor.robot.control_dofs_position(q)
            scene.step()

    executor.open_gripper()

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
            scene.step()

    for _ in range(80):
        scene.step()

    return True


def go_home():
    current = franka.get_qpos()
    if hasattr(current, "cpu"):
        current = current.cpu().numpy()
    for i in range(80):
        alpha = (i + 1) / 80.0
        q = (1.0 - alpha) * current + alpha * safe_home
        franka.control_dofs_position(q)
        scene.step()

# ---------------------------------------------------------------------------
# 4) PHASE 1 – TAMP plan for BASE pentagon only
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("PHASE 1: PLAN & BUILD BASE PENTAGON (b1..b5)")
print("=" * 80)

# Initial predicates + base-loc facts (NO top-loc here)
preds_base = extract_predicates(scene, franka, blocks_state)
print("\nInitial predicates before Phase 1:")
print_predicates(preds_base)

for slot in BASE_SLOT_NAMES:
    preds_base.add(f"BASE-LOC({slot})")

objects_base = BLOCK_IDS_BASE + BLOCK_IDS_TOP + BASE_SLOT_NAMES

goal_base = set()
for i, b in enumerate(BLOCK_IDS_BASE):
    slot = BASE_SLOT_NAMES[i]
    goal_base.add(f"ONTABLE({b})")
    goal_base.add(f"AT({b},{slot})")
    goal_base.add(f"CLEAR({b})")

goal_base.add("HANDEMPTY()")

print("\nBASE GOAL PREDICATES:")
for p in sorted(goal_base):
    print(" ", p)

problem_base = generate_pddl_problem(
    preds_base,
    goal_base,
    objects_base,
    problem_name="pentagon_base",
    domain_name="pentagonworld",
)

print("\nCalling Pyperplan for BASE plan...")
plan_base = call_pyperplan(domain_file, problem_base)

if not plan_base:
    print("No plan found for base pentagon.")
    sys.exit(1)

print("\nBase TAMP Plan:")
print(plan_to_string(plan_base))

print("\nExecuting BASE plan...")
for step_idx, (action_name, args) in enumerate(plan_base, start=1):
    print(f"\n[BASE] Step {step_idx}: {action_name.upper()}({', '.join(args)})")

    success = True

    if action_name == "pick-up":
        success = executor.pick_up(args[0])

    elif action_name == "put-down-base":
        _, slot = args  # (block, slot) – we only care about slot for geometry
        success = place_held_block_on_base_slot(slot)

    else:
        # No put-down-top here because there are no TOP-LOC facts in Phase 1
        print(f"[BASE] Unknown / unsupported action '{action_name}' – skipping.")
        success = True

    if not success:
        print(f"[BASE] ERROR executing {action_name}, aborting.")
        sys.exit(1)

    for _ in range(60):
        scene.step()

    preds_after = extract_predicates(scene, franka, blocks_state)
    if "HANDEMPTY()" in preds_after:
        go_home()

print("\n✓ PHASE 1 COMPLETE – BASE PENTAGON BUILT")

# Let things settle
for _ in range(150):
    scene.step()

# ---------------------------------------------------------------------------
# 5) PHASE 2 – TAMP plan for TOP pentagon only
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("PHASE 2: PLAN & BUILD TOP PENTAGON (b6..b10)")
print("=" * 80)

preds_top = extract_predicates(scene, franka, blocks_state)
print("\nPredicates before Phase 2:")
print_predicates(preds_top)

# Add both base-loc and top-loc now
for slot in BASE_SLOT_NAMES:
    preds_top.add(f"BASE-LOC({slot})")
for slot in TOP_SLOT_NAMES:
    preds_top.add(f"TOP-LOC({slot})")

objects_top = BLOCK_IDS_BASE + BLOCK_IDS_TOP + BASE_SLOT_NAMES + TOP_SLOT_NAMES

goal_top = set()
for i, b in enumerate(BLOCK_IDS_TOP):
    base_b = BLOCK_IDS_BASE[i]
    slot = TOP_SLOT_NAMES[i]
    goal_top.add(f"ON({b},{base_b})")
    goal_top.add(f"AT({b},{slot})")
    goal_top.add(f"CLEAR({b})")
goal_top.add("HANDEMPTY()")

print("\nTOP GOAL PREDICATES:")
for p in sorted(goal_top):
    print(" ", p)

problem_top = generate_pddl_problem(
    preds_top,
    goal_top,
    objects_top,
    problem_name="pentagon_top",
    domain_name="pentagonworld",
)

print("\nCalling Pyperplan for TOP plan...")
plan_top = call_pyperplan(domain_file, problem_top)

if not plan_top:
    print("No plan found for top pentagon.")
    sys.exit(1)

print("\nTop TAMP Plan:")
print(plan_to_string(plan_top))

print("\nExecuting TOP plan...")
for step_idx, (action_name, args) in enumerate(plan_top, start=1):
    print(f"\n[TOP] Step {step_idx}: {action_name.upper()}({', '.join(args)})")

    success = True

    if action_name == "pick-up":
        success = executor.pick_up(args[0])

    elif action_name == "put-down-base":
        # Planner COULD move base blocks, but A* should avoid pointless moves.
        _, slot = args
        success = place_held_block_on_base_slot(slot)

    elif action_name == "put-down-top":
        _, slot, _base_block = args  # base block is implied by geometry
        success = place_held_block_on_top_slot(slot)

    else:
        print(f"[TOP] Unknown / unsupported action '{action_name}' – skipping.")
        success = True

    if not success:
        print(f"[TOP] ERROR executing {action_name}, aborting.")
        sys.exit(1)

    for _ in range(60):
        scene.step()

    preds_after = extract_predicates(scene, franka, blocks_state)
    if "HANDEMPTY()" in preds_after:
        go_home()

print("\n✓ PHASE 2 COMPLETE – TOP PENTAGON BUILT")

# ---------------------------------------------------------------------------
# 6) FINAL VERIFICATION (base + top)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

for _ in range(150):
    scene.step()

final_preds = extract_predicates(scene, franka, blocks_state)
print_predicates(final_preds)

# Expected full goal = base-goal ∪ top-goal
full_goal = set(goal_base) | set(goal_top)

if full_goal.issubset(final_preds):
    print("=" * 80)
    print("SUCCESS! PENTAGON TOWER COMPLETE (TWO-PHASE TAMP)!")
    print("=" * 80)
else:
    print("Full symbolic goal not completely satisfied.")
    missing = full_goal - final_preds
    print("\nMissing predicates:", missing)

print("\Viewing final result. Press Ctrl+C to exit...")
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")
