"""
predicates_directional.py - DIRECTIONAL ADJACENCY DETECTION

Detects ADJACENT-X and ADJACENT-Y separately for precise placement.

ADJACENT-X(a, b): Block 'a' is to the RIGHT of block 'b' (+X direction)
ADJACENT-Y(a, b): Block 'a' is in FRONT of block 'b' (+Y direction)
"""

import numpy as np
from typing import Any, Dict, Set


BLOCK_SIZE = 0.04  # 4cm blocks

# Directional adjacency thresholds
MIN_ADJACENT_DIST = BLOCK_SIZE * 0.85   # 3.4cm
MAX_ADJACENT_DIST = BLOCK_SIZE * 1.4    # 5.6cm

LAYER_TOLERANCE = 0.025  # 2.5cm - blocks at similar Z are same layer

DEBUG_ADJACENCY = True  # Set to False to hide debug output


def extract_predicates_directional(scene: Any, robot: Any, blocks_state: Dict[str, Any]) -> Set[str]:
    """
    Extract predicates with DIRECTIONAL adjacency.
    
    ADJACENT-X(a, b): 'a' is to the RIGHT of 'b' (a.x > b.x, similar y)
    ADJACENT-Y(a, b): 'a' is in FRONT of 'b' (a.y > b.y, similar x)
    """
    predicates = set()
    
    # Get gripper state
    gripper_qpos = robot.get_qpos()
    if hasattr(gripper_qpos, "cpu"):
        gripper_qpos = gripper_qpos.cpu().numpy()
    
    gripper_width = float(gripper_qpos[-1] + gripper_qpos[-2])
    gripper_holding = gripper_width < 0.02
    
    if gripper_holding:
        # Find which block is held
        hand = robot.get_link("hand")
        hand_pos = np.array(hand.get_pos())
        
        for block_id, block_obj in blocks_state.items():
            block_pos = np.array(block_obj.get_pos())
            dist = np.linalg.norm(block_pos - hand_pos)
            if dist < 0.15:
                predicates.add(f"HOLDING({block_id})")
                break
    else:
        predicates.add("HANDEMPTY()")
    
    # Detect ON relationships
    for block_id, block_obj in blocks_state.items():
        block_pos = np.array(block_obj.get_pos())
        block_z = float(block_pos[2])
        
        # Check if on table (Z ≈ 0.02)
        if abs(block_z - 0.02) < 0.01:
            predicates.add(f"ONTABLE({block_id})")
        else:
            # Check if on another block
            for other_id, other_obj in blocks_state.items():
                if other_id == block_id:
                    continue
                
                other_pos = np.array(other_obj.get_pos())
                other_z = float(other_pos[2])
                
                # Check if directly above
                dx = abs(block_pos[0] - other_pos[0])
                dy = abs(block_pos[1] - other_pos[1])
                dz = block_z - other_z
                
                if dx < 0.015 and dy < 0.015 and 0.03 < dz < 0.05:
                    predicates.add(f"ON({block_id},{other_id})")
                    break
    
    # Detect CLEAR
    for block_id in blocks_state.keys():
        is_clear = True
        for on_pred in predicates:
            if on_pred.startswith("ON("):
                bottom_block = on_pred.split(",")[1].rstrip(")")
                if bottom_block == block_id:
                    is_clear = False
                    break
        
        if is_clear and block_id not in [p.split("(")[1].rstrip(")") for p in predicates if p.startswith("HOLDING(")]:
            predicates.add(f"CLEAR({block_id})")
    
    # Detect DIRECTIONAL ADJACENCY
    if DEBUG_ADJACENCY:
        print("\n[Directional Adjacency Detection]")
    
    block_list = list(blocks_state.keys())
    
    for block_a in block_list:
        pos_a = np.array(blocks_state[block_a].get_pos())
        
        for block_b in block_list:
            if block_a == block_b:
                continue
            
            pos_b = np.array(blocks_state[block_b].get_pos())
            
            # Only check blocks at similar Z height (same layer)
            if abs(pos_a[2] - pos_b[2]) > LAYER_TOLERANCE:
                continue
            
            # Calculate deltas
            dx = pos_a[0] - pos_b[0]  # a.x - b.x
            dy = pos_a[1] - pos_b[1]  # a.y - b.y
            
            # ADJACENT-X: 'a' is to the RIGHT of 'b' (+X direction)
            # Conditions: dx > 0 (a is right of b), dy ≈ 0 (same Y), distance is ~1 block
            if dx > 0 and abs(dy) < 0.09:  # Same Y row
                dist = abs(dx)
                if MIN_ADJACENT_DIST < dist < MAX_ADJACENT_DIST:
                    predicates.add(f"ADJACENT-X({block_a},{block_b})")
                    if DEBUG_ADJACENCY:
                        print(f"  ✓ ADJACENT-X({block_a},{block_b}): {block_a} is {dist*100:.1f}cm to the RIGHT of {block_b}")
            
            # ADJACENT-Y: 'a' is in FRONT of 'b' (+Y direction)
            # Conditions: dy > 0 (a is front of b), dx ≈ 0 (same X), distance is ~1 block
            if dy > 0 and abs(dx) < 0.09:  # Same X column
                dist = abs(dy)
                if MIN_ADJACENT_DIST < dist < MAX_ADJACENT_DIST:
                    predicates.add(f"ADJACENT-Y({block_a},{block_b})")
                    if DEBUG_ADJACENCY:
                        print(f"  ✓ ADJACENT-Y({block_a},{block_b}): {block_a} is {dist*100:.1f}cm in FRONT of {block_b}")
    
    return predicates


def print_predicates(predicates: Set[str]) -> None:
    """Pretty print predicates organized by type."""
    
    print("\n" + "=" * 60)
    print("PREDICATES:")
    print("=" * 60)
    
    # Separate by type
    on_preds = sorted([p for p in predicates if p.startswith("ON(")])
    ontable_preds = sorted([p for p in predicates if p.startswith("ONTABLE(")])
    adjacent_x_preds = sorted([p for p in predicates if p.startswith("ADJACENT-X(")])
    adjacent_y_preds = sorted([p for p in predicates if p.startswith("ADJACENT-Y(")])
    clear_preds = sorted([p for p in predicates if p.startswith("CLEAR(")])
    holding_preds = sorted([p for p in predicates if p.startswith("HOLDING(")])
    handempty_preds = sorted([p for p in predicates if p.startswith("HANDEMPTY")])
    
    if on_preds:
        print("\nStacked:")
        for p in on_preds:
            print(f"  {p}")
    
    if ontable_preds:
        print("\nOn table:")
        for p in ontable_preds:
            print(f"  {p}")
    
    if adjacent_x_preds:
        print("\nAdjacent-X (horizontal, +X to the right):")
        for p in adjacent_x_preds:
            inside = p[11:-1]  # Remove ADJACENT-X( and )
            a, b = inside.split(",")
            print(f"  {a} → {b}  ({a} is RIGHT of {b})")
    
    if adjacent_y_preds:
        print("\nAdjacent-Y (vertical, +Y in front):")
        for p in adjacent_y_preds:
            inside = p[11:-1]  # Remove ADJACENT-Y( and )
            a, b = inside.split(",")
            print(f"  {a} → {b}  ({a} is FRONT of {b})")
    
    if clear_preds:
        print("\nClear:")
        for p in clear_preds:
            print(f"  {p}")
    
    if holding_preds:
        print("\nGripper:")
        for p in holding_preds:
            print(f"  {p}")
    
    if handempty_preds:
        print("\nGripper:")
        for p in handempty_preds:
            print(f"  {p}")
    
    print("=" * 60 + "\n")