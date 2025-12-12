"""
Symbolic Abstraction (Lifting) Module
Converts Genesis scene state into PDDL predicates
Predicates:
- ON(block_a, block_b): block_a is on top of block_b
- ONTABLE(block): block is on the table
- CLEAR(block): nothing is on top of block
- HOLDING(block): gripper is holding block
- HANDEMPTY(): gripper is empty
"""

import numpy as np
from typing import Dict, Set, Any

BLOCK_SIZE = 0.04
TABLE_Z = 0.02
STACK_TOLERANCE = 0.015
XY_TOLERANCE = 0.02


def extract_predicates(scene: Any, franka: Any, blocks_state: Dict[str, Any]) -> Set[str]:
    #Convert current scene state to PDDL predicates
    predicates = set()
    
    # Get all block positions
    block_positions = {}
    for block_id, block in blocks_state.items():
        pos = np.array(block.get_pos())
        block_positions[block_id] = pos
    
    # Check gripper state
    hand = franka.get_link("hand")
    hand_pos = np.array(hand.get_pos())
    
    # Determine if holding any block
    holding = None
    for block_id, pos in block_positions.items():
        dist = np.linalg.norm(pos - hand_pos)
        if dist < 0.12:
            holding = block_id
            predicates.add(f"HOLDING({block_id})")
            break
    
    if holding is None:
        predicates.add("HANDEMPTY()")
    
    # For each block, determine its state
    for block_id, pos in block_positions.items():
        if block_id == holding:
            continue
        
        # Check if on table
        if abs(pos[2] - TABLE_Z) < STACK_TOLERANCE:
            predicates.add(f"ONTABLE({block_id})")
        
        # Check if on another block
        for other_id, other_pos in block_positions.items():
            if block_id == other_id or other_id == holding:
                continue
            
            expected_z = other_pos[2] + BLOCK_SIZE
            z_diff = abs(pos[2] - expected_z)
            xy_dist = np.sqrt((pos[0] - other_pos[0])**2 + (pos[1] - other_pos[1])**2)
            
            if z_diff < STACK_TOLERANCE and xy_dist < XY_TOLERANCE:
                predicates.add(f"ON({block_id},{other_id})")
                break
        
        # Check if clear
        is_clear = True
        for other_id, other_pos in block_positions.items():
            if block_id == other_id or other_id == holding:
                continue
            
            expected_z = pos[2] + BLOCK_SIZE
            z_diff = abs(other_pos[2] - expected_z)
            xy_dist = np.sqrt((other_pos[0] - pos[0])**2 + (other_pos[1] - pos[1])**2)
            
            if z_diff < STACK_TOLERANCE and xy_dist < XY_TOLERANCE:
                is_clear = False
                break
        
        if is_clear:
            predicates.add(f"CLEAR({block_id})")
    
    return predicates


def print_predicates(predicates: Set[str]) -> None:
    print("PREDICATES:")
    on_preds = sorted([p for p in predicates if p.startswith("ON(")])
    ontable_preds = sorted([p for p in predicates if p.startswith("ONTABLE(")])
    clear_preds = sorted([p for p in predicates if p.startswith("CLEAR(")])
    holding_preds = sorted([p for p in predicates if p.startswith("HOLDING(")])
    handempty_preds = sorted([p for p in predicates if p.startswith("HANDEMPTY(")])
    
    if on_preds:
        print("\nStacked:")
        for p in on_preds:
            print(f"  {p}")
    
    if ontable_preds:
        print("\nOn table:")
        for p in ontable_preds:
            print(f"  {p}")
    
    if clear_preds:
        print("\nClear:")
        for p in clear_preds:
            print(f"  {p}")
    
    if holding_preds:
        print("\nHolding:")
        for p in holding_preds:
            print(f"  {p}")
    
    if handempty_preds:
        print("\nGripper:")
        for p in handempty_preds:
            print(f"  {p}")
    