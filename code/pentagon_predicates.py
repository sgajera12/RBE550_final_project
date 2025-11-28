"""
pentagon_predicates.py

Symbolic abstraction for pentagon structure
Extracts predicates about block positions relative to pentagon edges
"""

import numpy as np
import math
from typing import Set, Dict, Any

from pentagon_geometry import PENTAGON_EDGES, PENTAGON_CENTER, BLOCK_SIZE


POSITION_TOLERANCE = 0.015  # 1.5cm tolerance for position matching
ROTATION_TOLERANCE = 15  # 15 degrees tolerance for rotation matching


def extract_pentagon_predicates(scene: Any, franka: Any, blocks_state: Dict[str, Any]) -> Set[str]:
    """
    Extract predicates for pentagon structure
    
    Predicates include:
    - Standard: ONTABLE, CLEAR, HOLDING, HANDEMPTY
    - Pentagon: AT-EDGE, EDGE-FREE, EDGE-OCCUPIED, IN-LAYER
    
    Args:
        scene: Genesis scene
        franka: Robot
        blocks_state: Dictionary of block objects
        
    Returns:
        Set of predicate strings
    """
    
    predicates = set()
    
    # Check gripper state
    gripper_holding = False
    held_block = None
    
    hand_link = franka.get_link("hand")
    hand_pos = np.array(hand_link.get_pos())
    
    # Check if holding any block
    for block_id, block in blocks_state.items():
        block_pos = np.array(block.get_pos())
        dist = np.linalg.norm(hand_pos - block_pos)
        
        if dist < 0.15:  # within 15cm of hand
            predicates.add(f"HOLDING({block_id})")
            gripper_holding = True
            held_block = block_id
            break
    
    if not gripper_holding:
        predicates.add("HANDEMPTY()")
    
    # Add pentagon edge definitions
    for edge_name in PENTAGON_EDGES.keys():
        predicates.add(f"PENTAGON-EDGE({edge_name})")
    
    # Add layer definitions
    predicates.add("LAYER(layer1)")
    predicates.add("LAYER(layer2)")
    
    # Track which edges are occupied
    occupied_edges_layer1 = set()
    occupied_edges_layer2 = set()
    
    # Check each block's position
    for block_id, block in blocks_state.items():
        if block_id == held_block:
            continue  # skip held block
        
        pos = np.array(block.get_pos())
        
        # Determine which layer based on Z height
        if abs(pos[2] - 0.02) < 0.015:
            layer = 1
            layer_name = "layer1"
        elif abs(pos[2] - 0.06) < 0.015:
            layer = 2
            layer_name = "layer2"
        else:
            # Not in a pentagon layer, treat as normal block
            predicates.add(f"ONTABLE({block_id})")
            predicates.add(f"CLEAR({block_id})")
            continue
        
        # Check against each edge
        matched_edge = None
        
        for edge_name, edge in PENTAGON_EDGES.items():
            expected_pos = edge.get_block_placement_position(layer=layer)
            
            # Calculate XY distance
            xy_dist = math.sqrt(
                (pos[0] - expected_pos[0])**2 + 
                (pos[1] - expected_pos[1])**2
            )
            
            if xy_dist < POSITION_TOLERANCE:
                # Block is at this edge position
                predicates.add(f"AT-EDGE({block_id},{edge_name},{layer_name})")
                predicates.add(f"EDGE-OCCUPIED({edge_name},{layer_name})")
                predicates.add(f"IN-LAYER({block_id},{layer_name})")
                predicates.add(f"CLEAR({block_id})")
                
                if layer == 1:
                    occupied_edges_layer1.add(edge_name)
                    predicates.add(f"ONTABLE({block_id})")
                else:
                    occupied_edges_layer2.add(edge_name)
                
                matched_edge = edge_name
                break
        
        if not matched_edge:
            # Block not at any edge position
            predicates.add(f"ONTABLE({block_id})")
            predicates.add(f"CLEAR({block_id})")
    
    # Mark free edges
    for edge_name in PENTAGON_EDGES.keys():
        if edge_name not in occupied_edges_layer1:
            predicates.add(f"EDGE-FREE({edge_name},layer1)")
        
        if edge_name not in occupied_edges_layer2:
            predicates.add(f"EDGE-FREE({edge_name},layer2)")
    
    return predicates


def count_pentagon_layer_blocks(predicates: Set[str], layer: int) -> int:
    """Count how many blocks are in a given layer"""
    layer_name = f"layer{layer}"
    count = 0
    
    for pred in predicates:
        if pred.startswith(f"IN-LAYER(") and layer_name in pred:
            count += 1
    
    return count


def get_free_edges(predicates: Set[str], layer: int) -> list:
    """Get list of free edge names for a given layer"""
    layer_name = f"layer{layer}"
    free_edges = []
    
    for pred in predicates:
        if pred.startswith(f"EDGE-FREE(") and layer_name in pred:
            parts = pred.split("(")[1].split(")")[0].split(",")
            edge_name = parts[0]
            free_edges.append(edge_name)
    
    return sorted(free_edges)


def get_occupied_edges(predicates: Set[str], layer: int) -> list:
    """Get list of occupied edge names for a given layer"""
    layer_name = f"layer{layer}"
    occupied_edges = []
    
    for pred in predicates:
        if pred.startswith(f"EDGE-OCCUPIED(") and layer_name in pred:
            parts = pred.split("(")[1].split(")")[0].split(",")
            edge_name = parts[0]
            occupied_edges.append(edge_name)
    
    return sorted(occupied_edges)


def print_pentagon_predicates(predicates: Set[str]) -> None:
    """Pretty print pentagon predicates with grouping"""
    
    print("\n" + "="*70)
    print("PREDICATES (PENTAGON STRUCTURE)")
    print("="*70)
    
    gripper = sorted([p for p in predicates if p.startswith("HOLDING(") or p.startswith("HANDEMPTY(")])
    at_edge = sorted([p for p in predicates if p.startswith("AT-EDGE(")])
    in_layer = sorted([p for p in predicates if p.startswith("IN-LAYER(")])
    edge_occupied = sorted([p for p in predicates if p.startswith("EDGE-OCCUPIED(")])
    edge_free = sorted([p for p in predicates if p.startswith("EDGE-FREE(")])
    
    if gripper:
        print("\nGripper State:")
        for p in gripper:
            print(f"  {p}")
    
    if at_edge:
        print("\nBlocks at Edges:")
        for p in at_edge:
            print(f"  {p}")
    
    if edge_occupied:
        print("\nOccupied Edges:")
        for p in edge_occupied:
            print(f"  {p}")
    
    if edge_free:
        print("\nFree Edges:")
        for p in edge_free:
            print(f"  {p}")
    
    layer1_count = count_pentagon_layer_blocks(predicates, 1)
    layer2_count = count_pentagon_layer_blocks(predicates, 2)
    
    print("\n" + "-"*70)
    print("SUMMARY:")
    print(f"  Layer 1: {layer1_count}/5 blocks")
    print(f"  Layer 2: {layer2_count}/5 blocks")
    print("="*70 + "\n")
