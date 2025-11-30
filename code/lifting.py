"""
lifting.py  –  Symbolic abstraction module for the Motion Planning project
---------------------------------------------------------------------------
Converts the continuous Genesis scene state into a symbolic (predicate-based)
representation suitable for task planning (pyperplan / PDDL).

Output example:
{
    ('ON', 'RED', 'GREEN'),
    ('ONTABLE', 'BLUE'),
    ('CLEAR', 'RED'),
    ('HANDEMPTY',)
}
"""

from typing import Dict, Set, Tuple, Any
import numpy as np


# Configuration constants
BLOCK_HEIGHT = 0.04          # cube size (m)
TABLE_Z = 0.02               # z-position of top of table
ON_Z_TOL = 0.015             # tolerance for ON relationship
XY_TOL = 0.03                # horizontal tolerance for alignment

# Helper utilities
def color_name_map() -> Dict[str, str]:
    """Mapping from block key to uppercase name (for predicates)."""
    return {
        "r": "RED",
        "g": "GREEN",
        "b": "BLUE",
        "y": "YELLOW",
        "m": "MAGENTA",
        "c": "CYAN",
    }


def block_centroid(block: Any) -> np.ndarray:
    """Return the (x, y, z) world position of the block's center."""
    pos = block.get_pos()
    if isinstance(pos, np.ndarray):
        return pos
    return np.array(pos, dtype=float)


# Main predicate extraction
def extract_predicates(
    blocks_state: Dict[str, Any],
    franka: Any,
) -> Set[Tuple[str, ...]]:
    """
    Extract a set of symbolic predicates from the current scene state.

    Args:
        blocks_state: dict of {color_key: genesis_block_entity}
        franka: RobotAdapter for the Franka arm (to access gripper state)

    Returns:
        A set of predicate tuples, e.g. {('ON','RED','GREEN'), ('CLEAR','RED')}
    """

    predicates: Set[Tuple[str, ...]] = set()
    names = color_name_map()

    # Determine ON / ONTABLE relations
    block_positions = {k: block_centroid(v) for k, v in blocks_state.items()}

    # keep track of which block has something on top of it
    has_block_above = {k: False for k in blocks_state}

    for top_key, top_pos in block_positions.items():
        found_support = False
        for bottom_key, bottom_pos in block_positions.items():
            if top_key == bottom_key:
                continue
            dz = top_pos[2] - bottom_pos[2]
            dx = abs(top_pos[0] - bottom_pos[0])
            dy = abs(top_pos[1] - bottom_pos[1])

            # check if top block is aligned above bottom block
            if abs(dz - BLOCK_HEIGHT) <= ON_Z_TOL and dx <= XY_TOL and dy <= XY_TOL:
                predicates.add(("ON", names[top_key], names[bottom_key]))
                has_block_above[bottom_key] = True
                found_support = True
                break  # assume only one support

        if not found_support:
            # likely sitting on table
            if abs(top_pos[2] - TABLE_Z) <= ON_Z_TOL:
                predicates.add(("ONTABLE", names[top_key]))

    # Determine CLEAR relations
    supported_blocks = {b for (_, _, b) in predicates if _ == "ON"}
    for k in blocks_state:
        if k not in supported_blocks:
            predicates.add(("CLEAR", names[k]))

    # Determine gripper / holding state
    qpos = franka.get_qpos()
    left_finger = qpos[-2]
    right_finger = qpos[-1]
    finger_gap = (left_finger + right_finger) / 2.0

    # Heuristic threshold for "grasped" vs "open"
    if finger_gap < 0.02:
        # Try to identify which block is being held
        hand_link = franka.get_link("hand")
        hand_pos = np.array(hand_link.get_pos())
        held_block = None
        for k, pos in block_positions.items():
            dist = np.linalg.norm(pos - hand_pos)
            if dist < 0.07:
                held_block = names[k]
                break
        if held_block:
            predicates.add(("HOLDING", held_block))
        else:
            predicates.add(("HOLDING", "UNKNOWN"))
    else:
        predicates.add(("HANDEMPTY",))

    return predicates


# Utility: pretty printing
def format_predicates(preds: Set[Tuple[str, ...]]) -> str:
    """Return a readable string representation for logging/debugging."""
    lines = []
    for p in sorted(preds):
        if len(p) == 1:
            lines.append(f"{p[0]}()")
        elif len(p) == 2:
            lines.append(f"{p[0]}({p[1]})")
        elif len(p) == 3:
            lines.append(f"{p[0]}({p[1]}, {p[2]})")
        else:
            lines.append(str(p))
    return "\n".join(lines)


# Quick self-test (optional)
if __name__ == "__main__":
    # This small test just ensures import works; full test requires Genesis running.
    print("lifting.py loaded successfully.")
