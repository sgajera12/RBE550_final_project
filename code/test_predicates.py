"""
test_predicates.py

Test the symbolic abstraction module
"""

import sys
import numpy as np
import genesis as gs

from scenes import create_scene_6blocks
from predicates import extract_predicates, print_predicates

# Initialize
gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

scene, franka, blocks_state = create_scene_6blocks()

print("="*80)
print("TESTING SYMBOLIC ABSTRACTION (LIFTING)")
print("="*80)

print("\nInitial scene: 6 blocks scattered on table")
print("Blocks:", list(blocks_state.keys()))

# Let physics settle
for _ in range(100):
    scene.step()

# Extract predicates
predicates = extract_predicates(scene, franka, blocks_state)
print_predicates(predicates)

# Verify expected predicates
expected = {
    "ONTABLE(r)", "ONTABLE(g)", "ONTABLE(b)", 
    "ONTABLE(y)", "ONTABLE(m)", "ONTABLE(c)",
    "CLEAR(r)", "CLEAR(g)", "CLEAR(b)",
    "CLEAR(y)", "CLEAR(m)", "CLEAR(c)",
    "HANDEMPTY()"
}

print("\nExpected predicates:")
for p in sorted(expected):
    status = "true" if p in predicates else "false"
    print(f"  {status} {p}")

if predicates == expected:
    print("\nSYMBOLIC ABSTRACTION WORKING CORRECTLY!")
else:
    print("\nPredicates don't match expected")
    print(f"\nMissing: {expected - predicates}")
    print(f"Extra: {predicates - expected}")

print("\nPress Ctrl+C to exit...")
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("\nExiting...")
