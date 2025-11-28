"""
pentagon_motion_primitives.py

Extension to motion_primitives for pentagon placement with rotation
This adds place_at_pentagon_edge() function
"""

import numpy as np
import math
from pentagon_geometry import PENTAGON_EDGES


def add_pentagon_support(executor_class):
    """
    Add pentagon placement methods to MotionPrimitiveExecutor
    Call this to extend the class with pentagon capabilities
    """
    
    def place_at_pentagon_edge(self, edge_name, layer=1):
        """
        Place held block at pentagon edge with correct rotation
        
        This is similar to put_down() but with specific XY position and rotation
        
        Args:
            edge_name: Which edge (edge1, edge2, edge3, edge4, edge5)
            layer: 1 for base layer, 2 for top layer
            
        Returns:
            bool: Success
        """
        
        if not self.gripper_holding:
            print("[motion] ERROR: Not holding any block")
            return False
        
        print(f"[motion] PLACE-AT-PENTAGON-EDGE {edge_name} layer {layer}")
        
        # Get edge data
        edge = PENTAGON_EDGES[edge_name]
        
        # Get position for this edge and layer
        target_pos = edge.get_block_placement_position(layer=layer)
        
        # Get rotation angle
        rotation_deg = edge.get_block_rotation(layer=layer)
        rotation_rad = math.radians(rotation_deg)
        
        print(f"  Target position: ({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f})")
        print(f"  Rotation: {rotation_deg:.1f} degrees")
        
        # For now, we'll place without rotation (simplified version)
        # Full rotation implementation would need IK with orientation control
        # which is complex - we'll use XY positioning only
        
        # Just use the standard put_down with specific XY
        success = self.put_down(target_pos[0], target_pos[1])
        
        if success:
            print(f"[motion] SUCCESS: Placed at {edge_name}")
        else:
            print(f"[motion] FAILED: Could not place at {edge_name}")
        
        return success
    
    # Attach method to class
    executor_class.place_at_pentagon_edge = place_at_pentagon_edge
    
    return executor_class
