"""
Calculates pentagon geometry for Goal 4
- Virtual pentagon at center
- Edge positions and normals
- Block placement positions with rotations
"""

import math
import numpy as np

# Virtual pentagon at table center
PENTAGON_CENTER = np.array([0.50, 0.0, 0.02])

# Block size
BLOCK_SIZE = 0.04  # 4cm cube

# Pentagon edge length (slightly larger than block to avoid collision)
PENTAGON_EDGE_LENGTH = BLOCK_SIZE + 0.005  # 4.5cm

# Calculate pentagon geometry
# For regular pentagon with edge length L:
# Apothem (center to edge midpoint) = L / (2 * tan(36))
APOTHEM = PENTAGON_EDGE_LENGTH / (2 * math.tan(math.radians(36)))
print(f"Pentagon apothem: {APOTHEM:.4f}m ({APOTHEM*100:.2f}cm)")

# Circumradius (center to vertex)
CIRCUMRADIUS = PENTAGON_EDGE_LENGTH / (2 * math.sin(math.radians(36)))
print(f"Pentagon circumradius: {CIRCUMRADIUS:.4f}m ({CIRCUMRADIUS*100:.2f}cm)")

# Define 5 edges
# Start from top edge and go clockwise
# Each edge is 72 degrees apart (360/5)

class PentagonEdge:
    """Represents one edge of the pentagon"""
    
    def __init__(self, edge_number):
        """
        Args:
            edge_number: 1 to 5, starting from top going clockwise
        """
        self.number = edge_number
        self.name = f"edge{edge_number}"
        
        # Angle of edge normal (perpendicular pointing outward)
        # Edge 1 is at top, normal points up (90 degrees)
        # Going clockwise, each edge is 72 degrees apart
        self.normal_angle = 90 - (edge_number - 1) * 72
        
        # Normalize to 0-360 range
        while self.normal_angle < 0:
            self.normal_angle += 360
        while self.normal_angle >= 360:
            self.normal_angle -= 360
        
        # Edge midpoint position (on apothem)
        self.normal_angle_rad = math.radians(self.normal_angle)
        self.midpoint = np.array([
            PENTAGON_CENTER[0] + APOTHEM * math.cos(self.normal_angle_rad),
            PENTAGON_CENTER[1] + APOTHEM * math.sin(self.normal_angle_rad),
            PENTAGON_CENTER[2]
        ])
        
        # Edge orientation (tangent angle, perpendicular to normal)
        self.tangent_angle = self.normal_angle - 90
        while self.tangent_angle < 0:
            self.tangent_angle += 360
        
        # Calculate edge endpoints (vertices)
        half_edge = PENTAGON_EDGE_LENGTH / 2.0
        tangent_rad = math.radians(self.tangent_angle)
        
        self.vertex1 = np.array([
            self.midpoint[0] - half_edge * math.cos(tangent_rad),
            self.midpoint[1] - half_edge * math.sin(tangent_rad),
            self.midpoint[2]
        ])
        
        self.vertex2 = np.array([
            self.midpoint[0] + half_edge * math.cos(tangent_rad),
            self.midpoint[1] + half_edge * math.sin(tangent_rad),
            self.midpoint[2]
        ])
    
    def get_block_placement_position(self, layer=1):
        """
        Get the position where block center should be placed
        For layer 1: Block sits ON the edge (outside pentagon)
        For layer 2: Block sits BETWEEN two layer 1 blocks (above)
        Args:
            layer: 1 for base layer, 2 for top layer
        Returns:
            np.array [x, y, z] position for block center
        """
        if layer == 1:
            # Block center is outside the pentagon edge
            # Distance from pentagon center to block center
            block_center_distance = APOTHEM + BLOCK_SIZE / 2.0
            
            x = PENTAGON_CENTER[0] + block_center_distance * math.cos(self.normal_angle_rad)
            y = PENTAGON_CENTER[1] + block_center_distance * math.sin(self.normal_angle_rad)
            z = PENTAGON_CENTER[2]  # table height
            
            return np.array([x, y, z])
        
        elif layer == 2:
            # Layer 2 blocks go between layer 1 blocks
            # Position is between this edge and next edge (clockwise)
            # Height is one block up
            next_edge_number = (self.number % 5) + 1
            next_edge = PentagonEdge(next_edge_number)
            
            # Average of this edge normal and next edge normal
            angle_between = (self.normal_angle + next_edge.normal_angle) / 2.0
            
            # Handle wrap-around (e.g., edge 5 to edge 1)
            if abs(self.normal_angle - next_edge.normal_angle) > 180:
                angle_between = ((self.normal_angle + next_edge.normal_angle) / 2.0 + 180) % 360
            
            angle_between_rad = math.radians(angle_between)
            
            # Same distance as layer 1
            block_center_distance = APOTHEM + BLOCK_SIZE / 2.0
            
            x = PENTAGON_CENTER[0] + block_center_distance * math.cos(angle_between_rad)
            y = PENTAGON_CENTER[1] + block_center_distance * math.sin(angle_between_rad)
            z = PENTAGON_CENTER[2] + BLOCK_SIZE  # one block height up
            
            return np.array([x, y, z])
        
        else:
            raise ValueError(f"Invalid layer: {layer}")
    
    def get_block_rotation(self, layer=1):
        """
        Get the rotation angle for block placement
        Mathematical approach:
        - All blocks start at 0 (1st side facing camera/user)
        - edge1 (normal=90, top) needs 0 rotation (1st side on edge)
        - Other edges: rotate relative to edge1
        Formula: rotation = -(normal_angle - 90) = 90 - normal_angle
        Args:
        layer: 1 or 2
        Returns:
        float: rotation angle in degrees (for robot wrist joint)
        """
        if layer == 1:
            # Reference: edge1 (normal=90) uses 0 rotation
            # Other edges rotate relative to this
            rotation = (90 - self.normal_angle) % 360
            return rotation
        
        elif layer == 2:
            # Layer 2 blocks face the same way as their position angle
            next_edge_number = (self.number % 5) + 1
            next_edge = PentagonEdge(next_edge_number)
            angle_between = (self.normal_angle + next_edge.normal_angle) / 2.0

            if abs(self.normal_angle - next_edge.normal_angle) > 180:
                angle_between = ((self.normal_angle + next_edge.normal_angle) / 2.0 + 180) % 360
            
            return (angle_between + 180) % 360
        
        else:
            raise ValueError(f"Invalid layer: {layer}")
    
    def __str__(self):
        return f"Edge {self.number}: midpoint=({self.midpoint[0]:.3f}, {self.midpoint[1]:.3f}), normal_angle={self.normal_angle:.1f}°"


# Create all 5 edges
PENTAGON_EDGES = {
    f"edge{i}": PentagonEdge(i) for i in range(1, 6)
}


def get_edge_by_number(edge_number):
    """Get edge object by number (1-5)"""
    return PENTAGON_EDGES[f"edge{edge_number}"]


def get_edge_by_name(edge_name):
    """Get edge object by name (edge1-edge5)"""
    return PENTAGON_EDGES[edge_name]


def print_pentagon_geometry():
    # To print 
    print("PENTAGON GEOMETRY")
    print(f"\nCenter: ({PENTAGON_CENTER[0]:.3f}, {PENTAGON_CENTER[1]:.3f}, {PENTAGON_CENTER[2]:.3f})")
    print(f"Edge length: {PENTAGON_EDGE_LENGTH:.4f}m ({PENTAGON_EDGE_LENGTH*100:.2f}cm)")
    print(f"Apothem: {APOTHEM:.4f}m ({APOTHEM*100:.2f}cm)")
    print(f"Circumradius: {CIRCUMRADIUS:.4f}m ({CIRCUMRADIUS*100:.2f}cm)")
    print("EDGES:")
    
    for edge_name in sorted(PENTAGON_EDGES.keys()):
        edge = PENTAGON_EDGES[edge_name]
        print(f"\n{edge}")
        print(f" Tangent angle: {edge.tangent_angle:.1f}°")
        print(f" Midpoint: ({edge.midpoint[0]:.4f}, {edge.midpoint[1]:.4f})")
        print(f" Vertex 1: ({edge.vertex1[0]:.4f}, {edge.vertex1[1]:.4f})")
        print(f" Vertex 2: ({edge.vertex2[0]:.4f}, {edge.vertex2[1]:.4f})")
    
    print("BLOCK PLACEMENT POSITIONS:")

    for edge_name in sorted(PENTAGON_EDGES.keys()):
        edge = PENTAGON_EDGES[edge_name]
        
        # Layer 1
        pos_l1 = edge.get_block_placement_position(layer=1)
        rot_l1 = edge.get_block_rotation(layer=1)
        print(f"\n{edge_name} Layer 1:")
        print(f"Position: ({pos_l1[0]:.4f}, {pos_l1[1]:.4f}, {pos_l1[2]:.4f})")
        print(f"Rotation: {rot_l1:.1f}°")
        
        # Layer 2
        pos_l2 = edge.get_block_placement_position(layer=2)
        rot_l2 = edge.get_block_rotation(layer=2)
        print(f"{edge_name} Layer 2:")
        print(f"Position: ({pos_l2[0]:.4f}, {pos_l2[1]:.4f}, {pos_l2[2]:.4f})")
        print(f"Rotation: {rot_l2:.1f}°")

if __name__ == "__main__":
    print_pentagon_geometry()