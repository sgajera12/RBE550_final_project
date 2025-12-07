"""
pentagon_geometry_simple.py - Simplified geometry using user's proven math

Uses the exact math that works well:
- Start at 0° and increment by 72°
- Top pentagon at 36° offset
- Simple and clean!
"""

import numpy as np
from typing import List, Tuple, Dict


class PentagonGeometry:
    """
    Calculate pentagon geometry using simplified math.
    
    Based on user's original implementation that works well.
    """
    
    def __init__(self, radius: float = 0.053, center: Tuple[float, float] = (0.50, 0.0)):
        """
        Initialize pentagon geometry.
        
        Args:
            radius: Distance from center to BLOCK CENTER (0.08m = 8cm)
                   This is the distance from pentagon center to each block's center
            center: (x, y) center position of pentagon
        """
        self.radius = radius
        self.center_x = center[0]
        self.center_y = center[1]
        self.block_size = 0.04
        self.table_z = 0.02
    
    def get_pentagon_position(self, index: int, rotation_offset: float = 0) -> Tuple[float, float, float, float]:
        """
        Calculate position for a block in pentagon formation.
        
        Args:
            index: Block index (0-4)
            rotation_offset: Rotation offset in degrees (0 for base, 36 for top)
        
        Returns:
            (x, y, z, rotation) - position and rotation in degrees
        """
        # Pentagon: 5 vertices, 72° apart
        # Start from 0° and go by 72° increments
        angle_deg = 0 + (index * 72) + rotation_offset
        angle_rad = np.deg2rad(angle_deg)
        
        x = self.center_x + self.radius * np.cos(angle_rad)
        y = self.center_y + self.radius * np.sin(angle_rad)
        
        # Z height
        if rotation_offset == 0:
            z = self.table_z  # Base on table
        else:
            z = self.table_z + self.block_size  # Top one block height above
        
        # Rotation angle for block
        rotation = angle_deg
        
        # Normalize rotation to [-180, 180] range
        while rotation < -180:
            rotation += 360
        while rotation > 180:
            rotation -= 360
        
        return (x, y, z, rotation)
    
    def calculate_base_positions(self, center: Tuple[float, float] = None) -> List[Tuple[float, float, float]]:
        """
        Calculate XYZ positions for base pentagon blocks.
        
        Returns:
            List of 5 (x, y, z) positions
        """
        if center:
            self.center_x, self.center_y = center
        
        positions = []
        for i in range(5):
            x, y, z, _ = self.get_pentagon_position(i, rotation_offset=0)
            positions.append((x, y, z))
        
        return positions
    
    def calculate_top_positions(self, center: Tuple[float, float] = None) -> List[Tuple[float, float, float]]:
        """
        Calculate XYZ positions for top pentagon blocks.
        
        Top pentagon rotated 36° (half of 72°) to bridge base blocks.
        
        Returns:
            List of 5 (x, y, z) positions
        """
        if center:
            self.center_x, self.center_y = center
        
        positions = []
        for i in range(5):
            x, y, z, _ = self.get_pentagon_position(i, rotation_offset=36)
            positions.append((x, y, z))
        
        return positions
    
    def get_support_blocks_for_top_block(self, top_block_idx: int) -> Tuple[int, int]:
        """
        Determine which base blocks support a given top block.
        
        Due to 36° rotation, top block i sits between base blocks.
        
        Args:
            top_block_idx: Index of top block (0-4)
            
        Returns:
            (base_idx1, base_idx2) - indices of two supporting base blocks
        """
        # Top block at angle: 0 + i*72 + 36
        # This is between base blocks at angles: i*72 and (i+1)*72
        base_idx1 = top_block_idx
        base_idx2 = (top_block_idx + 1) % 5
        
        return (base_idx1, base_idx2)
    
    def create_block_mapping(self, block_ids: List[str]) -> Dict[str, Dict]:
        """
        Create mapping of block IDs to their target positions and roles.
        
        Args:
            block_ids: List of 10 block IDs (first 5 are base, last 5 are top)
            
        Returns:
            Dictionary mapping block_id to position, role, orientation, supports
        """
        if len(block_ids) != 10:
            raise ValueError(f"Expected 10 block IDs, got {len(block_ids)}")
        
        mapping = {}
        
        # Get positions
        base_positions = self.calculate_base_positions()
        top_positions = self.calculate_top_positions()
        
        # First 5 blocks are base pentagon
        for i, block_id in enumerate(block_ids[:5]):
            _, _, _, rotation = self.get_pentagon_position(i, rotation_offset=0)
            mapping[block_id] = {
                'position': base_positions[i],
                'role': 'base',
                'index': i,
                'orientation': rotation,
                'support_blocks': None
            }
        
        # Last 5 blocks are top pentagon
        for i, block_id in enumerate(block_ids[5:]):
            base_idx1, base_idx2 = self.get_support_blocks_for_top_block(i)
            support_block_ids = [block_ids[base_idx1], block_ids[base_idx2]]
            _, _, _, rotation = self.get_pentagon_position(i, rotation_offset=36)
            
            mapping[block_id] = {
                'position': top_positions[i],
                'role': 'top',
                'index': i,
                'orientation': rotation,
                'support_blocks': support_block_ids
            }
        
        return mapping
    
    def visualize_plan(self, block_mapping: Dict[str, Dict]) -> str:
        """Generate text visualization of pentagon plan."""
        output = []
        output.append("=" * 60)
        output.append("PENTAGON STACKING PLAN")
        output.append("=" * 60)
        
        output.append("\nBASE PENTAGON (on table):")
        output.append("-" * 40)
        for block_id, info in block_mapping.items():
            if info['role'] == 'base':
                pos = info['position']
                output.append(f"  {block_id:3s}: ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})")
        
        output.append("\nTOP PENTAGON (rotated 36°):")
        output.append("-" * 40)
        for block_id, info in block_mapping.items():
            if info['role'] == 'top':
                pos = info['position']
                supports = info['support_blocks']
                output.append(f"  {block_id:3s}: ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}) - rests on {supports}")
        
        output.append("\n" + "=" * 60)
        return '\n'.join(output)


if __name__ == "__main__":
    # Test the geometry
    geo = PentagonGeometry()
    block_ids = ['r', 'g', 'b', 'y', 'o', 'r2', 'g2', 'b2', 'y2', 'o2']
    mapping = geo.create_block_mapping(block_ids)
    print(geo.visualize_plan(mapping))
    
    # Verify all blocks at correct radius
    print("\nVerification:")
    print("-" * 40)
    for block_id, info in mapping.items():
        pos = info['position']
        dist = np.sqrt((pos[0] - geo.center_x)**2 + (pos[1] - geo.center_y)**2)
        print(f"{block_id}: distance from center = {dist:.4f}m (target: {geo.radius}m)")
    
    # Check rotation angle
    top_angles = [mapping[bid]['orientation'] for bid in block_ids[5:]]
    base_angles = [mapping[bid]['orientation'] for bid in block_ids[:5]]
    print(f"\nRotation offset: {top_angles[0] - base_angles[0]}° (target: 36°)")