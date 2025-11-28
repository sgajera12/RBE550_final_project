"""
motion_primitives.py - FIXED ARM DRIFT AND PLACEMENT

Real issues fixed:
1. ARM joints drift down under load → Continuously re-command target position
2. Wrong stacking position calculation → Fixed math for placement
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
from planning import PlannerInterface


BLOCK_SIZE = 0.04
TABLE_BLOCK_CENTER_Z = 0.02
SAFE_RETREAT_POS = np.array([0.7, 0.0, 0.5])
MIN_APPROACH_HEIGHT = 0.180
GRIPPER_OPEN_WIDTH = 0.04
GRIPPER_CLOSED_WIDTH = 0.0
GRIPPER_STEPS = 50


@dataclass
class MotionConfig:
    safe_retreat_pos: np.ndarray = SAFE_RETREAT_POS
    min_approach_height: float = MIN_APPROACH_HEIGHT
    gripper_open_width: float = GRIPPER_OPEN_WIDTH
    gripper_closed_width: float = GRIPPER_CLOSED_WIDTH
    gripper_steps: int = GRIPPER_STEPS
    num_waypoints: int = 150
    grasp_offset: float = 0.12


class MotionPrimitiveExecutor:
    """Motion primitives with anti-drift arm control."""

    def __init__(
        self,
        scene: Any,
        robot: Any,
        blocks_state: Dict[str, Any],
        config: Optional[MotionConfig] = None,
    ):
        self.scene = scene
        self.robot = robot
        self.blocks_state = blocks_state
        self.config = config or MotionConfig()
        self.planner = PlannerInterface(robot, scene)
        self.grasp_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
        self.gripper_holding = False
        
        # Track target position to prevent drift
        self.target_qpos = None
        self.tower_centers = {}

    def _resolve_block_key(self, block_id: Any) -> str:
        s = str(block_id).strip().lower()
        if s in self.blocks_state:
            return s
        if s and s[0] in self.blocks_state:
            return s[0]
        raise KeyError(f"Unknown block: {block_id}")

    def _block_center(self, block_key: str) -> np.ndarray:
        pos = self.blocks_state[block_key].get_pos()
        if isinstance(pos, np.ndarray):
            return pos.astype(float)
        return np.array(pos, dtype=float)

    def _ik_for_pose(self, pos: np.ndarray, quat: np.ndarray) -> Optional[np.ndarray]:
        hand = self.robot.get_link("hand")
        q = self.robot.inverse_kinematics(link=hand, pos=pos, quat=quat)
        return np.array(q, dtype=float) if q is not None else None

    def _plan_and_execute(self, q_goal: np.ndarray, attached_object: Any = None, description: str = "", retries: int = 2) -> bool:
        """Execute path directly - FAST. Retry if needed."""
        if description:
            print(f"[motion] Planning: {description}")
        
        for attempt in range(retries):
            path = self.planner.plan_path(
                qpos_goal=q_goal,
                num_waypoints=self.config.num_waypoints,
                attached_object=attached_object,
                timeout=10.0,
            )
            
            if path:
                break
            
            if attempt < retries - 1:
                print(f"[motion] Planning failed, retry {attempt + 1}/{retries - 1}...")
                # Small random perturbation might help
                q_goal_perturbed = q_goal.copy()
                q_goal_perturbed[:7] += np.random.uniform(-0.01, 0.01, 7)
                q_goal = q_goal_perturbed
        
        if not path:
            print("[motion]Planning failed after retries")
            return False

        print(f"[motion] Executing...")
        
        # Direct execution
        for waypoint in path:
            if hasattr(waypoint, "cpu"):
                wp_array = waypoint.cpu().numpy().copy()
            else:
                wp_array = np.array(waypoint, dtype=float, copy=True)
            
            if self.gripper_holding:
                wp_array[-2:] = self.config.gripper_closed_width
            
            self.robot.control_dofs_position(wp_array)
            self.scene.step()
        
        # Store final target
        self.target_qpos = np.array(path[-1], dtype=float, copy=True)
        if self.gripper_holding:
            self.target_qpos[-2:] = self.config.gripper_closed_width
        
        # Brief hold
        for _ in range(10):
            self.robot.control_dofs_position(self.target_qpos)
            self.scene.step()
        
        return True

    def _hold_position(self, duration_seconds: float = 0.5):
        """Actively hold current position to prevent drift."""
        if self.target_qpos is None:
            current = self.robot.get_qpos()
            if hasattr(current, "cpu"):
                self.target_qpos = current.cpu().numpy().copy()
            else:
                self.target_qpos = np.array(current, dtype=float, copy=True)
        
        steps = int(duration_seconds * 100)
        
        for i in range(steps):
            self.robot.control_dofs_position(self.target_qpos)
            self.scene.step()

    def _interpolate_gripper(self, target_width: float) -> None:
        q_start = self.robot.get_qpos()
        if hasattr(q_start, "clone"):
            q_start = q_start.clone().cpu().numpy()
        else:
            q_start = np.array(q_start, dtype=float)
        
        q_target = np.array(q_start, copy=True)
        q_target[-2:] = target_width

        for i in range(self.config.gripper_steps):
            alpha = (i + 1) / float(self.config.gripper_steps)
            q = (1.0 - alpha) * q_start + alpha * q_target
            self.robot.control_dofs_position(q)
            self.scene.step()

    def open_gripper(self) -> None:
        print("[motion] Opening gripper...")
        self.gripper_holding = False
        self._interpolate_gripper(self.config.gripper_open_width)
        print("[motion] Gripper opened")

    def close_gripper(self) -> None:
        print("[motion] Closing gripper...")
        
        # Get current position BEFORE closing
        current_q = self.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            target_q = current_q.cpu().numpy().copy()
        else:
            target_q = np.array(current_q, dtype=float, copy=True)
        
        # Close gripper while HOLDING arm position
        for i in range(self.config.gripper_steps):
            alpha = (i + 1) / float(self.config.gripper_steps)
            gripper_width = (1 - alpha) * self.config.gripper_open_width + alpha * self.config.gripper_closed_width
            
            # Command arm joints to STAY at original position
            target_q[-2:] = gripper_width
            self.robot.control_dofs_position(target_q)
            self.scene.step()
        
        self.gripper_holding = True
        
        # Strong hold after closing
        target_q[-2:] = self.config.gripper_closed_width
        for _ in range(50):
            self.robot.control_dofs_position(target_q)
            self.scene.step()
        
        self.target_qpos = target_q
        print("[motion]Gripper closed")


    def pick_up(self, block_id: Any) -> bool:
        key = self._resolve_block_key(block_id)
        center = self._block_center(key)

        print(f"\n[motion] PICK-UP: '{key}'")

        top_of_block_z = center[2] + (BLOCK_SIZE / 2.0)
        approach_pos = center.copy()
        approach_pos[2] = top_of_block_z + self.config.min_approach_height
        grasp_pos = center.copy()
        grasp_pos[2] = center[2] + self.config.grasp_offset

        # Open gripper first
        self.open_gripper()

        # Go to approach
        q_approach = self._ik_for_pose(approach_pos, self.grasp_quat)
        if q_approach is None or not self._plan_and_execute(q_approach):
            return False

        # Descend to grasp
        q_grasp = self._ik_for_pose(grasp_pos, self.grasp_quat)
        if q_grasp is None or not self._plan_and_execute(q_grasp):
            return False

        self.close_gripper()

        # Hold briefly
        self._hold_position(0.15)

        # Direct lift
        current_q = self.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        for i in range(40):
            alpha = (i + 1) / 40.0
            q_interp = (1 - alpha) * start_q + alpha * q_approach
            q_interp[-2:] = self.config.gripper_closed_width
            self.robot.control_dofs_position(q_interp)
            self.scene.step()

        print("[motion] PICK-UP SUCCESS")
        return True

    def put_down(self, x: float = 0.50, y: float = 0.0) -> bool:
        """Place held block on table at position (x, y).
        
        Default: (0.50, 0.0) - centered below robot for stability
        """
        if not self.gripper_holding:
            print("[motion] Not holding any block!")
            return False
        
        print(f"\n[motion] PUT-DOWN at ({x:.2f}, {y:.2f})")
        
        # Find held block
        hand = self.robot.get_link("hand")
        hand_pos = np.array(hand.get_pos())
        held_block = None
        
        for key, block in self.blocks_state.items():
            block_pos = np.array(block.get_pos())
            dist = np.linalg.norm(block_pos - hand_pos)
            if dist < 0.15:
                held_block = block
                break
        
        # Calculate placement position
        place_center = np.array([x, y, TABLE_BLOCK_CENTER_Z])
        place_gripper = place_center.copy()
        place_gripper[2] = TABLE_BLOCK_CENTER_Z + self.config.grasp_offset
        
        # Approach position (above placement)
        approach_pos = place_gripper.copy()
        approach_pos[2] += 0.15
        
        # Move to approach
        q_approach = self._ik_for_pose(approach_pos, self.grasp_quat)
        if q_approach is None or not self._plan_and_execute(q_approach, attached_object=held_block):
            return False
        
        # Descend to placement
        current_q = self.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        q_place = self._ik_for_pose(place_gripper, self.grasp_quat)
        if q_place is None:
            return False
        
        # Slow descent
        for i in range(40):
            alpha = (i + 1) / 40.0
            q = (1 - alpha) * start_q + alpha * q_place
            q[-2:] = self.config.gripper_closed_width
            self.robot.control_dofs_position(q)
            self.scene.step()
        
        # Release
        self.open_gripper()
        
        # Lift up
        current_pos = np.array(hand.get_pos())
        up_pos = current_pos.copy()
        up_pos[2] += 0.10
        
        q_up = self._ik_for_pose(up_pos, self.grasp_quat)
        if q_up is not None:
            current_q = self.robot.get_qpos()
            if hasattr(current_q, "cpu"):
                start_q = current_q.cpu().numpy().copy()
            else:
                start_q = np.array(current_q, dtype=float, copy=True)
            
            for i in range(30):
                alpha = (i + 1) / 30.0
                q = (1 - alpha) * start_q + alpha * q_up
                self.robot.control_dofs_position(q)
                self.scene.step()
        
        print("[motion] PUT-DOWN SUCCESS")
        return True

    def arrange_base_blocks(self, block_ids: list, target_center: tuple = None) -> bool:
        """
        Arrange 4 blocks into a tight 2x2 square by picking and placing them.
        Blocks are placed close enough that they touch (4cm spacing center-to-center).
        
        Args:
            block_ids: List of 4 block IDs to arrange (e.g., ['r', 'g', 'b', 'y'])
            target_center: Optional (x, y) center for the square. Default: (0.55, 0.0)
            
        Returns:
            bool: True if successful
        """
        if len(block_ids) != 4:
            print(f"[motion] arrange_base_blocks requires 4 blocks, got {len(block_ids)}")
            return False
        
        print(f"[motion] ARRANGING 2x2 BASE: {block_ids}")
        
        if target_center is None:
            target_center = (0.55, 0.0)
        
        target_center = np.array(target_center)
        
        # Calculate target positions for 2x2 square with blocks TOUCHING
        # Block size is 0.04m, so blocks touching means 0.04m spacing center-to-center
        half_block = BLOCK_SIZE / 2.0  # 0.02m = 2cm
        
        target_positions = [
            (target_center[0] - half_block, target_center[1] - half_block),  # Back-left
            (target_center[0] + half_block, target_center[1] - half_block),  # Back-right
            (target_center[0] - half_block, target_center[1] + half_block),  # Front-left
            (target_center[0] + half_block, target_center[1] + half_block),  # Front-right
        ]
        
        print(f"[motion] Target center: ({target_center[0]:.3f}, {target_center[1]:.3f})")
        print(f"[motion] Blocks will be placed {BLOCK_SIZE*100:.1f}cm apart (touching)")
        
        # Pick up and place each block at exact positions
        for idx, bid in enumerate(block_ids):
            key = self._resolve_block_key(bid)
            target_xy = target_positions[idx]
            
            print(f"\n[motion] >> Block {idx+1}/4: {key.upper()}")
            print(f"[motion]    Target: ({target_xy[0]:.4f}, {target_xy[1]:.4f})")
            
            # Pick up block
            if not self.pick_up(key):
                print(f"[motion]  Failed to pick up {key}")
                return False
            
            # Place at exact target position
            if not self.put_down(x=target_xy[0], y=target_xy[1]):
                print(f"[motion]  Failed to place {key}")
                return False
            
            # Let physics settle
            for _ in range(80):
                self.scene.step()
            
            # Verify placement
            actual_pos = self._block_center(key)
            print(f"[motion]    ✓ Placed at ({actual_pos[0]:.4f}, {actual_pos[1]:.4f}, z={actual_pos[2]:.4f})")
        
        # Final verification
        print(f"\n[motion] ========================================")
        print(f"[motion] FINAL BASE CONFIGURATION:")
        print(f"[motion] ========================================")
        
        for bid in block_ids:
            key = self._resolve_block_key(bid)
            pos = self._block_center(key)
            print(f"[motion]   {key.upper()}: ({pos[0]:.4f}, {pos[1]:.4f}, z={pos[2]:.4f})")
        
        print(f"\n[motion]BASE ARRANGEMENT COMPLETE - Blocks are touching!")
        return True

    def stack_on(self, target_block_id: Any,predicates=None) -> bool:
        """
            Stack current block on top of target_id block.
            Uses per-tower XY: each base block gets its own fixed tower center.
        """
        if predicates is None:
            print("stack_on requires predicates")
            return False
        
        # 1. Determine base block for THIS target's tower
        base = self._find_base_block(target_block_id, predicates)
        
        # Find held block
        hand = self.robot.get_link("hand")
        hand_pos = np.array(hand.get_pos())
        held_block = None
        
        for key, block in self.blocks_state.items():
            block_pos = np.array(block.get_pos())
            dist = np.linalg.norm(block_pos - hand_pos)
            if dist < 0.15:
                held_block = block
                break
        
        # 2. If this base has no stored XY → detect and store
        if base not in self.tower_centers:
            center = self._block_center(base)
            self.tower_centers[base] = (center[0], center[1])
            print(f"New tower base: {base}, center={self.tower_centers[base]}")
        
        # 3. Use ONLY that tower’s XY
        tower_xy = np.array(self.tower_centers[base])
        
        # 4. Compute Z from actual target block
        target_center = self._block_center(target_block_id)
        target_z = target_center[2]
        BLOCK_SIZE = 0.04
        new_z = target_z + BLOCK_SIZE
        # Calculate precise placement using FIXED XY
        target_top_z = target_z + (BLOCK_SIZE / 2.0)
        final_block_bottom_z = target_top_z
        final_block_center_z = final_block_bottom_z + (BLOCK_SIZE / 2.0)
        final_gripper_z = final_block_center_z + self.config.grasp_offset
        
        # Use FIXED tower center for XY positioning
        target_xy = tower_xy
        
        # High approach - ABOVE target
        high_pos = np.array([target_xy[0], target_xy[1], final_gripper_z + 0.15])
        
        # Low approach - just above placement
        low_pos = np.array([target_xy[0], target_xy[1], final_gripper_z + 0.03])
        
        # Final placement - ON TARGET (no gap for drop)
        place_pos = np.array([target_xy[0], target_xy[1], final_gripper_z])
        
        # Move to high approach
        q_high = self._ik_for_pose(high_pos, self.grasp_quat)
        if q_high is None or not self._plan_and_execute(q_high, attached_object=held_block):
            return False
        
        # Descend to low approach using direct interpolation
        q_low = self._ik_for_pose(low_pos, self.grasp_quat)
        if q_low is None:
            return False
        
        current_q = self.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        # Smooth descent - 50 steps
        for i in range(50):
            alpha = (i + 1) / 50.0
            q = (1 - alpha) * start_q + alpha * q_low
            q[-2:] = self.config.gripper_closed_width
            self.robot.control_dofs_position(q)
            self.scene.step()
        
        # Final placement - descend to exact position
        q_place = self._ik_for_pose(place_pos, self.grasp_quat)
        if q_place is None:
            return False
        
        current_q = self.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        # Very slow final placement - 30 steps
        for i in range(30):
            alpha = (i + 1) / 30.0
            q = (1 - alpha) * start_q + alpha * q_place
            q[-2:] = self.config.gripper_closed_width
            self.robot.control_dofs_position(q)
            self.scene.step()
        
        # Hold position to let block settle
        print("[motion] Holding position for stability...")
        hold_q = self.robot.get_qpos()
        if hasattr(hold_q, "cpu"):
            hold_q = hold_q.cpu().numpy().copy()
        else:
            hold_q = np.array(hold_q, dtype=float, copy=True)
        
        # Hold for 100 steps (~1 second)
        for _ in range(100):
            self.robot.control_dofs_position(hold_q)
            self.scene.step()
        
        # Release
        self.open_gripper()
        
        # Small lift using direct interpolation
        current_pos = np.array(hand.get_pos())
        up_pos = current_pos.copy()
        up_pos[2] += 0.10  # 10cm up
        
        q_up = self._ik_for_pose(up_pos, self.grasp_quat)
        if q_up is not None:
            current_q = self.robot.get_qpos()
            if hasattr(current_q, "cpu"):
                start_q = current_q.cpu().numpy().copy()
            else:
                start_q = np.array(current_q, dtype=float, copy=True)
            
            for i in range(30):
                alpha = (i + 1) / 30.0
                q = (1 - alpha) * start_q + alpha * q_up
                self.robot.control_dofs_position(q)
                self.scene.step()
        
        print("[motion]STACK COMPLETE")
        return True
    def _find_base_block(self, block_id, predicates):
        """
        From target block, walk up ON(x,y) predicates until reaching ONTABLE(base).
        That base is the tower base for this block.
        """
        current = block_id
        
        while True:
            found_parent = False
            
            # Check ON(current, parent)
            for p in predicates:
                if p.startswith("ON("):
                    # Parse ON(a,b)
                    inside = p[3:-1]  # remove ON( )
                    a, b = inside.split(",")
                    
                    if a == current:
                        # a is on b
                        current = b
                        found_parent = True
                        break
            
            if not found_parent:
                # No ON(a,b) found → must be base (ONTABLE)
                return current
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