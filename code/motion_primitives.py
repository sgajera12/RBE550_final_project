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
            print("[motion] ❌ Planning failed after retries")
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
        print("[motion] ✅ Gripper opened")

    def close_gripper(self) -> None:
        print("[motion] Closing gripper...")
        self._interpolate_gripper(self.config.gripper_closed_width)
        self.gripper_holding = True
        
        # Brief hold - removed monitoring printouts
        q_closed = self.robot.get_qpos()
        if hasattr(q_closed, "cpu"):
            q_closed = q_closed.cpu().numpy().copy()
        else:
            q_closed = np.array(q_closed, dtype=float, copy=True)
        
        q_closed[-2:] = self.config.gripper_closed_width
        
        for i in range(50):  # Reduced from 100 to 50
            self.robot.control_dofs_position(q_closed)
            self.scene.step()
        
        self.target_qpos = q_closed
        print("[motion] ✅ Gripper closed")

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

        print("[motion] ✅ PICK-UP SUCCESS")
        return True

    def put_down(self, x: float = None, y: float = None) -> bool:
        """Place held block on table at position (x, y).
        
        If x, y not provided, places at current x, y below gripper.
        """
        if not self.gripper_holding:
            print("[motion] ❌ Not holding any block!")
            return False
        
        print(f"\n[motion] PUT-DOWN at ({x}, {y})")
        
        # If position not specified, use current x, y
        if x is None or y is None:
            hand = self.robot.get_link("hand")
            hand_pos = np.array(hand.get_pos())
            x = hand_pos[0] if x is None else x
            y = hand_pos[1] if y is None else y
        
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
        
        print("[motion] ✅ PUT-DOWN SUCCESS")
        return True

    def stack_on(self, target_block_id: Any) -> bool:
        if not self.gripper_holding:
            print("[motion] ❌ Not holding!")
            return False
        
        target_key = self._resolve_block_key(target_block_id)
        target_center = self._block_center(target_key)
        
        print(f"\n[motion] STACK on '{target_key}'")
        
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
        
        # Get FRESH target position
        target_center = self._block_center(target_key)
        
        # Calculate precise placement
        target_top_z = target_center[2] + (BLOCK_SIZE / 2.0)
        final_block_bottom_z = target_top_z
        final_block_center_z = final_block_bottom_z + (BLOCK_SIZE / 2.0)
        final_gripper_z = final_block_center_z + self.config.grasp_offset
        
        # High approach - ABOVE target
        high_pos = np.array([target_center[0], target_center[1], final_gripper_z + 0.15])
        
        # Low approach - just above placement
        low_pos = np.array([target_center[0], target_center[1], final_gripper_z + 0.03])
        
        # Final placement - ON TARGET (no gap for drop)
        place_pos = np.array([target_center[0], target_center[1], final_gripper_z])
        
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
        
        print("[motion] ✅ STACK COMPLETE")
        return True