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
    num_waypoints: int = 200
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

    def _plan_and_execute(self, q_goal: np.ndarray, attached_object: Any = None, description: str = "") -> bool:
        """Execute path while PREVENTING ARM DRIFT."""
        if description:
            print(f"[motion] Planning: {description}")
        
        path = self.planner.plan_path(
            qpos_goal=q_goal,
            num_waypoints=self.config.num_waypoints,
            attached_object=attached_object,
            timeout=10.0,
        )
        
        if not path:
            print("[motion] ❌ Planning failed")
            return False

        print(f"[motion] Executing {len(path)} waypoints...")
        
        for i, waypoint in enumerate(path):
            if hasattr(waypoint, "cpu"):
                wp_array = waypoint.cpu().numpy().copy()
            else:
                wp_array = np.array(waypoint, dtype=float, copy=True)
            
            # Keep gripper at current state
            if self.gripper_holding:
                wp_array[-2:] = self.config.gripper_closed_width
            
            self.robot.control_dofs_position(wp_array)
            self.scene.step()
        
        # CRITICAL: Store final target and HOLD IT
        self.target_qpos = np.array(path[-1], dtype=float, copy=True)
        if self.gripper_holding:
            self.target_qpos[-2:] = self.config.gripper_closed_width
        
        # Actively hold position briefly to prevent drift
        for _ in range(30):  # Reduced from 100 to 30 (0.3s)
            self.robot.control_dofs_position(self.target_qpos)
            self.scene.step()
        
        print("[motion] ✅ Complete and stable")
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

        print(f"\n{'='*70}")
        print(f"[motion] PICK-UP: '{key}'")
        print(f"{'='*70}\n")

        top_of_block_z = center[2] + (BLOCK_SIZE / 2.0)
        retreat_pos = self.config.safe_retreat_pos.copy()
        approach_pos = center.copy()
        approach_pos[2] = top_of_block_z + self.config.min_approach_height
        grasp_pos = center.copy()
        grasp_pos[2] = center[2] + self.config.grasp_offset

        self.open_gripper()

        q_retreat = self._ik_for_pose(retreat_pos, self.grasp_quat)
        if q_retreat is None or not self._plan_and_execute(q_retreat):
            return False

        q_approach = self._ik_for_pose(approach_pos, self.grasp_quat)
        if q_approach is None or not self._plan_and_execute(q_approach):
            return False

        q_grasp = self._ik_for_pose(grasp_pos, self.grasp_quat)
        if q_grasp is None or not self._plan_and_execute(q_grasp):
            return False

        self.close_gripper()

        # Brief stabilization after grasp
        self._hold_position(0.3)

        if not self._plan_and_execute(q_approach, attached_object=self.blocks_state[key]):
            self.gripper_holding = False
            return False

        if not self._plan_and_execute(q_retreat, attached_object=self.blocks_state[key]):
            self.gripper_holding = False
            return False

        print(f"\n{'='*70}")
        print("[motion] ✅ PICK-UP SUCCESS!")
        print(f"{'='*70}\n")
        return True

    def stack_on(self, target_block_id: Any) -> bool:
        if not self.gripper_holding:
            print("[motion] ❌ Not holding!")
            return False
        
        target_key = self._resolve_block_key(target_block_id)
        target_center = self._block_center(target_key)
        
        print(f"\n{'='*70}")
        print(f"[motion] STACK on '{target_key}' - FIXED PLACEMENT")
        print(f"{'='*70}\n")
        
        # Find held block
        hand = self.robot.get_link("hand")
        hand_pos = np.array(hand.get_pos())
        held_block = None
        
        for key, block in self.blocks_state.items():
            block_pos = np.array(block.get_pos())
            dist = np.linalg.norm(block_pos - hand_pos)
            if dist < 0.15:
                held_block = block
                print(f"  Holding: '{key}' (dist={dist:.3f}m)")
                break
        
        # CORRECT PLACEMENT CALCULATION
        # Get FRESH target position (in case it moved)
        target_center = self._block_center(target_key)
        print(f"  Fresh target position: {target_center}")
        
        # Target block top surface
        target_top_z = target_center[2] + (BLOCK_SIZE / 2.0)
        
        # Where held block's BOTTOM should end up (on target's top)
        final_block_bottom_z = target_top_z
        
        # Where held block's CENTER will be
        final_block_center_z = final_block_bottom_z + (BLOCK_SIZE / 2.0)
        
        # Where GRIPPER should be (grasp_offset above block center)
        final_gripper_z = final_block_center_z + self.config.grasp_offset
        
        # High approach - DIRECTLY ABOVE TARGET (use target's X, Y!)
        high_approach_pos = np.array([
            target_center[0],  # EXACT X of green
            target_center[1],  # EXACT Y of green
            final_gripper_z + 0.20  # 20cm above
        ])
        
        print(f"  Target X,Y: [{target_center[0]:.3f}, {target_center[1]:.3f}]")
        print(f"  Target top: z={target_top_z:.3f}m")
        print(f"  Final block bottom: z={final_block_bottom_z:.3f}m")
        print(f"  Final block center: z={final_block_center_z:.3f}m")
        print(f"  Final gripper: z={final_gripper_z:.3f}m")
        print(f"  High approach: [{high_approach_pos[0]:.3f}, {high_approach_pos[1]:.3f}, {high_approach_pos[2]:.3f}]")
        
        # Move to high approach - ABOVE TARGET
        print(f"\n[motion] Stage 1: Moving directly above target...")
        q_high = self._ik_for_pose(high_approach_pos, self.grasp_quat)
        if q_high is None:
            print("[motion] ❌ IK failed")
            return False
        
        if not self._plan_and_execute(q_high, attached_object=held_block):
            return False
        
        # Verify we're above the target
        current_hand_pos = np.array(hand.get_pos())
        print(f"  Current hand position: [{current_hand_pos[0]:.3f}, {current_hand_pos[1]:.3f}, {current_hand_pos[2]:.3f}]")
        xy_error = np.sqrt((current_hand_pos[0] - target_center[0])**2 + 
                          (current_hand_pos[1] - target_center[1])**2)
        print(f"  XY alignment error: {xy_error:.3f}m")
        
        if xy_error > 0.05:  # More than 5cm off
            print(f"  ⚠️  Not well aligned, adjusting...")
            # Try to move directly above
            adjustment_pos = np.array([
                target_center[0],
                target_center[1],
                current_hand_pos[2]  # Keep same height
            ])
            q_adjust = self._ik_for_pose(adjustment_pos, self.grasp_quat)
            if q_adjust is not None:
                q_adjust[-2:] = self.config.gripper_closed_width
                for _ in range(50):
                    self.robot.control_dofs_position(q_adjust)
                    self.scene.step()
                print(f"  ✅ Adjusted to be above target")
        
        # SIMPLE VERTICAL DESCENT - maintaining X,Y!
        print(f"\n[motion] Stage 2: Descending vertically...")
        
        # Get fresh hand position after any adjustment
        start_hand_pos = np.array(hand.get_pos())
        target_hand_z = final_gripper_z + 0.02  # 2cm above final
        descent_distance = start_hand_pos[2] - target_hand_z
        
        print(f"  Start z: {start_hand_pos[2]:.3f}m")
        print(f"  Target z: {target_hand_z:.3f}m")
        print(f"  Descending: {descent_distance:.3f}m")
        
        # Descend in small steps, MAINTAINING target X,Y
        steps = 150
        for i in range(steps):
            alpha = (i + 1) / float(steps)
            
            # Target position: FORCE it to be at target X,Y!
            target_pos = np.array([
                target_center[0],  # LOCK to green's X
                target_center[1],  # LOCK to green's Y
                start_hand_pos[2] - (descent_distance * alpha)  # Descend in Z
            ])
            
            # Compute IK
            q_desc = self._ik_for_pose(target_pos, self.grasp_quat)
            
            if q_desc is not None:
                q_desc[-2:] = self.config.gripper_closed_width
                self.robot.control_dofs_position(q_desc)
            else:
                # Maintain current
                current_q = self.robot.get_qpos()
                if hasattr(current_q, "cpu"):
                    current_q = current_q.cpu().numpy().copy()
                else:
                    current_q = np.array(current_q, dtype=float, copy=True)
                current_q[-2:] = self.config.gripper_closed_width
                self.robot.control_dofs_position(current_q)
            
            self.scene.step()
        
        # Final position check
        final_hand_pos = np.array(hand.get_pos())
        final_xy_error = np.sqrt((final_hand_pos[0] - target_center[0])**2 + 
                                (final_hand_pos[1] - target_center[1])**2)
        print(f"  Final hand: [{final_hand_pos[0]:.3f}, {final_hand_pos[1]:.3f}, {final_hand_pos[2]:.3f}]")
        print(f"  Final XY error: {final_xy_error:.3f}m")
        print("[motion] ✅ Descent complete")
        
        # Release
        print(f"\n[motion] Stage 3: Releasing...")
        self.open_gripper()
        
        # Let block drop and settle
        for _ in range(50):
            self.scene.step()
        
        # Simple straight-up ascent
        print(f"\n[motion] Stage 4: Ascending...")
        for i in range(100):
            current_pos = np.array(hand.get_pos())
            up_pos = current_pos.copy()
            up_pos[2] += 0.001  # 1mm per step
            
            q_up = self._ik_for_pose(up_pos, self.grasp_quat)
            if q_up is not None:
                self.robot.control_dofs_position(q_up)
            self.scene.step()
        
        print(f"\n{'='*70}")
        print("[motion] ✅ STACK COMPLETE!")
        print(f"{'='*70}\n")
        return True