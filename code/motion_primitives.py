"""
motion_primitives.py

Motion primitive layer for the TAMP project.

This module bridges symbolic actions (PICK-UP, STACK, PUT-DOWN, UNSTACK)
and low-level motion planning / control in Genesis.

Step 1 (this file version):
    - Implements a robust PICK-UP primitive with:
        * pre-grasp approach
        * IK-based pose computation
        * OMPL path planning
        * smooth gripper open/close

Later we will extend this with STACK / PUT-DOWN / UNSTACK primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from planning import PlannerInterface


# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

# Physical constants from scenes.py
BLOCK_SIZE = 0.04          # side length of cube
TABLE_BLOCK_CENTER_Z = 0.02  # z-position of block center when on table

# Motion tuning
APPROACH_HEIGHT = 0.1     # vertical offset above block for pre-grasp
GRASP_HEIGHT_OFFSET = 0.07 # how far above block center the gripper should be
GRIPPER_OPEN_WIDTH = 0.04  # joint value for fully open fingers
GRIPPER_CLOSED_WIDTH = 0.0 # joint value for closed fingers
GRIPPER_STEPS = 100         # interpolation steps for open/close


@dataclass
class MotionConfig:
    """Config bundle so we can tune primitive behaviour in one place."""
    approach_height: float = APPROACH_HEIGHT
    grasp_height_offset: float = GRASP_HEIGHT_OFFSET
    gripper_open_width: float = GRIPPER_OPEN_WIDTH
    gripper_closed_width: float = GRIPPER_CLOSED_WIDTH
    gripper_steps: int = GRIPPER_STEPS
    num_waypoints: int = 200   # number of waypoints for OMPL interpolation


class MotionPrimitiveExecutor:
    """High-level executor for motion primitives.

    Usage:
        executor = MotionPrimitiveExecutor(scene, franka, blocks_state)
        success = executor.pick_up("r")
    """

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

        # OMPL interface wrapper
        self.planner = PlannerInterface(robot, scene)

        # Fixed downward-facing hand orientation.
        # This matches the quaternion used in your demo script.
        self.grasp_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_block_key(self, block_id: Any) -> str:
        """Normalise various block identifiers to the keys in blocks_state.

        Accepts:
            'r', 'R', 'red', 'RED' -> 'r'
            'g', 'GREEN', etc.     -> 'g'
        """
        s = str(block_id).strip().lower()
        if s in self.blocks_state:
            return s
        # fall back to first character (e.g. 'red' -> 'r')
        if s and s[0] in self.blocks_state:
            return s[0]
        raise KeyError(f"Unknown block identifier: {block_id!r}")

    def _block_center(self, block_key: str) -> np.ndarray:
        """Get the current 3D world position of a block's center."""
        block = self.blocks_state[block_key]
        pos = block.get_pos()
        if isinstance(pos, np.ndarray):
            return pos.astype(float)
        return np.array(pos, dtype=float)

    def _ik_for_pose(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Solve inverse kinematics for the hand link at (pos, quat)."""
        hand = self.robot.get_link("hand")
        q = self.robot.inverse_kinematics(link=hand, pos=pos, quat=quat)
        if q is None:
            raise RuntimeError("IK failed to find a solution.")
        return np.array(q, dtype=float)

    def _plan_and_execute(self, q_goal: np.ndarray, attached_object: Any = None) -> bool:
        """Use OMPL to plan from current qpos to q_goal, then execute.

        Returns True on success (non-empty path), False on failure.
        """
        path = self.planner.plan_path(
            qpos_goal=q_goal,
            num_waypoints=self.config.num_waypoints,
            attached_object=attached_object,
        )
        if not path:
            print("[motion] plan_and_execute: OMPL returned empty path.")
            return False

        for waypoint in path:
            self.robot.control_dofs_position(waypoint)
            self.scene.step()
        return True

    def _interpolate_gripper(self, target_width: float) -> None:
        """Smoothly interpolate gripper fingers to target open width."""
        q_start = self.robot.get_qpos().clone() if hasattr(self.robot.get_qpos(), "clone") else np.array(self.robot.get_qpos(), dtype=float)
        q_target = np.array(q_start, copy=True)
        q_target[-2:] = target_width

        for i in range(self.config.gripper_steps):
            alpha = (i + 1) / float(self.config.gripper_steps)
            q = (1.0 - alpha) * q_start + alpha * q_target
            self.robot.control_dofs_position(q)
            self.scene.step()

    def open_gripper(self) -> None:
        """Open the gripper to a preset width."""
        self._interpolate_gripper(self.config.gripper_open_width)

    def close_gripper(self) -> None:
        """Close the gripper (simple joint interpolation)."""
        self._interpolate_gripper(self.config.gripper_closed_width)

    # ------------------------------------------------------------------
    # Public motion primitives
    # ------------------------------------------------------------------

    def pick_up(self, block_id: Any) -> bool:
        """Pick up the specified block from the table or a stack.

        High-level sequence:
            1. Resolve which block entity we mean.
            2. Compute pre-grasp position above block.
            3. Open gripper.
            4. Plan & execute path to pre-grasp.
            5. Plan & execute path down to grasp pose.
            6. Close gripper.
            7. Lift back to pre-grasp height.

        Returns
        -------
        bool
            True if all internal steps succeeded, False otherwise.
        """
        key = self._resolve_block_key(block_id)
        center = self._block_center(key)

        # Compute key z-positions
        top_of_block_z = center[2] + (BLOCK_SIZE)
        pre_grasp_pos = center.copy()
        pre_grasp_pos[2] = top_of_block_z + self.config.approach_height

        grasp_pos = center.copy()
        grasp_pos[2] = top_of_block_z + self.config.grasp_height_offset

        goal_pos = [2,0,2]
        # ------------------------------------------------------------------
        # Motion sequence
        # ------------------------------------------------------------------
        print(f"[motion] PICK-UP: block={key}, center={center}")

        # 1) Open gripper
        self.open_gripper()
        # 2) Move to pre-grasp pose
        try:
            q_pre = self._ik_for_pose(pre_grasp_pos, self.grasp_quat)
        except RuntimeError as exc:
            print("[motion] pick_up: IK failed for pre-grasp pose:", exc)
            return False

        if not self._plan_and_execute(q_pre):
            print("[motion] pick_up: failed to reach pre-grasp pose.")
            return False

        # 3) Move from pre-grasp down to grasp pose
        try:
            q_grasp = self._ik_for_pose(grasp_pos, self.grasp_quat)
        except RuntimeError as exc:
            print("[motion] pick_up: IK failed for grasp pose:", exc)
            return False

        if not self._plan_and_execute(q_grasp):
            print("[motion] pick_up: failed to reach grasp pose.")
            return False

        # 4) Close gripper
        self.close_gripper()

        # 3) Move from pre-grasp down to grasp pose
        try:
            q_goal = self._ik_for_pose(goal_pos, self.grasp_quat)
        except RuntimeError as exc:
            print("[motion] pick_up: IK failed for goal pose:", exc)
            return False
        
        # 5) Lift back to pre-grasp height (with the block)
        if not self._plan_and_execute(q_goal, attached_object=self.blocks_state[key]):
            # If this fails, we still consider the grasp likely succeeded, but
            # we report the motion failure so higher-level code can re-plan.
            print("[motion] pick_up: failed to lift after grasp.")
            return False

        # self.open_gripper()
        print("[motion] pick_up: completed successfully.")
        return True

    # Placeholders for future primitives (will be implemented after testing pick_up)
    def put_down(self, *args, **kwargs):
        raise NotImplementedError("put_down primitive not implemented yet.")

    def stack(self, *args, **kwargs):
        raise NotImplementedError("stack primitive not implemented yet.")

    def unstack(self, *args, **kwargs):
        raise NotImplementedError("unstack primitive not implemented yet.")
