"""
pentagon_motion_primitives.py - Motion primitives for pentagon stacking

Extends motion_primitives.py with pentagon-specific actions:
- place_at_position: Place block at exact (x,y,z) position
- stack_on_two: Stack block on two support blocks (for top pentagon)
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
from motion_primitives import MotionPrimitiveExecutor, MotionConfig


class PentagonMotionExecutor(MotionPrimitiveExecutor):
    """Extended motion primitives for pentagon stacking."""
    
    def __init__(
        self,
        scene: Any,
        robot: Any,
        blocks_state: Dict[str, Any],
        config: Optional[MotionConfig] = None,
    ):
        super().__init__(scene, robot, blocks_state, config)
        
        # Store block orientations (in degrees)
        # Each block in pentagon gets 72° increment (360°/5)
        self.block_orientations = {}
    
    def _apply_wrist_rotation(self, qpos: np.ndarray, angle_degrees: float) -> np.ndarray:
        """
        Apply rotation to wrist joint only (joint 6, the last arm joint).
        
        The Franka has 7 arm joints (0-6) + 2 gripper joints (7-8).
        Joint 6 is the wrist rotation joint.
        
        Args:
            qpos: Full joint position array [9 values: 7 arm + 2 gripper]
            angle_degrees: Desired wrist rotation in degrees
            
        Returns:
            Modified qpos with wrist rotated
        """
        qpos_rotated = qpos.copy()
        
        # Joint 6 is the wrist rotation (last arm joint before gripper)
        # Convert degrees to radians
        angle_rad = np.deg2rad(angle_degrees)
        
        # Set wrist joint angle
        qpos_rotated[6] = angle_rad
        
        return qpos_rotated
        
    def place_at_position(self, target_x: float, target_y: float, target_z: float = 0.02, 
                         orientation_degrees: float = 0.0) -> bool:
        """
        Place currently held block at exact (x, y, z) position with wrist rotation.
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate  
            target_z: Target Z coordinate (block center height, default 0.02 for table)
            orientation_degrees: Wrist rotation angle in degrees (0, 72, 144, 216, 288 for pentagon)
            
        Returns:
            True if successful
        """
        if not self.gripper_holding:
            print("[pentagon_motion] Not holding any block!")
            return False
        
        print(f"\n[pentagon_motion] PLACE AT POSITION: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})")
        print(f"[pentagon_motion] Wrist rotation: {orientation_degrees:.1f}°")
        
        # Find held block
        hand = self.robot.get_link("hand")
        hand_pos = np.array(hand.get_pos())
        held_block = None
        
        for key, block in self.blocks_state.items():
            block_pos = np.array(block.get_pos())
            dist = np.linalg.norm(block_pos - hand_pos)
            # Increased tolerance to 20cm - block may be slightly separated after pick-up
            if dist < 0.20:
                held_block = block
                print(f"[pentagon_motion] Identified held block: {key} (dist: {dist:.3f}m)")
                break
        
        if held_block is None:
            print("[pentagon_motion] Could not identify held block!")
            print(f"[pentagon_motion] Hand position: {hand_pos}")
            print("[pentagon_motion] Block distances:")
            for key, block in self.blocks_state.items():
                block_pos = np.array(block.get_pos())
                dist = np.linalg.norm(block_pos - hand_pos)
                print(f"  {key}: {dist:.3f}m")
            return False
        
        # Use standard gripper orientation (pointing down)
        grasp_quat = self.grasp_quat
        
        # Calculate placement gripper position
        place_gripper = np.array([target_x, target_y, target_z + self.config.grasp_offset])
        
        # Approach position (15cm above placement)
        approach_pos = place_gripper.copy()
        approach_pos[2] += 0.15
        
        # Move to approach
        q_approach = self._ik_for_pose(approach_pos, grasp_quat)
        if q_approach is None:
            print("[pentagon_motion] Failed IK for approach")
            return False
        
        # Apply wrist rotation to approach pose
        q_approach = self._apply_wrist_rotation(q_approach, orientation_degrees)
        
        if not self._plan_and_execute(q_approach, attached_object=held_block, description="Approach placement"):
            return False
        
        # Descend to placement
        q_place = self._ik_for_pose(place_gripper, grasp_quat)
        if q_place is None:
            print("[pentagon_motion] Failed IK for placement")
            return False
        
        # Apply wrist rotation to placement pose
        q_place = self._apply_wrist_rotation(q_place, orientation_degrees)
        
        current_q = self.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        # Slow descent with wrist rotation maintained
        for i in range(80):  # Slower for better precision (was 60)
            alpha = (i + 1) / 80.0
            q = (1 - alpha) * start_q + alpha * q_place
            q[-2:] = self.config.gripper_closed_width
            self.robot.control_dofs_position(q)
            self.scene.step()
        
        # Hold position briefly
        print("[pentagon_motion] Holding position...")
        for _ in range(120):  # Hold longer for stability (was 80)
            q_place[-2:] = self.config.gripper_closed_width
            self.robot.control_dofs_position(q_place)
            self.scene.step()
        
        # === CRITICAL: Lift BEFORE opening gripper to avoid disturbing adjacent blocks ===
        print("[pentagon_motion] Lifting before opening gripper...")
        current_pos = np.array(hand.get_pos())
        safe_open_pos = current_pos.copy()
        safe_open_pos[2] += 0.01  # Lift 10cm before opening (was 5cm)
        
        q_safe_open = self._ik_for_pose(safe_open_pos, grasp_quat)
        if q_safe_open is not None:
            q_safe_open = self._apply_wrist_rotation(q_safe_open, orientation_degrees)
            
            current_q = self.robot.get_qpos()
            if hasattr(current_q, "cpu"):
                start_q = current_q.cpu().numpy().copy()
            else:
                start_q = np.array(current_q, dtype=float, copy=True)
            
            # Lift with gripper STILL CLOSED
            for i in range(30):
                alpha = (i + 1) / 30.0
                q = (1 - alpha) * start_q + alpha * q_safe_open
                q[-2:] = self.config.gripper_closed_width  # Keep closed!
                self.robot.control_dofs_position(q)
                self.scene.step()
        
        # NOW safe to open gripper (lifted 5cm above block)
        self.open_gripper()
        
        # Lift up (maintain wrist rotation)
        # CRITICAL: Lift straight UP in Z-direction only to avoid disturbing adjacent blocks
        print("[pentagon_motion] Lifting straight up (Z-only movement)...")
        current_pos = np.array(hand.get_pos())
        up_pos = current_pos.copy()
        up_pos[2] += 0.20  # Lift 20cm more (was 15cm) - total 30cm from placement
        # Keep X and Y exactly the same - only change Z!
        
        q_up = self._ik_for_pose(up_pos, grasp_quat)
        if q_up is not None:
            q_up = self._apply_wrist_rotation(q_up, orientation_degrees)
            
            current_q = self.robot.get_qpos()
            if hasattr(current_q, "cpu"):
                start_q = current_q.cpu().numpy().copy()
            else:
                start_q = np.array(current_q, dtype=float, copy=True)
            
            # Slow vertical lift (50 steps for smooth Z-only motion)
            for i in range(50):
                alpha = (i + 1) / 50.0
                q = (1 - alpha) * start_q + alpha * q_up
                self.robot.control_dofs_position(q)
                self.scene.step()
        
        print(f"[pentagon_motion] ✓ PLACED at ({target_x:.4f}, {target_y:.4f}, {target_z:.4f}) @ {orientation_degrees:.1f}°")
        return True
    
    def stack_on_two(
        self, 
        support_block1_id: str, 
        support_block2_id: str,
        target_x: float,
        target_y: float,
        target_z: float,
        orientation_degrees: float = 0.0
    ) -> bool:
        """
        Stack currently held block on TWO support blocks at specific position with wrist rotation.
        
        This is the key primitive for the top pentagon - each block rests on
        two adjacent base blocks due to the 36° rotation.
        
        Args:
            support_block1_id: First support block ID
            support_block2_id: Second support block ID
            target_x: Target X coordinate for block center
            target_y: Target Y coordinate for block center
            target_z: Target Z coordinate for block center
            orientation_degrees: Wrist rotation angle in degrees
            
        Returns:
            True if successful
        """
        if not self.gripper_holding:
            print("[pentagon_motion] Not holding any block!")
            return False
        
        print(f"\n[pentagon_motion] ========================================")
        print(f"[pentagon_motion] STACK ON TWO BLOCKS")
        print(f"[pentagon_motion] Support: [{support_block1_id}, {support_block2_id}]")
        print(f"[pentagon_motion] Target: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})")
        print(f"[pentagon_motion] Wrist rotation: {orientation_degrees:.1f}°")
        print(f"[pentagon_motion] ========================================")
        
        # Find held block
        hand = self.robot.get_link("hand")
        hand_pos = np.array(hand.get_pos())
        held_block = None
        
        for key, block in self.blocks_state.items():
            block_pos = np.array(block.get_pos())
            dist = np.linalg.norm(block_pos - hand_pos)
            if dist < 0.25:
                held_block = block
                break
        
        if held_block is None:
            print("[pentagon_motion] Could not identify held block!")
            return False
        
        # Verify support blocks exist and get their positions
        try:
            support1_key = self._resolve_block_key(support_block1_id)
            support2_key = self._resolve_block_key(support_block2_id)
        except KeyError as e:
            print(f"[pentagon_motion] Support block not found: {e}")
            return False
        
        support1_pos = self._block_center(support1_key)
        support2_pos = self._block_center(support2_key)
        
        print(f"[pentagon_motion] Support1 ({support1_key}): ({support1_pos[0]:.4f}, {support1_pos[1]:.4f}, {support1_pos[2]:.4f})")
        print(f"[pentagon_motion] Support2 ({support2_key}): ({support2_pos[0]:.4f}, {support2_pos[1]:.4f}, {support2_pos[2]:.4f})")
        
        # Use standard gripper orientation (pointing down)
        grasp_quat = self.grasp_quat
        
        # Calculate placement gripper position
        place_gripper = np.array([target_x, target_y, target_z + self.config.grasp_offset])
        
        # High approach - well above target
        high_pos = place_gripper.copy()
        high_pos[2] += 0.20
        
        # Low approach - just above placement
        low_pos = place_gripper.copy()
        low_pos[2] += 0.05
        
        # Move to high approach
        print(f"[pentagon_motion] Moving to high approach...")
        q_high = self._ik_for_pose(high_pos, grasp_quat)
        if q_high is None:
            print("[pentagon_motion] Failed IK for high approach")
            return False
        
        # Apply wrist rotation
        q_high = self._apply_wrist_rotation(q_high, orientation_degrees)
        
        if not self._plan_and_execute(q_high, attached_object=held_block, description="High approach"):
            return False
        
        # Descend to low approach
        print(f"[pentagon_motion] Descending to low approach...")
        q_low = self._ik_for_pose(low_pos, grasp_quat)
        if q_low is None:
            print("[pentagon_motion] Failed IK for low approach")
            return False
        
        # Apply wrist rotation
        q_low = self._apply_wrist_rotation(q_low, orientation_degrees)
        
        current_q = self.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        # Smooth descent to low approach
        for i in range(60):
            alpha = (i + 1) / 60.0
            q = (1 - alpha) * start_q + alpha * q_low
            q[-2:] = self.config.gripper_closed_width
            self.robot.control_dofs_position(q)
            self.scene.step()
        
        # Final placement - very slow and controlled
        print(f"[pentagon_motion] Final placement...")
        q_place = self._ik_for_pose(place_gripper, grasp_quat)
        if q_place is None:
            print("[pentagon_motion] Failed IK for placement")
            return False
        
        # Apply wrist rotation
        q_place = self._apply_wrist_rotation(q_place, orientation_degrees)
        
        current_q = self.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        # Very slow final descent - 80 steps
        for i in range(80):
            alpha = (i + 1) / 80.0
            q = (1 - alpha) * start_q + alpha * q_place
            q[-2:] = self.config.gripper_closed_width
            self.robot.control_dofs_position(q)
            self.scene.step()
        
        # Extended hold for stability on two-point contact
        print("[pentagon_motion] Holding position for stability...")
        hold_q = q_place.copy()
        hold_q[-2:] = self.config.gripper_closed_width
        
        # Hold longer for two-point contact (150 steps ~1.5 seconds)
        for _ in range(200):
            self.robot.control_dofs_position(hold_q)
            self.scene.step()
        
        # Release very slowly
        print("[pentagon_motion] Releasing...")
        self.open_gripper()
        
        # Additional settling time after release
        for _ in range(80):
            self.scene.step()
        
        # Careful lift (maintain wrist rotation)
        current_pos = np.array(hand.get_pos())
        up_pos = current_pos.copy()
        up_pos[2] += 0.12
        
        q_up = self._ik_for_pose(up_pos, grasp_quat)
        if q_up is not None:
            q_up = self._apply_wrist_rotation(q_up, orientation_degrees)
            
            current_q = self.robot.get_qpos()
            if hasattr(current_q, "cpu"):
                start_q = current_q.cpu().numpy().copy()
            else:
                start_q = np.array(current_q, dtype=float, copy=True)
            
            for i in range(50):
                alpha = (i + 1) / 50.0
                q = (1 - alpha) * start_q + alpha * q_up
                self.robot.control_dofs_position(q)
                self.scene.step()
        
        # Final verification
        final_pos = np.array(held_block.get_pos())
        error = np.linalg.norm(final_pos - np.array([target_x, target_y, target_z]))
        
        print(f"[pentagon_motion] ✓ STACKED ON TWO")
        print(f"[pentagon_motion] Target: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f}) @ {orientation_degrees:.1f}°")
        print(f"[pentagon_motion] Actual: ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
        print(f"[pentagon_motion] Error: {error:.4f}m")
        print(f"[pentagon_motion] ========================================")
        
        return True
        """
        Stack currently held block on TWO support blocks at specific position with orientation.
        
        This is the key primitive for the top pentagon - each block rests on
        two adjacent base blocks due to the 36° rotation.
        
        Args:
            support_block1_id: First support block ID
            support_block2_id: Second support block ID
            target_x: Target X coordinate for block center
            target_y: Target Y coordinate for block center
            target_z: Target Z coordinate for block center
            orientation_degrees: Rotation angle in degrees
            
        Returns:
            True if successful
        """
        if not self.gripper_holding:
            print("[pentagon_motion] Not holding any block!")
            return False
        
        print(f"\n[pentagon_motion] ========================================")
        print(f"[pentagon_motion] STACK ON TWO BLOCKS")
        print(f"[pentagon_motion] Support: [{support_block1_id}, {support_block2_id}]")
        print(f"[pentagon_motion] Target: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})")
        print(f"[pentagon_motion] Orientation: {orientation_degrees:.1f}°")
        print(f"[pentagon_motion] ========================================")
        
        # Find held block
        hand = self.robot.get_link("hand")
        hand_pos = np.array(hand.get_pos())
        held_block = None
        
        for key, block in self.blocks_state.items():
            block_pos = np.array(block.get_pos())
            dist = np.linalg.norm(block_pos - hand_pos)
            if dist < 0.25:
                held_block = block
                break
        
        if held_block is None:
            print("[pentagon_motion] Could not identify held block!")
            return False
        
        # Verify support blocks exist and get their positions
        try:
            support1_key = self._resolve_block_key(support_block1_id)
            support2_key = self._resolve_block_key(support_block2_id)
        except KeyError as e:
            print(f"[pentagon_motion] Support block not found: {e}")
            return False
        
        support1_pos = self._block_center(support1_key)
        support2_pos = self._block_center(support2_key)
        
        print(f"[pentagon_motion] Support1 ({support1_key}): ({support1_pos[0]:.4f}, {support1_pos[1]:.4f}, {support1_pos[2]:.4f})")
        print(f"[pentagon_motion] Support2 ({support2_key}): ({support2_pos[0]:.4f}, {support2_pos[1]:.4f}, {support2_pos[2]:.4f})")
        
        # Calculate gripper quaternion with rotation
        grasp_quat = self._rotation_quaternion(orientation_degrees)
        
        # Calculate placement gripper position
        place_gripper = np.array([target_x, target_y, target_z + self.config.grasp_offset])
        
        # High approach - well above target
        high_pos = place_gripper.copy()
        high_pos[2] += 0.20
        
        # Low approach - just above placement
        low_pos = place_gripper.copy()
        low_pos[2] += 0.05
        
        # Move to high approach
        print(f"[pentagon_motion] Moving to high approach...")
        q_high = self._ik_for_pose(high_pos, grasp_quat)
        if q_high is None:
            print("[pentagon_motion] Failed IK for high approach")
            return False
        
        if not self._plan_and_execute(q_high, attached_object=held_block, description="High approach"):
            return False
        
        # Descend to low approach
        print(f"[pentagon_motion] Descending to low approach...")
        q_low = self._ik_for_pose(low_pos, grasp_quat)
        if q_low is None:
            print("[pentagon_motion] Failed IK for low approach")
            return False
        
        current_q = self.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        # Smooth descent to low approach
        for i in range(60):
            alpha = (i + 1) / 60.0
            q = (1 - alpha) * start_q + alpha * q_low
            q[-2:] = self.config.gripper_closed_width
            self.robot.control_dofs_position(q)
            self.scene.step()
        
        # Final placement - very slow and controlled
        print(f"[pentagon_motion] Final placement...")
        q_place = self._ik_for_pose(place_gripper, grasp_quat)
        if q_place is None:
            print("[pentagon_motion] Failed IK for placement")
            return False
        
        current_q = self.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        # Very slow final descent - 80 steps
        for i in range(80):
            alpha = (i + 1) / 80.0
            q = (1 - alpha) * start_q + alpha * q_place
            q[-2:] = self.config.gripper_closed_width
            self.robot.control_dofs_position(q)
            self.scene.step()
        
        # Extended hold for stability on two-point contact
        print("[pentagon_motion] Holding position for stability...")
        hold_q = self.robot.get_qpos()
        if hasattr(hold_q, "cpu"):
            hold_q = hold_q.cpu().numpy().copy()
        else:
            hold_q = np.array(hold_q, dtype=float, copy=True)
        
        # Hold longer for two-point contact (150 steps ~1.5 seconds)
        for _ in range(200):
            self.robot.control_dofs_position(hold_q)
            self.scene.step()
        
        # Release very slowly
        print("[pentagon_motion] Releasing...")
        self.open_gripper()
        
        # Additional settling time after release
        for _ in range(80):
            self.scene.step()
        
        # Careful lift
        current_pos = np.array(hand.get_pos())
        up_pos = current_pos.copy()
        up_pos[2] += 0.12
        
        q_up = self._ik_for_pose(up_pos, grasp_quat)
        if q_up is not None:
            current_q = self.robot.get_qpos()
            if hasattr(current_q, "cpu"):
                start_q = current_q.cpu().numpy().copy()
            else:
                start_q = np.array(current_q, dtype=float, copy=True)
            
            for i in range(50):
                alpha = (i + 1) / 50.0
                q = (1 - alpha) * start_q + alpha * q_up
                self.robot.control_dofs_position(q)
                self.scene.step()
        
        # Final verification
        final_pos = np.array(held_block.get_pos())
        error = np.linalg.norm(final_pos - np.array([target_x, target_y, target_z]))
        
        print(f"[pentagon_motion] ✓ STACKED ON TWO")
        print(f"[pentagon_motion] Target: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f}) @ {orientation_degrees:.1f}°")
        print(f"[pentagon_motion] Actual: ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
        print(f"[pentagon_motion] Error: {error:.4f}m")
        print(f"[pentagon_motion] ========================================")
        
        return True
        """
        Stack currently held block on TWO support blocks at specific position.
        
        This is the key primitive for the top pentagon - each block rests on
        two adjacent base blocks due to the 36° rotation.
        
        Args:
            support_block1_id: First support block ID
            support_block2_id: Second support block ID
            target_x: Target X coordinate for block center
            target_y: Target Y coordinate for block center
            target_z: Target Z coordinate for block center
            
        Returns:
            True if successful
        """
        if not self.gripper_holding:
            print("[pentagon_motion] Not holding any block!")
            return False
        
        print(f"\n[pentagon_motion] ========================================")
        print(f"[pentagon_motion] STACK ON TWO BLOCKS")
        print(f"[pentagon_motion] Support: [{support_block1_id}, {support_block2_id}]")
        print(f"[pentagon_motion] Target: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})")
        print(f"[pentagon_motion] ========================================")
        
        # Find held block
        hand = self.robot.get_link("hand")
        hand_pos = np.array(hand.get_pos())
        held_block = None
        
        for key, block in self.blocks_state.items():
            block_pos = np.array(block.get_pos())
            dist = np.linalg.norm(block_pos - hand_pos)
            if dist < 0.25:
                held_block = block
                break
        
        if held_block is None:
            print("[pentagon_motion] Could not identify held block!")
            return False
        
        # Verify support blocks exist and get their positions
        try:
            support1_key = self._resolve_block_key(support_block1_id)
            support2_key = self._resolve_block_key(support_block2_id)
        except KeyError as e:
            print(f"[pentagon_motion] Support block not found: {e}")
            return False
        
        support1_pos = self._block_center(support1_key)
        support2_pos = self._block_center(support2_key)
        
        print(f"[pentagon_motion] Support1 ({support1_key}): ({support1_pos[0]:.4f}, {support1_pos[1]:.4f}, {support1_pos[2]:.4f})")
        print(f"[pentagon_motion] Support2 ({support2_key}): ({support2_pos[0]:.4f}, {support2_pos[1]:.4f}, {support2_pos[2]:.4f})")
        
        # Calculate placement gripper position
        place_gripper = np.array([target_x, target_y, target_z + self.config.grasp_offset])
        
        # High approach - well above target
        high_pos = place_gripper.copy()
        high_pos[2] += 0.20
        
        # Low approach - just above placement
        low_pos = place_gripper.copy()
        low_pos[2] += 0.05
        
        # Move to high approach
        print(f"[pentagon_motion] Moving to high approach...")
        q_high = self._ik_for_pose(high_pos, self.grasp_quat)
        if q_high is None:
            print("[pentagon_motion] Failed IK for high approach")
            return False
        
        if not self._plan_and_execute(q_high, attached_object=held_block, description="High approach"):
            return False
        
        # Descend to low approach
        print(f"[pentagon_motion] Descending to low approach...")
        q_low = self._ik_for_pose(low_pos, self.grasp_quat)
        if q_low is None:
            print("[pentagon_motion] Failed IK for low approach")
            return False
        
        current_q = self.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        # Smooth descent to low approach
        for i in range(60):
            alpha = (i + 1) / 60.0
            q = (1 - alpha) * start_q + alpha * q_low
            q[-2:] = self.config.gripper_closed_width
            self.robot.control_dofs_position(q)
            self.scene.step()
        
        # Final placement - very slow and controlled
        print(f"[pentagon_motion] Final placement...")
        q_place = self._ik_for_pose(place_gripper, self.grasp_quat)
        if q_place is None:
            print("[pentagon_motion] Failed IK for placement")
            return False
        
        current_q = self.robot.get_qpos()
        if hasattr(current_q, "cpu"):
            start_q = current_q.cpu().numpy().copy()
        else:
            start_q = np.array(current_q, dtype=float, copy=True)
        
        # Very slow final descent - 80 steps
        for i in range(80):
            alpha = (i + 1) / 80.0
            q = (1 - alpha) * start_q + alpha * q_place
            q[-2:] = self.config.gripper_closed_width
            self.robot.control_dofs_position(q)
            self.scene.step()
        
        # Extended hold for stability on two-point contact
        print("[pentagon_motion] Holding position for stability...")
        hold_q = self.robot.get_qpos()
        if hasattr(hold_q, "cpu"):
            hold_q = hold_q.cpu().numpy().copy()
        else:
            hold_q = np.array(hold_q, dtype=float, copy=True)
        
        # Hold longer for two-point contact (150 steps ~1.5 seconds)
        for _ in range(200):
            self.robot.control_dofs_position(hold_q)
            self.scene.step()
        
        # Release very slowly
        print("[pentagon_motion] Releasing...")
        self.open_gripper()
        
        # Additional settling time after release
        for _ in range(80):
            self.scene.step()
        
        # Careful lift
        current_pos = np.array(hand.get_pos())
        up_pos = current_pos.copy()
        up_pos[2] += 0.12
        
        q_up = self._ik_for_pose(up_pos, self.grasp_quat)
        if q_up is not None:
            current_q = self.robot.get_qpos()
            if hasattr(current_q, "cpu"):
                start_q = current_q.cpu().numpy().copy()
            else:
                start_q = np.array(current_q, dtype=float, copy=True)
            
            for i in range(50):
                alpha = (i + 1) / 50.0
                q = (1 - alpha) * start_q + alpha * q_up
                self.robot.control_dofs_position(q)
                self.scene.step()
        
        # Final verification
        final_pos = np.array(held_block.get_pos())
        error = np.linalg.norm(final_pos - np.array([target_x, target_y, target_z]))
        
        print(f"[pentagon_motion] ✓ STACKED ON TWO")
        print(f"[pentagon_motion] Target: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})")
        print(f"[pentagon_motion] Actual: ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
        print(f"[pentagon_motion] Error: {error:.4f}m")
        print(f"[pentagon_motion] ========================================")
        
        return True