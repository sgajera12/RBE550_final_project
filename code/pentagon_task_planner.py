"""
pentagon_task_planner.py - TAMP task planner for pentagon stacking

Generates PDDL problem from current state, calls planner, and executes plan.
"""

from typing import List, Tuple, Set, Dict, Any
import numpy as np
from task_planner import call_pyperplan, plan_to_string
from pentagon_geometry import PentagonGeometry


def generate_pddl_problem(
    predicates: Set[str],
    goal_predicates: Set[str],
    blocks: List[str],
    problem_name: str = "pentagon_problem"
) -> str:
    """
    Generate PDDL problem file from predicates.
    
    CRITICAL: Domain name must match the domain file (pentagon-world)!
    """

    def format_pred(p: str) -> str:
        """Convert 'ON(r,g)' to '(on r g)'"""
        p = p.lower()
        if '(' not in p:
            return p

        pred_name = p.split('(')[0]
        args = p.split('(')[1].rstrip(')').split(',')

        if args[0]:
            return f"({pred_name} {' '.join(args)})"
        else:
            return f"({pred_name})"

    init_preds = '\n    '.join([format_pred(p) for p in predicates])
    goal_preds = '\n      '.join([format_pred(p) for p in goal_predicates])

    # CRITICAL: Domain name must match domain file!
    problem = f"""(define (problem {problem_name})
  (:domain pentagon-world)

  (:objects {' '.join(blocks)} - block)

  (:init
    {init_preds}
  )

  (:goal
    (and
      {goal_preds}
    )
  )
)
"""
    return problem


def print_plan_detailed(plan: List[Tuple[str, List[str]]], block_mapping: Dict[str, Dict]) -> None:
    """
    Print detailed plan with positions and orientations.
    
    Args:
        plan: List of (action_name, [args])
        block_mapping: Block mapping with positions and orientations
    """
    print("\n" + "=" * 80)
    print(" GENERATED PLAN - DETAILED VIEW")
    print("=" * 80)
    
    for i, (action, args) in enumerate(plan, 1):
        print(f"\n[Step {i:2d}] {action.upper()}({', '.join(args)})")
        
        # Add details based on action type
        if action == "pick-up":
            block_id = args[0]
            if block_id in block_mapping:
                pos = block_mapping[block_id]['position']
                print(f"         Pick from: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
        elif action == "place-in-base":
            block_id = args[0]
            if block_id in block_mapping:
                info = block_mapping[block_id]
                pos = info['position']
                orient = info['orientation']
                print(f"         Place at: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                print(f"         Orientation: {orient:.0f}°")
                print(f"         Pentagon position: {info['index']} (base)")
        
        elif action == "stack-two":
            block_id = args[0]
            support1 = args[1]
            support2 = args[2]
            if block_id in block_mapping:
                info = block_mapping[block_id]
                pos = info['position']
                orient = info['orientation']
                print(f"         Stack at: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                print(f"         Orientation: {orient:.0f}°")
                print(f"         Rests on: [{support1}, {support2}]")
                print(f"         Pentagon position: {info['index']} (top)")
        
        elif action == "complete-base":
            print(f"         → Base pentagon complete (5 blocks placed)")
        
        elif action == "complete-top":
            print(f"         → Top pentagon complete (5 blocks stacked)")
    
    print("\n" + "=" * 80)
    print(f" TOTAL: {len(plan)} actions")
    print("=" * 80)


class PentagonTaskPlanner:
    """Task planner for pentagon stacking using PDDL."""
    
    def __init__(
        self,
        blocks_state: Dict[str, Any],
        block_ids: List[str],
        domain_file: str = "/home/claude/pentagon_domain.pddl"
    ):
        """
        Initialize pentagon task planner.
        
        Args:
            blocks_state: Dictionary mapping block IDs to block objects
            block_ids: Ordered list of block IDs [base5, top5]
            domain_file: Path to PDDL domain file
        """
        self.blocks_state = blocks_state
        self.block_ids = block_ids
        self.domain_file = domain_file
        
        # Initialize geometry calculator
        self.geometry = PentagonGeometry()
        self.block_mapping = self.geometry.create_block_mapping(block_ids)
        
        print(self.geometry.visualize_plan(self.block_mapping))
        
    def detect_predicates(self) -> Set[str]:
        """
        Detect current state predicates from simulation.
        
        Returns:
            Set of predicates like: {'ONTABLE(r)', 'CLEAR(r)', 'HANDEMPTY()', ...}
        """
        predicates = set()
        
        # Always start with empty hand
        predicates.add("HANDEMPTY()")
        
        # Mark which blocks are base vs top (this is part of the problem definition)
        for block_id, info in self.block_mapping.items():
            if info['role'] == 'base':
                predicates.add(f"BASE-BLOCK({block_id})")
            else:
                predicates.add(f"TOP-BLOCK({block_id})")
        
        # NOTE: We don't add supports predicates because the planner
        # can choose ANY two base blocks, and we'll use the correct
        # geometry-based supports during execution!
        
        BLOCK_SIZE = 0.04
        TABLE_HEIGHT = 0.02
        ON_THRESHOLD = 0.025  # 2.5cm tolerance for "on" relationship
        POSITION_TOLERANCE = 0.02  # 2cm tolerance for "at target" (was 0.01)
        
        # Check each block
        for block_id in self.block_ids:
            block = self.blocks_state[block_id]
            block_pos = np.array(block.get_pos())
            block_z = block_pos[2]
            
            # Check if on table (within tolerance of table height)
            if abs(block_z - TABLE_HEIGHT) < 0.01:
                predicates.add(f"ONTABLE({block_id})")
                
                # CRITICAL: Check if base block is at its TARGET position
                if block_id in self.block_mapping:
                    info = self.block_mapping[block_id]
                    target_pos = np.array(info['position'])
                    
                    # Check XY distance to target (ignore Z)
                    xy_distance = np.linalg.norm(block_pos[:2] - target_pos[:2])
                    
                    if info['role'] == 'base' and xy_distance < POSITION_TOLERANCE:
                        # Base block is at target position!
                        predicates.add(f"PLACED-IN-BASE({block_id})")
                        print(f"[detect] {block_id} is at target position (dist={xy_distance:.4f}m)")
            
            # Check if block is clear (nothing on top)
            is_clear = True
            for other_id in self.block_ids:
                if other_id == block_id:
                    continue
                other = self.blocks_state[other_id]
                other_pos = np.array(other.get_pos())
                
                # Check if other block is above this one
                xy_dist = np.linalg.norm(other_pos[:2] - block_pos[:2])
                z_diff = other_pos[2] - block_z
                
                # If other block is roughly one block-height above and aligned in XY
                if z_diff > (BLOCK_SIZE * 0.5) and xy_dist < ON_THRESHOLD:
                    is_clear = False
                    predicates.add(f"ON({other_id},{block_id})")
            
            if is_clear:
                predicates.add(f"CLEAR({block_id})")
        
        return predicates
    
    def generate_goal_predicates(self) -> Set[str]:
        """
        Generate goal predicates for completed double pentagon.
        
        Returns:
            Set of goal predicates
        """
        goal = set()
        
        # Goal: All base blocks are placed
        for block_id, info in self.block_mapping.items():
            if info['role'] == 'base':
                goal.add(f"PLACED-IN-BASE({block_id})")
        
        # Goal: All top blocks are stacked on two
        for block_id, info in self.block_mapping.items():
            if info['role'] == 'top':
                goal.add(f"STACKED-ON-TWO({block_id})")
        
        # Goal: Gripper is empty at the end
        goal.add("HANDEMPTY()")
        
        return goal
    
    def plan_phase(self, phase: int) -> List[Tuple[str, List[str]]]:
        """
        Generate PDDL plan for a specific phase.
        
        Phase 1: Base pentagon only (5 blocks)
        Phase 2: Top pentagon only (5 blocks) - COMPLETELY INDEPENDENT
        
        Args:
            phase: 1 for base pentagon, 2 for top pentagon
            
        Returns:
            List of (action_name, [args])
        """
        if phase == 1:
            print("\n" + "=" * 60)
            print("PHASE 1: PLANNING BASE PENTAGON")
            print("=" * 60)
            goal_predicates = self._generate_base_goal()
            # Phase 1: Include ALL blocks (base + top)
            blocks_for_planning = self.block_ids
        elif phase == 2:
            print("\n" + "=" * 60)
            print("PHASE 2: PLANNING TOP PENTAGON (INDEPENDENT)")
            print("=" * 60)
            goal_predicates = self._generate_top_goal()
            
            # Phase 2: ONLY top blocks - completely independent problem!
            # No references to base blocks whatsoever
            blocks_for_planning = [
                block_id for block_id, info in self.block_mapping.items()
                if info['role'] == 'top'
            ]
            print(f"\n[Phase 2] Independent planning - only top blocks: {blocks_for_planning}")
            print(f"[Phase 2] Base pentagon is invisible to planner (handled by execution)")
        else:
            raise ValueError(f"Invalid phase: {phase} (must be 1 or 2)")
        
        # Detect current state
        init_predicates = self.detect_predicates()
        
        # Phase 2: Keep ONLY predicates about top blocks
        if phase == 2:
            filtered_predicates = set()
            top_block_ids = [
                block_id for block_id, info in self.block_mapping.items()
                if info['role'] == 'top'
            ]
            
            for pred in init_predicates:
                # Keep HANDEMPTY
                if pred.startswith("HANDEMPTY"):
                    filtered_predicates.add(pred)
                    continue
                
                # Keep TOP-BLOCK predicates
                if pred.startswith("TOP-BLOCK"):
                    filtered_predicates.add(pred)
                    continue
                
                # Keep ONTABLE for top blocks (CRITICAL!)
                if pred.startswith("ONTABLE"):
                    for top_id in top_block_ids:
                        if f"ONTABLE({top_id})" == pred:
                            filtered_predicates.add(pred)
                            break
                    continue
                
                # Keep CLEAR for top blocks (CRITICAL!)
                if pred.startswith("CLEAR"):
                    for top_id in top_block_ids:
                        if f"CLEAR({top_id})" == pred:
                            filtered_predicates.add(pred)
                            break
                    continue
                
                # Keep any other predicates ONLY about top blocks
                for top_id in top_block_ids:
                    if f"({top_id})" in pred:
                        # Make sure it doesn't reference base blocks
                        base_block_ids = ['r', 'g', 'b', 'y', 'o']
                        has_base_ref = False
                        for base_id in base_block_ids:
                            if f"({base_id})" in pred or f"({base_id}," in pred or f",{base_id})" in pred:
                                has_base_ref = True
                                break
                        if not has_base_ref:
                            filtered_predicates.add(pred)
                        break
            
            init_predicates = filtered_predicates
            print(f"\n[Phase 2] Completely independent - NO base block references")
        
        print(f"\nCurrent State ({len(init_predicates)} predicates):")
        for p in sorted(init_predicates):
            print(f"  {p}")
        
        print(f"\nPhase {phase} Goal ({len(goal_predicates)} predicates):")
        for p in sorted(goal_predicates):
            print(f"  {p}")
        
        # Generate PDDL problem
        problem_string = generate_pddl_problem(
            predicates=init_predicates,
            goal_predicates=goal_predicates,
            blocks=blocks_for_planning,  # CRITICAL: Only top blocks in Phase 2!
            problem_name=f"pentagon_phase{phase}"
        )
        
        print("\n" + "=" * 60)
        print("CALLING PLANNER")
        print("=" * 60)
        
        # Call planner
        plan = call_pyperplan(self.domain_file, problem_string)
        
        if plan:
            print("\n" + "=" * 60)
            print(f"PHASE {phase} PLAN GENERATED")
            print("=" * 60)
            print(plan_to_string(plan))
            print()
            
            # Print detailed plan
            print_plan_detailed(plan, self.block_mapping)
        else:
            print(f"\n❌ PHASE {phase} PLANNING FAILED - No plan found!")
        
        return plan
    
    def _generate_base_goal(self) -> Set[str]:
        """Generate goal predicates for base pentagon only."""
        goal = set()
        
        # Goal: All base blocks are placed
        for block_id, info in self.block_mapping.items():
            if info['role'] == 'base':
                goal.add(f"PLACED-IN-BASE({block_id})")
        
        # Goal: Gripper is empty at the end
        goal.add("HANDEMPTY()")
        
        return goal
    
    def _generate_top_goal(self) -> Set[str]:
        """Generate goal predicates for top pentagon only."""
        goal = set()
        
        # DON'T include base blocks in goal - they're already placed!
        # Including them causes planner to try re-placing them
        
        # Goal: All top blocks are stacked
        for block_id, info in self.block_mapping.items():
            if info['role'] == 'top':
                goal.add(f"STACKED-ON-TWO({block_id})")
        
        # Goal: Gripper is empty at the end
        goal.add("HANDEMPTY()")
        
        return goal
    
    def plan(self) -> List[Tuple[str, List[str]]]:
        """
        Generate and solve PDDL planning problem (single-phase).
        
        NOTE: For better results, use two_phase_planner.execute_two_phase_plan()
        which plans base and top separately!
        
        Returns:
            List of (action_name, [args]) tuples
        """
        print("\n⚠️  Using single-phase planning (plans all 10 blocks at once)")
        print("    Consider using two-phase planning for better results!")
        print()
        
        print("\n" + "=" * 60)
        print("GENERATING PDDL PROBLEM")
        print("=" * 60)
        
        # Detect current state
        init_predicates = self.detect_predicates()
        print("\nInitial State:")
        for pred in sorted(init_predicates):
            print(f"  {pred}")
        
        # Generate goal
        goal_predicates = self.generate_goal_predicates()
        print("\nGoal State:")
        for pred in sorted(goal_predicates):
            print(f"  {pred}")
        
        # Generate PDDL problem
        problem_string = generate_pddl_problem(
            predicates=init_predicates,
            goal_predicates=goal_predicates,
            blocks=self.block_ids,
            problem_name="pentagon_stacking"
        )
        
        print("\n" + "=" * 60)
        print("CALLING PLANNER")
        print("=" * 60)
        
        # Call planner
        plan = call_pyperplan(self.domain_file, problem_string)
        
        if plan:
            print("\n" + "=" * 60)
            print("PLAN GENERATED")
            print("=" * 60)
            print(plan_to_string(plan))
            print()
            
            # Print detailed plan with positions and orientations
            print_plan_detailed(plan, self.block_mapping)
        else:
            print("\n❌ PLANNING FAILED - No plan found!")
        
        return plan
    
    def execute_plan(
        self, 
        plan: List[Tuple[str, List[str]]],
        motion_executor: Any
    ) -> bool:
        """
        Execute a PDDL plan using motion primitives.
        
        Args:
            plan: List of (action_name, [args]) from planner
            motion_executor: PentagonMotionExecutor instance
            
        Returns:
            True if all actions executed successfully
        """
        print("\n" + "=" * 60)
        print("EXECUTING PLAN")
        print("=" * 60)
        
        for i, (action, args) in enumerate(plan, 1):
            print(f"\n[{i}/{len(plan)}] {action.upper()}({', '.join(args)})")
            
            success = False
            
            if action == "pick-up":
                block_id = args[0]
                success = motion_executor.pick_up(block_id)
                
            elif action == "put-down":
                block_id = args[0]
                # Standard put-down on table
                success = motion_executor.put_down()
                
            elif action == "place-in-base":
                block_id = args[0]
                # Place at precise pentagon position WITH ORIENTATION
                target_pos = self.block_mapping[block_id]['position']
                orientation = self.block_mapping[block_id]['orientation']
                success = motion_executor.place_at_position(
                    target_pos[0], target_pos[1], target_pos[2],
                    orientation_degrees=orientation
                )
                
            elif action == "stack-two" or action == "stack-on-two-blocks":
                block_id = args[0]
                # NO args[1] or args[2] - action only has one parameter!
                # Use geometry to find correct support blocks
                
                if block_id not in self.block_mapping:
                    print(f"[planner] Unknown block: {block_id}")
                    success = False
                    continue
                
                info = self.block_mapping[block_id]
                if info['support_blocks'] is None:
                    print(f"[planner] Block {block_id} has no support blocks defined")
                    success = False
                    continue
                
                # Use geometry-calculated supports, NOT planner's choice!
                support1_id, support2_id = info['support_blocks']
                
                print(f"[planner] Stacking {block_id} on geometry-calculated supports: ({support1_id}, {support2_id})")
                
                # Get target position and orientation for this block
                target_pos = info['position']
                orientation = info['orientation']
                
                success = motion_executor.stack_on_two(
                    support1_id, support2_id,
                    target_pos[0], target_pos[1], target_pos[2],
                    orientation_degrees=orientation
                )
                
            else:
                print(f"[planner] ⚠️  Unknown action: {action}")
                success = True  # Don't fail on unknown actions
            
            if not success:
                print(f"[planner] ❌ Action failed: {action}({', '.join(args)})")
                return False
            
            print(f"[planner] ✓ Action completed")
        
        print("\n" + "=" * 60)
        print("✓ PLAN EXECUTION COMPLETE")
        print("=" * 60)
        return True


def create_manual_plan(block_ids: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Create an algorithmic plan for pentagon stacking (fallback if planner fails).
    
    IMPORTANT: This is NOT hardcoded! Positions and support relationships are
    calculated from geometric primitives using PentagonGeometry.
    
    This creates a simple sequential plan:
    1. Place all 5 base blocks in calculated pentagon positions
    2. Stack all 5 top blocks on their calculated support blocks
    
    Args:
        block_ids: List of 10 block IDs [base5, top5]
        
    Returns:
        List of (action_name, [args])
    """
    plan = []
    
    # Calculate support blocks for top pentagon - NOT HARDCODED!
    geo = PentagonGeometry()
    block_mapping = geo.create_block_mapping(block_ids)
    
    # Phase 1: Place base pentagon
    print("\nGenerating algorithmic plan...")
    print("Phase 1: Place base pentagon (positions calculated from geometry)")
    
    for block_id in block_ids[:5]:
        plan.append(("pick-up", [block_id]))
        plan.append(("place-in-base", [block_id]))
    
    # Phase 2: Stack top pentagon
    print("Phase 2: Stack top pentagon (support relationships calculated)")
    
    for block_id in block_ids[5:]:
        # Support blocks are CALCULATED from geometry, not hardcoded!
        support_blocks = block_mapping[block_id]['support_blocks']
        plan.append(("pick-up", [block_id]))
        plan.append(("stack-two", [block_id, support_blocks[0], support_blocks[1]]))
    
    print(f"\nAlgorithmic plan generated: {len(plan)} actions")
    print("(All positions and relationships calculated from PentagonGeometry)")
    return plan