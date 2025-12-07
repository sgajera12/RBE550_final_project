"""
two_phase_planner.py - Two-phase planning for pentagon stacking

Much smarter approach:
1. Plan & execute base pentagon (5 blocks)
2. Re-plan & execute top pentagon (5 blocks on completed base)

This is more efficient and leads to better plans!
"""

from typing import List, Tuple
from pentagon_task_planner import PentagonTaskPlanner


def execute_two_phase_plan(task_planner: PentagonTaskPlanner, motion_executor) -> bool:
    """
    Execute pentagon stacking using two-phase planning.
    
    Phase 1: Plan and execute base pentagon
    Phase 2: Re-plan and execute top pentagon on completed base
    
    Args:
        task_planner: PentagonTaskPlanner instance
        motion_executor: Motion executor instance
        
    Returns:
        True if both phases succeeded
    """
    print("\n" + "=" * 70)
    print(" TWO-PHASE PLANNING APPROACH")
    print("=" * 70)
    print("  Phase 1: Build base pentagon (5 blocks)")
    print("  Phase 2: Stack top pentagon on completed base (5 blocks)")
    print("  ")
    print("  Benefits:")
    print("  - Simpler planning problems (fewer states)")
    print("  - Better ordering decisions")
    print("  - Can adapt to actual base configuration")
    print("=" * 70)
    
    # ========================================================================
    # PHASE 1: BASE PENTAGON
    # ========================================================================
    print("\n" + "🔷" * 35)
    print("PHASE 1: BASE PENTAGON")
    print("🔷" * 35)
    
    print("\n[Phase 1] Planning base pentagon...")
    phase1_plan = task_planner.plan_phase(1)
    
    if not phase1_plan:
        print("❌ Phase 1 planning failed!")
        return False
    
    print(f"\n[Phase 1] ✓ Plan generated: {len(phase1_plan)} actions")
    print(f"[Phase 1] Executing base pentagon...")
    
    success = task_planner.execute_plan(phase1_plan, motion_executor)
    
    if not success:
        print("\n❌ Phase 1 execution failed!")
        return False
    
    print("\n" + "✅" * 35)
    print("PHASE 1 COMPLETE - Base pentagon built!")
    print("✅" * 35)
    
    # ========================================================================
    # PHASE 2: TOP PENTAGON
    # ========================================================================
    print("\n" + "🔶" * 35)
    print("PHASE 2: TOP PENTAGON")
    print("🔶" * 35)
    
    print("\n[Phase 2] Re-planning with completed base...")
    phase2_plan = task_planner.plan_phase(2)
    
    if not phase2_plan:
        print("❌ Phase 2 planning failed!")
        return False
    
    print(f"\n[Phase 2] ✓ Plan generated: {len(phase2_plan)} actions")
    print(f"[Phase 2] Executing top pentagon...")
    
    success = task_planner.execute_plan(phase2_plan, motion_executor)
    
    if not success:
        print("\n❌ Phase 2 execution failed!")
        return False
    
    print("\n" + "✅" * 35)
    print("PHASE 2 COMPLETE - Top pentagon stacked!")
    print("✅" * 35)
    
    # ========================================================================
    # SUCCESS
    # ========================================================================
    print("\n" + "🎉" * 35)
    print("TWO-PHASE EXECUTION COMPLETE!")
    print("🎉" * 35)
    print("\n✓ Base pentagon: 5 blocks placed")
    print("✓ Top pentagon: 5 blocks stacked")
    print("✓ Total: 10 blocks in double pentagon structure")
    print()
    
    return True