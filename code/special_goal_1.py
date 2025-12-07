"""
pentagon_demo_two_phase.py - Pentagon stacking with TWO-PHASE planning

Much smarter approach:
- Phase 1: Plan & execute base pentagon
- Phase 2: Re-plan & execute top pentagon

Benefits:
- Simpler planning problems
- Better ordering decisions  
- More efficient
"""

import genesis as gs

# Try both import patterns
try:
    from scene_factory import create_scene_10blocks
except ImportError:
    from scenes import create_scene_10blocks

from pentagon_motion_primitives import PentagonMotionExecutor
from pentagon_task_planner import PentagonTaskPlanner
from pentagon_two_phase import execute_two_phase_plan


def main():
    """Main execution function with two-phase planning."""
    
    print("=" * 70)
    print(" PENTAGON STACKING TAMP DEMO - TWO-PHASE PLANNING")
    print("=" * 70)
    print()
    print("Objective:")
    print("  - Build base pentagon (5 blocks) on table")
    print("  - Build top pentagon (5 blocks) rotated 36° on top")
    print("  - Each top block rests on TWO base blocks")
    print()
    print("Planning Strategy:")
    print("  - Phase 1: Plan and execute base pentagon")
    print("  - Phase 2: Re-plan and execute top pentagon")
    print()
    print("Pentagon specifications:")
    print("  - Radius: 0.06m (6cm)")
    print("  - Block size: 0.04m (4cm cubes)")
    print("  - Rotation: 36° (one vertex offset)")
    print("=" * 70)
    
    # Step 1: Initialize Genesis
    print("\n[1/5] Initializing Genesis...")
    gs.init(backend=gs.cpu)
    print("[1/5] ✓ Genesis initialized")
    
    # Step 2: Create scene
    print("\n[2/5] Creating simulation scene...")
    scene, robot, blocks_state = create_scene_10blocks()
    
    # Define block IDs: first 5 are base, last 5 are top pentagon
    block_ids = ['r', 'g', 'b', 'y', 'o', 'r2', 'g2', 'b2', 'y2', 'o2']
    
    print(f"[2/5] ✓ Scene created with {len(blocks_state)} blocks")
    print(f"       Base blocks: {block_ids[:5]}")
    print(f"       Top blocks:  {block_ids[5:]}")
    
    # Step 3: Initialize motion primitives
    print("\n[3/5] Initializing motion primitives...")
    motion_executor = PentagonMotionExecutor(scene, robot, blocks_state)
    print("[3/5] ✓ Motion executor ready")
    
    # Step 4: Initialize task planner
    print("\n[4/5] Initializing task planner...")
    
    # Use relative path for domain file - try fixed version first
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    domain_file = os.path.join(current_dir, "pentagon_domain_simple.pddl")
    if not os.path.exists(domain_file):
        domain_file = os.path.join(current_dir, "pentagon_domain.pddl")
    
    print(f"       Using domain: {os.path.basename(domain_file)}")
    
    task_planner = PentagonTaskPlanner(
        blocks_state=blocks_state,
        block_ids=block_ids,
        domain_file=domain_file
    )
    print("[4/5] ✓ Task planner ready")
    
    # Step 5: Execute two-phase plan
    print("\n[5/5] Executing two-phase planning...")
    
    try:
        success = execute_two_phase_plan(task_planner, motion_executor)
        
        if success:
            print("\n" + "=" * 70)
            print(" ✅ PENTAGON STACKING SUCCEEDED!")
            print("=" * 70)
            print("\nFinal structure:")
            print("  - Base pentagon: 5 blocks on table")
            print("  - Top pentagon: 5 blocks rotated 36°")
            print("  - Each top block resting on TWO base blocks")
            print()
        else:
            print("\n" + "=" * 70)
            print(" ❌ PENTAGON STACKING FAILED")
            print("=" * 70)
            
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    except Exception as e:
        print(f"\n\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n💤 Exiting Genesis...")
    

if __name__ == "__main__":
    main()