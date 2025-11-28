"""
pentagon_task_planner.py

Task planning for pentagon structure
Based on task_planner.py but adapted for pentagon predicates
"""

import os
import subprocess
import tempfile
from typing import Set, List, Tuple


def generate_pentagon_pddl_problem(
    predicates: Set[str],
    goal_predicates: Set[str],
    blocks: List[str],
    edges: List[str],
    layers: List[str],
    problem_name: str = "pentagon_problem"
) -> str:
    """
    Generate PDDL problem for pentagon structure
    
    Args:
        predicates: Current state predicates
        goal_predicates: Goal state predicates
        blocks: List of block IDs
        edges: List of edge IDs (edge1, edge2, etc.)
        layers: List of layer IDs (layer1, layer2)
        problem_name: Problem name
    
    Returns:
        PDDL problem string
    """
    
    def format_pred(p: str) -> str:
        """Convert predicate to PDDL format"""
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
    
    # Objects include blocks, edges, and layers
    all_objects = blocks + edges + layers
    
    problem = f"""(define (problem {problem_name})
  (:domain pentagon)
  
  (:objects {' '.join(all_objects)})
  
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


def call_pyperplan_pentagon(domain_file: str, problem_string: str) -> List[Tuple[str, List[str]]]:
    """
    Call Pyperplan for pentagon problem
    
    Args:
        domain_file: Path to pentagon_domain.pddl
        problem_string: PDDL problem as string
        
    Returns:
        List of (action_name, [args]) tuples
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as f:
        problem_file = f.name
        f.write(problem_string)
    
    try:
        print(f"  Domain: {domain_file}")
        print(f"  Problem: {problem_file}")
        
        result = subprocess.run(
            ['pyperplan', domain_file, problem_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print("\n--- Pyperplan Output ---")
        print(result.stdout)
        print("--- End Output ---\n")
        
        if result.returncode != 0:
            print("ERROR: Pyperplan failed!")
            print("STDERR:", result.stderr)
            return []
        
        if 'no solution' in result.stdout.lower() or 'unsolvable' in result.stdout.lower():
            print("ERROR: Pyperplan says problem is unsolvable!")
            return []
        
        # Read solution file
        solution_file = problem_file + '.soln'
        
        if not os.path.exists(solution_file):
            print(f"WARNING: Solution file not found: {solution_file}")
            return []
        
        # Parse plan
        plan = []
        with open(solution_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('(') and line.endswith(')'):
                    # Parse action
                    action_str = line[1:-1]
                    parts = action_str.split()
                    
                    if parts:
                        action_name = parts[0]
                        args = parts[1:]
                        plan.append((action_name, args))
        
        # Clean up
        if os.path.exists(solution_file):
            os.remove(solution_file)
        
        if not plan:
            print("WARNING: No plan found in solution file!")
        else:
            print(f"SUCCESS: Found plan with {len(plan)} actions")
        
        return plan
    
    except subprocess.TimeoutExpired:
        print("ERROR: Pyperplan timed out after 30 seconds!")
        return []
    
    finally:
        if os.path.exists(problem_file):
            os.remove(problem_file)


def plan_to_string(plan: List[Tuple[str, List[str]]]) -> str:
    """Convert plan to readable string"""
    if not plan:
        return "No plan"
    
    result = []
    for i, (action, args) in enumerate(plan, 1):
        args_str = ', '.join(args)
        result.append(f"{i}. {action.upper()}({args_str})")
    
    return '\n'.join(result)
