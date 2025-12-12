"""
Task Planning Module - Interfaces with Pyperplan
"""

import os
import subprocess
import tempfile
from typing import Set, List, Tuple

#Generate PDDL problem file from predicates
def generate_pddl_problem(predicates: Set[str],goal_predicates: Set[str],blocks: List[str],problem_name: str = "blocks_problem") -> str:
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
    
    problem = f"""(define (problem {problem_name}) (:domain blocksworld) (:objects {' '.join(blocks)}) (:init {init_preds}) (:goal (and {goal_preds})))"""
    return problem


def call_pyperplan(domain_file: str, problem_string: str) -> List[Tuple[str, List[str]]]:
    """
    Calling Pyperplan to solve planning problem
    Returns:
        List of (action_name, [args])
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as f:
        problem_file = f.name
        f.write(problem_string)
    
    try:
        print(f"Domain: {domain_file}")
        print(f"Problem: {problem_file}")
        
        result = subprocess.run(['pyperplan', domain_file, problem_file],capture_output=True,text=True,timeout=30)
        
        print("\nPyperplan Output")
        print(result.stdout)
        print("End Output\n")
        
        if result.returncode != 0:
            print("Pyperplan failed!")
            print("STDERR:", result.stderr)
            return []
        
        # Check if no solution exists
        if 'no solution' in result.stdout.lower() or 'unsolvable' in result.stdout.lower():
            print("Pyperplan says problem is unsolvable!")
            return []
        
        # Pyperplan writes solution to problem_file.soln
        solution_file = problem_file + '.soln'
        
        if not os.path.exists(solution_file):
            print(f"Solution file not found: {solution_file}")
            return []
        
        # Read plan from solution file
        plan = []
        with open(solution_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('(') and line.endswith(')'):
                    # Parse "(pick-up r)" or "(stack r g)"
                    action_str = line[1:-1]  # Remove ( )
                    parts = action_str.split()
                    
                    if parts:
                        action_name = parts[0]
                        args = parts[1:]
                        plan.append((action_name, args))
        
        # Clean up solution file
        if os.path.exists(solution_file):
            os.remove(solution_file)
        
        if not plan:
            print("No plan found in solution file!")
        else:
            print(f"Found plan with {len(plan)} actions")
        
        return plan
    
    finally:
        if os.path.exists(problem_file):
            os.remove(problem_file)

# Convert plan to readable string
def plan_to_string(plan: List[Tuple[str, List[str]]]) -> str:
    if not plan:
        return "No plan"
    
    result = []
    for i, (action, args) in enumerate(plan, 1):
        args_str = ', '.join(args)
        result.append(f"{i}. {action.upper()}({args_str})")
    
    return '\n'.join(result)