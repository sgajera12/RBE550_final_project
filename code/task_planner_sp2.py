"""
task_planner_typed.py

Fixed task planner with support for typed PDDL domains.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from typing import Set, List, Tuple


def generate_pddl_problem(
    predicates: Set[str],
    goal_predicates: Set[str],
    blocks: List[str],
    problem_name: str = "blocks_problem",
    domain_name: str = "blocksworld-spatial"
) -> str:
    """Generate PDDL problem file from predicates with typed objects"""

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

    # ← FIXED: Declare objects with type
    typed_objects = ' '.join([f"{block} - block" for block in blocks])

    problem = f"""(define (problem {problem_name})
  (:domain {domain_name})

  (:objects {typed_objects})

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


def call_pyperplan(domain_file: str, problem_string: str) -> List[Tuple[str, List[str]]]:
    """
    Call Pyperplan to solve planning problem with A* and heuristic.

    Returns:
        List of (action_name, [args])
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as f:
        problem_file = f.name
        f.write(problem_string)

    try:
        print(f"  Domain: {domain_file}")
        print(f"  Problem: {problem_file}")

        cli_path = shutil.which("pyperplan")
        if cli_path is not None:
            # Use A* search with FF heuristic for smarter planning
            cmd = [cli_path, domain_file, problem_file, "-s", "astar", "-H", "hff"]
        else:
            # Fallback with smarter search
            cmd = [sys.executable, "-m", "pyperplan", domain_file, problem_file, "-s", "astar", "-H", "hff"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        print("\n--- Pyperplan STDOUT ---")
        print(result.stdout)
        print("--- End STDOUT ---\n")

        if result.stderr.strip():
            print("--- Pyperplan STDERR ---")
            print(result.stderr)
            print("--- End STDERR ---\n")

        if result.returncode != 0:
            print("Pyperplan failed!")
            return []

        # Try .soln file first
        solution_file = problem_file + '.soln'
        plan: List[Tuple[str, List[str]]] = []

        if os.path.exists(solution_file):
            with open(solution_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('(') and line.endswith(')'):
                        action_str = line[1:-1]
                        parts = action_str.split()
                        if parts:
                            action_name = parts[0]
                            args = parts[1:]
                            plan.append((action_name, args))
            os.remove(solution_file)

        # Fallback: parse stdout/stderr
        if not plan:
            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            for line in combined.splitlines():
                line = line.strip()
                if '(' in line and ')' in line:
                    start = line.find('(')
                    end = line.find(')', start)
                    if start != -1 and end != -1:
                        inner = line[start + 1:end]
                        parts = inner.split()
                        if parts:
                            action_name = parts[0]
                            args = parts[1:]
                            plan.append((action_name, args))

        if not plan:
            print("No plan found in solution file or output!")
        else:
            print(f"Found plan with {len(plan)} actions")

        return plan

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