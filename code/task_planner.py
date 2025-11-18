"""
task_planner.py

High–level symbolic task planner wrapper for the project.

This module does **not** know anything about Genesis or the robot.  Its job is:

1. Take a set of symbolic predicates that describe the current scene
   (Blocks World style: ON, ONTABLE, CLEAR, HOLDING, HANDEMPTY).
2. Take a set of goal predicates.
3. Write a PDDL *problem* file for a standard Blocks World domain.
4. Call the `pyperplan` planner to obtain a sequence of high–level actions.
5. Return that plan in a simple, easy-to-use Python format.

The actual motion planning / execution code will translate each returned
symbolic action into a motion primitive (pick, stack, …).

The planner expects the classic Blocks World STRIPS domain with the actions
`pick-up`, `put-down`, `stack`, and `unstack`.  A matching domain description
is embedded below and will be written to `blocks_domain.pddl` next to this file
(on first use).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Any
from pyperplan import planner as pyperplanner  # type: ignore
from pyperplan.search import breadth_first_search  # type: ignore

import os
import re
import subprocess

# -----------------------------------------------------------------------------
# Domain description (classic Blocks World)
# -----------------------------------------------------------------------------

BLOCKS_DOMAIN_PDDL = """(define (domain blocks)
 (:requirements :strips :typing)
 (:types block)
 (:predicates
   (on ?x - block ?y - block)
   (ontable ?x - block)
   (clear ?x - block)
   (handempty)
   (holding ?x - block)
 )
 (:action pick-up
  :parameters (?x - block)
  :precondition (and (clear ?x) (ontable ?x) (handempty))
  :effect (and (holding ?x)
               (not (ontable ?x))
               (not (clear ?x))
               (not (handempty)))
 )
 (:action put-down
  :parameters (?x - block)
  :precondition (holding ?x)
  :effect (and (ontable ?x)
               (clear ?x)
               (handempty)
               (not (holding ?x)))
 )
 (:action stack
  :parameters (?x - block ?y - block)
  :precondition (and (holding ?x) (clear ?y))
  :effect (and (on ?x ?y)
               (clear ?x)
               (handempty)
               (not (holding ?x))
               (not (clear ?y)))
 )
 (:action unstack
  :parameters (?x - block ?y - block)
  :precondition (and (on ?x ?y) (clear ?x) (handempty))
  :effect (and (holding ?x)
               (clear ?y)
               (not (on ?x ?y))
               (not (clear ?x))
               (not (handempty)))
 )
)"""


# Directory where this file lives – we store the domain / problem files here.
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DOMAIN_PATH = MODULE_DIR / "blocks_domain.pddl"


@dataclass
class SymbolicAction:
    """Simple container for a high–level action returned by the planner.

    Example instances:
        SymbolicAction(name="pick-up", args=("red",))
        SymbolicAction(name="stack", args=("red", "green"))
    """

    name: str
    args: Tuple[str, ...]


# -----------------------------------------------------------------------------
# Helpers to write domain / problem files
# -----------------------------------------------------------------------------


def ensure_domain_file(path: Path = DEFAULT_DOMAIN_PATH) -> Path:
    """Ensure the Blocks World domain PDDL file exists on disk.

    If the file does not exist it will be created with the embedded domain
    description.  The existing file is left untouched so you can experiment
    with your own domain variants if desired.
    """
    if not path.exists():
        path.write_text(BLOCKS_DOMAIN_PDDL)
    return path


def _normalise_object_name(name: Any) -> str:
    """Convert Python object identifiers to PDDL object names.

    For now we simply:
      * cast to string
      * strip whitespace
      * lower-case
    so "R", "r", and "Red" would become "r", "r", and "red".
    """
    return str(name).strip().lower()


def _pddl_literal_from_predicate(pred: Sequence[Any]) -> Optional[str]:
    """Convert an internal predicate tuple into a PDDL literal string.

    The expected internal representation is a tuple like:
        ("ON", "R", "G")
        ("ONTABLE", "R")
        ("CLEAR", "R")
        ("HOLDING", "R")
        ("HANDEMPTY",)

    Names are case-insensitive; they are always written lower-case in PDDL.
    Unknown or empty predicates are ignored by returning None.
    """
    if not pred:
        return None

    name = str(pred[0]).strip()
    if not name:
        return None

    # Normalise to lower-case for PDDL.
    pddl_name = name.lower()
    args = [_normalise_object_name(a) for a in pred[1:]]

    # Only support the classic Blocks World predicates here.
    allowed = {"on", "ontable", "clear", "holding", "handempty"}
    if pddl_name not in allowed:
        # Silently ignore extra predicates for now – they might be useful
        # for internal reasoning but are not part of the STRIPS domain.
        return None

    if args:
        return f"({pddl_name} {' '.join(args)})"
    else:
        return f"({pddl_name})"


def build_problem_pddl(
    initial_predicates: Iterable[Sequence[Any]],
    goal_predicates: Iterable[Sequence[Any]],
    objects: Iterable[Any],
    problem_name: str = "blocks-task",
    domain_name: str = "blocks",
) -> str:
    """Construct a PDDL problem string from predicates and object names.

    Args
    ----
    initial_predicates:
        Iterable of predicate tuples describing the current world.
    goal_predicates:
        Iterable of predicate tuples describing the desired goal.
    objects:
        Iterable of block identifiers (e.g., "R", "G", ...).  These will be
        normalised with `_normalise_object_name`.
    problem_name:
        Symbolic name for the problem (only used inside the PDDL header).
    domain_name:
        Must match the name in `BLOCKS_DOMAIN_PDDL` ("blocks" by default).
    """
    # Collect objects
    obj_names = sorted({_normalise_object_name(o) for o in objects})

    lines: List[str] = []
    lines.append(f"(define (problem {problem_name})")
    lines.append(f"  (:domain {domain_name})")
    if obj_names:
        lines.append(f"  (:objects {' '.join(obj_names)} - block)")

    # Initial state
    lines.append("  (:init")
    for pred in initial_predicates:
        lit = _pddl_literal_from_predicate(pred)
        if lit is not None:
            lines.append(f"    {lit}")
    lines.append("  )")

    # Goal
    lines.append("  (:goal")
    lines.append("    (and")
    for pred in goal_predicates:
        lit = _pddl_literal_from_predicate(pred)
        if lit is not None:
            lines.append(f"      {lit}")
    lines.append("    )")
    lines.append("  )")
    lines.append(")")

    return "\n".join(lines)


def _extract_objects_from_predicates(
    initial_predicates: Iterable[Sequence[Any]],
    goal_predicates: Iterable[Sequence[Any]],
) -> List[str]:
    """Derive the set of block names appearing in predicates."""
    names = set()
    for pred in list(initial_predicates) + list(goal_predicates):
        for arg in pred[1:]:
            if isinstance(arg, str) and arg.strip():
                names.add(_normalise_object_name(arg))
    return sorted(names)


# -----------------------------------------------------------------------------
# Pyperplan integration
# -----------------------------------------------------------------------------


def _parse_action_string(s: str) -> SymbolicAction:
    """Parse a single textual action like "(pick-up r)" into SymbolicAction.

    This is intentionally liberal and tries to handle several formats that
    pyperplan may produce, for example:

        (pick-up a)
        step 0: (stack a b)
        pick-up(a)
        UNSTACK(A,B)
        stack a b
    """
    s = s.strip()
    if not s:
        raise ValueError("empty action line")

    # Strip comments (everything after ';')
    if ";" in s:
        s = s.split(";", 1)[0].strip()

    # Remove leading labels like "0: ..." or "step 0: ..."
    if ":" in s and "(" in s and s.index(":") < s.index("("):
        s = s.split(":", 1)[1].strip()

    # Trim outer parentheses if present
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    # Form "name(arg1,arg2)" or "name(arg1)"
    m = re.match(r"^([a-zA-Z0-9_\-]+)\((.*)\)$", s)
    if m:
        name = m.group(1)
        arg_text = m.group(2).strip()
        if arg_text:
            args = tuple(
                a.strip().lower() for a in re.split(r"[,\s]+", arg_text) if a.strip()
            )
        else:
            args = ()
        return SymbolicAction(name=name.lower(), args=args)

    # Fallback: treat as space / comma separated tokens
    cleaned = re.sub(r"[(),]", " ", s)
    parts = [p for p in cleaned.split() if p]
    if not parts:
        raise ValueError("could not parse action line: %r" % s)
    name = parts[0].lower()
    args = tuple(p.lower() for p in parts[1:])
    return SymbolicAction(name=name, args=args)


def _run_pyperplan(domain_file: str, problem_file: str) -> List[SymbolicAction]:
    """Call pyperplan and return a list of SymbolicAction objects.

    We first try to use the Python API (``planner.search_plan``).  If that
    fails because the Python package is not installed, we fall back to calling
    the ``pyperplan`` command line tool and parsing the resulting ``.soln``
    file that pyperplan writes next to the problem file.
    """
    actions: List[SymbolicAction] = []

    try:
        # Preferred path: use pyperplan as a library.

        try:
            from pyperplan import planner as pyperplanner   # type: ignore
            from pyperplan.search import breadth_first_search  # type: ignore

            # Get function signature to decide how to call
            import inspect
            sig = inspect.signature(pyperplanner.search_plan)
            params = list(sig.parameters.keys())

            # Case 1: New versions with keyword 'heuristic'
            if "heuristic" in params:
                plan = pyperplanner.search_plan(
                    domain_file, problem_file, breadth_first_search, heuristic=None
                )

            # Case 2: Old versions with required 'heuristic_class'
            elif "heuristic_class" in params:
                plan = pyperplanner.search_plan(
                    domain_file, problem_file, breadth_first_search, None
                )

            # Case 3: Very old versions (no heuristic parameter)
            else:
                plan = pyperplanner.search_plan(
                    domain_file, problem_file, breadth_first_search
                )

            if plan is None:
                return []

            # Convert steps into SymbolicAction objects
            for step in plan:
                actions.append(_parse_action_string(str(step)))

            return actions

        except ImportError:
            # Fall back to CLI version
            pass

    except ImportError:
        # No pyperplan Python package – fall back to CLI invocation.
        pass

    # CLI fallback: call the ``pyperplan`` executable.
    try:
        completed = subprocess.run(
            ["pyperplan", domain_file, problem_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # `completed.stdout` / `stderr` may be useful for debugging but we do
        # not parse them – pyperplan always writes the actual plan into a .soln
        # file next to the problem.
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not find the 'pyperplan' executable and the pyperplan Python "
            "package is not installed. Please install pyperplan with "
            "'pip install pyperplan'."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "pyperplan returned a non-zero exit status (%d).\nstdout:\n%s\n\nstderr:\n%s"
            % (exc.returncode, exc.stdout, exc.stderr)
        ) from exc

    soln_path = problem_file + ".soln"
    if not os.path.exists(soln_path):
        # No solution file – treat as no plan found.
        return []

    with open(soln_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "(" not in line:
                continue
            try:
                actions.append(_parse_action_string(line))
            except ValueError:
                # Be robust against other diagnostic lines in the file.
                continue

    return actions


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def plan_blocks_world(
    initial_predicates: Iterable[Sequence[Any]],
    goal_predicates: Iterable[Sequence[Any]],
    objects: Optional[Iterable[Any]] = None,
    problem_name: str = "blocks-task",
) -> List[SymbolicAction]:
    """Plan a sequence of Blocks World actions using pyperplan.

    Args
    ----
    initial_predicates:
        Iterable of predicate tuples describing the initial state.
    goal_predicates:
        Iterable of predicate tuples describing the goal.
    objects:
        Optional iterable of block identifiers.  If omitted, we infer the set of
        blocks from the arguments that appear in ``initial_predicates`` and
        ``goal_predicates``.
    problem_name:
        Name used inside the PDDL problem header and as the problem filename
        ("<problem_name>.pddl").

    Returns
    -------
    List[SymbolicAction]
        High–level plan as a list of actions, or an empty list if no plan was
        found.
    """
    # Materialise predicates so we can iterate over them multiple times.
    init_list = [tuple(p) for p in initial_predicates]
    goal_list = [tuple(p) for p in goal_predicates]

    if objects is None:
        obj_list = _extract_objects_from_predicates(init_list, goal_list)
    else:
        obj_list = [_normalise_object_name(o) for o in objects]

    # Ensure domain file exists and write a fresh problem file.
    domain_path = ensure_domain_file(DEFAULT_DOMAIN_PATH)
    problem_path = MODULE_DIR / f"{problem_name}.pddl"
    problem_pddl = build_problem_pddl(
        init_list, goal_list, obj_list, problem_name=problem_name
    )
    problem_path.write_text(problem_pddl)

    # Delegate to pyperplan and parse its result.
    return _run_pyperplan(str(domain_path), str(problem_path))


# -----------------------------------------------------------------------------
# Small self-test when run directly
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Tiny three-block example that should yield a non-empty plan once
    # pyperplan is installed.  This is only for quick manual testing and
    # is not used by the main project code.
    #
    # Initial: A, B, C are on the table and clear; hand is empty.
    init = [
        ("ONTABLE", "A"),
        ("ONTABLE", "B"),
        ("ONTABLE", "C"),
        ("CLEAR", "A"),
        ("CLEAR", "B"),
        ("CLEAR", "C"),
        ("HANDEMPTY",),
    ]

    # Goal: build a simple tower A on B on C.
    goal = [
        ("ON", "A", "B"),
        ("ON", "B", "C"),
        ("CLEAR", "A"),
        ("ONTABLE", "C"),
    ]

    print("Writing domain/problem and calling pyperplan ...")
    try:
        plan = plan_blocks_world(init, goal, objects=["A", "B", "C"], problem_name="demo3")
    except Exception as exc:  # debug helper only
        print("Planning failed:", exc)
    else:
        if not plan:
            print("No plan found.")
        else:
            print("Plan:")
            for step in plan:
                print(" ", step)
