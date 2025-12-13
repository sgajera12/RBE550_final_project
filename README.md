# RBE 550 Final Project - Task and Motion Planning

**Team 5**

**Members**: Sahil Gajera, Ashish Sukumar, Anirudh Nallawar, Rajdeep Banerjee

**Course**: RBE 550 - Motion Planning, Fall 2025

**Institution**: Worcester Polytechnic Institute

---

## Quick Setup Guide

### Prerequisites
- **Ubuntu 22.04/24.04** (recommended) or Linux with apt
- **Python 3.8+**
- **4GB RAM minimum** (8GB+ recommended)
- **NVIDIA GPU** (optional, for faster simulation)

---

## Installation

### Step 1: Extract the Project
```bash
# Extract the zip file
unzip zip
cd ../code
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv env_genesis
source env_genesis/bin/activate
```

### Step 3: Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y libompl-dev
```

### Step 4: Install Python Packages
```bash
# Install Genesis simulator
pip install genesis-world

# Install task planner
pip install pyperplan

# Install motion planning library
pip install --pre --extra-index-url https://pypi.kavrakilab.org ompl

# Install other dependencies
pip install numpy scipy
```

### Step 5: Verify Installation
```bash
python3 -c "import genesis, pyperplan, ompl; print('All dependencies OK')"
```

---

## Running the Code

All goal files are in the `code/` directory. Make sure your virtual environment is activated.

### Activate Environment (do this every time)
```bash
source env_genesis/bin/activate
cd code
```

### Run Goals

**Goal 1 - Stack Blocks in Order**:
```bash
python3 goal1_scattered.py    # From scattered start
python3 goal1_stacked.py       # From pre-stacked start
```

**Goal 2 - Build Two Towers**:
```bash
python3 goal2_scattered.py    # From scattered start
python3 goal2_stacked.py       # From pre-stacked start
```

**Goal 3 - Tallest Tower**:
```bash
python3 goal3_tallest.py
```

**Goal 4 - Pentagon Structure and grid blocks**:
```bash
python3 goal4_task1.py        # Two-layer pentagon
python3 goal4_task2.py        # grid blocks
```

**Note for Goal 4**: If you see "OMPL start state out of bounds" error on first run:
- **Quick fix**: Just rerun the command - it usually works on second try
- **Permanent fix**: Edit `goal4_task1.py` line ~40:
  ```python
  # Change from:
  safe_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=float)
  
  # To:
  safe_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.039, 0.039], dtype=float)
  ```
  This ensures gripper values stay within robot joint limits.

### Running in CPU Mode
If you encounter GPU memory errors or don't have a GPU:
```bash
python3 goal1_scattered.py cpu
```

---

## What to Expect

When you run a goal, you'll see:

1. **Initialization**: Robot moves to home position
2. **Task Planning**: Pyperplan generates action sequence
3. **Execution**: Robot picks and places blocks
4. **Visualization**: 3D window shows the simulation
5. **Completion**: Success message when goal is achieved

Example output:
```
At home

Initial predicates:
  ONTABLE(b1)
  CLEAR(b1)
  ...

Calling Pyperplan...
Found plan with 10 actions

Step 1: PICK-UP(b1)
motion: PICK-UP: 'b1'
...

SUCCESS! Goal achieved.
```

---

## Project Structure

```
code/
├── Domain Files (PDDL)
│   ├── blocksworld.pddl              # Standard domain
│   ├── blocksworld_directional.pddl  # Extended domain
│   └── pentagon_blocksworld.pddl     # Pentagon domain
│
├── Core Modules
│   ├── scenes.py                     # Simulation setup
│   ├── motion_primitives.py          # Pick/place actions
│   ├── planning.py                   # OMPL motion planning
│   ├── predicates.py                 # State extraction
│   ├── task_planner.py               # PDDL/Pyperplan interface
│   └── robot_adapter.py              # Robot control
│
└── Goal Files (run these)
    ├── goal1_scattered.py
    ├── goal1_stacked.py
    ├── goal2_scattered.py
    ├── goal2_stacked.py
    ├── goal3_tallest.py
    ├── goal4_task1.py
    └── goal4_task2.py
```

---

## Key Implementation Details

### Task Planning
- Uses **PDDL** (Planning Domain Definition Language) for symbolic reasoning
- **Pyperplan** generates action sequences (pick-up, stack, put-down)
- Plans are domain-independent and work with any valid initial/goal state

### Motion Planning
- Uses **OMPL** (Open Motion Planning Library) for collision-free paths
- **RRT-Connect** algorithm for fast motion planning
- Accounts for robot kinematics and workspace obstacles

### Control Loop
```
1. Extract current state → Predicates
2. Generate symbolic plan → PDDL/Pyperplan
3. Execute each action:
   - Plan collision-free motion → OMPL
   - Execute trajectory → Robot control
   - Let physics settle
4. Verify goal achieved
```

## Additional Help

### Check Installation
```bash
python3 << END
import genesis as gs
import pyperplan
import ompl
import numpy
print("Genesis:", gs.__version__)
print("NumPy:", numpy.__version__)
print("All packages installed")
END
```

### Force Reinstall Everything
```bash
deactivate  # Exit virtual environment
rm -rf env_genesis
python3 -m venv env_genesis
source env_genesis/bin/activate
pip install genesis-world pyperplan numpy scipy
pip install --pre --extra-index-url https://pypi.kavrakilab.org ompl
```

### GPU Check
```bash
# Check if NVIDIA GPU is detected
nvidia-smi

# If not found, GPU mode won't work - use CPU mode instead
```

---

## Credits

- **Genesis Physics Simulator**: Genesis Team
- **OMPL**: Kavraki Lab, Rice University
- **Pyperplan**: University of Basel

---


**Last Updated**: December 2025
