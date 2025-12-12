"""Scene factory helpers.

Provide functions to create common demo scenes. Each factory returns a
tuple (scene, franka, blocks_state, end_effector) to be used by demos.
"""
from typing import Any, Dict, Tuple
import random
import time
random.seed(time.time())

import numpy as np
import genesis as gs
from robot_adapter import RobotAdapter


def _build_base_scene(camera_pos=(3, -1, 1.5), camera_lookat=(0.0, 0.0, 0.5)) -> gs.Scene:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=8),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=True,
    )
    return scene

def _elevate_robot_base(franka: Any) -> None:
    """Slightly raise robot base to avoid initial collisions."""
    base_pos = np.asarray(franka.get_pos(), dtype=float)
    new_pos = base_pos.copy()
    new_pos[2] += 0.01
    franka.set_pos(new_pos) 

def _rand_xy(base, noise=0.05):
        dx = random.uniform(-noise, noise)
        dy = random.uniform(-noise, noise)
        return (base[0] + dx, base[1] + dy, base[2])

def create_scene_6blocks() -> Tuple[Any, Any, Dict[str, Any], Any]:
    """Create the default demo scene (layout 1).
    Returns:
        scene, franka_adapter, blocks_state, end_effector
    """
    scene = _build_base_scene()

    # basic geometry
    plane = scene.add_entity(gs.morphs.Plane())
    # add some random noise up to 5 cm in x/y

    posR = _rand_xy((0.65, 0.0, 0.02))
    posG = _rand_xy((0.65, 0.2, 0.02))
    posB = _rand_xy((0.65, 0.4, 0.02))
    posY = _rand_xy((0.45, 0.0, 0.02))
    posM = _rand_xy((0.45, 0.2, 0.02))
    posC = _rand_xy((0.45, 0.4, 0.02))

    cubeR = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posR),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posG),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posB),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posY),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )
    cubeM = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posM),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0, 1.0)),
    )

    cubeC = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posC),
        surface=gs.options.surfaces.Plastic(color=(0, 1.0, 1.0)),
    )

    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)

    # build scene (construct physics/visuals)
    scene.build()

    # initial robot pose (7 arm joints + 2 gripper fingers)
    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    # slightly raise robot base to avoid initial collisions
    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {"r": cubeR, "g": cubeG, "b": cubeB, "y": cubeY, "m": cubeM, "c": cubeC}

    return scene, franka, blocks_state


def create_scene_stacked() -> Tuple[Any, Any, Dict[str, Any], Any]:
    """Create an alternative demo scene (layout 2) with cube positions. one on top of the other."""
    scene = _build_base_scene(camera_pos=(2.5, -1.2, 1.2), camera_lookat=(0.6, 0.0, 0.2))

    plane = scene.add_entity(gs.morphs.Plane())

    # slightly different positions
    startx, starty, _ = _rand_xy((0.45, 0.0, 0.02), noise=0.2) 
    cubeR = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.02)),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.06)),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.10)),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.14)),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )

    cubeM = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.18)),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0, 1.0)),
    )

    cubeC = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(startx, starty, 0.22)),
        surface=gs.options.surfaces.Plastic(color=(0, 1.0, 1.0)),
    )

    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)
    scene.build()

    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {"r": cubeR, "g": cubeG, "b": cubeB, "y": cubeY, "m": cubeM, "c": cubeC}

    return scene, franka, blocks_state

#for goal 3 tallest tower task
def create_scene_10blocks2ln() -> Tuple[Any, Any, Dict[str, Any], Any]:
    scene = _build_base_scene()

    # basic geometry
    plane = scene.add_entity(gs.morphs.Plane())
    # add some random noise up to 5 cm in x/y

    posR = (0.45, -0.40, 0.02)
    posG = (0.45, -0.20, 0.02)
    posB = (0.45, 0.00, 0.02)
    posY = (0.45, 0.20, 0.02)
    posO = (0.45, 0.40, 0.02)
    posR_2 = (0.65, -0.40, 0.02)
    posG_2 = (0.65, -0.20, 0.02)
    posB_2 = (0.65, 0.00, 0.02)
    posY_2 = (0.65, 0.20, 0.02)
    posO_2 = (0.65, 0.40, 0.02)
    cubeR = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posR),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posG),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posB),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posY),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )
    cubeO = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posO),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.6, 0.0)),
    )

    cubeR_2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posR_2),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeG_2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posG_2),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeB_2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos= posB_2),
        surface=gs.options.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    )
    cubeY_2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posY_2),
        surface=gs.options.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )
    cubeO_2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posO_2),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.6, 0.0)),
    )

    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)

    # build scene (construct physics/visuals)
    scene.build()

    # initial robot pose (7 arm joints + 2 gripper fingers)
    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    # slightly raise robot base to avoid initial collisions
    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {"r": cubeR, "g": cubeG, "b": cubeB, "y": cubeY, "o": cubeO,"r2": cubeR_2, "g2": cubeG_2, "b2": cubeB_2, "y2": cubeY_2, "o2": cubeO_2}

    return scene, franka, blocks_state

#  for goal 4 task 1
def create_scene_10blocks() -> Tuple[Any, Any, Dict[str, Any], Any]:
    """Create scene with 10 blocks for pentagon tower structure.
    
    Goal structure:
    - Base pentagon (5 blocks): b1-b5 arranged in pentagon shape
    - Top pentagon (5 blocks): b6-b10 rotated 45° and stacked on base
    
    Returns:
        scene, franka_adapter, blocks_state
    """
    scene = _build_base_scene(camera_pos=(2.5, -1.2, 1.2), camera_lookat=(0.55, 0.0, 0.1))

    plane = scene.add_entity(gs.morphs.Plane())

    # Scatter blocks in two rows for initial placement
    block_positions = [
        # Row 1: 5 blocks (will become base pentagon b1-b5)
        _rand_xy((0.60, -0.30, 0.02), noise=0.03),
        _rand_xy((0.60, 0.15, 0.02), noise=0.03),
        _rand_xy((0.60, 0.30, 0.02), noise=0.03),
        _rand_xy((0.60, 0.45, 0.02), noise=0.03),
        _rand_xy((0.50, 0.40, 0.02), noise=0.03),
        
        # Row 2: 5 blocks (will become top pentagon b6-b10)
        _rand_xy((0.30, -0.30, 0.02), noise=0.03),
        _rand_xy((0.30, 0.00, 0.02), noise=0.03),
        _rand_xy((0.30, 0.30, 0.02), noise=0.03),
        _rand_xy((0.30, 0.45, 0.02), noise=0.03),
        _rand_xy((0.40, 0.5, 0.02), noise=0.03),
    ]
    
    # Create blocks with distinct colors
    colors = [
        (1.0, 0.0, 0.0),   # Red - b1
        (1.0, 0.5, 0.0),   # Orange - b2
        (0.0, 1.0, 0.0),   # Green - b3
        (0.0, 0.0, 1.0),   # Blue - b4
        (1.0, 1.0, 0.0),   # Yellow - b5
        (1.0, 0.0, 0.0),   # Red - b6
        (1.0, 0.5, 0.0),   # Orange - b7
        (0.0, 1.0, 0.0),   # Green - b8
        (0.0, 0.0, 1.0),   # Blue - b9
        (1.0, 1.0, 0.0),   # Yellow - b10
    ]
    
    blocks_state = {}
    
    for i in range(10):
        cube = scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=block_positions[i]),
            surface=gs.options.surfaces.Plastic(color=colors[i]),
        )
        blocks_state[f"b{i+1}"] = cube

    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)

    scene.build()

    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    _elevate_robot_base(franka)

    return scene, franka, blocks_state

# for goal 4 task 2
def create_scene_3red_3green() -> Tuple[Any, Any, Dict[str, Any], Any]:
    """Create scene with 3 red blocks and 3 green blocks.
    
    Uses same placement strategy as create_scene_6blocks.
    Blocks are named: r1, r2, r3, g1, g2, g3
    
    Returns:
        scene, franka_adapter, blocks_state, end_effector
    """
    scene = _build_base_scene()

    # basic geometry
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Add random noise up to 5 cm in x/y (same as original)
    posR1 = _rand_xy((0.65, 0.0, 0.02))
    posR2 = _rand_xy((0.55, 0.2, 0.02))
    posR3 = _rand_xy((0.6, 0.4, 0.02))
    posG1 = _rand_xy((0.45, 0.0, 0.02))
    posG2 = _rand_xy((0.45, 0.2, 0.02))
    posG3 = _rand_xy((0.45, 0.4, 0.02))

    # Create 3 red blocks
    cubeR1 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posR1),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeR2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posR2),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    cubeR3 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posR3),
        surface=gs.options.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    )
    
    # Create 3 green blocks
    cubeG1 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posG1),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeG2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posG2),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )
    cubeG3 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=posG3),
        surface=gs.options.surfaces.Plastic(color=(0.0, 1.0, 0.0)),
    )

    # Add robot
    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)

    # Build scene (construct physics/visuals)
    scene.build()

    # Initial robot pose (7 arm joints + 2 gripper fingers)
    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    # Slightly raise robot base to avoid initial collisions
    _elevate_robot_base(franka)

    blocks_state: Dict[str, Any] = {
        "r1": cubeR1, 
        "r2": cubeR2, 
        "r3": cubeR3,
        "g1": cubeG1, 
        "g2": cubeG2, 
        "g3": cubeG3
    }

    return scene, franka, blocks_state