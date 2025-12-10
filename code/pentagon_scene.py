"""Scene factory helpers.

Provide functions to create common demo scenes. Each factory returns a
tuple (scene, franka, blocks_state, end_effector) to be used by demos.
"""
from typing import Any, Dict, Tuple
import random
import time
import numpy as np
import genesis as gs
from robot_adapter import RobotAdapter

random.seed(time.time())


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


def create_scene_16blocks() -> Tuple[Any, Any, Dict[str, Any], Any]:
    """Create scene with 16 blocks for 3D cross structure.
    
    Blocks are scattered initially and will be arranged into:
    - Bottom layer (4 blocks): Cross pattern
    - Middle layer (8 blocks): Square ring (3x3 minus center)
    - Top layer (4 blocks): Cross pattern
    
    Returns:
        scene, franka_adapter, blocks_state
    """
    scene = _build_base_scene(camera_pos=(2.5, -1.0, 1.3), camera_lookat=(0.55, 0.0, 0.06))

    plane = scene.add_entity(gs.morphs.Plane())
    
    # Create 16 blocks in scattered positions
    # Using yellow color for all blocks (like your reference image)
    block_positions = [
        # Row 1 (5 blocks)
        _rand_xy((0.30, -0.20, 0.02), noise=0.03),
        _rand_xy((0.30, -0.60, 0.02), noise=0.03),
        _rand_xy((0.30, 0.00, 0.02), noise=0.03),
        _rand_xy((0.30, 0.10, 0.02), noise=0.03),
        _rand_xy((0.30, 0.20, 0.02), noise=0.03),
        
        # Row 2 (6 blocks)
        # _rand_xy((0.70, -0.25, 0.02), noise=0.03),
        # _rand_xy((0.70, -0.15, 0.02), noise=0.03),
        # _rand_xy((0.70, -0.05, 0.02), noise=0.03),
        # _rand_xy((0.70, 0.05, 0.02), noise=0.03),
        # _rand_xy((0.70, 0.15, 0.02), noise=0.03),
        # _rand_xy((0.70, 0.25, 0.02), noise=0.03),
        
        # Row 3 (5 blocks)
        _rand_xy((-0.30, -0.20, 0.02), noise=0.03),
        _rand_xy((0.40, -0.50, 0.02), noise=0.03),
        _rand_xy((0.45, 0.20, 0.02), noise=0.03),
        _rand_xy((0.45, 0.30, 0.02), noise=0.03),
        _rand_xy((0.45, 0.40, 0.02), noise=0.03),
    ]
    
    # Block naming: b1, b2, ..., b16
    blocks = []
    blocks_state = {}
    
    for i in range(16):
        cube = scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=block_positions[i]),
            surface=gs.options.surfaces.Plastic(color=(1.0, 0.9, 0.0)),  # Yellow
        )
        blocks.append(cube)
        blocks_state[f"b{i+1}"] = cube

    franka_raw = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    franka = RobotAdapter(franka_raw, scene)

    scene.build()

    franka.set_qpos(np.array([0.0, -0.5, -0.2, -1.0, 0.0, 1.00, 0.5, 0.02, 0.02]))

    _elevate_robot_base(franka)

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