import sys
from typing import Dict, Tuple, Set
import numpy as np
import genesis as gs

# to import some functions from scenes.py
from scenes import create_scene_6blocks, create_scene_stacked

block_size  = 0.04   # boxe created with size=(0.04,0.04,0.04) in scenes.py taken value from there 
block_bot     = 0.00   # bottom of each cube is on table 
hor_cen_tol      = 0.02   # when 2 blocks are one top of each other, horizontal distance between centers should be 0 but given tolerence 
ver_tol  = 0.015  # if 2 blocks are stacked, top_height(bottom block) - bottom_height(top block) = 0, given a little tolerence
clear_gap   = 0.03   # to check if another cube is above or not 

# function to calculate z coordinate of top of block
def top_z(center_z: float) -> float:
    return center_z + block_size / 2.0

# function to calculate z coordinate of bottom of block
def bottom_z(center_z: float) -> float:
    return center_z - block_size / 2.0

# function to check horizontal distance between block to check if they are stacked one on another or side to side 
def xy_dist(a_xy, b_xy) -> float:
    return float(np.linalg.norm(np.asarray(a_xy[:2]) - np.asarray(b_xy[:2])))

# function ot get the position of the block from genesis world
def get_block_pose(entity) -> Tuple[Tuple[float,float,float], Tuple[float,float,float,float]]:
    pos = entity.get_pos()
    quat = (0.0, 0.0, 0.0, 1.0)
    return (tuple(pos), quat)

# function to calculate the predicate 
def get_predicates(blocks_state: Dict[str, any]) -> Set[Tuple[str, Tuple[str, ...]]]:
    preds: Set[Tuple[str, Tuple[str, ...]]] = set()
    # start with gripper is empty 
    preds.add(("HANDEMPTY", ()))  

    # sort all blocks and get their position from genesis
    names = sorted(list(blocks_state.keys()))
    centers = {n: get_block_pose(blocks_state[n])[0] for n in names}

    # loop to check if block is on table or on another block 
    for a in names:
        a_xyz = centers[a]
        # initial flag
        on_something = False
        for b in names:
            if a == b:
                continue
            b_xyz = centers[b]
            # check horizontal distance of centers of block a and b and apply tolerence
            aligned_xy = xy_dist(a_xyz, b_xyz) < hor_cen_tol

            # check vertical distance between 2 blocks and apply tolerence 
            touch_z  = abs(bottom_z(a_xyz[2]) - top_z(b_xyz[2])) < ver_tol
            if aligned_xy and touch_z:
                preds.add(("ON", (a, b)))
                on_something = True
        # if z of bottom is 0 it is on table
        if (not on_something) and abs(bottom_z(a_xyz[2]) - block_bot) < ver_tol:
            preds.add(("ONTABLE", (a,)))

    # similar loop to check if block has anything on top or not
    for a in names:
        a_xyz = centers[a]
        is_clear = True
        for b in names:
            if a == b:
                continue
            b_xyz = centers[b]
            aligned_xy = xy_dist(a_xyz, b_xyz) < hor_cen_tol
            near_top   = abs(bottom_z(b_xyz[2]) - top_z(a_xyz[2])) < clear_gap
            if aligned_xy and near_top:
                is_clear = False
                break
        if is_clear:
            preds.add(("CLEAR", (a,)))
    return preds

# function to print to console neatly 
def term_print(preds: Set[Tuple[str, Tuple[str, ...]]]) -> str:
    items = sorted(list(preds), key=lambda p: (p[0], p[1]))
    return "{" + ", ".join(f"{p}({','.join(args)})" if args else f"{p}()" for p, args in items) + "}"

# use functions from scenes,py to create genesis scene
def run_scene(scene_factory, title: str, interactive: bool = True):
    scene, franka, blocks_state = scene_factory()

    # Let physics settle
    for _ in range(50):
        scene.step()

    preds = get_predicates(blocks_state)
    print("Predicates:", term_print(preds))

    if interactive:
        try:
            while True:
                scene.step()
        except KeyboardInterrupt:
            pass
    return scene, franka, blocks_state


def main():
    use_gpu = any(arg.lower() == "gpu" for arg in sys.argv[1:])
    backend = gs.gpu if use_gpu else gs.cpu
    gs.init(backend=backend, logging_level='Warning', logger_verbose_time=False)

    # when running code give python predicates.py scattered or python predicates.py stacked to check if predicates are displayed properly 
    choice = "scattered"
    for arg in sys.argv[1:]:
        if arg.lower() in {"scattered", "stacked"}:
            choice = arg.lower()
            break

    if choice == "scattered":
        run_scene(create_scene_6blocks, "Scene: 6 blocks scattered", interactive=True)
    elif choice == "stacked":
        run_scene(create_scene_stacked, "Scene: 6 blocks stacked", interactive=True)
        

if __name__ == "__main__":
    main()
