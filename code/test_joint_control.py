"""
test_joint_control.py

Interactive robot joint control for debugging
- Control each joint individually with keyboard
- Test gripper rotation and movement
- Display current joint angles in real-time

Controls:
  1/Q - Joint 1 (base rotation)
  2/W - Joint 2 (shoulder)
  3/E - Joint 3 (elbow)
  4/R - Joint 4 (forearm roll)
  5/T - Joint 5 (wrist pitch)
  6/Y - Joint 6 (wrist roll)
  7/U - Joint 7 (wrist yaw)
  8/I - Gripper open
  9/O - Gripper close
  
  Numbers = positive direction (increase angle)
  Letters = negative direction (decrease angle)
  
  H - Home position
  P - Print current joint angles
  ESC - Exit

Usage:
    python test_joint_control.py [gpu]
"""

import sys
import numpy as np
import genesis as gs
from scenes import create_scene_6blocks

# Initialize
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    gs.init(backend=gs.gpu, logging_level="Warning", logger_verbose_time=False)
else:
    gs.init(backend=gs.cpu, logging_level="Warning", logger_verbose_time=False)

scene, franka, blocks_state = create_scene_6blocks()

# Strong gripper control
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 2000, 2000]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 200, 200]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -200, -200]),
    np.array([87, 87, 87, 87, 12, 12, 12, 200, 200]),
)

print("="*80)
print("JOINT CONTROL TEST")
print("="*80)
print("\nControls:")
print("  1/Q - Joint 1 (base)")
print("  2/W - Joint 2 (shoulder)")
print("  3/E - Joint 3 (elbow)")
print("  4/R - Joint 4 (forearm)")
print("  5/T - Joint 5 (wrist pitch)")
print("  6/Y - Joint 6 (wrist roll)")
print("  7/U - Joint 7 (wrist yaw)")
print("  8/I - Open gripper")
print("  9/O - Close gripper")
print("  H   - Home position")
print("  P   - Print angles")
print("="*80)

# Home position
home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=float)

# Move to home
print("\nMoving to home position...")
current = franka.get_qpos()
if hasattr(current, "cpu"):
    current = current.cpu().numpy()

for i in range(200):
    alpha = (i + 1) / 200.0
    q = (1 - alpha) * current + alpha * home
    franka.control_dofs_position(q)
    scene.step()

print("At home position\n")

# Get current position
def get_current_q():
    q = franka.get_qpos()
    if hasattr(q, "cpu"):
        q = q.cpu().numpy()
    return q.copy()

def print_joints():
    q = get_current_q()
    print("\nCurrent joint angles:")
    for i in range(7):
        print(f"  Joint {i+1}: {np.degrees(q[i]):7.2f}° ({q[i]:6.3f} rad)")
    print(f"  Gripper:  {q[7]:.4f}, {q[8]:.4f}")
    
    # Get end-effector position
    hand = franka.get_link("hand")
    pos = np.array(hand.get_pos())
    print(f"\nEnd-effector position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

# Step size for joint movement
STEP = 0.05  # radians (~3 degrees)
GRIPPER_STEP = 0.005

print("\nReady! Use keyboard controls...")
print("(Note: Terminal must be in focus, commands are instant)")

# Manual control mode
current_q = get_current_q()

try:
    import sys
    import termios
    import tty
    
    # Save terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    
    try:
        # Set terminal to raw mode for immediate key capture
        tty.setcbreak(sys.stdin.fileno())
        
        while True:
            # Read one character
            char = sys.stdin.read(1)
            
            if char == '\x1b':  # ESC key
                print("\nExiting...")
                break
            
            # Update current position
            current_q = get_current_q()
            
            # Joint controls
            if char == '1':
                current_q[0] += STEP
                print(f"Joint 1: {np.degrees(current_q[0]):.2f}°")
            elif char == 'q':
                current_q[0] -= STEP
                print(f"Joint 1: {np.degrees(current_q[0]):.2f}°")
            elif char == '2':
                current_q[1] += STEP
                print(f"Joint 2: {np.degrees(current_q[1]):.2f}°")
            elif char == 'w':
                current_q[1] -= STEP
                print(f"Joint 2: {np.degrees(current_q[1]):.2f}°")
            elif char == '3':
                current_q[2] += STEP
                print(f"Joint 3: {np.degrees(current_q[2]):.2f}°")
            elif char == 'e':
                current_q[2] -= STEP
                print(f"Joint 3: {np.degrees(current_q[2]):.2f}°")
            elif char == '4':
                current_q[3] += STEP
                print(f"Joint 4: {np.degrees(current_q[3]):.2f}°")
            elif char == 'r':
                current_q[3] -= STEP
                print(f"Joint 4: {np.degrees(current_q[3]):.2f}°")
            elif char == '5':
                current_q[4] += STEP
                print(f"Joint 5: {np.degrees(current_q[4]):.2f}°")
            elif char == 't':
                current_q[4] -= STEP
                print(f"Joint 5: {np.degrees(current_q[4]):.2f}°")
            elif char == '6':
                current_q[5] += STEP
                print(f"Joint 6: {np.degrees(current_q[5]):.2f}°")
            elif char == 'y':
                current_q[5] -= STEP
                print(f"Joint 6: {np.degrees(current_q[5]):.2f}°")
            elif char == '7':
                current_q[6] += STEP
                print(f"Joint 7: {np.degrees(current_q[6]):.2f}°")
            elif char == 'u':
                current_q[6] -= STEP
                print(f"Joint 7: {np.degrees(current_q[6]):.2f}°")
            elif char == '8' or char == 'i':
                current_q[7] = 0.04  # Open
                current_q[8] = 0.04
                print("Gripper: OPEN")
            elif char == '9' or char == 'o':
                current_q[7] = 0.0  # Close
                current_q[8] = 0.0
                print("Gripper: CLOSE")
            elif char == 'h' or char == 'H':
                current_q = home.copy()
                print("Moving to HOME")
            elif char == 'p' or char == 'P':
                print_joints()
                continue
            else:
                continue
            
            # Send command and step simulation
            for _ in range(10):  # Hold for 10 steps
                franka.control_dofs_position(current_q)
                scene.step()
    
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

except ImportError:
    print("\nKeyboard control not available on this system")
    print("Showing automated joint sequence instead...\n")
    
    # Automated sequence - move each joint
    print("AUTOMATED JOINT TEST SEQUENCE")
    print("="*80)
    
    def move_joint(joint_idx, target_angle, name):
        current_q = get_current_q()
        start_angle = current_q[joint_idx]
        
        print(f"\n{name} (Joint {joint_idx+1}): {np.degrees(start_angle):.1f}° → {np.degrees(target_angle):.1f}°")
        
        for i in range(100):
            alpha = (i + 1) / 100.0
            current_q[joint_idx] = (1 - alpha) * start_angle + alpha * target_angle
            franka.control_dofs_position(current_q)
            scene.step()
        
        # Hold
        for _ in range(30):
            franka.control_dofs_position(current_q)
            scene.step()
    
    # Test each joint
    move_joint(0, 0.5, "Base rotation")
    move_joint(0, -0.5, "Base rotation back")
    move_joint(0, 0.0, "Base to center")
    
    move_joint(1, -0.3, "Shoulder lift")
    move_joint(1, -1.2, "Shoulder down")
    move_joint(1, -0.785, "Shoulder to home")
    
    move_joint(2, 0.5, "Elbow bend")
    move_joint(2, -0.5, "Elbow extend")
    move_joint(2, 0.0, "Elbow to center")
    
    move_joint(6, 1.3, "Wrist rotate")
    move_joint(6, 0.3, "Wrist rotate back")
    move_joint(6, 0.785, "Wrist to home")
    
    # Gripper test
    print("\nGripper: OPEN → CLOSE → OPEN")
    current_q = get_current_q()
    
    # Open
    for i in range(50):
        current_q[7] = 0.04
        current_q[8] = 0.04
        franka.control_dofs_position(current_q)
        scene.step()
    
    # Close
    for i in range(50):
        alpha = (i + 1) / 50.0
        current_q[7] = (1 - alpha) * 0.04 + alpha * 0.0
        current_q[8] = (1 - alpha) * 0.04 + alpha * 0.0
        franka.control_dofs_position(current_q)
        scene.step()
    
    # Open again
    for i in range(50):
        alpha = (i + 1) / 50.0
        current_q[7] = (1 - alpha) * 0.0 + alpha * 0.04
        current_q[8] = (1 - alpha) * 0.0 + alpha * 0.04
        franka.control_dofs_position(current_q)
        scene.step()
    
    print("\nJoint test sequence complete!")
    print("\nFinal joint angles:")
    print_joints()
    
    print("\nPress Ctrl+C to exit...")
    try:
        while True:
            franka.control_dofs_position(current_q)
            scene.step()
    except KeyboardInterrupt:
        print("\nExiting...")