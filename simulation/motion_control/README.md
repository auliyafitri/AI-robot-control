# Motion Control for UR10 using MoveIt

This folder contains python script for controling the movement of the robot (in Gazebo) using *MoveIt*. Since *MoveIt*'s python API is in 2.7, the script is also written in Python 2.7.

### Standard procedure
#### 1. Start Gazebo simulation in Terminal 1:
```sh
roslaunch ur_gazebo pickbot.launch
```

<sub>Notice that the process in Terminal 1 can take a fairly big amount of time. And after that, you will see Warnings 
saying vacuum_gripper_joint missing. This is fine because we deliberately set `vacuum_gripper_joint` to be `revolute` 
instead of `fixed`, so that Gazebo is able to identify where the `vacuum_gripper_link` is. </sub>



#### 2. Wait until Terminal 1 to print out *"You can start planning now!"*. Then open a new terminal (Terminal 2) and start one of the script in this folder:
- **pickbot_motion_control.py**
  - listens to ROS messages of target positions from the environment script and perform the actions accordingly.It will publish a ROS message to tell when the action is finished by MoveIt.

- **ur10_keyboard_motion.py**
  - A script to move the UR10 with keyboard input
  - You need to install `tkinter` for python2 first. 
    ```sh
    sudo apt-get install python-tk
    ```
  - Press the **arrow key** to move in x-y-plane
  - Press Z or X to move along z-axis.
  - Press A or S to rotate `wrist_3_joint` 
  - Escape key to exit


### About End-Effector
A `dummy_vacuum_gripper_link` is currently used as end-effector for *MoveIt*, it's **almost** connected with `ee_link` at the same position as `vacuum_gripper_link` but with a small offset.
- (0, 0.296, 0)   Origin for `vacuum_gripper_joint`
- (0, 0.29615, 0) Origin for `dummy_vaccum_gripper_joint`