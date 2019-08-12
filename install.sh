#!/usr/bin/env bash

apt-get update
apt-get dist-upgrade

apt-get install python-scipy

# Installing Gazebo-ROS Compatibility Packages
apt-get install -y --allow-unauthenticated ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control ros-kinetic-ros-controllers ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control ros-kinetic-joint-state-controller ros-kinetic-joint-state-publisher ros-kinetic-effort-controllers ros-kinetic-moveit
source /opt/ros/kinetic/setup.bash

# Installing Python packages
pip install keras tensorflow gym

# Installing Python packages for python3
apt-get install -y python3-pip
python3 -m pip install pyyaml rospkg catkin_pkg exception defusedxml empy numpy transformations joblib cloudpickle trollius

# Make all python files executable in the robot_training package
find ./ -name "*.py" -exec chmod +x {} \;

# Install submodules
# openAI gym
cd ./python_pkgs/gym
python3 -m pip install -e .
# pygazebo
cd ../pygazebo
python setup.py develop
# openAI Baselines
cd ../../algorithms/baselines
python3 -m pip install -e .
# stable-baselines
# TO DO

# Install ROS Required Plugins such as Vacuum Gripper from ARIAC 2017 project
cd /home/mark/SimBot
catkin_make
source devel/setup.bash
# openai_ros
rosdep install openai_ros
