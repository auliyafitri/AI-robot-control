#!/usr/bin/env bash
# Setup Ubuntu to Install Packages from the Open Source Robotics Foundation (OSRF)
sudo sh -c '
  echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" \
    > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

apt-get update
apt-get dist-upgrade

apt-get install python-scipy

# Installing Gazebo
sudo apt-get update
sudo apt-get install -y gazebo7 gazebo7-plugin-base gazebo7-common libgazebo7

# Installing Gazebo-ROS Compatibility Packages
sudo apt-get install -y --allow-unauthenticated ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control ros-kinetic-ros-controllers ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control ros-kinetic-joint-state-controller ros-kinetic-joint-state-publisher ros-kinetic-effort-controllers ros-kinetic-moveit
source /opt/ros/kinetic/setup.bash

# Installing Python packages
sudo pip install keras tensorflow gym

# Installing Python packages for python3
sudo apt-get install -y python3-pip
sudo python3 -m pip install tensorflow pyyaml rospkg catkin_pkg exception defusedxml empy numpy transformations joblib cloudpickle tensorflow-gpu trollius

# Make all python files executable in the robot_training package
find ./ -name "*.py" -exec chmod +x {} \;

# Install submodules
# openAI gym
cd ./python_pkgs/gym
sudo python3 -m pip install -e .
# pygazebo
cd ../pygazebo
sudo python setup.py develop
# openAI Baselines
cd ../../algorithms/baselines
sudo python3 -m pip install -e .
# stable-baselines
# TO DO

# Install ROS Required Plugins such as Vacuum Gripper from ARIAC 2017 project
cd ~/catkin_ws/src
git clone https://bitbucket.org/osrf/ariac -b ariac_2017
git clone https://github.com/YangLiu14/gazebo_ros_pkgs.git
cd ..
catkin_make
source devel/setup.bash
# openai_ros
rosdep install openai_ros