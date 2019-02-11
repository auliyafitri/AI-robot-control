# AI-robot-control

## Installing ROS including all needed Packages and setting up Workspace:
### Setup Ubuntu to Install Packages from the Open Source Robotics Foundation (OSRF)
```sudo sh -c '
  echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" \
    > /etc/apt/sources.list.d/gazebo-stable.list'```
```wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -```
### Installing Gazebo
```sudo apt-get update```
```sudo apt-get install gazebo7 gazebo7-plugin-base gazebo7-common libgazebo7```
### Installing ROS
### Setup Ubuntu to Install Packages from ROS
```sudo sh -c '
  echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros-latest.list'```
```sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net --recv-key 0xB01FA116```
### Installing ROS Desktop
```sudo apt-get update```
```sudo apt-get install ros-kinetic-desktop```
```sudo rosdep init; rosdep update```
### Installing Gazebo-ROS Compatibility Packages
```sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control```
### Installing Additional Dependencies
```sudo apt-get install ros-kinetic-ros-controllers```
```source /opt/ros/kinetic/setup.bash```
### Setting Up a Catkin Workspace
```mkdir -p ~/catkin_ws/src```
```cd ~/catkin_ws/src```
```catkin_make```
```source devel/setup.bash```


## Cloning the OpenAi Ros Package:
## Cloning and installing the OpenAI Baseline and installing the virtual Environment:
## Cloning the pickbot_sumulation and pickbot_traiing Packages:
## How to run the Training: 



``` code```