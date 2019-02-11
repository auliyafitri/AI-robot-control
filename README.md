# AI-robot-control

## Installing ROS including all needed Packages and setting up Workspace:
### Setup Ubuntu to Install Packages from the Open Source Robotics Foundation (OSRF)
```
sudo sh -c '
  echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" \
    > /etc/apt/sources.list.d/gazebo-stable.list'
```
```
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
```
### Installing Gazebo
```
sudo apt-get update
```
```
sudo apt-get install gazebo7 gazebo7-plugin-base gazebo7-common libgazebo7
```
### Installing ROS
### Setup Ubuntu to Install Packages from ROS
```
sudo sh -c '
  echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros-latest.list'
```
```
sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net --recv-key 0xB01FA116
```
### Installing ROS Desktop
```
sudo apt-get update
```
```
sudo apt-get install ros-kinetic-desktop
```
```
sudo rosdep init; rosdep update
```
### Installing Gazebo-ROS Compatibility Packages
```
sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control
```
### Installing Additional Dependencies
```
sudo apt-get install ros-kinetic-ros-controllers
```
```
source /opt/ros/kinetic/setup.bash
```
### Setting Up a Catkin Workspace
```
mkdir -p ~/catkin_ws/src
```
```
cd ~/catkin_ws/src
```
```
catkin_make
```
```
source devel/setup.bash
```


## Cloning the OpenAi Ros Package:
```
cd catkin_ws/src
```
```
git clone https://bitbucket.org/theconstructcore/openai_ros.git
```
```
cd ..
```
```
catkin_make
```
## Cloning and installing the OpenAI Baseline and installing the virtual Environment:
### cloning Baseline package into src folder
```
cd catkin_ws/src
```
```
git clone https://github.com/openai/baselines.git
```
### installing pipenv for creating a virtual environment in order to use python 3 fpr baseline and python 2 for ros/simulation
```
Pip install –-user pipenv 
```
```
cd catkin_ws/src/baselines
```
### this might cause issues due to problems with path variables. Make sure where pipenv is installed. Must be included in the path system 
```
pipenv install --python=python3
```
### to run the virtual environment you must be in the baselinefolder.
```
pipenv shell
```
### installing all the nessesary packages into the virtual environment
```
pip install tensorflow
```
```
pip install -e .
```
```
pip install pyyaml
```
```
pip install rospgk
```
```
pip install catkin_pkg
```
```
pip install exception
```
```
pip install defusedxml
```
```
pip install empy
```
```
pip install numpy
```

## Cloning the pickbot_sumulation and pickbot_traiing Packages:
## How to run the Training: 



``` code```