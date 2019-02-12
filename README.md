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
cd ~/catkin_ws
```
```
catkin_make
```
```
source devel/setup.bash
```
### (add the setup.bash directory to your bashrc file)
```
gedit ~/.bashrc
```
```
source ~/your_ws/devel/setup.bash
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
pip install –-user pipenv 
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
pip install rospkg
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
### Now close the terminal with the activated virtual environment

## Cloning the pickbot_sumulation and pickbot_training Packages:
```
cd catkin_wa/src
```
```
git clone https://github.com/PhilipKurrek/AI-robot-control.git
```
```
cd ..
```
```
catkin_make
```
### Now we need to change the path of the mesh files in the pickbot.world file
### Go to the following directory
```
catkin_ws/src/Pickbot/pickbot_simulation/worlds
```
### Open the pickbot.world file and change the four pathes  of dae and stl files to your correct path
```
/home/robotics/catkin_ws/src/Pickbot/pickbot_simulation/meshes/ur10/collision/Pickbot_Schubladenregal_offen.stl
```
### Make all python files executable in the pickbot_training package
```
cd catkin_ws/src/Pickbot/pickbot/training/src
```
### Navigate to the different subfolders 1_OpenAI Baselines/DeepQ, 2_Environment, 3_Evaluation and 4_Own Impelemtations of Algorythms and make in each direktory all python files executable by entering into terminal in each directory
```
chmod +x *
```
## How to run the Training: 
### Start the simulation
```
roslaunch pickbot_simulation main.launch
```
### To use my own implementation of DeepQ-Learning and Q-Learning run
```
roslaunch pickbot_training start_training_Deep_Q_Learning.launch
```
```
roslaunch pickbot_training start_training_Q_Learning.launch
```
### To run the Baseline DeepQ Algorythm you need to activate the virtual environment 
```
cd catkin_ws/src/baselines
```
```
pipenv shell
```
```
cd ..
```
```
cd Pickbot/pickbot_training/src/1_OpenAi Baselines/DeepQ
```
```
python train_pickbot.py
```