# AI-robot-control

## Installing ROS including all needed Packages and setting up Workspace:
### Setup Ubuntu to Install Packages from the Open Source Robotics Foundation (OSRF)
```
sudo sh -c '
  echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" \
    > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
```
### Installing Gazebo
```
sudo apt-get update
sudo apt-get install gazebo7 gazebo7-plugin-base gazebo7-common libgazebo7
```
### Installing ROS
### Setup Ubuntu to Install Packages from ROS
```
sudo sh -c '
  echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net --recv-key 0xB01FA116
```
### Installing ROS Desktop
```
sudo apt-get update
sudo apt-get install ros-kinetic-desktop
sudo rosdep init; rosdep update
```
### Installing Gazebo-ROS Compatibility Packages
```
sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control
```
### Installing Additional Dependencies
```
sudo apt-get install ros-kinetic-ros-controllers
source /opt/ros/kinetic/setup.bash
```
### Enable ROS from command line
#### open bashrc
```
gedit ~/.bashrc
```
#### then add this line in the end of the file
```
source /opt/ros/kinetic/setup.bash
```
### Installing OpenAI gym
```
git clone https://github.com/openai/gym
cd gym
sudo pip install -e .
```
### Installing Keras
```
sudo pip install keras
```
### Installing Tensorflow
```
sudo pip install tensorflow
```
#### If you have problem with enum34 when installing tensorflow, uninstall them first by 'sudo apt-get remove python-enum34'

### Installing Pygazebo
#### note: should be installed from develop branch in pygazebo github repo, not with pip or easy_install
```
git clone --single-branch --branch develop https://github.com/jpieper/pygazebo.git
cd pygazebo/
sudo python setup.py develop
```
### Setting Up a Catkin Workspace
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```
### Add the setup.bash directory to your bashrc file
```
gedit ~/.bashrc
```
#### add this line in the end of the file
```
source ~/your_ws/devel/setup.bash
```


## Cloning the OpenAI ROS Package:
```
cd catkin_ws/src
git clone https://bitbucket.org/theconstructcore/openai_ros.git
cd ..
catkin_make
```
## Cloning and installing the OpenAI Baseline and installing the virtual Environment:
### cloning Baseline package into src folder
```
cd catkin_ws/src
git clone https://github.com/openai/baselines.git
```
### installing pipenv for creating a virtual environment in order to use python 3 fpr baseline and python 2 for ros/simulation
#### if you have not installed pip, install it using 'easy_install pip' then open new terminal to continue
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
### to run the virtual environment you must be in the baseline folder
```
pipenv shell
```
### installing all the necessary packages into the virtual environment
```
pip install tensorflow
pip install -e .
pip install pyyaml
pip install rospkg
pip install catkin_pkg
pip install exception
pip install defusedxml
pip install empy
pip install numpy
```
### Append PYTHONPATH to recognize OpenCV, needed for running the algorithms
* When running pipenv shell, you should see the virtualenv ID at the top, something like '/home/robotics/Envs/baselines-qhDvDYR-/bin/activate'
* copy it then replace the 'bin/activate' by 'lib/python3.5/site-packages', add as PYTHONPATH
* Do it inside the virtualenv
```
export PYTHONPATH='/home/robotics/Envs/baselines-qhDvDYR-/lib/python3.5/site-packages':$PYTHOPATH
```
#### You need to do this everytime you open the virtualenv
##### To avoid that, add PYTHOPATH in the activate file, something like '/home/robotics/Envs/baselines-qhDvDYR-/bin/activate'
```
echo $PYTHONPATH
```
##### Copy them, then open the activate file
```
gedit /home/robotics/Envs/baselines-qhDvDYR-/bin/activate
```
##### Add these two line in the end
```
PYTHONPATH="<paste here what you get after you echo $PYTHONPATH>"
export PYTHONPATH
```

### Now close the terminal with the activated virtual environment

## Cloning the pickbot_simulation and pickbot_training Packages:
```
cd catkin_ws/src
git clone https://github.com/PhilipKurrek/AI-robot-control.git
cd ..
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
### Navigate to the different subfolders 1_OpenAI_Baselines/DeepQ, 2_Environment, 3_Evaluation and 4_Own_Impelemtations_of_Algorythms and make in each direktory all python files executable by entering into terminal in each directory
```
chmod +x *
```
### Install ROS Required Plugins such as Vacuum Gripper from ARIAC 2017 project
```
cd catkin_ws/src
git clone https://bitbucket.org/osrf/ariac -b ariac_2017
cd ..
catkin_make
```
#### P.S. You can delete the ARIAC folder in src folder afterwards, we don't really need it

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
pipenv shell
```
```
cd ..
cd Pickbot/pickbot_training/src/1_OpenAI_Baselines/DeepQ
python train_pickbot_DeepQ.py
```
### To reuse the saved model in order to replay the learned policy
```
cd catkin_ws/src/baselines
pipenv shell
```
```
cd ..
cd Pickbot/pickbot_training/src/1_OpenAI_Baselines/DeepQ
python enjoy_pickbot.py
```

## Issues
* If you cannot found the rostopic /gripper_contactsensor_1_state, repeat Installing Gazebo-ROS Compatibility Packages by 'sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control'
