# AI-robot-control

This repository contains code for industrial robot learning enabled with reinforcement learning algorithm that run in selected environments.
The repository contains the following:
* algorithms : implementation of RL algorithms from OpenAI baselines and stable-baselines
* environments : robot environments
* evaluations : results of training 
* experiments : scripts for running the training and for replaying the models
* models : saved models
* python_pkgs : python packages used
* simulation : robot definition, including pickbot 

## Prerequisites
* python 2.7
* python 3.5
* ROS

**If ROS has not installed yet, go to [installation](INSTALL.md) for information on how to install ROS and catkin workspace then continue below**

## Get the code of the AI-Robot-Control packages
**! Assuming your catkin workspace is named catkin_ws**
```
cd ~/catkin_ws/src
git clone --recurse-submodules https://github.com/PhilipKurrek/AI-robot-control.git
cd AI-robot-control
git submodule update --init --recursive
sudo python3 -m pip install -e .
chmod +x install.sh && . install.sh
```
If pip is not installed, run
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py && python3 get-pip.py
```

## How to run the Training: 
**Start the simulation**
```
roslaunch simulation main.launch
```
**To run the Baseline DeepQ Algorithm**
```
cd ~/catkin_ws/src/AI-Robot-Control/experiments/reach/DeepQ
python3 train_pickbot-v0_DeepQ_discrete.py
```
**To reuse the saved model in order to replay the learned policy**
```
cd ~/catkin_ws/src/AI-Robot-Control/experiments/reach/DeepQ
python3 enjoy_pickbot.py
```
**To use algorithm implementation of DeepQ-Learning and Q-Learning**
* Q-Learning
```
roslaunch robot_training start_training_Q_Learning.launch
```
* Deep Q Learning
```
roslaunch robot_training start_training_Deep_Q_Learning.launch
```