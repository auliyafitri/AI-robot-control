#!/usr/bin/env python3

import gym
import os
import sys
import rospkg
import rospy
import datetime

from environments import pickbot_env_registration, gazebo_connection
import evaluations
import models

from baselines import deepq
from baselines import logger

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')


def callback(lcl, _glb):
    # stop training if average reward exceeds 450
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 450
    return is_solved


def main():
    # unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    # create node
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)

    env_id = "Pickbot-v0"
    task_name = "reach"

    # Create needed folders for log file and models
    logdir = os.path.dirname(evaluations.__file__) + '/' + task_name + '/deepq/' + env_id + '/' + timestamp + '/'
    modelsdir = os.path.dirname(models.__file__) + '/' + task_name + '/deepq/' + env_id + '/' + timestamp + '/'

    # Generate tensorboard file
    format_strs = ['stdout', 'log', 'csv', 'tensorboard']
    logger.configure(os.path.abspath(logdir), format_strs)

    env = gym.make(env_id)
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=1000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
    )
    print("Saving model to " + modelsdir)
    act.save(modelsdir + "model.pkl")


if __name__ == '__main__':
    main()
