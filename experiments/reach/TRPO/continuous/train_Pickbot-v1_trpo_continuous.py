#!/usr/bin/env python3
import gym
import os
import datetime
import rospy

from environments import pickbot_env_registration, gazebo_connection
import evaluations
import models

from baselines.trpo_mpi import trpo_mpi, defaults
from baselines.bench import Monitor
from baselines import logger
from baselines.common.cmd_util import make_vec_env

from environments.joint_array_publisher import JointArrayPub
from std_msgs.msg import String

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')

alg_kwargs = defaults.mara_mlp()

num_env = 1
env_id = "PickbotReachContinuous-v1"
# env_id = "PickbotReachContinuousDoorHandle-v1"
env_type = "classic_control"
seed = None
task_name = "reach"
# last_timestamp = "2019-06-11_12h28min"
# last_timestamp = "2019-06-18_09h26min"
last_timestamp = "2019-06-18_21h00min"

# Create needed folders for log file and models
logdir = os.path.dirname(evaluations.__file__) + '/' + task_name + '/trpo/' + env_id + '/' + timestamp + '/'
modelsdir = os.path.dirname(models.__file__) + '/' + task_name + '/trpo/' + env_id + '/' + timestamp + '/'
load_path = os.path.dirname(models.__file__) + '/' + task_name + '/trpo/' + env_id + '/' + last_timestamp + '/'

print("modeldir:  {}".format(modelsdir))
print("load_path: {}".format(load_path + "model"))

# Generate tensorboard file
format_strs = ['stdout', 'log', 'csv', 'tensorboard']
logger.configure(os.path.abspath(logdir), format_strs)


def mov_complete_callback(data):
    msg = data.data
    print(msg)


def main():
    # unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    # create node
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)

    env = make_vec_env(env_id, env_type, num_env, seed,
                       wrapper_kwargs=Monitor,
                       start_index=0,
                       reward_scale=1.0,
                       flatten_dict_observations=True,
                       gamestate=None)

    # Normal training

    # act = trpo_mpi.learn(
    #     env=env,
    #     network='mlp',
    #     total_timesteps=1000000,
    #     save_path=modelsdir,
    #     load_path=load_path + "policy_best"
    # )

    # Training hyper-paramerters from MARA

    # act = trpo_mpi.learn(
    #     env=env,
    #     network='mlp',
    #     save_path=modelsdir,
    #     **alg_kwargs
    # )

    # Transfer learning

    act = trpo_mpi.learn(
        env=env,
        network='mlp',
        save_path = modelsdir,
        load_path=load_path + "policy_best",
        **alg_kwargs
    )

    print("Saving model to " + modelsdir)
    act.save(modelsdir + "model")


if __name__ == '__main__':
    main()
