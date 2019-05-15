# -*- coding: utf-8 -*-
# modified from baselines/baselines/deepq/experiments/enjoy_cartpole.py

import gym
import rospy
import os
import numpy as np

from environments import pickbot_env_registration, gazebo_connection
import models

from baselines import deepq
from baselines.common.cmd_util import make_vec_env
from baselines.bench import Monitor
from baselines.common.vec_env import VecEnv

num_env = 1
env_id = "Pickbot-v0"
env_type = "classic_control"
seed = None
task_name = "reach"
timestamp = "2019-04-29_13h33min"

modelsdir = os.path.dirname(models.__file__) + '/' + task_name + '/deepq/' + env_id + '/' + timestamp + '/'


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

    act = deepq.learn(
        env=env,
        network='mlp',
        total_timesteps=0,
        load_path=modelsdir + "model"
    )

    obs, done = env.reset(), False
    episode_rew = 0

    while True:
        obs, rew, done, _ = env.step(act.step(obs)[0])
        episode_rew += rew[0] if isinstance(env, VecEnv) else rew
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            print('episode_rew={}'.format(episode_rew))
            episode_rew = 0
            obs = env.reset()


if __name__ == '__main__':
    main()
