# -*- coding: utf-8 -*-
# modified from baselines/baselines/deepq/experiments/enjoy_cartpole.py

import gym
import rospy

from environments import pickbot_env_npstate, gazebo_connection

from baselines import deepq


def main():
    rospy.init_node('Pickbot_Training', anonymous=True, log_level=rospy.FATAL)
    env = gym.make("Pickbot-v0")
    act = deepq.learn(
            env, 
            network='mlp', 
            total_timesteps=0, 
            load_path="pickbot_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()


