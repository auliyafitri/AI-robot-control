# -*- coding: utf-8 -*-
# modified from baselines/baselines/deepq/experiments/enjoy_cartpole.py

import gym
import rospy
import rospkg
import sys

from baselines import deepq

rospack = rospkg.RosPack()
Env_path=rospack.get_path('pickbot_training')+"/src/2_Environment"
sys.path.insert(0,Env_path)
import pickbot_env_npstate 

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


