# -*- coding: utf-8 -*-
# modified from baselines/baselines/deepq/experiments/enjoy_cartpole.py

import gym
import rospy
import rospkg
import sys
import datetime

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
            load_path="pickbot_model_2019-03-20_simresearch.pkl")

    while True:
        time_start = datetime.datetime.now().replace(microsecond=0)
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)
        time_finish = datetime.datetime.now().replace(microsecond=0)
        print("Runtime for this episode: %s " % str(time_finish-time_start))
        print("")


if __name__ == '__main__':
    main()


