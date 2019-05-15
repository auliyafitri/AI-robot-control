
#IN CONTINUOUS ENV THE RANDOM SPAWN MUST BE COMMENTED 

import numpy as np
import random
import gym
import sys
import datetime
import rospkg
import rospy
import sys

from environments import pickbot_env_registration
from environments.gazebo_connection import GazeboConnection


def main():
    #unpause Simulation so that robot receives data on all topics
    gc = GazeboConnection()
    gc.unpauseSim()
    #create node 
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)
    env = gym.make("Pickbot-v1")
    env.reset()
    
    for i in range(10000):
        x = np.random.uniform(low=-2.9, high=2.9, size=None)
        action = []
        for i in range(6):
            act = np.random.uniform(low=-2.9, high=2.9, size=None)
            action.append(act)
        # action = [1.7003532462030622, -0.9988946153372558, 1.4019861182133537, -2.277473508541637, -1.5707587583812623, 0.0010351167324680333]#env.action_space.sample()
        print(" Step: {}".format(action))
        next_state, reward, done, info = env.step(action)

        if done:
            env.reset()
            
    
if __name__ == '__main__':
    main()
