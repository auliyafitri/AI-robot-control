
#IN CONTINUOUS ENV THE RANDOM SPAWN MUST BE COMMENTED 

import numpy as np
import random
import gym
import sys
import datetime
import rospkg
import rospy
import sys



rospack = rospkg.RosPack()
Env_path=rospack.get_path('pickbot_training')+"/src/2_Environment"
sys.path.insert(0,Env_path)
from pickbot_env_continuous import PickbotEnv
import gazebo_connection



def main():
    #unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    #create node 
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)
    env = gym.make("Pickbot-v1")

    PickbotEnv().turn_off_gripper()
    rospy.sleep(1)
    env.reset()
    
    for i in range(10):
        PickbotEnv().turn_on_gripper()
        action = [1.7003532462030622, -0.9988946153372558, 1.4019861182133537, -2.277473508541637, -1.5707587583812623, 0.0010351167324680333]#env.action_space.sample()
        print (" Step: "+str(i))
        next_state, reward, done, info=env.step(action)
        rospy.sleep(1)
        action = [1.7003532462030622, -1.6, 1.4019861182133537, -2.277473508541637, -1.5707587583812623, 0.0010351167324680333]
        next_state, reward, done, info=env.step(action)
        rospy.sleep(1)
        action = [1.7003532462030622, -0.9988946153372558, 1.4019861182133537, -2.277473508541637, -1.5707587583812623, 0.0010351167324680333]#env.action_space.sample()
        next_state, reward, done, info=env.step(action)
        #if done==True:
        PickbotEnv().turn_off_gripper()
        rospy.sleep(1)
        env.reset()
            
    
if __name__ == '__main__':
    main()
