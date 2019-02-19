
#IN CONTINUOUS ENV THE RANDOM SPAWN MUST BE COMMENTED 

from pickbot_env_continuous import PickbotEnv
import numpy as np
import random





import gym
import sys
import datetime
import rospkg
import rospy
import sys

from baselines import deepq


rospack = rospkg.RosPack()
Env_path=rospack.get_path('pickbot_training')+"/src/2_Environment"
sys.path.insert(0,Env_path)
import pickbot_env_npstate  
import gazebo_connection


def callback(lcl, _glb):
    # stop training if average reward exceeds 450
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 450
    return is_solved


def main():
    #unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    #create node 
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)
    env = gym.make("Pickbot-v1")
    """
    env.reset()
    print (env.step([1.2, -1.40, -1.6, -1.9, -1.4, 0.2]))
    print(env.step([1, -1.50, -1.6, -2, -1.1, 0.2]))
    print(env.step([0,0,0,0,0,0]))
    print(env.step([0,0,2,0,0,0]))
    """
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
