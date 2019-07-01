
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
import pickbot_env_continuous  
import gazebo_connection


def main():
    #unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    #create node 
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)
    env = gym.make("Pickbot-v1")

    
    for i in range(1000):
        env.reset()
        
        next_state, reward, done, info=env.step(env.action_space.sample())
        print (" Step: "+str(i))
       
            
    
if __name__ == '__main__':
    main()
