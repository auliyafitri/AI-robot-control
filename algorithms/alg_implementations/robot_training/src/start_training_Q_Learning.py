#!/usr/bin/env python3

'''
    Original Training code made by Ricardo Tellez <rtellez@theconstructsim.com>
    Moded by Miguel Angel Rodriguez <duckfrost@theconstructsim.com>
    Visit our website at www.theconstructsim.com
'''

import rospy
import rospkg
import sys
import random
import gym
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Float64

from gym import wrappers
from baselines import logger
import qlearn

# import our training environment
from environments import pickbot_env_stringstate


if __name__ == '__main__':
    # Generate tensorboard file
    rospack = rospkg.RosPack()
    outdir = rospack.get_path('robot_training') + '/training_results'
    format_strs = ['stdout', 'log', 'csv', 'tensorboard']
    logger.configure(outdir, format_strs)

    rospy.init_node('Pickbot_Training', anonymous=True, log_level=rospy.INFO)

    # Create the Gym environment
    env = gym.make('Pickbot-v0')
    print("Gym Make done")

    env = wrappers.Monitor(env, outdir, force=True, write_upon_reset=True)
    print("Monitor Wrapper started")

    reward_pub = rospy.Publisher('/pickbot/reward', Float64, queue_size=1)
    episode_reward_pub = rospy.Publisher('/pickbot/episode_reward', Float64, queue_size=1)

    #load parameters from YAML File
    Alpha = rospy.get_param("/alpha")
    Epsilon = rospy.get_param("/epsilon")
    Gamma = rospy.get_param("/gamma")
    epsilon_discount = rospy.get_param("/epsilon_discount")
    nepisodes = rospy.get_param("/nepisodes")
    nsteps = rospy.get_param("/nsteps")

    #initialize empty list for episode rewards 
    episode_rewards = []

        
    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                    alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    highest_reward = 0
    

    # loop over Episodes
    for x in range(nepisodes):

        #rospy.loginfo ("STARTING Episode #"+str(x))
        cumulated_reward = 0
        cumulated_reward_msg = Float64()
        episode_reward_msg = Float64()

        # Initialize the environment and get first state of the robot
        state = env.reset()

        #reset done to false at the beginning of each episode
        done = False

        # Discount Exporationrate Epsilon each Episode
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
            
        #loop over steps 
        for i in range(nsteps):
            #choose an action
            action = qlearn.chooseAction(state) 
            # execute a step and get back (last_position, next_state, reward, done, info)
            nextState, reward, done, info = env.step(action=action)

            #cumulate episode reward
            cumulated_reward += reward
            # We publish the cumulated reward
            cumulated_reward_msg.data = cumulated_reward
            reward_pub.publish(cumulated_reward_msg)

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            # Make the algorithm learn based on the results
            qlearn.learn(state, action, reward, nextState)

            # if done break the episode 
            if not(done):
                state = nextState
            else:
                rospy.logdebug ("DONE")
                break

        # publishing episode reward
        episode_reward_msg.data = cumulated_reward
        episode_reward_pub.publish(episode_reward_msg)
        episode_rewards.append(cumulated_reward)

        print("\n| "+ "Episode: "+ str(x) +"/"+ str(nepisodes)+" | "+"Episode-Reward: "+str(cumulated_reward)+" | "+"Alpha: "+str(Alpha) +" | "+"Gamma: "+str(Gamma) +" | "+"Epsilon: "+str(qlearn.epsilon) +" | "+"Steps: "+str(i))  # Solved? Steps in episode

    
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Episode_Rewards')
        
    plt.savefig(outdir+'/Rewards_Q_learn.png')
    plt.show()
    
    env.close()
    