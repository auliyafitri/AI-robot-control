#!/usr/bin/env python
# Inspired by https://keon.io/deep-q-learning/

import rospy
import rospkg
import sys
import random
import gym
import math
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from std_msgs.msg import Float64

# import our training environment   
rospack = rospkg.RosPack()
Env_path=rospack.get_path('pickbot_training')+"/src/2_Environment"
sys.path.insert(0,Env_path)
import pickbot_env_npstate


#import liveplot
from gym import wrappers    

class DQNPickbot():
    def __init__(self, gamma=1.0, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, alpha=0.01, alpha_decay=0.01, n_episodes=1000, n_steps=120, state_size=9, n_actions=12, batch_size=64):
        
        self.memory = deque(maxlen=800000)
        self.env = gym.make('Pickbot-v0')  
        print "Gym Make done"    

        self.rospack = rospkg.RosPack()
        self.outdir = self.rospack.get_path('pickbot_training')+'/training_results'
        self.env = wrappers.Monitor(self.env, self.outdir, force=True, write_upon_reset=True)
        print "Monitor Wrapper started"
        #start Liveplot not working yet 
        #self.plotter = LivePlot(self.outdir)

        self.reward_pub = rospy.Publisher('/pickbot/reward', Float64, queue_size=1)
        self.episode_reward_pub = rospy.Publisher('/pickbot/episode_reward', Float64, queue_size=1)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.alpha_decay = alpha_decay

        self.n_episodes = n_episodes
        self.n_steps = n_steps
        
        self.state_size = state_size
        self.n_actions = n_actions

        self.batch_size = batch_size

        self.model = self._build_model()
        

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        return model
        

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def preprocess_state(self, state):
        return np.reshape(state, [1, self.state_size])

    """
    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))
    """
    
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  


    def replay(self, batch_size):
        #target:        reward wenn done sonst reward+gamma*np.amax(predict(next state))
        #prediction:    model.predict(state)  
        
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit(x=np.array(x_batch), y=np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        """
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \                          
                       np.amax(self.model.predict(next_state)[0])       #muss * und nicht / sein 
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        #fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
        """
        """
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict( else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])state)
            y_target[0][action] = reward if done
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        """

    def run(self):
                
        rate = rospy.Rate(30)
        
        episode_rewards = []

        # loop over Episodes
        for e in range(self.n_episodes):
            
            rospy.loginfo ("STARTING Episode #"+str(e))
            cumulated_reward = 0
            cumulated_reward_msg = Float64()
            episode_reward_msg = Float64()
            
            # reset the env and get state
            state = self.preprocess_state(self.env.reset())

            # inital Position 
            # write into YAML File
            #last_position = [1.5,-1.2,1.4,-1.87,-1.57,0]
            #with open('last_position.yml', 'w') as yaml_file:
            #    yaml.dump(last_position, yaml_file, default_flow_style=False)

            #reset done to false at the beginning of each episode
            done = False

            #loop over steps 
            for i in range(self.n_steps):
    
                #choose an action
                action = self.choose_action(state)
                # execute a step and get back (next_state, reward, done, info)
                next_state, reward, done, info = self.env.step(action=action)
                #reshape next_state
                next_state = self.preprocess_state(next_state)
                #remember all the values (state, action, reward, next_state, done) for learning
                self.remember(state, action, reward, next_state, done)
                """
                print "STEP Nr. "+str(i)
                if next_state.item(0,7)==0:
                    col1=True
                else:
                    col1=False
                if next_state.item(0,8)==0:
                    col2=True
                else:
                    col2=False
                print "Contact Force 1: "+str(next_state.item(0,7))+" Contact Force 2: "+str(next_state.item(0,8)) + " Contacts 1 and 2  are ZERO: "+str(col1)+", "+str(col2)
                """
                #cumulate episode reward
                cumulated_reward += reward
                # We publish the cumulated reward
                cumulated_reward_msg.data = cumulated_reward
                self.reward_pub.publish(cumulated_reward_msg)
                
                # if done break the episode 
                if not(done):
                    state = next_state
                else:
                    rospy.logdebug ("DONE")
                    break

            # publishing episode reward
            episode_reward_msg.data = cumulated_reward
            self.episode_reward_pub.publish(episode_reward_msg)
            episode_rewards.append(cumulated_reward)
            # learning
            self.replay(self.batch_size)
            
            print "\n| "+ "Episode: "+ str(e) +"/"+ str(self.n_episodes)+" | "+"Episode-Reward: "+str(cumulated_reward)+" | "+"Alpha: "+str(self.alpha) +" | "+"Gamma: "+str(self.gamma) +" | "+"Epsilon: "+str(self.epsilon) +" | "+"Steps: "+str(i) # Solved? Steps in episode

        plt.plot(episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Episode_Rewards')
        
        plt.savefig(self.outdir+'/Rewards_Deep_Q_Learn.png')
        plt.show()
        
        self.env.close()
        
if __name__ == '__main__':
    # Init ROS Node
    rospy.init_node('Pickbot_Training', anonymous=True, log_level=rospy.FATAL)
    
    # Load Parameters from YAML File
    gamma =  rospy.get_param('/pickbot_v0/gamma')
    epsilon = rospy.get_param('/pickbot_v0/epsilon')
    epsilon_decay = rospy.get_param('/pickbot_v0/epsilon_decay')
    epsilon_min = rospy.get_param('/pickbot_v0/epsilon_min')
    alpha = rospy.get_param('/pickbot_v0/alpha')
    alpha_decay = rospy.get_param('/pickbot_v0/alpha_decay')

    n_episodes = rospy.get_param('/pickbot_v0/episodes_training')
    n_steps = rospy.get_param('/pickbot_v0/n_steps')
   
    state_size = rospy.get_param('/pickbot_v0/state_size')
    n_actions = rospy.get_param('/pickbot_v0/n_actions')
    
    batch_size = rospy.get_param('/pickbot_v0/batch_size')

    agent = DQNPickbot(gamma=gamma, 
                        epsilon=epsilon,
                        epsilon_decay=epsilon_decay,
                        epsilon_min=epsilon_min,
                        alpha=alpha,
                        alpha_decay=alpha_decay,
                        n_episodes=n_episodes,
                        n_steps=n_steps,
                        state_size=state_size,
                        n_actions=n_actions,
                        batch_size=batch_size)

    agent.run()

"""
Parameter to play with:


gamma discount factor for future reward 
epsilon exploration rate 
epsilon_decay decay for exploration rate
epsilon_min minimum exporation
alpha learning rate for neural net
alpha_decay
n_episodes
n_steps
batch_size

Network Architecture
    Shape (Layers, Type)
    Activation Function
    Otimizer
Rewardstructure
Observations
"""