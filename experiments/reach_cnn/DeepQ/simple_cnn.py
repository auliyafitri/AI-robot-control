import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from environments import pickbot_reach_env, gazebo_connection

timestamp = datetime.datetime.now()

class SimpleCNN(nn.Module):
    def __init__(self, number_actions):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        # (480, 640)
        self.fc1 = nn.Linear(in_features=self.count_neurons(1, 80, 80), out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


def callback(lcl, _glb):
    # stop training if average reward exceeds 450
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 450
    return is_solved


def main():
    # unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    # create node
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)
    env = gym.make("Pickbot-v2")


    print("Saving model to pickbot_model_"+str(timestamp)+".pkl")
    # act.save("pickbot_model_"+str(timestamp)+".pkl")

if __name__ == '__main__':
    main()
