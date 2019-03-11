import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

import sys
import rospy
import rospkg
rospack = rospkg.RosPack()
Env_path = rospack.get_path('pickbot_training')+"/src/2_Environment"
sys.path.insert(0, Env_path)
import pickbot_env_continuous
import gazebo_connection

import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    # unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    # create node
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)

    env = gym.make('Pickbot-v1')

    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=200000)

    print("Saving model to pickbot_model_ddpg_continuous_" + timestamp + ".pkl")
    model.save("pickbot_model_ddpg_continuous_" + timestamp)


if __name__ == '__main__':
    main()
