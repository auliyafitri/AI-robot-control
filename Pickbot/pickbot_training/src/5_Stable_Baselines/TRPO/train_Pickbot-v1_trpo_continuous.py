import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO

import sys
import rospkg
import rospy
rospack = rospkg.RosPack()
Env_path = rospack.get_path('pickbot_training')+"/src/2_Environment"
sys.path.insert(0,Env_path)
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

    model = TRPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=200000)

    print("Saving model to pickbot_model_trpo_continuous_"+timestamp+".pkl")
    model.save("pickbot_model_trpo_continuous_"+timestamp)


if __name__ == '__main__':
    main()