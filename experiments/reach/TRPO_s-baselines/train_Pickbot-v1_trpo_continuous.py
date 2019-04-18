import gym
import rospy

from environments import pickbot_env_continuous, gazebo_connection

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO


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