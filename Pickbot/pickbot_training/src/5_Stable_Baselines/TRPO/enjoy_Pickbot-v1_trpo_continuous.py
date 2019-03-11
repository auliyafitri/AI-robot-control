import gym

from stable_baselines import TRPO

import sys
import rospkg
import rospy
rospack = rospkg.RosPack()
Env_path = rospack.get_path('pickbot_training')+"/src/2_Environment"
sys.path.insert(0,Env_path)
import pickbot_env_continuous
import gazebo_connection


def main():
    # unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    # create node
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)

    env = gym.make('Pickbot-v1')

    model = TRPO.load("pickbot_model_trpo_continuous_2019-03-11 10:37:55")

    while True:
        obs, done = env.reset(), False
        action, _states = model.predict(obs)
        episode_rew = 0
        while not done:
            obs, rewards, done, info = env.step(action)
            episode_rew += rewards
            print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
