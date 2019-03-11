import gym

from stable_baselines import TRPO

import sys
import rospkg
import rospy
rospack = rospkg.RosPack()
Env_path = rospack.get_path('pickbot_training')+"/src/2_Environment"
sys.path.insert(0,Env_path)
import pickbot_env_npstate
import gazebo_connection


def main():
    # unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    # create node
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)

    env = gym.make('Pickbot-v0')

    model = TRPO.load("pickbot_model_trpo_discrete_2019-03-11 10:22:01")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        done = False
        episode_rew = 0
        while not done:
            obs, rewards, done, info = env.step(action)
            episode_rew += rewards
            print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
