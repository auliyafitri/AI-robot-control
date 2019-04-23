import rospy
import gym
from environments import pickbot_env_continuous, gazebo_connection

from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG


def main():
    # unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    # create node
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)

    env = gym.make('Pickbot-v1')

    model = DDPG.load("pickbot_model_ddpg_continuous_2019-03-11 12:45:38")

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
