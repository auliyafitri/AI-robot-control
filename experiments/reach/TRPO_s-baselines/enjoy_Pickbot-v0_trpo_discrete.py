import gym
import rospy

from environments import pickbot_env_npstate, gazebo_connection

from stable_baselines import TRPO


def main():
    # unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    # create node
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)

    env = gym.make('Pickbot-v0')

    model = TRPO.load("pickbot_model_trpo_discrete_2019-03-11 10:22:01")

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
