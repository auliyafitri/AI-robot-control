import gym
import rospy
import datetime
import os

import evaluations
import models

from environments import gazebo_connection
from environments.pickbot_reach_cam_env import PickbotReachCamEnv
from baselines import logger

from baselines import deepq
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')


task_name = "reach_cnn"
logdir = os.path.dirname(evaluations.__file__) + '/' + task_name + '/deepQ' + '/' + timestamp + '/'
format_strs = ['stdout', 'log', 'csv', 'tensorboard']
logger.configure(os.path.abspath(logdir), format_strs)

modelsdir = os.path.dirname(models.__file__) + '/' + task_name + '/deepQ' + '/' + timestamp + '/'

def callback(lcl, _glb):
    # stop training if average reward exceeds 450
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 450
    return is_solved


def main():
    # unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)

    env = PickbotReachCamEnv(is_discrete=True, random_position=True)
    model = deepq.models.cnn_to_mlp(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1), (64, 3, 1), (128, 3, 1)],
                                    hiddens=[512, 256],
                                    dueling=False)
    act = deepq.learn(
        env,
        network=model,
        lr=1e-3,
        total_timesteps=1000000000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=100,
        checkpoint_path=modelsdir,
        callback=callback
    )
    print("Saving model to kuka_cam_model.pkl")
    act.save("pickbot_cam_model.pkl")


if __name__ == '__main__':
    main()
