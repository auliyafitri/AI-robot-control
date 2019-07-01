import gym
import os
import sys
import datetime
import rospkg
import rospy

from baselines.ppo2 import ppo2
from baselines.bench import Monitor
from baselines import logger
from baselines.common.cmd_util import make_vec_env

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')
rospack = rospkg.RosPack()
Env_path = rospack.get_path('pickbot_training') + "/src/2_Environment"
sys.path.insert(0, Env_path)
import pickbot_env_npstate
import gazebo_connection

"""
make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None):
"""

num_env = 1
env_id = "Pickbot-v0"
env_type = "classic_control"
seed = None

# Create needed folders
logdir = './log/' + env_id + '/ppo2/' + timestamp

# Generate tensorboard file
format_strs = ['stdout', 'log', 'csv', 'tensorboard']
logger.configure(os.path.abspath(logdir), format_strs)


def main():
    # unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    # create node
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)

    env = make_vec_env(env_id, env_type, num_env, seed,
                       wrapper_kwargs=Monitor,
                       start_index=0,
                       reward_scale=1.0,
                       flatten_dict_observations=True,
                       gamestate=None)

    act = ppo2.learn(
        env=env,
        network='mlp',
        total_timesteps=10000
    )
    print("Saving model to pickbot_model_" + str(timestamp))
    act.save("./pickbot_model_" + str(timestamp))
    # Environment Object: <baselines.common.vec_env.dummy_vec_env.DummyVecEnv object at 0x7fbcfa356fd0>


if __name__ == '__main__':
    main()
