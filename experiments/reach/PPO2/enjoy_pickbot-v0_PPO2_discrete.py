import gym
import sys
import datetime
import rospkg
import rospy
import numpy as np

from baselines.ppo2 import ppo2
from baselines.bench import Monitor
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env import VecEnv

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
        total_timesteps=0,
        load_path="pickbot_model_2019-03-26_08h50min"
    )

    obs, done = env.reset(), False
    episode_rew = 0

    while True:
        obs, rew, done, _ = env.step(act.step(obs)[0])
        episode_rew += rew[0] if isinstance(env, VecEnv) else rew
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            print('episode_rew={}'.format(episode_rew))
            episode_rew = 0
            obs = env.reset()


if __name__ == '__main__':
    main()
