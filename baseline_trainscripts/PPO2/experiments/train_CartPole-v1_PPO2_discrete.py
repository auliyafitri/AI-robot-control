import gym
import os

from baselines.ppo2 import ppo2
from baselines.bench import Monitor
from baselines import logger
from baselines.common.cmd_util import make_vec_env, make_env


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    num_env = 5
    env_id = "CartPole-v1"
    env_type = "classic_control"
    seed = None

    # Create needed folders
    logdir = './log/' + env_id + '/trpo_mpi'

    # Generate tensorboard file
    format_strs = ['stdout', 'log', 'csv', 'tensorboard']
    logger.configure(os.path.abspath(logdir), format_strs)

    env = make_vec_env(env_id, env_type, num_env, seed,
                       wrapper_kwargs=None,
                       start_index=0,
                       reward_scale=1.0,
                       flatten_dict_observations=True,
                       gamestate=None)

    act = ppo2.learn(
        env=env,
        network='mlp',
        total_timesteps=40000,
        save_interval=2
    )
    print("Saving model to cartpole_model.pkl")
    act.save("./cartpole_model")


if __name__ == '__main__':
    main()
