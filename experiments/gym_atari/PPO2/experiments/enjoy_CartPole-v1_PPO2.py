import gym
import numpy as np

from baselines.ppo2 import ppo2
from baselines.bench import Monitor
from baselines.common.cmd_util import make_vec_env, make_env
from baselines.common.vec_env import VecEnv


def main():
    num_env=1
    env_id="CartPole-v1"
    env_type="classic_control"
    seed=None

    env = make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None)
    
    print(env)
    act = ppo2.learn(
        env=env,
        network='mlp',
        total_timesteps=0,
        load_path="./cartpole_model"
    )

    obs, done = env.reset(), False
    episode_rew = 0

    while True:
        # actions, _, _, _ = act.step(obs)

        obs, rew, done, _ = env.step(act.step(obs)[0])
        episode_rew += rew[0] if isinstance(env, VecEnv) else rew
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            print('episode_rew={}'.format(episode_rew))
            episode_rew = 0
            obs = env.reset()

        # while not done:
        #     env.render()
        #     obs, rew, done, _ = env.step(act(obs[None])[0])
        #     episode_rew += rew
        # print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
