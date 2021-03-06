import gym

from baselines.ppo2 import ppo2
from baselines.bench import Monitor
from baselines.common.cmd_util import make_vec_env, make_env


def main():
    num_env=1
    env_id="Pendulum-v0"
    env_type="classic_control"
    seed=None

    env = make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None)
    
    
    act=ppo2.learn(
        env=env,
        network='mlp',
        total_timesteps=0,
        load_path="Pendulum-v0.pkl"
    )

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
