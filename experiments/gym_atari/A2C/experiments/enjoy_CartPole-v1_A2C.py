import gym

from baselines.a2c import a2c
from baselines.bench import Monitor
from baselines.common.cmd_util import make_vec_env, make_env


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
    
    act = a2c.learn(env, network='mlp', total_timesteps=0, load_path="cartpole_model.pkl")

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
