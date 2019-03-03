import gym

from baselines.acktr import acktr
from baselines.bench import Monitor
from baselines.common.cmd_util import make_vec_env, make_env


def main():
    num_env=1
    env_id="Pendulum-v0"
    env_type="classic_control"
    seed=1

    env = make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None)
    
    act=acktr.learn(
        env=env,
        network='mlp',
        total_timesteps=50000,
        seed=seed
    )
    print("Saving model to pendulum_model.pkl")
    act.save("pendulum_model.pkl")

if __name__ == '__main__':
    main()
