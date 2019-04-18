import gym

from baselines.acer import acer
from baselines.bench import Monitor
from baselines.common.cmd_util import make_vec_env, make_env

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

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
    
    act=acer.learn(
        env=env,
        network='mlp',
        total_timesteps=80000
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")

if __name__ == '__main__':
    main()
