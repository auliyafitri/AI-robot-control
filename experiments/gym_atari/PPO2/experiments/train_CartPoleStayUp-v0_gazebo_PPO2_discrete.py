import gym

from baselines.ppo2 import ppo2
import rospy

from openai_ros.task_envs.cartpole_stay_up import stay_up
from baselines.common.cmd_util import make_vec_env, make_env
from baselines.bench import Monitor



num_env=1
env_id="CartPoleStayUp-v0"
env_type="classic_control"
seed=None


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    rospy.init_node('cartpole_gym', anonymous=True, log_level=rospy.FATAL)

    env = make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=Monitor,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None)

    act=ppo2.learn(
        env=env,
        network='mlp',
        total_timesteps=100000
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
