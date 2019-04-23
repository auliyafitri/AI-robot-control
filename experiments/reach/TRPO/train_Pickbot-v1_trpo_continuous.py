import rospy

from environments import pickbot_env_continuous, gazebo_connection

from baselines.trpo_mpi import trpo_mpi
from baselines.bench import Monitor
from baselines.common.cmd_util import make_vec_env, make_env


def main():
    gazebo_connection.GazeboConnection().unpauseSim()
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)

    num_env=1
    env_id="Pickbot-v1"
    env_type="classic_control"
    seed=None

    env = make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None)
    
    
    act=trpo_mpi.learn(
        env=env,
        network='mlp',
        total_timesteps=2000000
    )
    


if __name__ == '__main__':
    main()
