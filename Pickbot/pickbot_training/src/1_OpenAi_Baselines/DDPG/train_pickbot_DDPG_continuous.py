import gym
import sys
import datetime
import rospkg
import rospy


#from baselines import ddpg
from baselines.ddpg import ddpg


from baselines.bench import Monitor
from baselines.common.cmd_util import make_vec_env, make_env

rospack = rospkg.RosPack()
Env_path=rospack.get_path('pickbot_training')+"/src/2_Environment"
sys.path.insert(0,Env_path)
import pickbot_env_continuous
import gazebo_connection

"""
make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None):
"""

num_env=1
env_id="Pickbot-v1"
env_type="classic_control"
seed=None


def main():
    #unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    #create node 
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)

    env = make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None)
    
    act=ddpg.learn(
        env=env,
        network='mlp',
        total_timesteps=100000
    )
    #Environment Object: <baselines.common.vec_env.dummy_vec_env.DummyVecEnv object at 0x7fbcfa356fd0>










if __name__ == '__main__':
    main()