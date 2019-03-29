import gym
import sys
import datetime
import rospkg
import rospy
import sys
import datetime

from baselines import deepq
timestamp=datetime.datetime.now()

rospack = rospkg.RosPack()
Env_path=rospack.get_path('pickbot_training')+"/src/2_Environment"
sys.path.insert(0,Env_path)
import pickbot_env_npstate  
import gazebo_connection


def callback(lcl, _glb):
    # stop training if average reward exceeds 450
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 450
    return is_solved


def main():
    #unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    #create node 
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)
    env = gym.make("Pickbot-v0")
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
    )
    print("Saving model to pickbot_model_"+str(timestamp)+".pkl")
    act.save("pickbot_model_"+str(timestamp)+".pkl")

if __name__ == '__main__':
    main()
