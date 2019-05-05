import gym
import os
import datetime
import rospy

from environments import pickbot_env_registration, gazebo_connection
import evaluations
import models

from baselines.trpo_mpi import trpo_mpi
from baselines.bench import Monitor
from baselines import logger
from baselines.common.cmd_util import make_vec_env

from environments.joint_array_publisher import JointArrayPub
from std_msgs.msg import String

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')


num_env = 1
env_id = "PickbotReachContinuous-v1"
env_type = "classic_control"
seed = None
task_name = "reach"

# Create needed folders for log file and models
logdir = os.path.dirname(evaluations.__file__) + '/' + task_name + '/trpo/' + env_id + '/' + timestamp + '/'
modelsdir = os.path.dirname(models.__file__) + '/' + task_name + '/trpo/' + env_id + '/' + timestamp + '/'

# Generate tensorboard file
format_strs = ['stdout', 'log', 'csv', 'tensorboard']
logger.configure(os.path.abspath(logdir), format_strs)


def mov_complete_callback(data):
    msg = data.data
    print(msg)


def main():
    # unpause Simulation so that robot receives data on all topics
    gazebo_connection.GazeboConnection().unpauseSim()
    # create node
    rospy.init_node('pickbot_gym', anonymous=True, log_level=rospy.FATAL)

    env = make_vec_env(env_id, env_type, num_env, seed,
                       wrapper_kwargs=Monitor,
                       start_index=0,
                       reward_scale=1.0,
                       flatten_dict_observations=True,
                       gamestate=None)



    a = [1.5,-1.2,1.4,-1.87,-1.57,0]
    b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # rate = rospy.Rate(10)

    jointPub = JointArrayPub()
    mov_sub = rospy.Subscriber('/pickbot/movement_complete/', String, mov_complete_callback)


    # for i in range(10):
    #     jointPub.pub_joints_to_moveit(a)
    #     jointPub.pub_joints_to_moveit(b)
    #     rate.sleep()
    jointPub.pub_joints_to_moveit(a)

    print(">>>>>>>>>>>>>>>>>>> I am waiting for a ros message")
    rospy.wait_for_message("/pickbot/movement_complete", String)
    print(">>>>>>>>>>>>>>>>>>> Waiting complete")

    jointPub.pub_joints_to_moveit(b)

    print(">>>>>>>>>>>>>>>>>>> I am waiting for a ros message")
    rospy.wait_for_message("/pickbot/movement_complete", String)
    print(">>>>>>>>>>>>>>>>>>> Waiting complete")

    act = trpo_mpi.learn(
        env=env,
        network='mlp',
        total_timesteps=1000000,
        save_path=modelsdir
    )

    print("Saving model to " + modelsdir)
    act.save(modelsdir + "model")


if __name__ == '__main__':
    main()
