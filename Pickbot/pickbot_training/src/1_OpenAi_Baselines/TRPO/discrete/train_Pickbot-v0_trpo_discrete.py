import gym
import os
import sys
import datetime
import rospkg
import rospy

from baselines.trpo_mpi import trpo_mpi, defaults
from baselines.common.models import mlp
from baselines.bench import Monitor
from baselines import logger
from baselines.common.cmd_util import make_vec_env

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')
rospack = rospkg.RosPack()
Env_path = rospack.get_path('pickbot_training') + "/src/2_Environment"
sys.path.insert(0, Env_path)
import pickbot_env_npstate
import gazebo_connection


num_env = 1
env_id = "Pickbot-v0"
env_type = "classic_control"
seed = None

# Create needed folders
logdir = './log/' + env_id + '/trpo_mpi/' + timestamp

# Generate tensorboard file
format_strs = ['stdout', 'log', 'csv', 'tensorboard']
logger.configure(os.path.abspath(logdir), format_strs)

def make_env(alg_kwargs):

    # num_env = 1
    # env_id = "Pickbot-v0"
    # env_type = "classic_control"
    # seed = None
    #
    # env = make_vec_env(env_id, env_type, num_env, seed,
    #                    wrapper_kwargs=None,
    #                    start_index=0,
    #                    reward_scale=1.0,
    #                    flatten_dict_observations=True,
    #                    gamestate=None)

    env = gym.make(alg_kwargs['env_name'])
    # env.set_episode_size(alg_kwargs['timesteps_per_batch'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env


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

    act = trpo_mpi.learn(
        env=env,
        network='mlp',
        total_timesteps=3000
    )
    print("Saving model to pickbot_model_" + str(timestamp))
    act.save("./pickbot_model_" + str(timestamp))

    """
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    # logger.session().__enter__()
    

    # Get dictionary from baselines/trpo_mpi/defaults
    alg_kwargs = defaults.pickbot_mlp()

    # Create needed folders
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')
    logdir = '/tmp/roslearn/' + alg_kwargs['env_name'] + '/trpo_mpi/' + timestamp

    # Generate tensorboard file
    format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)

    with open(logger.get_dir() + "/parameters.txt", 'w') as out:
        out.write(
            'num_layers = ' + str(alg_kwargs['num_layers']) + '\n'
            + 'num_hidden = ' + str(alg_kwargs['num_hidden']) + '\n'
            + 'layer_norm = ' + str(alg_kwargs['layer_norm']) + '\n'
            + 'timesteps_per_batch = ' + str(alg_kwargs['timesteps_per_batch']) + '\n'
            + 'max_kl = ' + str(alg_kwargs['max_kl']) + '\n'
            + 'cg_iters = ' + str(alg_kwargs['cg_iters']) + '\n'
            + 'cg_damping = ' + str(alg_kwargs['cg_damping']) + '\n'
            + 'total_timesteps = ' + str(alg_kwargs['total_timesteps']) + '\n'
            + 'gamma = ' + str(alg_kwargs['gamma']) + '\n'
            + 'lam = ' + str(alg_kwargs['lam']) + '\n'
            + 'seed = ' + str(alg_kwargs['seed']) + '\n'
            + 'ent_coef = ' + str(alg_kwargs['ent_coef']) + '\n'
            + 'vf_iters = ' + str(alg_kwargs['vf_iters']) + '\n'
            + 'vf_stepsize = ' + str(alg_kwargs['vf_stepsize']) + '\n'
            + 'normalize_observations = ' + str(alg_kwargs['normalize_observations']) + '\n'
            + 'env_name = ' + alg_kwargs['env_name'] + '\n'
            + 'transfer_path = ' + str(alg_kwargs['transfer_path']))

    env = make_env(alg_kwargs)
    transfer_path = alg_kwargs['transfer_path']

    # Remove unused parameters for training
    alg_kwargs.pop('env_name')
    alg_kwargs.pop('trained_path')
    alg_kwargs.pop('transfer_path')

    network = mlp(num_layers=alg_kwargs['num_layers'], num_hidden=alg_kwargs['num_hidden'],
                  layer_norm=alg_kwargs['layer_norm'])

    if transfer_path is not None:
        # Do transfer learning
        trpo_mpi.learn(env=env, network=network, load_path=transfer_path, **alg_kwargs)
    else:
        trpo_mpi.learn(env=env, network=network, **alg_kwargs)

    
    saver = tf.train.Saver()
    saver.save(sess, '/tmp/model')

    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    #
    # with tf.Session() as ses:
    #     saver.restore(ses, '/tmp/model')
    #     obs, done = env.reset(), False
    #     episode_rew = 0
    #     while not done:
    #         obs, rew, done, _ = env.step(act(obs[None])[0])
    #         episode_rew += rew
    #     print("Episode reward", episode_rew)
    """


if __name__ == '__main__':
    main()
