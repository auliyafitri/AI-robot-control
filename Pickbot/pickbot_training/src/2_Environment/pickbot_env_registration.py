import sys
import rospkg

from gym.envs.registration import register

rospack = rospkg.RosPack()
Env_path = rospack.get_path('pickbot_training') + "/src/2_Environment"
sys.path.insert(0, Env_path)
from pickbot_env_npstate import PickbotEnv

register(
    id='PickbotReach-v0',
    entry_point='pickbot_env_npstate:PickbotEnv',
    max_episode_steps=120,
)

register(
    id='PickbotReach-v1',
    entry_point='pickbot_env_npstate:PickbotEnv',
    kwargs={
        'random_object': False,
        'random_position': True,
    },
    max_episode_steps=120,
)

register(
    id='PickbotReach-v2',
    entry_point='pickbot_env_npstate:PickbotEnv',
    kwargs={
        'random_object': True,
        'random_position': False,
        'use_object_type': True,
        'populate_object': True,
    },
    max_episode_steps=120,
)

register(
    id='PickbotReach-v3',
    entry_point='pickbot_env_npstate:PickbotEnv',
    kwargs={
        'random_object': True,
        'random_position': True,
        'use_object_type': True,
        'populate_object': True,
    },
    max_episode_steps=120,
)
