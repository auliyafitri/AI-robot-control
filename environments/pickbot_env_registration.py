import sys
import rospkg

from gym.envs.registration import register

from environments.pickbot_env_npstate import PickbotEnv
from environments.pickbot_env_continuous import PickbotEnv

register(
    id='PickbotReach-v0',
    entry_point='environments.pickbot_env_npstate:PickbotEnv',
    max_episode_steps=240,
)

register(
    id='PickbotReach-v1',
    entry_point='environments.pickbot_env_npstate:PickbotEnv',
    kwargs={
        'random_object': False,
        'random_position': True,
    },
    max_episode_steps=240,
)

register(
    id='PickbotReach-v2',
    entry_point='environments.pickbot_env_npstate:PickbotEnv',
    kwargs={
        'random_object': True,
        'random_position': False,
        'use_object_type': True,
        'populate_object': True,
    },
    max_episode_steps=240,
)

register(
    id='PickbotReach-v3',
    entry_point='environments.pickbot_env_npstate:PickbotEnv',
    kwargs={
        'random_object': True,
        'random_position': True,
        'use_object_type': True,
        'populate_object': True,
    },
    max_episode_steps=240,
)

register(
    id='PickbotReachContinuous-v0',
    entry_point='environments.pickbot_env_continuous:PickbotEnv',
    max_episode_steps=300,
)

register(
    id='PickbotReachContinuous-v1',
    entry_point='environments.pickbot_env_continuous:PickbotEnv',
    kwargs={
        'random_object': False,
        'random_position': True,
    },
    max_episode_steps=300,
)

register(
    id='PickbotReachContinuous-v2',
    entry_point='environments.pickbot_env_continuous:PickbotEnv',
    kwargs={
        'random_object': True,
        'random_position': False,
        'use_object_type': True,
        'populate_object': True,
    },
    max_episode_steps=300,
)

register(
    id='PickbotReachContinuous-v3',
    entry_point='environments.pickbot_env_continuous:PickbotEnv',
    kwargs={
        'random_object': True,
        'random_position': True,
        'use_object_type': True,
        'populate_object': True,
    },
    max_episode_steps=300,
)