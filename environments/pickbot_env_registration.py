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

# MoveIt Env
register(
    id='PickbotReachContinuous-v0',
    entry_point='environments.pickbot_reach_env:PickbotEnv',
    max_episode_steps=300,
)

# MoveIt Env
register(
    id='PickbotReachContinuous-v1',
    entry_point='environments.pickbot_reach_env:PickbotEnv',
    kwargs={
        'random_object': False,
        'random_position': True,
    },
    max_episode_steps=300,
)

# MoveIt Env
register(
    id='PickbotReachContinuous-v2',
    entry_point='environments.pickbot_reach_env:PickbotEnv',
    kwargs={
        'random_object': True,
        'random_position': False,
        'use_object_type': True,
        'populate_object': True,
    },
    max_episode_steps=300,
)

# MoveIt Env
register(
    id='PickbotReachContinuous-v3',
    entry_point='environments.pickbot_reach_env:PickbotEnv',
    kwargs={
        'random_object': True,
        'random_position': True,
        'use_object_type': True,
        'populate_object': True,
    },
    max_episode_steps=300,
)

register(
    id='PickbotReachContinuousDoorHandle-v0',
    entry_point='environments.pickbot_env_continuous:PickbotEnv',
    kwargs={
        'env_object_type': 'door_handle',
    },
    max_episode_steps=300,
)

register(
    id='PickbotReachContinuousDoorHandle-v1',
    entry_point='environments.pickbot_env_continuous:PickbotEnv',
    kwargs={
        'random_object': False,
        'random_position': True,
        'env_object_type': 'door_handle',
        'joint_increment': None,
        'sim_time_factor': 0.001,
    },
    max_episode_steps=300,
)

register(
    id='PickbotReachContinuousDoorHandle-v2',
    entry_point='environments.pickbot_env_continuous:PickbotEnv',
    kwargs={
        'random_object': True,
        'random_position': False,
        'use_object_type': True,
        'populate_object': True,
        'env_object_type': 'door_handle',
    },
    max_episode_steps=300,
)

register(
    id='PickbotReachContinuousDoorHandle-v3',
    entry_point='environments.pickbot_env_continuous:PickbotEnv',
    kwargs={
        'random_object': True,
        'random_position': True,
        'use_object_type': True,
        'populate_object': True,
        'env_object_type': 'door_handle',
    },
    max_episode_steps=300,
)

# Combox
register(
    id='PickbotReachDiscreteCombox-v0',
    entry_point='environments.pickbot_env_npstate:PickbotEnv',
    kwargs={
        'env_object_type': 'combox',
    },
    max_episode_steps=300,
)

register(
    id='PickbotReachContinuousCombox-v0',
    entry_point='environments.pickbot_env_continuous:PickbotEnv',
    kwargs={
        'env_object_type': 'combox',
    },
    max_episode_steps=300,
)

# Pick
register(
    id='PickbotPickDiscreteDoorHandle-v1',
    entry_point='environments.pickbot_lift_npstate:PickbotEnv',
    kwargs={
        'env_object_type': 'door_handle',
        'random_position': False,
        'sim_time_factor': 0.001,
        'joint_increment_value': 0.174
    },
    max_episode_steps=500,
)
