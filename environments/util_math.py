import math


def compute_reward(observation, done_reward, invalid_contact):
    """
    Calculates the reward in each Step
    Reward for:
    Distance:       Reward for Distance to the Object
    Contact:        Reward for Contact with one contact sensor and invalid_contact must be false. As soon as both
                    contact sensors have contact and there is no invalid contact the goal is considered to be reached
                    and the episode is over. Reward is then set in is_done

    Calculates the Reward for the Terminal State
    Done Reward:    Reward when episode is Done. Negative Reward for Crashing and going into set Joint Limits.
                    High positive reward for having contact with both contact sensors and not having an invalid collision
    """
    reward_contact = 0

    # Reward for Distance to encourage approaching the box
    distance = observation[0]
    # reward_distance = 1 - math.pow(distance / max_distance, 0.4)
    relative_distance = observation[-1] - distance
    reward_distance = relative_distance * 20 if relative_distance < 0 else relative_distance * 10

    # Reward for Contact
    contact_1 = observation[7]
    contact_2 = observation[8]

    if contact_1 == 0 and contact_2 == 0:
        reward_contact = 0
    elif contact_1 != 0 and contact_2 == 0 and not invalid_contact or contact_1 == 0 and contact_2 != 0 and \
            not invalid_contact:
        reward_contact = 2000
        reward_distance = 0

    total_reward = reward_distance + reward_contact + done_reward

    # print("reward_distance: {}".format(reward_distance))

    return total_reward


def compute_reward_orient(observation, done_reward, invalid_contact):
    """
    Calculates the reward in each Step
    Reward for:
    Distance:       Reward for Distance to the Object
    Contact:        Reward for Contact with one contact sensor and invalid_contact must be false. As soon as both
                    contact sensors have contact and there is no invalid contact the goal is considered to be reached
                    and the episode is over. Reward is then set in is_done

    Calculates the Reward for the Terminal State
    Done Reward:    Reward when episode is Done. Negative Reward for Crashing and going into set Joint Limits.
                    High positive reward for having contact with both contact sensors and not having an invalid collision
    """

    # Reward for Distance to encourage approaching the box
    distance = observation[0]
    # reward_distance = 1 - math.pow(distance / max_distance, 0.4)
    relative_distance = observation[-2] - distance
    reward_distance = relative_distance * 20 if relative_distance < 0 else relative_distance * 10

    # Reward for orientation
    orient_differences = observation[-1]

    reward_orient = 0
    if not invalid_contact:
        reward_orient = (1 - orient_differences/math.pi) * 10

    total_reward = reward_distance * reward_orient + done_reward

    print("distance: {} orient:{} total:{}".format(reward_distance, reward_orient, total_reward))

    return total_reward
