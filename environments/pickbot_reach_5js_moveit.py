#!/usr/bin/env python

# IMPORT
import gym
import rospy
import numpy as np
import sys
import os
import yaml
import math
import time
import random
import datetime
import rospkg
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
from transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply, quaternion_conjugate

# OTHER FILES
import environments.util_env as U
import environments.util_math as UMath
from environments.gazebo_connection import GazeboConnection
from environments.controllers_connection import ControllersConnection
from environments.joint_publisher import JointPub
from environments.joint_array_publisher import JointArrayPub
from baselines import logger


# MESSAGES/SERVICES
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import Image
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Point, Quaternion, Vector3, Pose
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from openai_ros.msg import RLExperimentInfo
from moveit_msgs.msg import MoveGroupActionFeedback

from simulation.msg import VacuumGripperState
from simulation.srv import VacuumGripperControl


# DEFINE ENVIRONMENT CLASS
class PickbotEnv(gym.Env):

    def __init__(self, joint_increment_value=0.02, sim_time_factor=0.001, running_step=0.001, random_object=False,
                 random_position=False, use_object_type=False, env_object_type='free_shapes', load_init_pos=False, added_target_height=0):
        """
        initializing all the relevant variables and connections
        :param joint_increment_value: increment of the joints
        :param running_step: gazebo simulation time factor
        :param random_object: spawn random object in the simulation
        :param random_position: change object position in each reset
        :param use_object_type: assign IDs to objects and used them in the observation space
        :param env_object_type: object type for environment, free_shapes for boxes while others are related to use_case
            'door_handle', 'combox', ...
        """

        # Assign Parameters
        self._joint_increment_value = joint_increment_value
        self.running_step = running_step
        self._random_object = random_object
        self._random_position = random_position
        self._use_object_type = use_object_type
        self._load_init_pos = load_init_pos
        self._added_target_height = added_target_height

        # Assign MsgTypes
        self.joints_state = JointState()
        self.contact_1_state = ContactsState()
        self.contact_2_state = ContactsState()
        self.collision = Bool()
        self.camera_rgb_state = Image()
        self.camera_depth_state = Image()
        self.contact_1_force = Vector3()
        self.contact_2_force = Vector3()
        self.gripper_state = VacuumGripperState()
        
        # MoveIt! ======
        self.movement_complete = Bool()
        self.movement_complete.data = False
        self.moveit_action_feedback = MoveGroupActionFeedback()
        self.feedback_list = []
        # ==============

        self._list_of_observations = ["elbow_joint_state",
                                      "shoulder_lift_joint_state",
                                      "shoulder_pan_joint_state",
                                      "wrist_1_joint_state",
                                      "wrist_2_joint_state",
                                      "wrist_3_joint_state",
                                      "vacuum_gripper_pos_x",
                                      "vacuum_gripper_pos_y",
                                      "vacuum_gripper_pos_z",
                                      "vacuum_gripper_ori_w",
                                      "vacuum_gripper_ori_x",
                                      "vacuum_gripper_ori_y",
                                      "vacuum_gripper_ori_z",
                                      "object_pos_x",
                                      "object_pos_y",
                                      "object_pos_z",
                                      "object_ori_w",
                                      "object_ori_x",
                                      "object_ori_y",
                                      "object_ori_z",
                                      ]

        # if self._use_object_type:
        #     self._list_of_observations.append("object_type")

        # Establishes connection with simulator
        """
        1) Gazebo Connection 
        2) Controller Connection
        3) Joint Publisher 
        """
        self.gazebo = GazeboConnection(sim_time_factor=sim_time_factor)
        self.controllers_object = ControllersConnection()
        self.pickbot_joint_publisher_object = JointPub()
        self.publisher_to_moveit_object = JointArrayPub()

        # Define Subscribers as Sensordata
        """
        1) /pickbot/joint_states
        2) /gripper_contactsensor_1_state
        3) /gripper_contactsensor_2_state
        4) /gz_collisions
        5) /pickbot/gripper/state
        6) /camera_rgb/image_raw   
        7) /camera_depth/depth/image_raw
        """

        # PID Controller
        # rospy.Subscriber("/pickbot/joint_states", JointState, self.joints_state_callback)

        # MoveIt! Controller
        rospy.Subscriber("/joint_states", JointState, self.joints_state_callback)
        rospy.Subscriber("/pickbot/movement_complete", Bool, self.movement_complete_callback)
        rospy.Subscriber("/move_group/feedback", MoveGroupActionFeedback, self.move_group_action_feedback_callback, queue_size=4)
        
        rospy.Subscriber("/gripper_contactsensor_1_state", ContactsState, self.contact_1_callback)
        rospy.Subscriber("/gripper_contactsensor_2_state", ContactsState, self.contact_2_callback)
        rospy.Subscriber("/gz_collisions", Bool, self.collision_callback)
        # rospy.Subscriber("/pickbot/gripper/state", VacuumGripperState, self.gripper_state_callback)
        # rospy.Subscriber("/camera_rgb/image_raw", Image, self.camera_rgb_callback)
        # rospy.Subscriber("/camera_depth/depth/image_raw", Image, self.camera_depth_callback)

        # Define Action and state Space and Reward Range
        """
        Action Space: Box Space with 6 values.
        
        State Space: Box Space with 20 values. It is a numpy array with shape (20,)

        Reward Range: -infinity to infinity 
        """

        # Use first 5 of joint as action
        # Directly use joint_positions as action
        if self._joint_increment_value is None:
            high_action = (math.pi - 0.05) * np.ones(5)
            low_action = -high_action
        else:  # Use joint_increments as action
            high_action = self._joint_increment_value * np.ones(5)
            low_action = -high_action

        self.action_space = spaces.Box(low_action, high_action)

        self.obs_dim = 20
        high = np.inf * np.ones(self.obs_dim)
        low = -high

        self.observation_space = spaces.Box(low, high)

        # if self._use_object_type:
        #     high = np.append(high, 9)
        #     low = np.append(low, 0)

        self.reward_range = (-np.inf, np.inf)

        self._seed()
        self.done_reward = 0

        # set up everything to publish the Episode Number and Episode Reward on a rostopic
        self.episode_num = 0
        self.accumulated_episode_reward = 0
        self.episode_steps = 0
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
        self.reward_list = []
        self.episode_list = []
        self.step_list = []
        self.csv_name = logger.get_dir() + '/result_log'
        print("CSV NAME")
        print(self.csv_name)
        self.csv_success_exp = logger.get_dir() + '/success_exp' + datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mmin') + '.csv'
        self.successful_attempts = 0

        # variable to store last observation
        self.old_obs = self.get_obs()

        # object name: name of the target object
        # object type: index of the object name in the object list
        # object list: pool of the available objects, have at least one entry
        self.object_name = ''
        self.object_type_str = ''
        self.object_type = 0
        self.object_list = U.get_target_object(env_object_type)
        print("object list {}".format(self.object_list))
        self.object_initial_position = Pose(position=Point(x=-0.13, y=0.848, z=1.06),
                                            orientation=quaternion_from_euler(0.002567, 0.102, 1.563))

        # select first object, set object name and object type
        # if object is random, spawn random object
        # else get the first entry of object_list
        # self.set_target_object([0, 0, 0, 0, 0, 0])
        self.set_target_object(random_object=self._random_object, random_position=self._random_position)

        # get maximum distance to the object to calculate reward, renewed in the reset function
        self.max_distance, _ = U.get_distance_gripper_to_object()
        # The closest distance during training
        self.min_distance = 999

        # get samples from reaching task
        if self._load_init_pos:
            import environments
            self.init_samples = U.load_samples_from_prev_task(os.path.dirname(environments.__file__) +
                                                              "/contacts_sample/door_sample/success_exp2019-05-21_11h41min.csv")

    # Callback Functions for Subscribers to make topic values available each time the class is initialized
    def joints_state_callback(self, msg):
        self.joints_state = msg

    def contact_1_callback(self, msg):
        self.contact_1_state = msg.states

    def contact_2_callback(self, msg):
        self.contact_2_state = msg.states

    def collision_callback(self, msg):
        self.collision = msg.data

    def camera_rgb_callback(self, msg):
        self.camera_rgb_state = msg

    def camera_depth_callback(self, msg):
        self.camera_depth_state = msg

    def gripper_state_callback(self, msg):
        self.gripper_state = msg

    def movement_complete_callback(self, msg):
        self.movement_complete = msg

    def move_group_action_feedback_callback(self, msg):
        self.moveit_action_feedback = msg

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset The Robot to its initial Position and restart the Controllers 
        1) Publish the initial joint_positions to MoveIt
        2) Busy waiting until the movement is completed by MoveIt
        3) set target_object to random position
        4) Check all Systems work
        5) Create YAML Files for contact forces in order to get the average over 2 contacts
        6) Create YAML Files for collision to make shure to see a collision due to high noise in topic
        7) Get Observations and return current State
        8) Publish Episode Reward and set accumulated reward back to 0 and iterate the Episode Number
        9) Return State
        """

        init_joint_pos = [0, -1.57, 1.57, 0, 0, 0]
        self.publisher_to_moveit_object.set_joints(init_joint_pos)

        while not self.movement_complete.data:
            pass
        # print(">>>>>>>>>>>>>>>>>>> RESET: Waiting complete")

        start_ros_time = rospy.Time.now()
        while True:
            elapsed_time = rospy.Time.now() - start_ros_time
            if np.isclose(init_joint_pos, self.joints_state.position, rtol=0.0, atol=0.01).all():
                break
            elif elapsed_time > rospy.Duration(2): # time out
                break

        self.set_target_object(random_object=self._random_object, random_position=self._random_position)
        self._check_all_systems_ready()

        with open('contact_1_force.yml', 'w') as yaml_file:
            yaml.dump(0.0, yaml_file, default_flow_style=False)
        with open('contact_2_force.yml', 'w') as yaml_file:
            yaml.dump(0.0, yaml_file, default_flow_style=False)
        with open('collision.yml', 'w') as yaml_file:
            yaml.dump(False, yaml_file, default_flow_style=False)
        observation = self.get_obs()
        # self.object_position = observation[9:12]

        state = U.get_state(observation)
        self._update_episode()
        return state
    
    def step(self, action):
        """
        Given the action selected by the learning algorithm,
        we perform the corresponding movement of the robot
        return: the state of the robot, the corresponding reward for the step and if its done(terminal State)

        1) Read last joint positions by getting the observation before acting
        2) Get the new joint positions according to chosen action (actions here are the joint increments)
        3) Publish the joint_positions to MoveIt, meanwhile busy waiting, until the movement is complete
        4) Get new observation after performing the action
        5) Convert Observations into States
        6) Check if the task is done or crashing happens, calculate done_reward and pause Simulation again
        7) Calculate reward based on Observatin and done_reward
        8) Return State, Reward, Done
        """

        print("====================================================================")

        self.movement_complete.data = False

        # 1) Read last joint positions by getting the observation before acting
        self.old_obs = self.get_obs()
        last_position = self.old_obs[:6]
        
        # 2) Get the new joint positions according to chosen action (actions here are the joint increments)
        if self._joint_increment_value is None:
            next_action_position = np.append(action, last_position[-1])
        else:
            next_action_position = self.get_action_to_position(action, last_position)
        print("next action position: {}".format(np.around(next_action_position, decimals=3)))

        # 3) Move to position and wait for moveit to complete the execution
        self.publisher_to_moveit_object.pub_joints_to_moveit(next_action_position)
        while not self.movement_complete.data:
            pass

        # Busy waiting until all the joints reach the next_action_position (first the third joints are reversed)
        start_ros_time = rospy.Time.now()
        while True:
            # Check collision:
            # invalid_collision = self.get_collisions()
            # if invalid_collision:
            #     print(">>>>>>>>>> Collision: RESET <<<<<<<<<<<<<<<")
            #     observation = self.get_obs()
            #     print("joints after reset collision : {} ".format(observation[:6]))

            #     # calculate reward immediately
            #     distance_error = observation[6:9] - observation[13:16]
            #     orientation_error = quaternion_multiply(observation[9:13], quaternion_conjugate(observation[16:]))

            #     rewardDist = UMath.rmseFunc(distance_error)
            #     # rewardOrientation = 2 * np.arccos(abs(orientation_error[0]))

            #     reward = UMath.computeReward(rewardDist, invalid_collision)
            #     print("Reward this step after colliding {}".format(reward))
            #     self.accumulated_episode_reward += reward
            #     return U.get_state(observation), reward, True, {}

            elapsed_time = rospy.Time.now() - start_ros_time
            if np.isclose(next_action_position, self.joints_state.position, rtol=0.0, atol=0.01).all():
                break
            elif elapsed_time > rospy.Duration(2):  # time out
                print("TIME OUT, joints haven't reach positions")
                break

        # 4) Get new observation
        observation = self.get_obs()
        

        # 5) Convert Observations into state
        state = U.get_state(observation)

        # 6) Check if its done, calculate done_reward
        done, done_reward, invalid_contact = self.is_done(observation)

        # 7) Calculate reward based on Observation and done_reward and update the accumulated Episode Reward
        distance_error = observation[6:9] - observation[13:16]
        orientation_error = quaternion_multiply(observation[9:13], quaternion_conjugate(observation[16:]))

        rewardDist = UMath.rmseFunc(distance_error)
        rewardOrientation = 2 * np.arccos(abs(orientation_error[0]))

        reward = UMath.computeReward(rewardDist, rewardOrientation, invalid_contact) + done_reward
        print("Reward this step {}".format(reward))

        self.accumulated_episode_reward += reward
        self.episode_steps += 1

        # 8) Return State, Reward, Done
        return state, reward, done, {}

    def _check_all_systems_ready(self):
        """
        Checks that all subscribers for sensortopics are working

        1) /pickbot/joint_states
        2) /gripper_contactsensor_1_state
        3) /gripper_contactsensor_2_state
        7) Collisions

        not used so far
        4) /camera_rgb/image_raw
        5) /camera_depth/depth/image_raw

        """
        self.check_joint_states()
        self.check_contact_1()
        self.check_contact_2()
        self.check_collision()
        # self.check_rgb_camera()
        # self.check_rgbd_camera()
        # self.check_gripper_state()
        rospy.logdebug("ALL SYSTEMS READY")

    def check_joint_states(self):
        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                # joint_states_msg = rospy.wait_for_message("/pickbot/joint_states", JointState, timeout=0.1)
                joint_states_msg = rospy.wait_for_message("/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                rospy.logdebug("Current joint_states not ready yet, retrying==>" + str(e))
                print("EXCEPTION: Joint States not ready yet, retrying.")

    def check_contact_1(self):
        contact_1_states_msg = None
        while contact_1_states_msg is None and not rospy.is_shutdown():
            try:
                contact_1_states_msg = rospy.wait_for_message("/gripper_contactsensor_1_state", ContactsState,
                                                              timeout=0.1)
                self.contact_1_state = contact_1_states_msg.states
                rospy.logdebug("Contactsensor 1 READY")
            except Exception as e:
                rospy.logdebug("Contactsensor 1 not ready yet, retrying==>" + str(e))
                print("EXCEPTION: Contactsensor 1 not ready yet, retrying.")

    def check_contact_2(self):
        contact_2_states_msg = None
        while contact_2_states_msg is None and not rospy.is_shutdown():
            try:
                contact_2_states_msg = rospy.wait_for_message("/gripper_contactsensor_2_state", ContactsState,
                                                              timeout=0.1)
                self.contact_2_state = contact_2_states_msg.states
                rospy.logdebug("Contactsensor 2 READY")
            except Exception as e:
                rospy.logdebug("Contactsensor 2 not ready yet, retrying==>" + str(e))
                print("EXCEPTION: Contactsensor 2 not ready yet, retrying.")

    def check_collision(self):
        collision_msg = None
        while collision_msg is None and not rospy.is_shutdown():
            try:
                collision_msg = rospy.wait_for_message("/gz_collisions", Bool, timeout=0.1)
                self.collision = collision_msg.data
                rospy.logdebug("collision READY")
            except Exception as e:
                rospy.logdebug("EXCEPTION: Collision not ready yet, retrying==>" + str(e))

    def check_rgb_camera(self):
        camera_rgb_states_msg = None
        while camera_rgb_states_msg is None and not rospy.is_shutdown():
            try:
                camera_rgb_states_msg = rospy.wait_for_message("/camera_rgb/image_raw", Image, timeout=0.1)
                self.camera_rgb_state = camera_rgb_states_msg
                rospy.logdebug("rgb_image READY")
            except Exception as e:
                rospy.logdebug("EXCEPTION: rgb_image not ready yet, retrying==>" + str(e))

    def check_rgbd_camera(self):
        camera_depth_states_msg = None
        while camera_depth_states_msg is None and not rospy.is_shutdown():
            try:
                camera_depth_states_msg = rospy.wait_for_message("/camera_depth/depth/image_raw", Image, timeout=0.1)
                self.camera_depth_state = camera_depth_states_msg
                rospy.logdebug("rgbd_image READY")
            except Exception as e:
                rospy.logdebug("EXCEPTION: rgbd_image not ready yet, retrying==>" + str(e))

    def check_gripper_state(self):
        gripper_state_msg = None
        while gripper_state_msg is None and not rospy.is_shutdown():
            try:
                gripper_state_msg = rospy.wait_for_message("/pickbot/gripper/state", VacuumGripperState, timeout=0.1)
                self.gripper_state = gripper_state_msg
                rospy.logdebug("gripper_state READY")
            except Exception as e:
                rospy.logdebug("EXCEPTION: gripper_state not ready yet, retrying==>" + str(e))

    # Set target object
    # randomize: spawn object randomly from the object pool. If false, object will be the first entry of the object list
    # random_position: spawn object with random position
    def set_target_object(self, random_object=False, random_position=False):
        """
        Set target object
        :param random_object: spawn object randomly from the object pool. If false, object will be the first entry of the object list
        :param random_position: spawn object with random position
        """
        if random_object:
            rand_object = random.choice(self.object_list)
            self.object_name = rand_object["name"]
            self.object_type_str = rand_object["type"]
            self.object_type = self.object_list.index(rand_object)
            init_pos = rand_object["init_pos"]
            self.object_initial_position = Pose(position=Point(x=init_pos[0], y=init_pos[1], z=init_pos[2]),
                                                orientation=quaternion_from_euler(init_pos[3], init_pos[4], init_pos[5]))
        else:
            self.object_name = self.object_list[0]["name"]
            self.object_type_str = self.object_list[0]["type"]
            self.object_type = 0
            init_pos = self.object_list[0]["init_pos"]
            self.object_initial_position = Pose(position=Point(x=init_pos[0], y=init_pos[1], z=init_pos[2]),
                                                orientation=quaternion_from_euler(init_pos[3], init_pos[4], init_pos[5]))

        if random_position:
            if self.object_type_str == "door_handle":
                box_pos = U.get_random_door_handle_pos()
            else:
                box_pos = Pose(position=Point(x=np.random.uniform(low=-0.3, high=0.3, size=None),
                                              y=np.random.uniform(low=0.9, high=1.1, size=None),
                                              z=1.05),
                               orientation=quaternion_from_euler(0, 0, 0))
        else:
            box_pos = self.object_initial_position

        U.change_object_position(self.object_name, box_pos)
        print("Current target: ", self.object_name)

        # change red point position
        red_dot_position = box_pos
        red_dot_position.position.z = red_dot_position.position.z + self._added_target_height
        U.change_object_position("red_dot", red_dot_position)

    def get_action_to_position(self, action, last_position):
        """
        Take the last published joint and increment/decrement one joint according to action chosen
        :param action: output of NN
        :param last_position: array of 6 value
        :return: list with all joint positions according to chosen action
        """

        action_position = np.asarray(last_position)[:5] + action
        action_position = np.append(action_position, last_position[-1])
        # clip action that is going to be published to make sure to avoid losing control of controllers
        x = np.clip(action_position, -2.9, 2.9)

        return x.tolist()

    def get_obs(self):
        """
        Returns the state of the robot needed for Algorithm to learn
        The state will be defined by a List (later converted to numpy array) of the:

        self._list_of_observations = ["elbow_joint_state",
                              "shoulder_lift_joint_state",
                              "shoulder_pan_joint_state",
                              "wrist_1_joint_state",
                              "wrist_2_joint_state",
                              "wrist_3_joint_state",
                              "vacuum_gripper_pos_x",
                              "vacuum_gripper_pos_y",
                              "vacuum_gripper_pos_z",
                              "vacuum_gripper_ori_w",
                              "vacuum_gripper_ori_x",
                              "vacuum_gripper_ori_y",
                              "vacuum_gripper_ori_z",
                              "object_pos_x",
                              "object_pos_y",
                              "object_pos_z",
                              "object_ori_w",
                              "object_ori_x",
                              "object_ori_y",
                              "object_ori_z + added_target_height if any",
                              ]

        :return: observation
        """

        # Get Joints Data out of Subscriber
        joints_state = self.joints_state.position

        for joint in self.joints_state.position:
            if joint > math.pi or joint < -math.pi:
                print(self.joints_state.name)
                print(self.joints_state.position)
                sys.exit("Joint exceeds limit")

        vacuum_gripper_geometry = U.get_link_state("vacuum_gripper_link")

        target_geometry = U.get_link_state("target")

        # Concatenate the information that defines the robot state
        state = np.r_[np.reshape(joints_state, -1),
                      np.reshape(vacuum_gripper_geometry, -1),
                      np.reshape(target_geometry, -1)]

        return state

    def get_contact_force_1(self):
        """
        Get Contact Force of contact sensor 1
        Takes average over 2 contacts so the chances are higher that both sensors say there is contact the same time due to sensor noise
        :returns force value
        """

        # get Force out of contact_1_state
        if not self.contact_1_state:
            contact1_force = 0.0
        else:
            for state in self.contact_1_state:
                self.contact_1_force = state.total_wrench.force
                contact1_force_np = np.array((self.contact_1_force.x, self.contact_1_force.y, self.contact_1_force.z))
                force_magnitude_1 = np.linalg.norm(contact1_force_np)
                contact1_force = force_magnitude_1

        # read last contact force 1 value out of yaml
        with open("contact_1_force.yml", 'r') as stream:
            try:
                last_contact_1_force = (yaml.load(stream, Loader=yaml.Loader))
            except yaml.YAMLError as exc:
                print(exc)
        # write new contact_1_force value in yaml
        with open('contact_1_force.yml', 'w') as yaml_file:
            yaml.dump(contact1_force, yaml_file, default_flow_style=False)
        # calculate average force
        average_contact_1_force = (last_contact_1_force + contact1_force) / 2

        return average_contact_1_force

    def get_contact_force_2(self):
        """
        Get Contact Force of contact sensor 2
        Takes average over 2 contacts so the chances are higher that both sensors say there is contact the same time due to sensor noise
        :returns force value
        """

        # get Force out of contact_2_state
        if not self.contact_2_state:
            contact2_force = 0.0
        else:
            for state in self.contact_2_state:
                self.contact_2_force = state.total_wrench.force
                contact2_force_np = np.array((self.contact_2_force.x, self.contact_2_force.y, self.contact_2_force.z))
                force_magnitude_2 = np.linalg.norm(contact2_force_np)
                contact2_force = force_magnitude_2

        # read last contact_2_force value out of yaml
        with open("contact_2_force.yml", 'r') as stream:
            try:
                last_contact_2_force = (yaml.load(stream, Loader=yaml.Loader))
            except yaml.YAMLError as exc:
                print(exc)
        # write new contact force 2 value in yaml
        with open('contact_2_force.yml', 'w') as yaml_file:
            yaml.dump(contact2_force, yaml_file, default_flow_style=False)
        # calculate average force
        average_contact_2_force = (last_contact_2_force + contact2_force) / 2

        return average_contact_2_force

    def get_collisions(self):
        """
        Checks all the collisions by listening to rostopic /gz_collisions wich is republishing the gazebo topic (gz topic -e /gazebo/default/physics/contacts).
        The Publisher is started in a different node out of the simulation launch file.
        Stores last value yaml file and if one of the two values is showing a invalid collision it returns a invalid collision.
        This is to make shure seeing collisions due to high sensor noise and publish rate.

        If one of the 2 Messages is True it returns True.
        returns:
            False:  if no contacts or just valid ones -> Box/Shelf, Wrist3/Box, VacuumGripper/Box
            True:   if any other contact occurs which is invalid
        """

        # read last contact_2_force value out of yaml
        with open("collision.yml", 'r') as stream:
            try:
                last_collision = (yaml.load(stream, Loader=yaml.Loader))
            except yaml.YAMLError as exc:
                print(exc)
        # write new contact force 2 value in yaml
        with open('collision.yml', 'w') as yaml_file:
            yaml.dump(self.collision, yaml_file, default_flow_style=False)

        # Check if last_collision or self.collision is True. IF one s true return True else False
        if self.collision == True or last_collision == True:
            return True
        else:
            return False

    def is_done(self, observations):
        """Checks if episode is done based on observations given.

        Done when:
        -Successfully reached goal: Contact with both contact sensors and contact is a valid one(Wrist3 or/and Vacuum Gripper with unit_box)
        -Crashing with itself, shelf, base
        -Joints are going into limits set
        """

        done = False
        done_reward = 0
        reward_reached_goal = 1000
        reward_crashing = -2000
        reward_no_motion_plan = 0
        reward_joint_range = 0

        # Check if there are invalid collisions
        invalid_collision = self.get_collisions()

        if self.moveit_action_feedback.status.text == "No motion plan found. No execution attempted." or  \
                self.moveit_action_feedback.status.text == "Solution found but controller failed during execution" or \
                self.moveit_action_feedback.status.text == "Motion plan was found but it seems to be invalid (possibly due to postprocessing).Not executing.":

            print(">>>>>>>>>>>> NO MOTION PLAN!!! <<<<<<<<<<<<<<<")
            done = True
            done_reward = reward_no_motion_plan
            invalid_collision = True

        # Successfully reached_goal: orientation of the end-effector and target is less than threshold also
        # distance is less than threshold
        distance_gripper_to_target = np.linalg.norm(observations[6:9] - observations[13:16])
        orientation_error = quaternion_multiply(observations[9:13], quaternion_conjugate(observations[16:]))
        # print("check distance {} and orientation err {} ".format(distance_gripper_to_target, orientation_error))

        # if distance_gripper_to_target < 0.05 and np.abs(orientation_error[0]) < 0.1:
        if distance_gripper_to_target < 0.001:
            done = True
            print("Success! Distance {} ".format(distance_gripper_to_target))
            done_reward = reward_reached_goal

        # Successfully reached goal: Contact with both contact sensors and there is no invalid contact
        # if observations[7] != 0 and observations[8] != 0 and not invalid_collision:
        #     done = True
        #     print('>>>>>> Success!')
        #     done_reward = reward_reached_goal
        #     # save state in csv file
        #     U.append_to_csv(self.csv_success_exp, observations)
        #     self.successful_attempts += 1
        #     print("Successful contact so far: {} attempts".format(self.successful_attempts))

        # Crashing with itself, shelf, base
        # if invalid_collision:
        #     done = True
        #     print('>>>>>>>>>>>>>>>>>>>> crashing')
        #    done_reward = reward_crashing

        joint_exceeds_limits = False
        for joint_pos in self.joints_state.position:
            joint_correction = []
            if joint_pos < -math.pi or joint_pos > math.pi:
                joint_exceeds_limits = True
                done = True
                done_reward = reward_joint_range
                print('>>>>>>>>>>>>>>>>>>>> joint exceeds limit <<<<<<<<<<<<<<<<<<<<<<<')
                joint_correction.append(-joint_pos)
            else:
                joint_correction.append(0.0)

        if joint_exceeds_limits:
            print("is_done: Joints: {}".format(np.round(self.joints_state.position, decimals=3)))
            self.publisher_to_moveit_object.pub_joints_to_moveit([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            while not self.movement_complete.data:
                pass
            self.publisher_to_moveit_object.pub_relative_joints_to_moveit(joint_correction)
            while not self.movement_complete.data:
                pass
            print('>>>>>>>>>>>>>>>> joint corrected <<<<<<<<<<<<<<<<<')

        return done, done_reward, invalid_collision

    def _update_episode(self):
        """
        Publishes the accumulated reward of the episode and
        increases the episode number by one.
        :return:
        """
        if self.episode_num > 0:
            self._publish_reward_topic(
                self.accumulated_episode_reward,
                self.episode_steps,
                self.episode_num
            )

        self.episode_num += 1
        self.accumulated_episode_reward = 0
        self.episode_steps = 0

    def _publish_reward_topic(self, reward, steps, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)
        self.reward_list.append(reward)
        self.episode_list.append(episode_number)
        self.step_list.append(steps)
        list = str(reward) + ";" + str(episode_number) + ";" + str(steps) + "\n"

        with open(self.csv_name + '.csv', 'a') as csv:
            csv.write(str(list))
