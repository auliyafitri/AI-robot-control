#!/usr/bin/env python

# IMPORT
import os
import gym
import rospy
import numpy as np
# import cv2
import time
import random
import sys
import yaml
import math
import datetime
import rospkg
from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register
from transformations import quaternion_from_euler

# OTHER FILES
import environments.util_env as U
import environments.util_math as UMath
from environments.gazebo_connection import GazeboConnection
from environments.controllers_connection import ControllersConnection
from environments.joint_publisher import JointPub
from environments.joint_array_publisher import JointArrayPub
from baselines import logger

# from cv_bridge import CvBridge, CvBridgeError
# from scipy.misc import imsave

# MESSAGES/SERVICES
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import Image
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import Point, Quaternion, Vector3
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from openai_ros.msg import RLExperimentInfo
from moveit_msgs.msg import MoveGroupActionFeedback

from simulation.msg import VacuumGripperState
from simulation.srv import VacuumGripperControl


# DEFINE ENVIRONMENT CLASS
class PickbotReachEnv(gym.Env):

    def __init__(self, joint_increment=0.02, sim_time_factor=0.001, random_object=False, random_position=False,
                 use_object_type=False, populate_object=False, env_object_type='free_shapes', is_discrete=False,
                 sparse=True, load_experience=False):
        """
        initializing all the relevant variables and connections
        :param sim_time_factor: gazebo simulation time factor
        :param random_object: spawn random object in the simulation
        :param random_position: change object position in each reset
        :param use_object_type: assign IDs to objects and used them in the observation space
        :param populate_object: to populate object(s) in the simulation using sdf file
        :param env_object_type: object type for environment, free_shapes for boxes while others are related to use_case
            'door_handle', 'combox', ...
        """
        # Parameters for action
        self._is_discrete = is_discrete
        self._xy_increment = 0.01
        self._z_increment = 0.005
        self._rpy_increment = 1.0
        self._wrist_3_joint_increment = math.pi / 20
        self._joint_increment = joint_increment
        self._use_z_axis = True
        self._use_angle_as_action = False
        self._use_rpy_as_action = True
        self._action_bound = 1
        self._load_init_pos = load_experience  # for randomized initial state
        self._is_sparse = sparse

        # Parameters for target-object
        self._random_object = random_object
        self._random_position = random_position
        self._use_object_type = use_object_type
        self._populate_object = populate_object

        # Assign MsgTypes
        self.joints_state = JointState()
        self.contact_1_state = ContactsState()
        self.contact_2_state = ContactsState()
        self.collisions = Bool()
        # self.camera_rgb_state = Image()
        # self.camera_depth_state = Image()

        self._height = 480
        self._width = 640
        # self.realsense_rgb = Image()
        # self.realsense_depth = Image()

        self.contact_1_force = Vector3()
        self.contact_2_force = Vector3()
        self.gripper_state = VacuumGripperState()
        self.movement_complete = Bool()
        self.movement_complete.data = False
        self.moveit_action_feedback = MoveGroupActionFeedback()
        self.feedback_list = []

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
        1) /joint_states
        2) /gripper_contactsensor_1_state
        3) /gripper_contactsensor_2_state
        4) /gz_collisions

        not used so far but available in the environment 
        5) /pickbot/gripper/state
        6) /camera_rgb/image_raw   
        7) /camera_depth/depth/image_raw
        """
        rospy.Subscriber("/joint_states", JointState, self.joints_state_callback)
        rospy.Subscriber("/gripper_contactsensor_1_state", ContactsState, self.contact_1_callback)
        rospy.Subscriber("/gripper_contactsensor_2_state", ContactsState, self.contact_2_callback)
        rospy.Subscriber("/gz_collisions", Bool, self.collision_callback)
        rospy.Subscriber("/pickbot/movement_complete", Bool, self.movement_complete_callback)
        rospy.Subscriber("/move_group/feedback", MoveGroupActionFeedback, self.move_group_action_feedback_callback, queue_size=4)
        # rospy.Subscriber('/intel_realsense_camera/rgb/image_raw', Image, self.realsense_rgb_callback)
        # rospy.Subscriber('/intel_realsense_camera/depth/image_raw', Image, self.realsense_depth_callback)
        rospy.Subscriber("/pickbot/gripper/state", VacuumGripperState, self.gripper_state_callback)

        # Define Action and state Space and Reward Range
        """
        Action Space: Box Space with 6 values.
        
        State Space: Box Space with 12 values. It is a numpy array with shape (12,)

        Reward Range: -infinity to infinity 
        """
        ################################################
        # Action space                                 #
        ################################################
        if self._is_discrete:
            # 6 joints
            self.action_space = spaces.Discrete(12)
        else:
            action_dim = 6
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        ################################################
        # Action space                                 #
        ################################################

        ################################################
        # Observation space                            #
        ################################################
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
        self.obs_dim = 20
        high = np.inf * np.ones(self.obs_dim)
        low = -high

        self.observation_space = spaces.Box(low, high)
        ################################################
        # Observation space                            #
        ################################################

        self._list_of_status = {"distance_gripper_to_object": -1,
                                "contact_1_force": -1,
                                "contact_2_force": -1,
                                "gripper_pos": -1,
                                "gripper_ori": -1,
                                "object_pos": -1,
                                "object_ori": -1,
                                "min_distance_gripper_to_object": -1}
        # if self._use_object_type:
        #     self._list_of_status.append("object_type")

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
        self.csv_success_exp = "success_exp" + datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mmin') + ".csv"
        self.success_2_contacts = 0
        self.success_1_contact = 0

        # object name: name of the target object
        # object type: index of the object name in the object list
        # object list: pool of the available objects, have at least one entry
        self.object_name = ''
        self.object_type_str = ''
        self.object_type = 0
        self.object_list = U.get_target_object(env_object_type)
        print("object list {}".format(self.object_list))
        self.object_initial_position = Pose(position=Point(x=-0.13, y=0.848, z=1.06),  # x=0.0, y=0.9, z=1.05
                                            orientation=quaternion_from_euler(0.002567, 0.102, 1.563))
        self.object_position = self.object_initial_position

        if self._populate_object:
            # populate objects from object list
            self.populate_objects()

        # select first object, set object name and object type
        # if object is random, spawn random object
        # else get the first entry of object_list
        self.set_target_object(random_object=self._random_object, random_position=self._random_position)

        # get maximum distance to the object to calculate reward, renewed in the reset function
        self.max_distance, _ = U.get_distance_gripper_to_object()
        self.min_distance = 999

        # get samples from reaching task
        if self._load_init_pos:
            import environments
            self.init_samples = U.load_samples_from_prev_task(os.path.dirname(environments.__file__) +
                                                              "/contacts_sample/door_sample/observation_record_2019-08-02.csv")

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
        6) Create YAML Files for collision to make sure to see a collision due to high noise in topic
        7) Get Observations and return current State
        8) Publish Episode Reward and set accumulated reward back to 0 and iterate the Episode Number
        9) Return State
        """

        self.set_initial_position()
        self.turn_off_gripper()
        self._check_all_systems_ready()

        with open('contact_1_force.yml', 'w') as yaml_file:
            yaml.dump(0.0, yaml_file, default_flow_style=False)
        with open('contact_2_force.yml', 'w') as yaml_file:
            yaml.dump(0.0, yaml_file, default_flow_style=False)
        with open('collision.yml', 'w') as yaml_file:
            yaml.dump(False, yaml_file, default_flow_style=False)
        observation = self.get_obs()
        self.get_status()
        self.object_position = self._list_of_status["object_pos"]

        # get maximum distance to the object to calculate reward
        self.max_distance, _ = U.get_distance_gripper_to_object()
        self.min_distance = self.max_distance
        self._update_episode()

        return observation

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
        7) Calculate reward based on Observation and done_reward
        8) Return State, Reward, Done
        """
        # print("############################")
        # print("action: {}".format(action))

        # 1) Read last joint positions by getting the observation before acting
        old_observation = self.get_obs()
        self.movement_complete.data = False
        old_status = self.get_status()
        gripper_pos = U.get_gripper_position()
        gripper_geom = U.get_link_state("vacuum_gripper_link")

        # 2) Get the new joint positions according to chosen action (actions here are the joint increments)
        if self._is_discrete:
            dj2 = [-self._joint_increment, self._joint_increment, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][action]
            dj1 = [0, 0, -self._joint_increment, self._joint_increment, 0, 0, 0, 0, 0, 0, 0, 0][action]
            dj0 = [0, 0, 0, 0, -self._joint_increment, self._joint_increment, 0, 0, 0, 0, 0, 0][action]
            dj3 = [0, 0, 0, 0, 0, 0, -self._joint_increment, self._joint_increment, 0, 0, 0, 0][action]
            dj4 = [0, 0, 0, 0, 0, 0, 0, 0, -self._joint_increment, self._joint_increment, 0, 0][action]
            dj5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -self._joint_increment, self._joint_increment][action]

            realAction = [dj2, dj1, dj0, dj3, dj4, dj5]

            next_action_position = self.get_action_to_position(realAction, old_observation[:6])
        else:
            next_action_position = action

        # 3) Move to position and wait for moveit to complete the execution
        self.publisher_to_moveit_object.pub_joints_to_moveit(next_action_position)
        while not self.movement_complete.data:
            pass
        self.movement_complete.data = False

        # self.publisher_to_moveit_object.pub_relative_joints_to_moveit([0, 0, 0, 0, 0, realAction[-1]])
        # while not self.movement_complete.data:
        #     pass

        is_time_out = False
        start_ros_time = rospy.Time.now()
        while True:
            elapsed_time = rospy.Time.now() - start_ros_time
            if np.isclose(next_action_position, self.joints_state.position, rtol=0.0, atol=0.01).all():
                break
            elif elapsed_time > rospy.Duration(2):
                is_time_out = True
                print(">>> Time Out!")
                print("########################################")
                print("Old  position: {}".format(np.round(old_observation[:6], decimals=4)))
                print("Next position: {}".format(np.round(next_action_position, decimals=4)))
                break

        # 4) Get new status and update min_distance after performing the action
        new_observation = self.get_obs()
        new_status = self.get_status()

        new_gripper_position = np.append(new_status["gripper_pos"], self.joints_state.position[-1])
        if is_time_out:
            print("New Gripper position: {}".format(np.round(new_gripper_position, decimals=4)))
            print("########################################")

        if new_status["distance_gripper_to_object"] < self.min_distance:
            self.min_distance = new_status["distance_gripper_to_object"]

        # Turn on gripper and try to grip
        if new_status["gripper_pos"][-1] <= 1.1:
            self.movement_complete.data = False

            self.turn_on_gripper()
            gripping_pos = np.append(new_status["gripper_pos"][0:2], 1.0395)  # this data is only for the cube
            self.publisher_to_moveit_object.pub_pose_to_moveit(gripping_pos)  # grip

            if self.moveit_action_feedback.status.text == "No motion plan found. No execution attempted." or \
                    self.moveit_action_feedback.status.text == "Solution found but controller failed during execution" or \
                    self.moveit_action_feedback.status.text == "Motion plan was found but it seems to be invalid (possibly due to postprocessing).Not executing.":
                print(">>>>>>>>>>>> Gripping not succeed <<<<<<<<<<<<<<<")
                while not self.movement_complete.data:
                    pass
                self.movement_complete.data = False
            else:
                while not self.movement_complete.data:
                    pass
                time.sleep(1)
                self.movement_complete.data = False
                if self.is_gripper_attached():
                    # pick up
                    self.publisher_to_moveit_object.pub_relative_pose_to_moveit(0.3, is_discrete=True, axis='z')
                    while not self.movement_complete.data:
                        pass

        # 5) Convert Observations into state
        state = U.get_state(new_observation)

        # 6) Check if its done, calculate done_reward
        done, done_reward, invalid_contact = self.is_done(new_status)

        # 7) Calculate reward based on Observation and done_reward and update the accumulated Episode Reward
        # reward = UMath.compute_reward(new_observation, done_reward, invalid_contact)
        if self._is_sparse:
            reward = done_reward
        else:
            reward = UMath.computeReward(status=new_status, collision=invalid_contact) + done_reward

        if done: print("Reward this step: {}".format(reward))
        self.accumulated_episode_reward += reward

        self.episode_steps += 1

        return state, reward, done, {}

    def is_done(self, status):
        """Checks if episode is done based on observations given.

        Done when:
        -Successfully reached goal: Contact with both contact sensors and contact is a valid one(Wrist3 or/and Vavuum Gripper with unit_box)
        -Crashing with itself, shelf, base
        -Joints are going into limits set
        """
        ####################################################################
        #                        Plan0: init                               #
        ####################################################################
        # done = False
        # done_reward = 0
        # reward_reached_goal = 2000
        # reward_crashing = -200
        # reward_no_motion_plan = -50
        # reward_joint_range = -150

        ####################################################################################
        # Plan1: Reach a point in 3D space (usually right above the target object)         #
        # Reward only dependent on distance. Nu punishment for crashing or joint_limits    #
        ####################################################################################
        done = False
        done_reward = 0
        if self._is_sparse:
            reward_reached_goal = 1
            reward_crashing = 0
            reward_no_motion_plan = 0
            reward_joint_range = 0
        else:
            reward_reached_goal = 100
            reward_crashing = 0
            reward_no_motion_plan = 0
            reward_joint_range = 0

        # Check if there are invalid collisions
        invalid_collision = self.get_collisions()

        if self.is_gripper_attached():
            done = True
            done_reward = reward_reached_goal
            print(">>>>>>>>>>>> Success! Gripper is attached! <<<<<<<<<<<<<<<")

        # TODO: this only works for the dummy door handle
        # the gripper tried to grasp but did not succeed
        if self._list_of_status["gripper_pos"][-1] <= 1.039:
            done = True
            if not self.is_gripper_attached():
                print(">>>>>>>>>>>> reset gripper not succeed <<<<<<<<<<<<<<<")

        # print("##################{}: {}".format(self.moveit_action_feedback.header.seq, self.moveit_action_feedback.status.text))
        if self.moveit_action_feedback.status.text == "No motion plan found. No execution attempted." or \
                self.moveit_action_feedback.status.text == "Solution found but controller failed during execution" or \
                self.moveit_action_feedback.status.text == "Motion plan was found but it seems to be invalid (possibly due to postprocessing).Not executing.":
            print(">>>>>>>>>>>> NO MOTION PLAN!!! <<<<<<<<<<<<<<<")
            done = True
            done_reward = reward_no_motion_plan

        # Successfully reached goal: Contact with at least one contact sensor and there is no invalid contact
        if status["contact_1_force"] != 0 and status["contact_2_force"] != 0 and not invalid_collision:
            done = True
            print('>>>>>>>>>>>>> get two contacts <<<<<<<<<<<<<<<<<<')
            done_reward = reward_reached_goal
            # save state in csv file
            U.dict_to_csv(self.csv_success_exp, status)
            self.success_2_contacts += 1
            print("Successful 2 contacts so far: {} attempts".format(self.success_2_contacts))

        if status["contact_1_force"] != 0 or status["contact_2_force"] != 0 and not invalid_collision:
            done = True
            print('>>>>>>>>>>>>> get one contacts <<<<<<<<<<<<<<<<<<')
            self.success_1_contact += 1
            print("Successful 1 contact so far: {} attempts".format(self.success_1_contact))

        # Check if the box has been moved compared to the last observation
        # target_pos = U.get_target_position()
        # if not np.allclose(self.object_position, target_pos, rtol=0.0, atol=0.0001):
        #     print(">>>>>>>>>>>>>>>>>>> Target moved <<<<<<<<<<<<<<<<<<<<<<<")
        #     done = True

        # Crashing with itself, shelf, base
        if invalid_collision:
            done = True
            print('>>>>>>>>>>>>>>>>>>>> crashing <<<<<<<<<<<<<<<<<<<<<<<')
            done_reward = reward_crashing

        ##################################################################################
        # Joint Safety                                                                   #
        ##################################################################################
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
            self.movement_complete.data = False
            self.publisher_to_moveit_object.pub_joints_to_moveit([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            while not self.movement_complete.data:
                pass
            self.movement_complete.data = False
            self.publisher_to_moveit_object.pub_relative_joints_to_moveit(joint_correction)
            while not self.movement_complete.data:
                pass
            print('>>>>>>>>>>>>>>>> joint corrected <<<<<<<<<<<<<<<<<')
        ##################################################################################
        # Joint Safety                                                                   #
        ##################################################################################

        return done, done_reward, invalid_collision

    def get_action_to_position(self, action, last_position):
        """
        takes the last position and adds the increments for each joint
        returns the new position       
        """
        action_position = np.asarray(last_position) + action
        # clip action that is going to be published to -2.9 and 2.9 to make sure to avoid losing control of controllers
        x = np.clip(action_position, -(math.pi-0.05), math.pi-0.05)

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
                              "object_ori_z",
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

    def get_status(self):
        """
        Returns the state of the robot needed for Algorithm to learn
        The state will be defined by a List (later converted to numpy array) of the:

        1)          Distance from desired point in meters
        2-7)        States of the 6 joints in radiants
        8,9)        Force in contact sensor in Newtons
        10,11,12)   x, y, z Position of object?

        MISSING
        10)     RGBD image

        :return: observation
        """

        # Get Distance Object to Gripper and Objectposition from Service Call. Needs to be done a second time cause
        # we need the distance and position after the Step execution
        distance_gripper_to_object, position_xyz_object = U.get_distance_gripper_to_object()
        vacuum_gripper_pose = U.get_link_state("vacuum_gripper_link")
        target_pose = U.get_link_state("target")


        # Get Joints Data out of Subscriber
        joint_states = self.joints_state

        for joint in joint_states.position:
            if joint > 2 * math.pi or joint < -2 * math.pi:
                print(joint_states.name)
                print(np.around(joint_states.position, decimals=3))
                sys.exit("Joint exceeds limit")

        # Get Contact Forces out of get_contact_force Functions to be able to take an average over some iterations
        # otherwise chances are high that not both sensors are showing contact the same time
        contact_1_force = self.get_contact_force_1()
        contact_2_force = self.get_contact_force_2()

        # Stack all information into Observations List
        self._list_of_status["distance_gripper_to_object"] = distance_gripper_to_object
        self._list_of_status["contact_1_force"] = contact_1_force
        self._list_of_status["contact_2_force"] = contact_2_force
        self._list_of_status["gripper_pos"] = vacuum_gripper_pose[0:3]
        self._list_of_status["gripper_ori"] = vacuum_gripper_pose[3:]
        self._list_of_status["object_pos"] = target_pose[0:3]
        self._list_of_status["object_ori"] = target_pose[3:]
        self._list_of_status["min_distance_gripper_to_object"] = self.min_distance

        return self._list_of_status

    def set_initial_position(self):
        # load sample from previous training result with 25% probability,
        # the rest is randomized near initial position [1.57, -1.479, 1.41, -1.66, -1.57, -0.08]

        # set joints using moveit to initial position
        self.movement_complete.data = False
        init_joint_pos = [1.57, -1.479, 1.41, -1.66, -1.57, -0.08]
        self.publisher_to_moveit_object.set_joints(init_joint_pos)

        # if self.moveit_action_feedback.status.text == "No motion plan found. No execution attempted." or \
        #         self.moveit_action_feedback.status.text == "Solution found but controller failed during execution" or \
        #         self.moveit_action_feedback.status.text == "Motion plan was found but it seems to be invalid (possibly due to postprocessing).Not executing.":
        #      print("<<<<<<<<<<<<<<<<< Set init position no motion plan found")

        while not self.movement_complete.data:
            pass
        self.movement_complete.data = False

        # draw if the arm position is from experience of randomized
        draw = np.random.choice([0, 1], p=[0.75, 0.25])
        if self._load_init_pos and draw == 1:
            sample_ep = random.choice(self.init_samples)

            # change object position
            obj_pos_rpy = sample_ep[-7:]
            obj_pos_quat = Pose(position=Point(x=obj_pos_rpy[0], y=obj_pos_rpy[1], z=obj_pos_rpy[2]),
                                orientation=obj_pos_rpy[-4:])
            U.change_object_position(self.object_name, obj_pos_quat)

            # set joints using moveit
            # print("Joints from samples: {}".format(sample_ep[0:6]))
            self.publisher_to_moveit_object.pub_joints_to_moveit(sample_ep[0:6])

            while not self.movement_complete.data:
                pass
            self.movement_complete.data = False

            # move a little back in z axis
            if not self._is_sparse:
                new_status = self.get_status()
                new_gripper_pos = np.append(new_status["gripper_pos"][0:2], 1.15)
                self.publisher_to_moveit_object.pub_pose_to_moveit(new_gripper_pos)

                while not self.movement_complete.data:
                    pass
                self.movement_complete.data = False

        else:
            # change object position
            self.set_target_object(random_object=self._random_object, random_position=self._random_position)

            # randomize joints position near initial position, ensuring it's possible to go there
            joint_possible = False
            while not joint_possible:
                new_joint_pos = np.zeros(len(init_joint_pos))
                for i in range(len(init_joint_pos)):
                    new_joint_pos[i] = init_joint_pos[i] + np.random.uniform(-np.pi/6, np.pi/6)

                # print("new joint position {}".format(new_joint_pos))
                self.publisher_to_moveit_object.pub_joints_to_moveit(new_joint_pos)

                if self.moveit_action_feedback.status.text == "No motion plan found. No execution attempted." or \
                        self.moveit_action_feedback.status.text == "Solution found but controller failed during execution" or \
                        self.moveit_action_feedback.status.text == "Motion plan was found but it seems to be invalid (possibly due to postprocessing).Not executing.":
                    # print("generate new initial position of arm")
                    while not self.movement_complete.data:
                        pass
                    self.movement_complete.data = False
                else:
                    joint_possible = True
                    # print("new joint position will be executed")

                    # Busy waiting for moveit to complete the movement
                    while not self.movement_complete.data:
                        pass
                    start_ros_time = rospy.Time.now()
                    while True:
                        elapsed_time = rospy.Time.now() - start_ros_time
                        if np.isclose(new_joint_pos, self.joints_state.position, rtol=0.0, atol=0.01).all():
                            break
                        elif elapsed_time > rospy.Duration(2):  # time out
                            break
                    self.movement_complete.data = False

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
                               orientation=quaternion_from_euler(0, 0,
                                                                 np.random.uniform(low=-np.pi, high=np.pi, size=None)))
        else:
            box_pos = self.object_initial_position

        U.change_object_position(self.object_name, box_pos)
        print("Current target: ", self.object_name)

    def get_contact_force_1(self):
        """
        Get Contact Force of contact sensor 1
        Takes average over 2 contacts so the chances are higher that both sensors say there is contact the same time due to sensor noise 
        :returns force value
        """

        # get Force out of contact_1_state
        if self.contact_1_state == []:
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
        if self.contact_2_state == []:
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
            True:   if any other contact occures wich is invalid 
        """

        # read last contact_2_force value out of yaml
        with open("collision.yml", 'r') as stream:
            try:
                last_collision = (yaml.load(stream, Loader=yaml.Loader))
            except yaml.YAMLError as exc:
                print(exc)
        # write new contact force 2 value in yaml
        with open('collision.yml', 'w') as yaml_file:
            yaml.dump(self.collisions, yaml_file, default_flow_style=False)

        # Check if last_collision or self.collision is True. IF one s true return True else False
        if self.collisions or last_collision:
            return True
        else:
            return False

    def populate_objects(self):
        """
        populate objects, called in init
        :return: -
        """
        if not self._random_object:  # only populate the first object
            U.spawn_object(self.object_list[0], self.object_initial_position)
        else:
            rand_x = np.random.uniform(low=-0.35, high=0.35, size=(len(self.object_list),))
            rand_y = np.random.uniform(low=2.2, high=2.45, size=(len(self.object_list),))
            for idx, obj in enumerate(self.object_list):
                box_pos = Pose(position=Point(x=rand_x[idx],
                                              y=rand_y[idx],
                                              z=1.05))
                U.spawn_object(obj, box_pos)

    def turn_on_gripper(self):
        """
        turn on the Gripper by calling the service
        """
        try:
            turn_on_gripper_service = rospy.ServiceProxy('/pickbot/gripper/control', VacuumGripperControl)
            enable = True
            turn_on_gripper_service(enable)
        except rospy.ServiceException as e:
            rospy.loginfo("Turn on Gripper service call failed:  {0}".format(e))

    def turn_off_gripper(self):
        """
        turn off the Gripper by calling the service
        """
        try:
            turn_off_gripper_service = rospy.ServiceProxy('/pickbot/gripper/control', VacuumGripperControl)
            enable = False
            turn_off_gripper_service(enable)
        except rospy.ServiceException as e:
            rospy.loginfo("Turn off Gripper service call failed:  {0}".format(e))

    def is_gripper_attached(self):
        gripper_state = None
        while gripper_state is None and not rospy.is_shutdown():
            try:
                gripper_state = rospy.wait_for_message("/pickbot/gripper/state", VacuumGripperState, timeout=0.1)
            except Exception as e:
                rospy.logdebug("Current gripper_state not ready yet, retrying==>" + str(e))
        return gripper_state.attached

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

    def _check_all_systems_ready(self):
        """
        Checks that all subscribers for sensortopics are working

        1) /joint_states
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
        self.check_rgb_camera()
        self.check_rgbd_camera()
        self.check_gripper_state()
        rospy.logdebug("ALL SYSTEMS READY")

    def check_joint_states(self):
        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
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
                self.collisions = collision_msg.data
                rospy.logdebug("collision READY")
            except Exception as e:
                rospy.logdebug("EXCEPTION: Collision not ready yet, retrying==>" + str(e))

    def check_rgb_camera(self):
        camera_rgb_states_msg = None
        while camera_rgb_states_msg is None and not rospy.is_shutdown():
            try:
                camera_rgb_states_msg = rospy.wait_for_message("/intel_realsense_camera/rgb/image_raw", Image, timeout=0.1)
                self.camera_rgb_state = camera_rgb_states_msg
                rospy.logdebug("rgb_image READY")
            except Exception as e:
                rospy.logdebug("EXCEPTION: rgb_image not ready yet, retrying==>" + str(e))

    def check_rgbd_camera(self):
        camera_depth_states_msg = None
        while camera_depth_states_msg is None and not rospy.is_shutdown():
            try:
                camera_depth_states_msg = rospy.wait_for_message("/intel_realsense_camera/depth/image_raw", Image, timeout=0.1)
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

    # Callback Functions for Subscribers to make topic values available each time the class is initialized
    def joints_state_callback(self, msg):
        self.joints_state = msg
        self.joints_state.position = self.joints_state.position[0:6]

    def contact_1_callback(self, msg):
        self.contact_1_state = msg.states

    def contact_2_callback(self, msg):
        self.contact_2_state = msg.states

    def collision_callback(self, msg):
        self.collisions = msg.data

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

    # def realsense_rgb_callback(self, msg):
    #     self.realsense_rgb = msg
    #
    # def realsense_depth_callback(self, msg):
    #     self.realsense_depth = msg

