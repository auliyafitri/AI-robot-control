#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point
from gazebo_msgs.srv import SetModelConfiguration
from gazebo_msgs.srv import SetModelConfigurationRequest

import math
import random
import numpy as np

class JointArrayPub(object):
    def __init__(self):
        self.joint_pub = rospy.Publisher('/pickbot/target_joint_positions', JointState, queue_size=10)
        self.geomsg_pub = rospy.Publisher('/pickbot/target_pose', Pose, queue_size=10)
        self.relative_geomsg_pub = rospy.Publisher('/pickbot/relative_target_pose', Pose, queue_size=10)
        self.relative_joint_pub = rospy.Publisher('/pickbot/relative_joint_positions', JointState, queue_size=10)
        self.init_pos = [1.5, -1.2, 1.4, -1.87, -1.57, 0]

    def set_init_pose(self):
        """
        Sets joints to initial position 
        :return:
        """
        self.check_publishers_connection()
        self.pub_joints_to_moveit(self.init_pos)
        # self.reset_joints =  rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)



    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(100)  # 10hz
        while (self.joint_pub.get_num_connections() == 0):
            rospy.logdebug("No susbribers to _joint1_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("joint_pub Publisher Connected")

    def pub_joints_to_moveit(self, joints_array):
        self.check_publishers_connection()
        
        jointState = JointState()
        jointState.header = Header()
        jointState.header.stamp = rospy.Time.now()
        jointState.name = [ 'elbow_joint', 'shoulder_lift_joint','shoulder_pan_joint', 
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        jointState.position = joints_array
        jointState.velocity = []
        jointState.effort = []
        self.joint_pub.publish(jointState)
        # print("I've published: {}".format(jointState.position))

    def pub_pose_to_moveit(self, position):
        self.check_publishers_connection()
        geomsg = Pose(position=Point(x=position[0], y=position[1], z=position[2]))
        self.geomsg_pub.publish(geomsg)

    def pub_relative_pose_to_moveit(self, distance, is_discrete, axis=None):
        """
        :param distance: could be float(Discrete action space) or list of float(Continuous action space)
        :param is_discrete: True: only move along one axis, False: move along x,y,z axis simultaneously
        :param axis: which axis to move along. Only needed for Discrete action space
        :return:
        """
        self.check_publishers_connection()
        if is_discrete:
            if axis == 'x':
                geomsg = Pose(position=Point(x=distance, y=0, z=0))
            elif axis == 'y':
                geomsg = Pose(position=Point(x=0, y=distance, z=0))
            elif axis == 'z'
                geomsg = Pose(position=Point(x=0, y=0, z=distance))
        else:
            geomsg = Pose(position=Point(x=distance[0], y=distance[1], z=distance[2]))
        self.relative_geomsg_pub.publish(geomsg)

    def pub_relative_joints_to_moveit(self, joints_array):
        self.check_publishers_connection()

        jointState = JointState()
        jointState.header = Header()
        jointState.header.stamp = rospy.Time.now()
        jointState.name = ['elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint',
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        jointState.position = joints_array
        jointState.velocity = []
        jointState.effort = []
        self.relative_joint_pub.publish(jointState)


    def set_joints(self, array=[1.5,-1.2,1.4,-1.87,-1.57,0]):
        # reset_req = SetModelConfigurationRequest()
        # reset_req.model_name = 'pickbot'
        # reset_req.urdf_param_name = 'robot_description'
        # reset_req.joint_names =[ 'elbow_joint', 'shoulder_lift_joint','shoulder_pan_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        # reset_req.joint_positions = array
        # res = self.reset_joints(reset_req)
        self.pub_joints_to_moveit(self.init_pos)


if __name__=="__main__":
    rospy.init_node('joint_array_publisher_node')
    joint_publisher = JointArrayPub()
