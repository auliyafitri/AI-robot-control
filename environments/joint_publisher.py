#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetModelConfiguration
from gazebo_msgs.srv import SetModelConfigurationRequest

import math
import random
import numpy as np

class JointPub(object):
    def __init__(self):

        self.publishers_array = []
        self._joint1_pub = rospy.Publisher('/pickbot/joint3_position_controller/command', Float64, queue_size=1)
        self._joint2_pub = rospy.Publisher('/pickbot/joint2_position_controller/command', Float64, queue_size=1)
        self._joint3_pub = rospy.Publisher('/pickbot/joint1_position_controller/command', Float64, queue_size=1)
        self._joint4_pub = rospy.Publisher('/pickbot/joint4_position_controller/command', Float64, queue_size=1)
        self._joint5_pub = rospy.Publisher('/pickbot/joint5_position_controller/command', Float64, queue_size=1)
        self._joint6_pub = rospy.Publisher('/pickbot/joint6_position_controller/command', Float64, queue_size=1)


        self.publishers_array.append(self._joint1_pub)
        self.publishers_array.append(self._joint2_pub)
        self.publishers_array.append(self._joint3_pub)
        self.publishers_array.append(self._joint4_pub)
        self.publishers_array.append(self._joint5_pub)
        self.publishers_array.append(self._joint6_pub)
        self.init_pos = [1.5,-1.2,1.4,-1.87,-1.57,0] #[0, 0, 0, 0, 0, 0]

        self.reset_joints =  rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)

    def set_init_pose(self):
        """
        Sets joints to initial position 
        :return:
        """
        #self.check_publishers_connection()
        self.move_joints(self.init_pos)

    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(100)  # 10hz
        while (self._joint1_pub.get_num_connections() == 0):
            rospy.logdebug("No susbribers to _joint1_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_joint1_pub Publisher Connected")

        while (self._joint2_pub.get_num_connections() == 0):
            rospy.logdebug("No susbribers to _joint2_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_joint3_pub Publisher Connected")

        while (self._joint2_pub.get_num_connections() == 0):
            rospy.logdebug("No susbribers to _joint3_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_joint3_pub Publisher Connected")

        while (self._joint4_pub.get_num_connections() == 0):
            rospy.logdebug("No susbribers to _joint4_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_joint4_pub Publisher Connected")

        while (self._joint5_pub.get_num_connections() == 0):
            rospy.logdebug("No susbribers to _joint5_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_joint5_pub Publisher Connected")

        while (self._joint6_pub.get_num_connections() == 0):
            rospy.logdebug("No susbribers to _joint6_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_joint6_pub Publisher Connected")
        rospy.logdebug("All Publishers READY")


    def move_joints(self, joints_array):    

        i = 0
        self.check_publishers_connection()
        for Publisher_object in self.publishers_array:
            joint_value = Float64()
            joint_value.data = joints_array[i]
            rospy.logdebug("JointsPos>>"+str(joint_value))
            Publisher_object.publish(joint_value)
            i += 1
            
    
    def set_joints(self, array=[1.5,-1.2,1.4,-1.87,-1.57,0]):
        reset_req = SetModelConfigurationRequest()
        reset_req.model_name = 'pickbot'
        reset_req.urdf_param_name = 'robot_description'
        reset_req.joint_names =[ 'elbow_joint', 'shoulder_lift_joint','shoulder_pan_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        reset_req.joint_positions = array
        res = self.reset_joints(reset_req) 







    def joint_mono_des_callback(self, msg):
        rospy.logdebug(str(msg.joint_state.position))

        self.move_joints(msg.joint_state.position)



    def start_loop(self, rate_value):
        rospy.loginfo("Start Loop")
        pos1 = [3,3]
        pos2 = [-3,-3]
        position = "pos1"
        rate = rospy.Rate(rate_value)
        while not rospy.is_shutdown():
          if position == "pos1":
            self.move_joints(pos1)
            position = "pos2"
          else:
            self.move_joints(pos2)
            position = "pos1"
          rate.sleep()

    def start_sinus_loop(self, rate_value):
        rospy.logdebug("Start Loop")
        w = 0.0
        x = 2.0*math.sin(w)
        #pos_x = [0.0,0.0,x]
        #pos_x = [x, 0.0, 0.0]
        pos_x = [0.0, x, 0.0]
        rate = rospy.Rate(rate_value)
        while not rospy.is_shutdown():
            self.move_joints(pos_x)
            w += 0.05
            x = 2.0 * math.sin(w)
            #pos_x = [0.0, 0.0, x]
            #pos_x = [x, 0.0, 0.0]
            pos_x = [0.0, x, 0.0]
            rate.sleep()



if __name__=="__main__":
    rospy.init_node('joint_publisher_node')
    joint_publisher = JointPub()
