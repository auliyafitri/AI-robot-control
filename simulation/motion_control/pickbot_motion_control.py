#!/usr/bin/env python
# python2.7
import os
import cv2
import sys
import time
import math
import rospy
import copy
import tf
import numpy as np
import moveit_commander 
import moveit_msgs.msg 
import geometry_msgs.msg 
# from robotiq_c_model_control.msg import _CModel_robot_output as outputMsg

from cv_bridge import CvBridge, CvBridgeError
from scipy.misc import imsave



# MESSAGES/SERVICES
from std_msgs.msg import String
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState, Image
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

##___GLOBAL VARIABLES___###


##___INITIALIZATION___###
moveit_commander.roscpp_initialize(sys.argv) #initialize the moveit commander
rospy.init_node('move_group_python_interface', anonymous=True) #initialize the node 
robot = moveit_commander.RobotCommander() #define the robot
scene = moveit_commander.PlanningSceneInterface() #define the scene
group = moveit_commander.MoveGroupCommander("manipulator") #define the planning group (from the moveit packet 'manipulator' planning group)
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory) #publisher that publishes a plan to the topic: '/move_group/display_planned_path'
# gripper_publisher = rospy.Publisher('CModelRobotOutput', outputMsg.CModel_robot_output)
tf_listener = tf.TransformListener()
tf_broadcaster = tf.TransformBroadcaster()
group.set_end_effector_link("gripper_contactsensor_link_1") #TODO: change it to vacuum_gripper_link


def joint_position_sub():
    rospy.Subscriber('/pickbot/target_joint_positions', JointState, joint_callback)
    rospy.Subscriber('/pickbot/relative_joint_positions', JointState, relative_joint_callback)


def joint_callback(data):
    pos = data.position
    pub = rospy.Publisher('/pickbot/movement_complete', Bool, queue_size=10)
    complete_msg = Bool()
    complete_msg.data = False
    check_publishers_connection(pub)
    pub.publish(complete_msg)

    assign_joint_value(pos[2], pos[1], pos[0], pos[3], pos[4], pos[5])

    # check_publishers_connection(pub)
    complete_msg.data = True
    pub.publish(complete_msg)


def pose_callback(data):
    pos = data.position
    pub = rospy.Publisher('/pickbot/movement_complete', Bool, queue_size=10)
    complete_msg = Bool()
    complete_msg.data = False
    check_publishers_connection(pub)
    pub.publish(complete_msg)

    assign_pose_target(pos.x, pos.y, pos.z, 'nil', 'nil', 'nil', 'nil')

    complete_msg.data = True
    pub.publish(complete_msg)


def relative_pose_callbak(data):
    pos = data.position
    pub = rospy.Publisher('/pickbot/movement_complete', Bool, queue_size=10)
    complete_msg = Bool()
    complete_msg.data = False
    check_publishers_connection(pub)
    pub.publish(complete_msg)

    if pos.x != 0:
        relative_pose_target(axis_world='x', distance=pos.x)
    if pos.y != 0:
        relative_pose_target(axis_world='y', distance=pos.y)
    if pos.z != 0:
        relative_pose_target(axis_world='z', distance=pos.z)

    complete_msg.data = True
    pub.publish(complete_msg)


def relative_joint_callback(data):
    pos = data.position
    pub = rospy.Publisher('/pickbot/movement_complete', Bool, queue_size=10)
    complete_msg = Bool()
    complete_msg.data = False
    check_publishers_connection(pub)
    pub.publish(complete_msg)

    relative_joint_value(pos[2], pos[1], pos[0], pos[3], pos[4], pos[5])

    # check_publishers_connection(pub)
    complete_msg.data = True
    pub.publish(complete_msg)


def check_publishers_connection(pub):
    """
    Checks that all the publishers are working
    :return:
    """
    rate = rospy.Rate(100)  # 10hz
    while (pub.get_num_connections() == 0):
        rospy.logdebug("No subscribers to _joint1_pub yet so we wait and try again")
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            # This is to avoid error when world is rested, time when backwards.
            pass
    rospy.logdebug("joint_pub Publisher Connected")


###___REGRASP FUNCTION___###
## Regrasp thin object by simultaneously tiliting end-effector and widening grip (unit: mm)
def regrasp(theta, length, phi_target, axis, direction, tilt_axis, tilt_direction): # Assumption is that initial conditions are phi = 0 and opposite = length
     
    resol = 1 # set resolution of incremental movements with respect to phi (unit: degrees)
    rate_hz = 10 # set speed of regrasp by setting update frequency (hz)
    phi_current = 0.0
    i = 1
    while phi_current < phi_target: 
        
        opposite = length * math.sin(math.radians(90-phi_current))

        center_of_rotation = get_instantaneous_center(opposite, rate_hz)

        width = opposite / math.tan(math.radians(90-phi_current+1))
        position = int((width - 146.17)/(-0.6584)) # Gripper position from a range of (0-255)
        phi_current = phi_current + resol
        i += 1 
        set_gripper_position(position) #increment gripper width   
        if axis is 'x':
            TurnArcAboutAxis('x', center_of_rotation[1], center_of_rotation[2], resol, direction, 'yes', tilt_axis, tilt_direction)       
        if axis is 'y':
            TurnArcAboutAxis('y', center_of_rotation[2], center_of_rotation[0], resol, direction, 'yes', tilt_axis, tilt_direction)       
        if axis is 'z':
            TurnArcAboutAxis('z', center_of_rotation[0], center_of_rotation[1], resol, direction, 'yes', tilt_axis, tilt_direction)       
        #print 'Position: ', position, ' CoR: ', center_of_rotation #' phi_current: ', phi_current, ' width: ', width, ' opposite: ', opposite #debug
    

## Get instantaneous center of rotation for regrasp() function
def get_instantaneous_center(opposite, rate_hz):
    rate = rospy.Rate(rate_hz)       
    displacement = 0.277-(opposite/2)/1000
    
    tf_listener.waitForTransform('/base_link', '/ee_link', rospy.Time(), rospy.Duration(4.0))
    (trans1, rot1) = tf_listener.lookupTransform('/base_link', '/ee_link', rospy.Time(0)) #listen to transform between base_link2ee_link
    base2eelink_matrix = tf_listener.fromTranslationRotation(trans1, rot1) #change base2eelink from transform to matrix
    eelink2eetip_matrix = tf_listener.fromTranslationRotation((displacement, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)) #change eelink2eetip from transform to matrix
    base2eetip_matrix = np.matmul(base2eelink_matrix, eelink2eetip_matrix) #combine transformation: base2eetip = base2eelink x eelink2eetip
    scale, shear, rpy_angles, translation_vector, perspective = tf.transformations.decompose_matrix(base2eetip_matrix) #change base2eetip from matrix to transform
    quaternion = tf.transformations.quaternion_from_euler(rpy_angles[0], rpy_angles[1], rpy_angles[2])
    rate.sleep()
    return translation_vector
    #print translation_vector, quaternion #debug
    #print base2eetip_matrix #debug
    #print base2eelink_matrix #debug


###___TURN ARC FUNCTION___###
## Turns about a reference center point in path mode or tilt mode 
## User specifies axis:['x'/'y'/'z'], Center of Circle: [y,z / z,x / x,y], Arc turn angle: [degrees], Direction: [1/-1], Tilt Mode: ['yes'/'no'], End_effector tilt axis: ['x'/'y'/'z'], Tilt direction: [1/-1]   
def TurnArcAboutAxis(axis, CenterOfCircle_1, CenterOfCircle_2, angle_degree, direction, tilt, tilt_axis, tilt_direction):
    
    pose_target = group.get_current_pose().pose #create a pose variable. The parameters can be seen from "$ rosmsg show Pose"
    waypoints = []
    waypoints.append(pose_target)
    resolution = 360 #Calculation of resolution by (180/resolution) degrees 
    #define the axis of rotation
    if axis is 'x' :
        position_1 = pose_target.position.y
        position_2 = pose_target.position.z
    if axis is 'y' :
        position_1 = pose_target.position.z
        position_2 = pose_target.position.x
    if axis is 'z' :
        position_1 = pose_target.position.x
        position_2 = pose_target.position.y

    circle_radius = ((position_1 - CenterOfCircle_1)**2 + (position_2 - CenterOfCircle_2)**2)**0.5 #Pyth. Theorem to find radius
    
    #calculate the global angle with respect to 0 degrees based on which quadrant the end_effector is in 
    if position_1 > CenterOfCircle_1 and position_2 > CenterOfCircle_2:
        absolute_angle = math.asin(math.fabs(position_2 - CenterOfCircle_2) / circle_radius)
    if position_1 < CenterOfCircle_1 and position_2 > CenterOfCircle_2:
        absolute_angle = math.pi - math.asin(math.fabs(position_2 - CenterOfCircle_2) / circle_radius)
    if position_1 < CenterOfCircle_1 and position_2 < CenterOfCircle_2:
        absolute_angle = math.pi + math.asin(math.fabs(position_2 - CenterOfCircle_2) / circle_radius)
    if position_1 > CenterOfCircle_1 and position_2 < CenterOfCircle_2:
        absolute_angle = 2.0*math.pi - math.asin(math.fabs(position_2 - CenterOfCircle_2) / circle_radius)
    
    theta = 0 # counter that increases the angle     
    while theta < angle_degree/180.0 * math.pi:
        if axis is 'x' :
            pose_target.position.y = circle_radius * math.cos(theta*direction+absolute_angle)+CenterOfCircle_1 #equation of circle from polar to cartesian x = r*cos(theta)+dx
            pose_target.position.z = circle_radius * math.sin(theta*direction+absolute_angle)+CenterOfCircle_2 #equation of cirlce from polar to cartesian y = r*sin(theta)+dy 
        if axis is 'y' :
            pose_target.position.z = circle_radius * math.cos(theta*direction+absolute_angle)+CenterOfCircle_1
            pose_target.position.x = circle_radius * math.sin(theta*direction+absolute_angle)+CenterOfCircle_2
        if axis is 'z' :
            pose_target.position.x = circle_radius * math.cos(theta*direction+absolute_angle)+CenterOfCircle_1
            pose_target.position.y = circle_radius * math.sin(theta*direction+absolute_angle)+CenterOfCircle_2
        
        ## Maintain orientation with respect to turning axis  
        if tilt is 'yes':      
            pose_target = TiltAboutAxis(pose_target, resolution, tilt_axis, tilt_direction)

        waypoints.append(copy.deepcopy(pose_target))
        theta+=math.pi/resolution # increment counter, defines the number of waypoints 
    del waypoints[:2]
    plan_execute_waypoints(waypoints)

            
def TiltAboutAxis(pose_target, resolution, tilt_axis, tilt_direction):
    quaternion = (
        pose_target.orientation.x,
        pose_target.orientation.y,
        pose_target.orientation.z,
        pose_target.orientation.w)
            
   # euler = quaternion_to_euler(quaternion[0], quaternion[1], quaternion[2], quaternion[3])     
    euler = tf.transformations.euler_from_quaternion(quaternion) # convert quaternion to euler
    roll = euler[0]
    pitch = euler[1]
    yaw = euler [2]   
    # increment the orientation angle
    if tilt_axis is 'x' :
        roll += tilt_direction*math.pi/resolution
    if tilt_axis is 'y' :
        pitch += tilt_direction*math.pi/resolution
    if tilt_axis is 'z' :
        yaw += tilt_direction*math.pi/resolution
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw) # convert euler to quaternion
    # store values to pose_target
    pose_target.orientation.x = quaternion[0]
    pose_target.orientation.y = quaternion[1]
    pose_target.orientation.z = quaternion[2]
    pose_target.orientation.w = quaternion[3]
    return pose_target


def assign_joint_value(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5):
    # group.set_max_velocity_scaling_factor(0.1)
    group_variable_values = group.get_current_joint_values() #create variable that stores joint values
    # print("group: {}".format(np.round(group_variable_values, decimals=3)))

    #Assign values to joints
    group_variable_values[0] = joint_0
    group_variable_values[1] = joint_1
    group_variable_values[2] = joint_2
    group_variable_values[3] = joint_3
    group_variable_values[4] = joint_4
    group_variable_values[5] = joint_5

    group.set_joint_value_target(group_variable_values) #set target joint values for 'manipulator' group


    # group.plan() #call plan function to plan the path (visualize on rviz)
    group.go(group_variable_values, wait=True) #execute plan on real/simulation (gazebo) robot
    group.stop()
    # rospy.sleep(0.1)


def assign_pose_target(pos_x, pos_y, pos_z, orient_x, orient_y, orient_z, orient_w):
    # group.set_max_velocity_scaling_factor(0.1)
    pose_target = group.get_current_pose() # create a pose variable. The parameters can be seen from "$ rosmsg show Pose"

    #Assign values
    if pos_x is 'nil':
        pass
    else:     
        pose_target.pose.position.x = pos_x
    if pos_y is 'nil':
        pass
    else:    
        pose_target.pose.position.y = pos_y
    if pos_z is 'nil':
        pass
    else:    
        pose_target.pose.position.z = pos_z
    if orient_x is 'nil':
        pass
    else: 
        pose_target.pose.orientation.x = orient_x
    if orient_y is 'nil':
        pass
    else: 
        pose_target.pose.orientation.y = orient_y
    if orient_z is 'nil':
        pass
    else: 
        pose_target.pose.orientation.z = orient_z
    if orient_w is 'nil':
        pass
    else:     
        pose_target.pose.orientation.w = orient_w
    
    group.set_pose_target(pose_target) #set pose_target as the goal pose of 'manipulator' group
    # plan2 = group.plan()
    group.go(pose_target, wait=True)
    group.stop()
    # rospy.sleep(2)


def relative_joint_value(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5):
    group.set_max_velocity_scaling_factor(0.1)
    group_variable_values = group.get_current_joint_values() #create variable that stores joint values

    #Assign values to joints
    group_variable_values[0] += joint_0
    group_variable_values[1] += joint_1
    group_variable_values[2] += joint_2
    group_variable_values[3] += joint_3
    group_variable_values[4] += joint_4
    group_variable_values[5] += joint_5

    group.set_joint_value_target(group_variable_values) #set target joint values for 'manipulator' group
 
    # plan1 = group.plan()
    group.go(group_variable_values, wait=True)
    group.stop()
    # rospy.sleep(2) #sleep 2 seconds


def relative_pose_target(axis_world, distance):
    group.set_max_velocity_scaling_factor(0.1)
    pose_target = group.get_current_pose() #create a pose variable. The parameters can be seen from "$ rosmsg show Pose"
    if axis_world is 'x':
        pose_target.pose.position.x += distance
    if axis_world is 'y':
        pose_target.pose.position.y += distance
    if axis_world is 'z':
        pose_target.pose.position.z += distance
    group.set_pose_target(pose_target) #set pose_target as the goal pose of 'manipulator' group 

    # plan2 = group.plan()
    group.go(pose_target, wait=True)
    group.stop()
    # rospy.sleep(2)


def plan_execute_waypoints(waypoints):
    (plan3, fraction) = group.compute_cartesian_path(waypoints, 0.01, 0) #parameters(waypoints, resolution_1cm, jump_threshold)
    plan= group.retime_trajectory(robot.get_current_state(), plan3, 0.1) #parameter that changes velocity
    group.execute(plan) 


###___STATUS ROBOT___###
def manipulator_status():
    #You can get a list with all the groups of the robot like this:
    print("Robot Groups:")
    print(robot.get_group_names())

    #You can get the current values of the joints like this:
    print("Current Joint Values:")
    print(group.get_current_joint_values())

    #You can also get the current Pose of the end effector of the robot like this:
    print("Current Pose:")
    print(group.get_current_pose())

    #Finally you can check the general status of the robot like this:
    print("Robot State:")
    print(robot.get_current_state())


def get_distance_gripper_to_object():
    """
    Get the Position of the endeffektor and the object via rosservice call /gazebo/get_model_state and /gazebo/get_link_state
    Calculate distance between them

    In this case

    Object:     unite_box_0 link
    Gripper:    vacuum_gripper_link ground_plane
    """

    try:
        model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        blockName = "unit_box_0"
        relative_entity_name = "link"
        object_resp_coordinates = model_coordinates(blockName, relative_entity_name)
        Object = np.array((object_resp_coordinates.pose.position.x, object_resp_coordinates.pose.position.y,
                            object_resp_coordinates.pose.position.z))

        print("Object: {}".format(Object))

    except rospy.ServiceException as e:
        rospy.loginfo("Get Model State service call failed:  {0}".format(e))
        print("Exception get model state")

    try:
        model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        LinkName = "vacuum_gripper_link"
        ReferenceFrame = "ground_plane"
        resp_coordinates_gripper = model_coordinates(LinkName, ReferenceFrame)
        Gripper = np.array((resp_coordinates_gripper.link_state.pose.position.x,
                            resp_coordinates_gripper.link_state.pose.position.y,
                            resp_coordinates_gripper.link_state.pose.position.z))
        
        print("Gripper position: {},{},{}".format(resp_coordinates_gripper.link_state.pose.position.x,
                                                resp_coordinates_gripper.link_state.pose.position.y,
                                                resp_coordinates_gripper.link_state.pose.position.z))

    except rospy.ServiceException as e:
        rospy.loginfo("Get Link State service call failed:  {0}".format(e))
        print("Exception get Gripper position")
    distance = np.linalg.norm(Object - Gripper)

    return distance, Object


if __name__ == '__main__':

    # group.set_end_effector_link('vacuum_gripper_link')

    rospy.Subscriber('/pickbot/target_joint_positions/', JointState, joint_callback)
    rospy.Subscriber('/pickbot/relative_joint_positions', JointState, relative_joint_callback)
    rospy.Subscriber('/pickbot/target_pose', Pose, pose_callback)
    rospy.Subscriber('/pickbot/relative_target_pose', Pose, relative_pose_callbak)
    print("listening to joint states now")
    rospy.spin()