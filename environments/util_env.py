import os
import yaml
import numpy as np
import rospy
import rospkg
import csv
import random
import environments

from transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import Point, Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import SpawnModel

from simulation.srv import VacuumGripperControl


def get_target_object(object_type='free_shapes'):
    # get list of target object
    targetobj_fname = os.path.dirname(environments.__file__) + '/object_information.yml'
    with open(targetobj_fname, "r") as stream:
        out = yaml.load(stream)
        filtered_object = list(filter(lambda x: x["type"] == object_type, out['items']))
        return filtered_object


def get_state(observation):
    """
    convert observation list into a numpy array
    """
    x = np.asarray(observation)
    return x


def turn_on_gripper():
    """
    turn on the Gripper by calling the service
    """
    try:
        turn_on_gripper_service = rospy.ServiceProxy('/pickbot/gripper/control', VacuumGripperControl)
        enable = True
        turn_on_gripper_service(enable)
    except rospy.ServiceException as e:
        rospy.loginfo("Turn on Gripper service call failed:  {0}".format(e))


def turn_off_gripper():
    """
    turn off the Gripper by calling the service
    """
    try:
        turn_off_gripper_service = rospy.ServiceProxy('/pickbot/gripper/control', VacuumGripperControl)
        enable = False
        turn_off_gripper_service(enable)
    except rospy.ServiceException as e:
        rospy.loginfo("Turn off Gripper service call failed:  {0}".format(e))


def spawn_object(object_name, model_position, model_sdf=None):
    """
    spawn object using gazebo service
    :param object_name: name of the object
    :param model_position: position of the spawned object
    :param model_sdf: description of the object in sdf format
    :return: -
    """
    if model_sdf is None:  # take sdf file from default folder
        # get model from sdf file
        rospack = rospkg.RosPack()
        sdf_fname = rospack.get_path('simulation') + "/meshes/environments/sdf/" + object_name + ".sdf"
        with open(sdf_fname, "r") as f:
            model_sdf = f.read()

    try:
        spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        spawn_model(object_name, model_sdf, "/", model_position, "world")
        print("SPAWN %s finished" % object_name)
    except rospy.ServiceException as e:
        rospy.loginfo("Spawn Model service call failed:  {0}".format(e))


def spawn_urdf_object(object_name, model_position, model_urdf=None):
    """
    spawn object using gazebo service
    :param object_name: name of the object
    :param model_position: position of the spawned object
    :param model_sdf: description of the object in sdf format
    :return: -
    """
    if model_urdf is None:  # take sdf file from default folder
        # get model from sdf file
        rospack = rospkg.RosPack()
        urdf_fname = rospack.get_path('simulation') + "/urdf/" + object_name + ".urdf"
        with open(urdf_fname, "r") as f:
            model_urdf = f.read()

    try:
        spawn_model = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)
        spawn_model(object_name, model_urdf, "/", model_position, "world")
        print("SPAWN %s finished" % object_name)
    except rospy.ServiceException as e:
        rospy.loginfo("Spawn Model service call failed:  {0}".format(e))


def delete_object(object_name):
    """
    delete object using gazebo service
    :param object_name: the name of the object
    :return: -
    """
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        delete_model(object_name)
    except rospy.ServiceException as e:
        rospy.loginfo("Delete Model service call failed:  {0}".format(e))


def change_object_position(object_name, model_position):
    """
    change object postion using gazebo service
    :param object_name: name of the object
    :param model_position: destination
    :return: -
    """
    try:
        change_position = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        box = ModelState()
        box.model_name = object_name
        box.pose.position.x = model_position.position.x
        box.pose.position.y = model_position.position.y
        box.pose.position.z = model_position.position.z
        box.pose.orientation.x = model_position.orientation[1]
        box.pose.orientation.y = model_position.orientation[2]
        box.pose.orientation.z = model_position.orientation[3]
        box.pose.orientation.w = model_position.orientation[0]
        change_position(box)
    except rospy.ServiceException as e:
        rospy.loginfo("Set Model State service call failed:  {0}".format(e))


def get_target_position():
    try:
        model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        linkNameTarget = "target"
        ReferenceFrame = "ground_plane"
        object_resp_coordinates = model_coordinates(linkNameTarget, ReferenceFrame)
        Object = np.array((object_resp_coordinates.link_state.pose.position.x, object_resp_coordinates.link_state.pose.position.y, object_resp_coordinates.link_state.pose.position.z))

    except rospy.ServiceException as e:
        rospy.loginfo("Get Link State service call failed:  {0}".format(e))
        print("Exception get link state")
    
    return Object

def get_gripper_position():
    try:
        model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        LinkName = "vacuum_gripper_link"
        ReferenceFrame = "ground_plane"
        resp_coordinates_gripper = model_coordinates(LinkName, ReferenceFrame)
        Gripper = np.array((resp_coordinates_gripper.link_state.pose.position.x,
                            resp_coordinates_gripper.link_state.pose.position.y,
                            resp_coordinates_gripper.link_state.pose.position.z))

    except rospy.ServiceException as e:
        rospy.loginfo("Get Link State service call failed:  {0}".format(e))
        print("Exception get Gripper position")

    return Gripper


def get_distance_gripper_to_object(height=None):
    """
    Get the Position of the endeffektor and the object via rosservice /gazebo/get_link_state
    Calculate distance between them

    In this case

    Object:     unite_box_0 link
    Gripper:    vacuum_gripper_link ground_plane
    """

    try:
        model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        linkNameTarget = "target"
        ReferenceFrame = "ground_plane"
        object_resp_coordinates = model_coordinates(linkNameTarget, ReferenceFrame)
        Object = np.array((object_resp_coordinates.link_state.pose.position.x, 
                            object_resp_coordinates.link_state.pose.position.y,
                            object_resp_coordinates.link_state.pose.position.z if height is None else object_resp_coordinates.link_state.pose.position.z + height/2))

    except rospy.ServiceException as e:
        rospy.loginfo("Get Link State service call failed:  {0}".format(e))
        print("Exception get link state")

    try:
        model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        LinkName = "vacuum_gripper_link"
        ReferenceFrame = "ground_plane"
        resp_coordinates_gripper = model_coordinates(LinkName, ReferenceFrame)
        Gripper = np.array((resp_coordinates_gripper.link_state.pose.position.x,
                            resp_coordinates_gripper.link_state.pose.position.y,
                            resp_coordinates_gripper.link_state.pose.position.z))

    except rospy.ServiceException as e:
        rospy.loginfo("Get Link State service call failed:  {0}".format(e))
        print("Exception get Gripper position")
    distance = np.linalg.norm(Object - Gripper)

    return distance, Object


def get_random_door_handle_pos():
    positions = [[-0.26, 0.848, 1.1, 0, 0, 1.57],
                 [-0.195, 0.848, 1.1, 0, 0, 1.57],
                 [-0.13, 0.848, 1.1, 0, 0, 1.57],
                 [-0.065, 0.848, 1.1, 0, 0, 1.57],
                 [0, 0.848, 1.1, 0, 0, 1.57],
                 [0.065, 0.848, 1.1, 0, 0, 1.57],
                 [0.13, 0.848, 1.1, 0, 0, 1.57],
                 [0.196, 0.848, 1.1, 0, 0, 1.57],
                 [-0.224, 0.995, 1.1, 0, 0, -1.57],
                 [-0.159, 0.995, 1.1, 0, 0, -1.57],
                 [-0.094, 0.995, 1.1, 0, 0, -1.57],
                 [-0.029, 0.995, 1.1, 0, 0, -1.57],
                 [0.036, 0.995, 1.1, 0, 0, -1.57],
                 [0.101, 0.995, 1.1, 0, 0, -1.57],
                 [0.166, 0.995, 1.1, 0, 0, -1.57],
                 [0.231, 0.995, 1.1, 0, 0, -1.57]]

    rand = random.choice(positions)
    random_door_pos = Pose(position=Point(x=rand[0], y=rand[1], z=rand[2]),
                           orientation=quaternion_from_euler(rand[3], rand[4], rand[5]))
    return random_door_pos


def append_to_csv(csv_filename, anarray):
    outfile = open(csv_filename, 'a')
    writer = csv.writer(outfile)
    writer.writerow(anarray)
    outfile.close()

def dict_to_csv(csv_filename, dictionary):
    with open(csv_filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictionary.items():
            writer.writerow([key, value])

def load_samples_from_prev_task(filename):  # currently for door handle only
    output = np.genfromtxt(filename, delimiter=',')
    action = output[:, 0:6]
    pos = output[:, -7:]
    # pos_orient = np.empty((0, 7))

    # for i in range(len(pos)):
    #     _pos = pos[i]
    #     if pos[i, 1] > 0.99:
    #         _ori = [0, 0, -1.57]
    #     else:
    #         _ori = [0, 0, 1.57]
    #     _pos_ori = np.append(_pos, _ori, axis=0)
    #     pos_orient = np.append(pos_orient, [_pos_ori], axis=0)

    samples = np.append(action, pos, axis=1)
    return samples


def get_obj_orient():
    try:
        model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        linkNameTarget = "target"
        ReferenceFrame = "ground_plane"
        object_resp_geometry = model_coordinates(linkNameTarget, ReferenceFrame)
        object_orient = np.array((object_resp_geometry.link_state.pose.orientation.w,
                                  object_resp_geometry.link_state.pose.orientation.x,
                                  object_resp_geometry.link_state.pose.orientation.y,
                                  object_resp_geometry.link_state.pose.orientation.z))

        object_orient = euler_from_quaternion(object_orient)

    except rospy.ServiceException as e:
        rospy.loginfo("Get Link State service call failed:  {0}".format(e))
        print("Exception get link state")

    try:
        model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        LinkName = "vacuum_gripper_link"
        ReferenceFrame = "ground_plane"
        resp_coordinates_gripper = model_coordinates(LinkName, ReferenceFrame)
        gripper_orient = np.array((resp_coordinates_gripper.link_state.pose.orientation.w,
                                   resp_coordinates_gripper.link_state.pose.orientation.x,
                                   resp_coordinates_gripper.link_state.pose.orientation.y,
                                   resp_coordinates_gripper.link_state.pose.orientation.z))

        gripper_orient = euler_from_quaternion(gripper_orient)

    except rospy.ServiceException as e:
        rospy.loginfo("Get Link State service call failed:  {0}".format(e))
        print("Exception get Gripper position")

    return object_orient, gripper_orient


def get_link_state(link_name):
    """
    Get the Position and quaternion of link with ground plane as reference
    """

    try:
        model_geometry = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        linkNameTarget = link_name
        ReferenceFrame = "ground_plane"
        object_resp_geometry = model_geometry(linkNameTarget, ReferenceFrame)
        geometry = np.array((object_resp_geometry.link_state.pose.position.x,
                             object_resp_geometry.link_state.pose.position.y,
                             object_resp_geometry.link_state.pose.position.z,
                             object_resp_geometry.link_state.pose.orientation.w,
                             object_resp_geometry.link_state.pose.orientation.x,
                             object_resp_geometry.link_state.pose.orientation.y,
                             object_resp_geometry.link_state.pose.orientation.z))

    except rospy.ServiceException as e:
        rospy.loginfo("Get Link State service call failed:  {0}".format(e))
        print("Exception get link state")

    return geometry


if __name__ == '__main__':
    print(get_target_object())
