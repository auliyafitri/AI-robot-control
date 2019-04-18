#!/usr/bin/env python3
import sys
import trollius
from trollius import From

import pygazebo
import pygazebo.msg.contacts_pb2
import pygazebo.msg.contact_pb2

import rospy
from std_msgs.msg import Bool

# global variables
allowed_comb = {
    'wrist_3_link': ['target'],
    'vacuum_gripper_link': ['target'],
    'target': ['wrist_3_link', 'vacuum_gripper_link']
}

ignored_comb = ['container', 'target']

collision_1 = ''
collision_2 = ''

contacts_result = []

# ROS Publisher
pub = rospy.Publisher('gz_collisions', Bool, queue_size=1)
rospy.init_node('gz_subscriber', anonymous=True)
rate = rospy.Rate(10)


@trollius.coroutine
def publish_loop():
    manager = yield From(pygazebo.connect())

    # print('connected')

    def callback(data):
        global contacts_result
        message = pygazebo.msg.contacts_pb2.Contacts.FromString(data)

        if not message.contact:  # no contact
            # print('No Collision')
            if not rospy.is_shutdown():
                # rospy.loginfo(False)
                pub.publish(False)
        else:  # have contact(s)
            # print(len(message.contact))
            dump_object(message)

            # print contacts_result
            _done = False
            if len(contacts_result) > 0:
                for i in contacts_result:
                    _done = _done or i

            if not rospy.is_shutdown():
                # rospy.loginfo(_done)
                pub.publish(_done)

            # empty the contacts list
            contacts_result = []

    subscriber = manager.subscribe('/gazebo/default/physics/contacts',
                                   'gazebo.msgs.Contacts',
                                   callback)

    while True:
        yield From(subscriber.wait_for_connection())
        yield From(trollius.sleep(0.01))


'''        
Iterator for repeated object
1. Assign collision_1
2. Assign collision_2 then check whether collision_1 and collision_2 are allowed combinations
    if yes, return true
    else, return false
3. Publish the bool value to ROS topic gz_collisions
4. Empty collision_1 and collision_2
'''


def dump_object(obj):
    global contacts_result
    for descriptor in obj.DESCRIPTOR.fields:
        value = getattr(obj, descriptor.name)
        # print(descriptor.type, descriptor.name)

        if descriptor.type == descriptor.TYPE_MESSAGE:
            if descriptor.label == descriptor.LABEL_REPEATED:
                map(dump_object, value)
            else:
                dump_object(value)
        else:
            global collision_1
            global collision_2

            if 'collision1' in descriptor.full_name:
                if 'target' in value:
                    collision_1 = 'target'
                else:  # container, pickbot
                    collision_1 = value.split('::')[1]  # container; wrist_1_link; wrist_2_link; wrist_3_link
                # print "%s: collision_1 %s" % (descriptor.full_name, collision_1)
            elif 'collision2' in descriptor.full_name:
                if 'target' in value:
                    collision_2 = 'target'
                else:  # container, pickbot
                    collision_2 = value.split('::')[1]  # container; wrist_1_link; wrist_2_link; wrist_3_link
                # print "%s: collision_2 %s" % (descriptor.full_name, collision_2)

            while not (collision_2 == ''):  # both collision_1 and collision_2 are assigned
                if not_to_be_ignored():  # not container and target
                    # print "collisions: %s and %s" % (collision_1, collision_2)
                    # print is_collision_allowed()
                    contacts_result.append(is_collision_allowed())
                # reinitialize both var
                collision_1 = ''
                collision_2 = ''


def is_collision_allowed():
    global collision_1
    global collision_2
    global allowed_comb

    if collision_1 in allowed_comb:
        if collision_2 in allowed_comb[collision_1]:
            return False
        else:
            return True
    else:
        return True


def not_to_be_ignored():
    global collision_1
    global collision_2
    global ignored_comb
    return not ((collision_1 in ignored_comb) and (collision_2 in ignored_comb))


def subsc_listen():
    global loop
    loop = trollius.get_event_loop()
    loop.run_until_complete(publish_loop())


if __name__ == '__main__':
    try:
        subsc_listen()
    except KeyboardInterrupt:
        # except rospy.ROSInterruptException:
        # pass
        sys.exit()
