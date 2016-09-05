#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

import actionlib
import action_controller.msg

def pub_image():
    rospy.init_node('ImagePublisher', anonymous=True)


    img = cv2.imread('living_room.png',cv2.IMREAD_COLOR)

    # I want to publish the Canny Edge Image and the original Image
    msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")


    client = actionlib.SimpleActionClient('dense_caption', action_controller.msg.DenseCaptionAction)
    
    # Waits until the action server has started up and started
    # listening for goals.
    client.wait_for_server()
    
    # Creates a goal to send to the action server.
    goal = action_controller.msg.DenseCaptionGoal(msg_frame, 300, 100, 20, 0.7, 0.3)
    
    # Sends the goal to the action server.
    client.send_goal(goal, done_cb=done_cb)
    
    # Waits for the server to finish performing the action.
    client.wait_for_result()
    
    # Prints out the result of executing the action
    return client.get_result()  # A FibonacciResult
    
    print "Not waiting for the result anymore"

def done_cb(goal_status, result):
    print 'Done'
    print result


if __name__ == '__main__':
    try:
        pub_image()
    except rospy.ROSInterruptException:
        pass