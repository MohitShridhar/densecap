#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

import actionlib
import action_controller.msg
import copy

imgs = []
result_img = None

def pub_image():
    rospy.init_node('ImagePublisher', anonymous=True)

    client = actionlib.SimpleActionClient('dense_caption', action_controller.msg.DenseCaptionAction)
    client.wait_for_server()

    # Run captions
    img = cv2.imread('zebra.jpg',cv2.IMREAD_COLOR)
    imgs.append(img)
    msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    goal = action_controller.msg.DenseCaptionGoal(1, msg_frame, 210, 80, 20, 0.7, 0.3, True)
    client.send_goal(goal, done_cb=done_cb)
    client.wait_for_result()

    img = cv2.imread('terman_eng.jpg',cv2.IMREAD_COLOR)
    imgs.append(img)
    msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    goal = action_controller.msg.DenseCaptionGoal(2, msg_frame, 210, 80, 20, 0.7, 0.3, True)
    client.send_goal(goal, done_cb=done_cb)
    client.wait_for_result()

    img = cv2.imread('family.jpg',cv2.IMREAD_COLOR)
    imgs.append(img)
    msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    goal = action_controller.msg.DenseCaptionGoal(3, msg_frame, 210, 80, 20, 0.7, 0.3, True)
    client.send_goal(goal, done_cb=done_cb)
    client.wait_for_result()

    img = cv2.imread('living_room.png',cv2.IMREAD_COLOR)
    imgs.append(img)
    msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    goal = action_controller.msg.DenseCaptionGoal(4, msg_frame, 210, 80, 20, 0.7, 0.3, True)
    client.send_goal(goal, done_cb=done_cb)
    client.wait_for_result()

    # Query test
    client = actionlib.SimpleActionClient('dense_query', action_controller.msg.DenseImageQueryAction)
    client.wait_for_server()    
    goal = action_controller.msg.DenseImageQueryGoal("a man wearing a shirt", 6.0)
    client.send_goal(goal, done_cb=query_done_cb)
    client.wait_for_result()

    result = client.get_result()

    for i in range(0, 3):
        img = imgs[result.frame_ids[i]-1]
        offset = i*4

        x1 = int(result.boxes[0+offset])
        y1 = int(result.boxes[1+offset])
        x2 = int(result.boxes[0+offset]+result.boxes[2+offset])
        y2 = int(result.boxes[1+offset]+result.boxes[2+offset])

        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

        cv2.imshow('result', img)
        cv2.waitKey(0)

    # return client.get_result()  
    

def done_cb(goal_status, result):
    print 'Done'
    print result

def query_done_cb(goal_status, result):
    print 'Done'
    print result

if __name__ == '__main__':
    try:
        pub_image()
    except rospy.ROSInterruptException:
        pass