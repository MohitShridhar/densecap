#!/usr/bin/env python

import rospy
import json
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

import actionlib
import action_controller.msg
import copy

import time

imgs = []
result_img = None

build_map = True

def pub_image():
    rospy.init_node('ImagePublisher', anonymous=True)

    client = actionlib.SimpleActionClient('dense_caption', action_controller.msg.DenseCaptionAction)
    client.wait_for_server()

    # Generic Objects
    img = cv2.imread('zebra.jpg',cv2.IMREAD_COLOR)
    imgs.append(img)
    msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    if build_map:
        goal = action_controller.msg.DenseCaptionGoal(1, msg_frame, 300, 40, 20, 0.7, 0.3, True)
        client.send_goal(goal, done_cb=done_cb)
        client.wait_for_result()

    img = cv2.imread('terman_eng.jpg',cv2.IMREAD_COLOR)
    imgs.append(img)
    msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    if build_map:
        goal = action_controller.msg.DenseCaptionGoal(2, msg_frame, 300, 40, 20, 0.7, 0.3, True)
        client.send_goal(goal, done_cb=done_cb)
        client.wait_for_result()

    img = cv2.imread('living_room.png',cv2.IMREAD_COLOR)
    imgs.append(img)
    msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    if build_map:
        goal = action_controller.msg.DenseCaptionGoal(3, msg_frame, 300, 40, 20, 0.7, 0.3, True)
        client.send_goal(goal, done_cb=done_cb)
        client.wait_for_result()

    img = cv2.imread('pisa.jpeg',cv2.IMREAD_COLOR)
    imgs.append(img)
    msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    if build_map:
        goal = action_controller.msg.DenseCaptionGoal(4, msg_frame, 300, 40, 20, 0.7, 0.3, True)
        client.send_goal(goal, done_cb=done_cb)
        client.wait_for_result()

    img = cv2.imread('clock.png',cv2.IMREAD_COLOR)
    imgs.append(img)
    msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    if build_map:
        goal = action_controller.msg.DenseCaptionGoal(5, msg_frame, 300, 40, 20, 0.7, 0.3, True)
        client.send_goal(goal, done_cb=done_cb)
        client.wait_for_result()

    img = cv2.imread('results/ref1/office.jpg',cv2.IMREAD_COLOR)
    imgs.append(img)
    msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    if build_map:
        goal = action_controller.msg.DenseCaptionGoal(6, msg_frame, 300, 40, 20, 0.7, 0.3, True)
        client.send_goal(goal, done_cb=done_cb)
        client.wait_for_result()


    # Amazon Picking Challenge

    # img = cv2.imread('results/ref2/shelf.jpg',cv2.IMREAD_COLOR)
    # imgs.append(img)
    # msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    # if build_map:
    #     goal = action_controller.msg.DenseCaptionGoal(0, msg_frame, 1240, 500, 500, 0.7, 0.3, True)
    #     client.send_goal(goal, done_cb=done_cb)
    #     client.wait_for_result()


    # Ultimate Clock Challenge
    # img = cv2.imread('clock.png',cv2.IMREAD_COLOR)
    # imgs.append(img)
    # msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    # if build_map:
        # goal = action_controller.msg.DenseCaptionGoal(1, msg_frame, 300, 40, 20, 0.7, 0.3, True)
        # client.send_goal(goal, done_cb=done_cb)
        # client.wait_for_result()

    # img = cv2.imread('results/ref3/grand.jpg',cv2.IMREAD_COLOR)
    # imgs.append(img)
    # msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    # if build_map:
        # goal = action_controller.msg.DenseCaptionGoal(2, msg_frame, 300, 40, 20, 0.7, 0.3, True)
        # client.send_goal(goal, done_cb=done_cb)
        # client.wait_for_result()

    # img = cv2.imread('results/ref3/on_the_wall.jpg',cv2.IMREAD_COLOR)
    # imgs.append(img)
    # msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    # if build_map:
        # goal = action_controller.msg.DenseCaptionGoal(3, msg_frame, 300, 40, 20, 0.7, 0.3, True)
        # client.send_goal(goal, done_cb=done_cb)
        # client.wait_for_result()

    # img = cv2.imread('results/ref3/timezones.jpg',cv2.IMREAD_COLOR)
    # imgs.append(img)
    # msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
    # if build_map:
        # goal = action_controller.msg.DenseCaptionGoal(4, msg_frame, 300, 40, 20, 0.7, 0.3, True)
        # client.send_goal(goal, done_cb=done_cb)
        # client.wait_for_result()    


    # Read test cases:
    with open('experiments/data/small_set.json') as data_file:
        data = json.load(data_file)

    total_tests = 0
    total_rank1_successes = 0
    total_rank2_successes = 0
    total_rank3_successes = 0

    start_time = time.time()

    # Query tests:
    for test in data["set"]:
        query = test["query"]
        client = actionlib.SimpleActionClient('dense_query', action_controller.msg.DenseImageQueryAction)
        client.wait_for_server()    
        goal = action_controller.msg.DenseImageQueryGoal(query, 6.0)
        client.send_goal(goal, done_cb=query_done_cb)
        client.wait_for_result()

        result = client.get_result()

        total_tests += 1
        if result.frame_ids[0] == test["gt_frame"]:
            total_rank1_successes += 1
        elif result.frame_ids[1] == test["gt_frame"]:
            total_rank2_successes += 1
        elif result.frame_ids[2] == test["gt_frame"]:
            total_rank3_successes += 1 

    print("Time: --- %s seconds ---" % (time.time() - start_time))

    total_successes = total_rank1_successes + total_rank2_successes + total_rank3_successes

    print "Rank Dist: Rank1 - %d, Rank2 - %d, Rank3 - %d" % (total_rank1_successes, total_rank2_successes, total_rank3_successes)
    print "Success Rate: %d/%d = %f" % (total_successes, total_tests, total_successes/(1.0*total_tests)*100.0 )

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