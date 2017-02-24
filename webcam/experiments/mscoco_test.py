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

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

from skimage import img_as_ubyte
import random

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

result_img = None

dataDir='/home/mohitshridhar/Programs/coco/'
dataType='train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)

MAX_IMAGES = 120
MAX_QUERIES = 60

# MAX_IMAGES = 2
# MAX_QUERIES = 3

captions = []

def pub_image():
    rospy.init_node('ImagePublisher', anonymous=True)

    client = actionlib.SimpleActionClient('dense_caption', action_controller.msg.DenseCaptionAction)
    client.wait_for_server()

    # catergory settings
    coco=COCO(annFile)
    # catIds = coco.getCatIds(catNms=['person','dog', 'clock', 'laptop']);
    imgIds = coco.getImgIds();

    # load coco test image
    for i in range(MAX_IMAGES):
        coco_img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
        I = io.imread('http://mscoco.org/images/%d'%(coco_img['id']))
        img = img_as_ubyte(I)        
        msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")
        goal = action_controller.msg.DenseCaptionGoal(i, msg_frame, 300, 40, 20, 0.7, 0.3, True)
        client.send_goal(goal, done_cb=done_cb)
        client.wait_for_result()

        # load and annotations
        capFile = '%s/annotations/captions_%s.json'%(dataDir,dataType)
        coco_caps=COCO(capFile)
        annIds = coco_caps.getAnnIds(imgIds=coco_img['id']);
        anns = coco_caps.loadAnns(annIds)
        for annos in anns:
            captions.append([i, coco_img['id'], annos["caption"]])

    total_tests = 0
    total_rank1_successes = 0
    total_rank2_successes = 0
    total_rank3_successes = 0

    start_time = time.time()

    # Query tests:
    r = list(range(0,len(captions)))
    random.shuffle(r)

    for i in r:
        query = captions[i][2]
        gt = captions[i][0]

        client = actionlib.SimpleActionClient('dense_query', action_controller.msg.DenseImageQueryAction)
        client.wait_for_server()    
        goal = action_controller.msg.DenseImageQueryGoal(query, 6.0)
        client.send_goal(goal, done_cb=query_done_cb)
        client.wait_for_result()

        result = client.get_result()

        total_tests += 1
        if result.frame_ids[0] == gt:
            total_rank1_successes += 1
        elif result.frame_ids[1] == gt:
            total_rank2_successes += 1
        elif result.frame_ids[2] == gt:
            total_rank3_successes += 1 
        # else:
            # print "Failed Query: %s, gt - %d" % (query, gt)

        if total_tests >= MAX_QUERIES:
            break

        # print "Progress: %d/%d" % (total_tests, MAX_QUERIES)

    print("Time: --- %s seconds ---" % (time.time() - start_time))

    total_successes = total_rank1_successes + total_rank2_successes + total_rank3_successes

    print "Rank Dist: Rank1 - %d, Rank2 - %d, Rank3 - %d" % (total_rank1_successes, total_rank2_successes, total_rank3_successes)
    print "Success Rate: %d/%d = %f" % (total_successes, total_tests, total_successes/(1.0*total_tests)*100.0 )

    return client.get_result()  
    

def done_cb(goal_status, result):
    print 'Done'
    print result

def query_done_cb(goal_status, result):
    # print 'Done'
    # print result
    print "%f\t%d\t%d\t%d" % (result.search_time, result.meteor_ranks[0], result.meteor_ranks[1], result.meteor_ranks[2])

if __name__ == '__main__':
    try:
        pub_image()
    except rospy.ROSInterruptException:
        pass