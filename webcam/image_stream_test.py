#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from action_controller.srv import DenseCaption

import cv2
from cv_bridge import CvBridge, CvBridgeError


def pub_image():
    rospy.init_node('ImagePublisher', anonymous=True)


    img = -cv2.imread('terman_eng.jpg',cv2.IMREAD_COLOR)

    # I want to publish the Canny Edge Image and the original Image
    msg_frame = CvBridge().cv2_to_imgmsg(img, "bgr8")

    rospy.wait_for_service('dense_captioning')
    try:
        densecap_srv = rospy.ServiceProxy('dense_captioning', DenseCaption)
        resp1 = densecap_srv(msg_frame, 250, 50, 10, 0.7, 0.3)
        # return resp1.su
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


    print "Done"


if __name__ == '__main__':
    try:
        pub_image()
    except rospy.ROSInterruptException:
        pass