#! /usr/bin/env python3

##
# @package tapefollow
# @brief A ROS program which enables a robot to follow a line using CV2. 

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

##
# @brief Callback method to process image data and convert into steering input for robot
# @param data Image data to be processed.
# @retval 

def callback(data):

    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(1)
    


##
# @brief Listener method to retrieve data from ROS camera, and send data to callback function to be processed
    
def listener():

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/R1/pi_camera/image_raw", Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()