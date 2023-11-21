#!/usr/bin/env python3

##
# @file opencv.py
#
# @brief Python script for PID line movement of ENPH353 robot in gazebo
#
# @section author_doxygen_example Author(s)
# - Created by Avery Wong on 09/24/2023.
#



# Imports

from __future__ import print_function
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock

class robot_controller:

  # Functions
  def __init__(self):
    
    self.clock_sub = rospy.Subscriber("/clock",Clock)
    self.bridge = CvBridge()
    self.move_pub = rospy.Publisher("/R1/cmd_vel",Twist,queue_size=1)
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    #self.tracker_pub = rospy.Publisher("/score_tracker")

  def callback(self,data):
    move = Twist()
    move.linear.x = 0.5
    move.angular.z = 0.5    
    self.move_pub.publish(move)

def main(args):
    rospy.init_node('controller', anonymous=True)
    controller = robot_controller()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)