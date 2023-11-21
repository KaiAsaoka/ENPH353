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

    self.times = 0
    self.timens = 0
    self.t0s = 0
    
    self.clock_sub = rospy.Subscriber("/clock",Clock,self.clock_callback)
    self.bridge = CvBridge()
    self.move_pub = rospy.Publisher("/R1/cmd_vel",Twist,queue_size=1)
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.image_callback)
    self.tracker_pub = rospy.Publisher("/score_tracker",String,queue_size=1)

    startComp(self)


  
  def clock_callback(self,data):
    
    if(self.times == 0):
       self.t0s = data.clock.secs
       t0ns = data.clock.nsecs

    self.times = data.clock.secs
    self.timens = data.clock.nsecs

  def image_callback(self,data):
    place = 0
  
def startComp(self):
  while(self.times - self.t0s < 3):
     continue
  
  self.tracker_pub.publish(str('Team16,joebot,0,NA'))
  print("START")

  while(self.times - self.t0s < 5):
     continue
  
  move = Twist()
  move.linear.x = 0.5  
  self.move_pub.publish(move)
  print("MOVE")

  while(self.times - self.t0s < 7):
     continue
  
  stopComp(self)
  move.linear.x = 0.0
  self.move_pub.publish(move)
  
  
  
def stopComp(self):
  self.tracker_pub.publish(str('Team16,joebot,-1,NA')) 
  print("STOP")
    

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