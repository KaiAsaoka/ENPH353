#! /usr/bin/env python3

##
# @package tapefollow
# @brief A ROS program which enables a robot to follow a line using CV2. 

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys
from geometry_msgs.msg import Twist
import csv
from collections import namedtuple
import pickle
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


import Levenshtein
##
# @brief Callback method
# @retval 
class navigation():


    def __init__(self):
        
        testTruck = False
        testgrass = True
        testYoda = False
        testTunnel = False
        
        self.sift = cv2.SIFT_create()
        self.grassy = False
        self.tunnel = False
        self.car = False
        self.turntotun = False
        self.grasscount = 0
		## Feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        self.capture = False
        self.currentTime = 0
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.template_path = "/home/fizzer/ros_ws/src/2023_competition/media_src/clue_banner.png"
        self.img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
		## Features
        self.kp_image, self.desc_image = self.sift.detectAndCompute(self.img, None)
        self.bridge = CvBridge()
        self.move_pub = rospy.Publisher("/R1/cmd_vel",Twist,queue_size=1)
        self.pastman = False
        self.roadSpeed = 0.5
        self.grassSpeed = 0.3
        #### SET TRUE FOR REAL RUN
        self.predictions = False
        self.grassy2 = False
        
        self.climb = False
        print("Loaded template image file: " + self.template_path)

        self.white_count = 0
        self.times = 0
        self.capture_time = 99999 # big number
        self.turnstart = 0

        self.clock_sub = rospy.Subscriber("/clock",Clock,self.clock_callback)
        self.score_pub = rospy.Publisher("/score_tracker",String,queue_size=1)
        rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        rospy.Subscriber("/read_sign", Int32, self.callback)
        rospy.Subscriber("/predict_sign", Int32, self.predict_callback)
        
        #### SET FALSE FOR REAL RUN
        self.start = True

        if testgrass == True:
                self.predictions = False
                self.grassy = True
                self.pastman = True
                self.grassSpeed = 0.3
                self.roadSpeed = 0.3
                
        if testYoda == True:
                self.predictions = False
                self.grassy = True
                self.pastman = True
                self.grassSpeed = 0
                self.roadSpeed = 0
                
        if testTunnel == True:
                self.predictions = False
                self.turntotun = True
                self.tunnel = True
                self.car = True
                self.grassy = True
                self.pastman = True
                self.grassSpeed = 0
                self.roadSpeed = 0
            
    

        #rostopic pub /read_sign std_msgs/Int32 "data: 0"
        if self.predictions:
            self.start = False

            model_path = '/home/fizzer/ros_ws/src/my_controller/src/pickle/sign_detection_weights.h5'
            self.model = tf.keras.models.load_model(model_path)
            self.chr_vec = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N"
                            ,"O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]

        print("navigation init success")

    def predict_callback(self,data):
        num = data.data
        if(num == 0):
            message = str('TEAM16,joebot,0,START')
            self.score_pub.publish(message)
            self.start = True
        elif(num == -1):
            message = str('TEAM16,joebot,-1,END')
            self.score_pub.publish(message)
        else:
            self.makePrediction()

    def clock_callback(self,data):
        
        self.times = data.clock.secs
        self.timens = data.clock.nsecs
        
        
        
    def callback(self,data):

        # Label returns two words, (clue, cause)
        # Each word is comprised of a guess and a truth
        # Ex. Clue: Clue[0] -> guess in contour form
        #           Clue[1] -> guess in chr form
        #           Clue[1][0] -> first chr
        #           Clue[0][0] -> first chr in contour form
        #           len(Clue[1]) = len(Clue[0])
        # @param clue sign: clue contour for clue word
        # @param clue truth[SIGNID]: csv value for clue word
        # Other params follow same idea
        # First sign is 0
        
        signid = data.data
        full = self.readSign(signid, False)
        print(full[0][1])
        print("callback worked")
    
    def image_callback(self, data):
        
        if(self.start):
            
            #### UNCOMMENT FOR REAL RUN
            self.image_raw = data
            self.tapefollow(data) 
            #self.turn(data)
             
            WIDTH = 600
            HEIGHT = 400
            
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            
            # Apply blue color mask
    
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            lower_blue = np.array([115, 128, 95])
            upper_blue = np.array([120, 255, 204])
            blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

            #####
            hsv_tunnel = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            tunnel_lower = np.array([0, 70, 60])
            tunnel_upper = np.array([10, 100, 90])
            tunnel_mask = cv2.inRange(hsv_tunnel, tunnel_lower, tunnel_upper)
            
            #tunnel_copy = hsv_tunnel.copy()
            tunnel_contours, _ = cv2.findContours(tunnel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tunnel = cv2.drawContours(frame.copy(), tunnel_contours, -1, (0, 255, 0), 1)
            cv2.imshow("Tunnel", cv2.cvtColor(tunnel, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            #print(largest_contour_area)
            #####
            ### Find contours
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = frame.copy()
            dst = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
            
            
            if contours:
            
                largest_contour = max(contours, key=cv2.contourArea)
                
                signtresh = 30000  # minimum size of sign vector
                
                if (cv2.contourArea(largest_contour)) > signtresh:
                    
                    if self.capture == False:
                        
                        #### UNCOMMENT FOR REAL RUN
                        self.capture = True
                        
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        cv2.drawContours(contour, [approx], 0, (0, 255, 0), 2)
    
                        if(len(approx) >= 4):
                            tl, tr, bl, br = rectangle_positions(approx)
                        
                            pts1 = np.float32([tl,tr,bl,br])
                            pts2 = np.float32([[0,0],[WIDTH,0],[0,HEIGHT],[WIDTH,HEIGHT]])
    
                            perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    
                            # Apply the perspective transform
                            dst = cv2.warpPerspective(frame, perspective_matrix, (600, 400))
                            BORDER_WIDTH = 50
                            BORDER_HEIGHT = 50
                            #dst = dst[BORDER_HEIGHT:HEIGHT-BORDER_HEIGHT,
                            #        BORDER_WIDTH:WIDTH-BORDER_WIDTH]
                    
                        ### Create Contours to find Letters
    
                        hsv_image = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)
                        lower_blue = np.array([115, 128, 95])
                        upper_blue = np.array([120, 255, 204])
                        self.dstmask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
                        self.letters, self.letters_hierarchy = cv2.findContours(self.dstmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                        self.clue_sign, self.cause_sign = cleanLetterContours(self.letters,self.letters_hierarchy)
                        
                        if self.predictions:
                            move = Twist()
                            move.linear.x = 0
                            move.linear.y = 0
                            move.linear.z = 0
                            self.move_pub.publish(move)

                            self.wait_time = self.times
                            self.makePrediction()

                            while self.times - self.wait_time < 2:
                                continue


                        #cv2.imshow("isoletter", isoletter)
                        #cv2.imshow("test", test)
                        #cv2.imshow("Binary", blue_mask)
                        
                        # cv2.imshow("Contour Crop", cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
    
                        
                        # cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
                        
                        ### Screens
                        lettermask = dst.copy()
                        letterimage = cv2.drawContours(lettermask, self.letters, -1, (0, 255, 0), 1)    
                        cv2.imshow("Letter Image", cv2.cvtColor(letterimage, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
    
                        dstup = dst.copy()
                        uletterimage = cv2.drawContours(dstup, self.clue_sign, -1, (0, 255, 0), 1)
                        cv2.imshow("dst up", cv2.cvtColor(uletterimage, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                
                        dstdown = dst.copy()
                        dletterimage = cv2.drawContours(dstdown, self.cause_sign, -1, (0, 255, 0), 1)
                        cv2.imshow("dst down", cv2.cvtColor(dletterimage, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
    
                        self.capture_time = self.times
    
                elif self.times-self.capture_time < 3 and self.capture == True: # lower sign treshold value for reset
                    self.capture = True
    
                else: # reset capture state
                    self.capture = False
                
    
    
            #cv2.imshow("Binary", blue_mask)    
    
            cv2.imshow("Contour Image", cv2.cvtColor(contour, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    

    def tapefollow(self, data):
        

        if self.grassy == False: #Road detection
            
            self.roadFollow(data)
            
           
                
        elif self.tunnel == False: # grassy area
           
            self.grassFollow(data)
            
        elif self.car == False:
            
            self.carTunnel(data)
            
        elif self.turntotun == False:
            self.pidcar(data)
        
        elif self.climb  == False:
            self.turn(data)

        elif self.grassy2 == False:
            print("started tunnel climb")
            self.tunnelClimb(data)
        else:
            print("TOUCHING GRASS!!!!!!")
            self.grassFollow(data)
            
            
            
        
    def roadFollow(self, data):
        
            SPEED = self.roadSpeed
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            

    
            h=430
            
            ## Define the coordinates of the region of interest (ROI)
            roi_x1, roi_y1, roi_x2, roi_y2 = 0 + 200, h, 1280 - 200, h+10  # Adjust these coordinates as needed
            ## Default Resolution x = 320, y = 240

            ## Crop the image to the ROI
            roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            
            hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
            lower_white = hsvConv (0, 0, 32)
            upper_white = hsvConv (0, 0, 34)
            white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
            ## Define the lower and upper bounds for the color you want to detect (here, it's blue)

            cv2.imshow("white mask", cv2.cvtColor(white_mask, cv2.COLOR_RGB2BGR))
            ## Find contours in the binary mask
            pidcontours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cx = 0
            cxnet = 0
            moments = 0
            cxavg = 640
            
            pid_img = cv2.drawContours(roi_image.copy(), pidcontours, -1, (0, 255, 0), 1)

            ## Iterate through the contours and find the position of color change within the ROI
            
            if len(pidcontours) == 0:
                self.grasscount += 1
                if self.grasscount == 2:
                    self.grassy = True
                    print("grass time!!")
                    
            for contour in pidcontours:
                
                
                ## Calculate the centroid of the contour
                M = cv2.moments(contour)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])

                    ## Add the ROI offset to get the position within the original image
                    cx += roi_x1
                    
                    cxnet += cx
                    moments += 1

                    #print(f"Position of color change within ROI: ({cx}, {cy})")
   
            move = Twist()
            move.linear.x = SPEED
            cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            
            if self.pastman == False:
                if self.scanforred(frame) == True:
                    move.linear.x = 0
                    if self.scanforman(frame) == True:
                        self.pastman = True
                        print("self.pastman =" + str(self.pastman))
                        
            if self.pastman == True:
                if self.scanfortruck(frame) == True:
                    move.linear.x = 0.05
                    
            
            
            if moments != 0:
                cxavg = cxnet / moments
            
                turn0 = 0
                turn1 = 2
                turn2 = 3
                turn3 = 4
                turn4 = 5
                turn5 = 6
    
                if cxavg >= 0 and cxavg < 128:
                    move.angular.z = turn5
                    cv2.putText(frame, str(cxavg) + " LEFT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 128 and cxavg < 256:
                    move.angular.z = turn4
                    cv2.putText(frame, str(cxavg) + " LEft", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 256 and cxavg < 384:
                    move.angular.z = turn3
                    cv2.putText(frame, str(cxavg) + " Left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 384 and cxavg < 512:
                    move.angular.z = turn2
                    cv2.putText(frame, str(cxavg) + " left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 512 and cxavg < 630:
                    move.angular.z = turn1
                    cv2.putText(frame, str(cxavg) + " l", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
                elif cxavg >= 630 and cxavg < 650:
                    move.angular.z = turn0
                    cv2.putText(frame, str(cxavg) + " none", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
                elif cxavg >= 650 and cxavg < 768:
                    move.angular.z = -turn1
                    cv2.putText(frame, str(cxavg) + " r", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 768 and cxavg < 896:
                    move.angular.z = -turn2
                    cv2.putText(frame, str(cxavg) + " right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 896 and cxavg < 1024:
                    move.angular.z = -turn3
                    cv2.putText(frame, str(cxavg) + " Right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 1024 and cxavg < 1152:
                    move.angular.z = -turn4
                    cv2.putText(frame, str(cxavg) + " RIght", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                else:
                    move.angular.z = -turn5
                    cv2.putText(frame, str(cxavg) + " RIGHT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
            center_coordinates = (int(cxavg), int(h))  # Change the coordinates as needed
            #print (center_coordinates)
            radius = 30
            color = (0, 0, 255)  # Red color in BGR format
            thickness = -1 # Thickness of the circle's border (use -1 for a filled circle)
            # Process the frame here (you can add your tracking code or other operations)
            frame_with_circle = cv2.circle(pid_img, center_coordinates, radius, color, thickness)



            cv2.imshow("PID", cv2.cvtColor(frame_with_circle, cv2.COLOR_RGB2BGR))
            #cv2.imshow("pidimg", cv2.cvtColor(white_mask, cv2.COLOR_RGB2BGR))
            #cv2.imshow("hsv", cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            self.move_pub.publish(move)


    def grassFollow(self,data):
            GRASSSPEED = self.grassSpeed
            
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            
            h=430
            
            ## Define the coordinates of the region of interest (ROI)
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+100  # Adjust these coordinates as needed
            ## Default Resolution x = 320, y = 240
            
            ## Crop the image to the ROI
            roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
  
            
            hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
            lower_white = hsvConv (30, 10, 60)
            upper_white = hsvConv (75, 30, 90)
            white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
            ## Define the lower and upper bounds for the color you want to detect (here, it's blue)

            cv2.imshow("white mask", cv2.cvtColor(white_mask, cv2.COLOR_RGB2BGR))
            ## Find contours in the binary mask
            pidcontours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
                        
            min_area = 1000
            max_area = 100000
            
            pidcontours = [contour for contour in pidcontours if min_area < cv2.contourArea(contour) < max_area]
            
            cx = 0
            cxnet = 0
            moments = 0
            cxavg = 640
            
            pid_img = cv2.drawContours(roi_image.copy(), pidcontours, -1, (0, 255, 0), 1)

            
            ## Iterate through the contours and find the position of color change within the ROI
            for contour in pidcontours:

                ## Calculate the centroid of the contour
                M = cv2.moments(contour)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])

                    ## Add the ROI offset to get the position within the original image
                    cx += roi_x1
                    
                    cxnet += cx
                    moments += 1

                    #print(f"Position of color change within ROI: ({cx}, {cy})")
             
            move = Twist()
            move.linear.x = GRASSSPEED
            cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            
            if moments != 0:
                cxavg = cxnet / moments
            
                turn0 = 0
                turn1 = 0.75
                turn2 = 1
                turn3 = 1.2
                turn4 = 2
                turn5 = 3
                
                if cxavg >= 0 and cxavg < 128:
                    move.angular.z = turn5
                    cv2.putText(frame, str(cxavg) + " LEFT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 128 and cxavg < 256:
                    move.angular.z = turn4
                    cv2.putText(frame, str(cxavg) + " LEft", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 256 and cxavg < 384:
                    move.angular.z = turn3
                    cv2.putText(frame, str(cxavg) + " Left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 384 and cxavg < 512:
                    move.angular.z = turn2
                    cv2.putText(frame, str(cxavg) + " left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 512 and cxavg < 630:
                    move.angular.z = turn1
                    cv2.putText(frame, str(cxavg) + " l", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
                elif cxavg >= 630 and cxavg < 650:
                    move.angular.z = turn0
                    cv2.putText(frame, str(cxavg) + " none", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
                elif cxavg >= 650 and cxavg < 768:
                    move.angular.z = -turn1
                    cv2.putText(frame, str(cxavg) + " r", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 768 and cxavg < 896:
                    move.angular.z = -turn2
                    cv2.putText(frame, str(cxavg) + " right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 896 and cxavg < 1024:
                    move.angular.z = -turn3
                    cv2.putText(frame, str(cxavg) + " Right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 1024 and cxavg < 1152:
                    move.angular.z = -turn4
                    cv2.putText(frame, str(cxavg) + " RIght", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                else:
                    move.angular.z = -turn5
                    cv2.putText(frame, str(cxavg) + " RIGHT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
            center_coordinates = (int(cxavg), int(h))  # Change the coordinates as needed
            #print (center_coordinates)
            radius = 30
            color = (0, 0, 255)  # Red color in BGR format
            thickness = -1 # Thickness of the circle's border (use -1 for a filled circle)
            # Process the frame here (you can add your tracking code or other operations)
            frame_with_circle = cv2.circle(pid_img, center_coordinates, radius, color, thickness)

            if self.grassy2 == False and self.scanfortunnel(frame):
                self.tunnel = True

            cv2.imshow("PID", cv2.cvtColor(frame_with_circle, cv2.COLOR_RGB2BGR))
            #cv2.imshow("pidimg", cv2.cvtColor(white_mask, cv2.COLOR_RGB2BGR))
            #cv2.imshow("hsv", cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            self.move_pub.publish(move)

    def tunnelClimb(self, data):
        
            
            SPEED = 0.1
            
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    
            h=0
            ## Define the coordinates of the region of interest (ROI)
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+720  # Adjust these coordinates as needed

            ## Crop the image to the ROI
            roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            hsv_tunnel = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
            tunnel_lower = np.array([0, 70, 60])
            tunnel_upper = np.array([10, 100, 90])
            tunnel_mask = cv2.inRange(hsv_tunnel, tunnel_lower, tunnel_upper)
            
            #tunnel_copy = hsv_tunnel.copy()
            tunnel_contours, _ = cv2.findContours(tunnel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tunnel = cv2.drawContours(frame.copy(), tunnel_contours, -1, (0, 255, 0), 1)
            cv2.imshow("Tunnel", cv2.cvtColor(tunnel, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            ## Define the lower and upper bounds for the color you want to detect (here, it's blue)

            cv2.imshow("tunnel mask", cv2.cvtColor(tunnel_mask, cv2.COLOR_RGB2BGR))
            ## Find contours in the binary mask
            pidcontours, _ = cv2.findContours(tunnel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_area = 50
            cx = 0
            cxnet = 0
            moments = 0
            cxavg = 640
            pid_img = cv2.drawContours(roi_image.copy(), pidcontours, -1, (0, 255, 0), 1)
            
            ## Iterate through the contours and find the position of color change within the ROI
            if self.scanforwhite(frame,1700,2200):
                self.grassy2 = True
                print("grass time!!")
    
            for contour in pidcontours:
                ## Calculate the centroid of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    ## Add the ROI offset to get the position within the original image
                    cx += roi_x1
                    cxnet += cx
                    moments += 1
                    #print(f"Position of color change within ROI: ({cx}, {cy})")
            move = Twist()
            move.linear.x = SPEED
            cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            if moments != 0:
                cxavg = cxnet / moments
                turn0 = 0.25
                turn1 = 0.25
                turn2 = 0.25
                turn3 = 0.25
                turn4 = 0.25
                turn5 = 0.25
                if cxavg >= 0 and cxavg < 128:
                    move.angular.z = turn5
                    cv2.putText(frame, str(cxavg) + " LEFT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                elif cxavg >= 128 and cxavg < 256:
                    move.angular.z = turn4
                    cv2.putText(frame, str(cxavg) + " LEft", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                elif cxavg >= 256 and cxavg < 384:
                    move.angular.z = turn3
                    cv2.putText(frame, str(cxavg) + " Left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                elif cxavg >= 384 and cxavg < 512:
                    move.angular.z = turn2
                    cv2.putText(frame, str(cxavg) + " left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                elif cxavg >= 512 and cxavg < 630:
                    move.angular.z = turn1
                    cv2.putText(frame, str(cxavg) + " l", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                elif cxavg >= 630 and cxavg < 650:
                    move.angular.z = turn0
                    cv2.putText(frame, str(cxavg) + " none", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                elif cxavg >= 650 and cxavg < 768:
                    move.angular.z = -turn1
                    cv2.putText(frame, str(cxavg) + " r", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                elif cxavg >= 768 and cxavg < 896:
                    move.angular.z = -turn2
                    cv2.putText(frame, str(cxavg) + " right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                elif cxavg >= 896 and cxavg < 1024:
                    move.angular.z = -turn3
                    cv2.putText(frame, str(cxavg) + " Right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                elif cxavg >= 1024 and cxavg < 1152:
                    move.angular.z = -turn4
                    cv2.putText(frame, str(cxavg) + " RIght", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                else:
                    move.angular.z = -turn5
                    cv2.putText(frame, str(cxavg) + " RIGHT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            center_coordinates = (int(cxavg), int(h))  # Change the coordinates as needed
            #print (center_coordinates)
            radius = 30
            color = (0, 0, 255)  # Red color in BGR format
            thickness = -1 # Thickness of the circle's border (use -1 for a filled circle)
            # Process the frame here (you can add your tracking code or other operations)
            frame_with_circle = cv2.circle(pid_img, center_coordinates, radius, color, thickness)
            cv2.imshow("TUNNEL PID", cv2.cvtColor(frame_with_circle, cv2.COLOR_RGB2BGR))
            #cv2.imshow("pidimg", cv2.cvtColor(white_mask, cv2.COLOR_RGB2BGR))
            #cv2.imshow("hsv", cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            self.move_pub.publish(move)

    def carTunnel(self, data):
        
            SPEED = self.roadSpeed
            
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    
            h=0
            ## Define the coordinates of the region of interest (ROI)
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+720  # Adjust these coordinates as needed

            ## Crop the image to the ROI
            roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            hsv_tunnel = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
            tunnel_lower = np.array([5, 130, 75])
            tunnel_upper = np.array([15, 140, 190])
            tunnel_mask = cv2.inRange(hsv_tunnel, tunnel_lower, tunnel_upper)

            #tunnel_copy = hsv_tunnel.copy()
            tunnel_contours, _ = cv2.findContours(tunnel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)            
            pid_img = cv2.drawContours(frame, tunnel_contours, -1, (0, 255, 0), 1)

            cx = 0
            cxnet = 0
            moments = 0
            cxavg = 640

            ## Iterate through the contours and find the position of color change within the ROI
            
                    
            for contour in tunnel_contours:
                
                
                ## Calculate the centroid of the contour
                M = cv2.moments(contour)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])

                    ## Add the ROI offset to get the position within the original image
                    cx += roi_x1
                    
                    cxnet += cx
                    moments += 1

                    #print(f"Position of color change within ROI: ({cx}, {cy})")
   
            move = Twist()
            move.linear.x = SPEED
            cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            
            if moments != 0:
                cxavg = cxnet / moments + 250
            
                turn0 = 0
                turn1 = .25
                turn2 = .5
                turn3 = .75
                turn4 = 1
                turn5 = 1.25
    
                if cxavg >= 0 and cxavg < 128:
                    move.angular.z = turn5
                    cv2.putText(frame, str(cxavg) + " LEFT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 128 and cxavg < 256:
                    move.angular.z = turn4
                    cv2.putText(frame, str(cxavg) + " LEft", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 256 and cxavg < 384:
                    move.angular.z = turn3
                    cv2.putText(frame, str(cxavg) + " Left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 384 and cxavg < 512:
                    move.angular.z = turn2
                    cv2.putText(frame, str(cxavg) + " left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 512 and cxavg < 630:
                    move.angular.z = turn1
                    cv2.putText(frame, str(cxavg) + " l", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
                elif cxavg >= 630 and cxavg < 650:
                    move.angular.z = turn0
                    cv2.putText(frame, str(cxavg) + " none", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
                elif cxavg >= 650 and cxavg < 768:
                    move.angular.z = -turn1
                    cv2.putText(frame, str(cxavg) + " r", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 768 and cxavg < 896:
                    move.angular.z = -turn2
                    cv2.putText(frame, str(cxavg) + " right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 896 and cxavg < 1024:
                    move.angular.z = -turn3
                    cv2.putText(frame, str(cxavg) + " Right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 1024 and cxavg < 1152:
                    move.angular.z = -turn4
                    cv2.putText(frame, str(cxavg) + " RIght", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                else:
                    move.angular.z = -turn5
                    cv2.putText(frame, str(cxavg) + " RIGHT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
            center_coordinates = (int(cxavg), int(h))  # Change the coordinates as needed
            #print (center_coordinates)
            radius = 30
            color = (0, 0, 255)  # Red color in BGR format
            thickness = -1 # Thickness of the circle's border (use -1 for a filled circle)
            # Process the frame here (you can add your tracking code or other operations)
            frame_with_circle = cv2.circle(pid_img, center_coordinates, radius, color, thickness)



            cv2.imshow("PID", cv2.cvtColor(frame_with_circle, cv2.COLOR_RGB2BGR))
            #cv2.imshow("pidimg", cv2.cvtColor(white_mask, cv2.COLOR_RGB2BGR))
            #cv2.imshow("hsv", cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            if self.scanforcar(frame) == True:
                self.car = True
            
            self.move_pub.publish(move)
            
            
    def scanfortunnel(self, frameorig):
        
            frame = frameorig.copy()
        
            h=0
            ## Define the coordinates of the region of interest (ROI)
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+720  # Adjust these coordinates as needed

            ## Crop the image to the ROI
            roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            hsv_tunnel = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
            tunnel_lower = np.array([5, 130, 180])
            tunnel_upper = np.array([15, 140, 190])
            tunnel_mask = cv2.inRange(hsv_tunnel, tunnel_lower, tunnel_upper)

            #tunnel_copy = hsv_tunnel.copy()
            tunnel_contours, _ = cv2.findContours(tunnel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)            
            pid_img = cv2.drawContours(frame, tunnel_contours, -1, (0, 255, 0), 1)
            cv2.imshow("tunnel cont", cv2.cvtColor(pid_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            min_area = 20
            
            if len(tunnel_contours) != 0 and cv2.contourArea(max(tunnel_contours, key=cv2.contourArea)) > min_area:
                print("tunnel pog!!")
                return True
            else:
                return False
            
    def pidcar(self, data):
        
            SPEED = self.roadSpeed
            
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    
            h=0
            ## Define the coordinates of the region of interest (ROI)
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+720  # Adjust these coordinates as needed

            ## Crop the image to the ROI
            roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
            lower_red1 = hsvConv (0, 50, 35)
            upper_red1 = hsvConv (11, 60, 45)
            
            lower_red2 = hsvConv (350, 50, 35)
            upper_red2 = hsvConv (360, 60, 45)
            
            car_masklow = cv2.inRange(hsv_image, lower_red1, upper_red1)
            car_maskhigh = cv2.inRange(hsv_image, lower_red2, upper_red2)

            car_mask = cv2.bitwise_or(car_masklow, car_maskhigh)
            ## Define the lower and upper bounds for the color you want to detect (here, it's blue)

            cv2.imshow("car mask", cv2.cvtColor(car_mask, cv2.COLOR_RGB2BGR))
            ## Find contours in the binary mask
            pidcontours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            cx = 0
            cxnet = 0
            moments = 0
            cxavg = 640
            
            pid_img = cv2.drawContours(roi_image.copy(), pidcontours, -1, (0, 255, 0), 1)

            ## Iterate through the contours and find the position of color change within the ROI
            
            if len(pidcontours) == 0:
                self.grasscount += 1
                if self.grasscount == 2:
                    self.grassy = True
                    print("grass time!!")
                    
            for contour in pidcontours:
                
                
                ## Calculate the centroid of the contour
                M = cv2.moments(contour)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])

                    ## Add the ROI offset to get the position within the original image
                    cx += roi_x1
                    
                    cxnet += cx
                    moments += 1

                    #print(f"Position of color change within ROI: ({cx}, {cy})")
   
            move = Twist()
            move.linear.x = SPEED
            cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            
            if moments != 0:
                cxavg = cxnet / moments
            
                turn0 = 0
                turn1 = .25
                turn2 = .5
                turn3 = .75
                turn4 = 1
                turn5 = 1.25
    
                if cxavg >= 0 and cxavg < 128:
                    move.angular.z = turn5
                    cv2.putText(frame, str(cxavg) + " LEFT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 128 and cxavg < 256:
                    move.angular.z = turn4
                    cv2.putText(frame, str(cxavg) + " LEft", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 256 and cxavg < 384:
                    move.angular.z = turn3
                    cv2.putText(frame, str(cxavg) + " Left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 384 and cxavg < 512:
                    move.angular.z = turn2
                    cv2.putText(frame, str(cxavg) + " left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 512 and cxavg < 630:
                    move.angular.z = turn1
                    cv2.putText(frame, str(cxavg) + " l", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
                elif cxavg >= 630 and cxavg < 650:
                    move.angular.z = turn0
                    cv2.putText(frame, str(cxavg) + " none", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
                elif cxavg >= 650 and cxavg < 768:
                    move.angular.z = -turn1
                    cv2.putText(frame, str(cxavg) + " r", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 768 and cxavg < 896:
                    move.angular.z = -turn2
                    cv2.putText(frame, str(cxavg) + " right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 896 and cxavg < 1024:
                    move.angular.z = -turn3
                    cv2.putText(frame, str(cxavg) + " Right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 1024 and cxavg < 1152:
                    move.angular.z = -turn4
                    cv2.putText(frame, str(cxavg) + " RIght", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                else:
                    move.angular.z = -turn5
                    cv2.putText(frame, str(cxavg) + " RIGHT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
            center_coordinates = (int(cxavg), int(h))  # Change the coordinates as needed
            #print (center_coordinates)
            radius = 30
            color = (0, 0, 255)  # Red color in BGR format
            thickness = -1 # Thickness of the circle's border (use -1 for a filled circle)
            # Process the frame here (you can add your tracking code or other operations)
            frame_with_circle = cv2.circle(pid_img, center_coordinates, radius, color, thickness)

            min_area = 360
            
            if len(pidcontours) != 0 and cv2.contourArea(max(pidcontours, key=cv2.contourArea)) > min_area:
                self.turntotun = True          
                self.turnstart = self.times 

            cv2.imshow("PID", cv2.cvtColor(frame_with_circle, cv2.COLOR_RGB2BGR))
            #cv2.imshow("pidimg", cv2.cvtColor(white_mask, cv2.COLOR_RGB2BGR))
            #cv2.imshow("hsv", cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            self.move_pub.publish(move)
    
    def turn(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        while( self.times - self.turnstart < 2):
            move = Twist()
            move.linear.x = 0
            move.linear.y = 0
            move.linear.z = 0

            move.angular.z = 1.5
            self.move_pub.publish(move)
        
        move = Twist()
        move.linear.x = 0.1
        move.angular.z = 0
        self.move_pub.publish(move)

        bot_thresh = 0
        top_thresh = 10000
        if(self.scanforwhite(frame,bot_thresh,top_thresh)):
            self.climb = True
            print("climbing")


    def scanforcar(self, frameorig):
    
        frame = frameorig.copy()
    
        h=0
        ## Define the coordinates of the region of interest (ROI)
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+720  # Adjust these coordinates as needed

        ## Crop the image to the ROI
        roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
        lower_red1 = hsvConv (0, 50, 35)
        upper_red1 = hsvConv (11, 60, 45)
        
        lower_red2 = hsvConv (350, 50, 35)
        upper_red2 = hsvConv (360, 60, 45)
        
        car_masklow = cv2.inRange(hsv_image, lower_red1, upper_red1)
        car_maskhigh = cv2.inRange(hsv_image, lower_red2, upper_red2)

        car_mask = cv2.bitwise_or(car_masklow, car_maskhigh)

        ## Define the lower and upper bounds for the color you want to detect (here, it's blue)


        cv2.imshow("red mask", cv2.cvtColor(car_mask, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        ## Find contours in the binary mask
        redcont, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pid_img = cv2.drawContours(frame, redcont, -1, (0, 255, 0), 1)
        cv2.imshow("car cont", cv2.cvtColor(pid_img, cv2.COLOR_RGB2BGR))

        min_area = 2
        
        if len(redcont) != 0 and cv2.contourArea(max(redcont, key=cv2.contourArea)) > min_area:
            
            return True
        else:
            return False
    # Read sign takes a perspective transformed image and a sign to read
    #
    def scanforwhite(self,frameorig,bot,top):
            
            frame = frameorig.copy()

            h=430
            
            ## Define the coordinates of the region of interest (ROI)
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+100  # Adjust these coordinates as needed
            ## Default Resolution x = 320, y = 240
            
            ## Crop the image to the ROI
            roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
  
            
            hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
            lower_white = hsvConv (30, 10, 60)
            upper_white = hsvConv (75, 30, 90)
            white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

            pidcontours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
                        
            min_area = 1000
            max_area = 100000
            
            pidcontours = [contour for contour in pidcontours if min_area < cv2.contourArea(contour) < max_area]

            largest_contour_area = 0
            largest_contour = None
            
            # Iterate through each contour
            for contour in pidcontours:
                area = cv2.contourArea(contour)
                if area > largest_contour_area:
                    largest_contour_area = area
                    largest_contour = contour

            white_count_threshold = 10
            

            if(top > largest_contour_area > bot):
                self.white_count += 1
            else:
                self.white_count = 0

            print(largest_contour_area)
            pid_img = cv2.drawContours(roi_image.copy(), pidcontours, -1, (0, 255, 0), 1)
            cv2.imshow("SCANFORWHITE", cv2.cvtColor(pid_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            print(self.white_count > white_count_threshold)
            return self.white_count > white_count_threshold


    def readSign(self,signid,savepickle):


        ### Load CSV data in
        csv_file_path = '/home/fizzer/ros_ws/src/2023_competition/enph353/enph353_gazebo/scripts/plates.csv'
        clue_truth,cause_truth = loadCsv(csv_file_path)

        clue1,cause1,full = createClueCause(self.clue_sign,clue_truth[signid],self.cause_sign,cause_truth[signid])

        return full
    
    def createPickle(self,full):
        
        ### Prepare dataset for export to colab
        if(len(full) != 0):
            X_dataset_orig, Y_dataset_orig = findFullIndex(full)
            data_to_save = transform_X_Y(X_dataset_orig,Y_dataset_orig)
            
            pickle_file_path = '/home/fizzer/ros_ws/src/my_controller/src/pickle/X_Y_data.pkl'
            with open(pickle_file_path, 'wb') as file:
                pickle.dump(data_to_save, file)
            print("Save Successful")
        else:
            print("full is null, try again")

    def appendPickle(self,full):
        ### Prepare dataset for export to colab
        if(len(full) != 0):
            X_dataset_orig, Y_dataset_orig = findFullIndex(full)
            data_to_save = transform_X_Y(X_dataset_orig,Y_dataset_orig)
            pickle_file_path =  '/home/fizzer/ros_ws/src/my_controller/src/pickle/X_Y_data.pkl'
            try:
                with open(pickle_file_path, 'rb') as file:
                    existing_data = pickle.load(file)

            except (FileNotFoundError, EOFError):
                # If the file doesn't exist or is empty, create an empty list
                existing_data = []

            new_to_save = [np.concatenate((existing_data[0],data_to_save[0]))
                           ,np.concatenate((existing_data[1],data_to_save[1]))]
            
            

            with open(pickle_file_path, 'wb') as file:
                    pickle.dump(new_to_save, file) 
            print("Save Successful")      
        else:
            print("full is null, try again")
            
    def scanforred(self, frameorig):
        
            frame = frameorig.copy()
        
            h=500
            ## Define the coordinates of the region of interest (ROI)
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+120  # Adjust these coordinates as needed

            ## Crop the image to the ROI
            roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
            lower_red = hsvConv (0, 99, 99)
            upper_red = hsvConv (1, 100, 100)
            red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
            ## Define the lower and upper bounds for the color you want to detect (here, it's blue)


            # cv2.imshow("red mask", cv2.cvtColor(red_mask, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            ## Find contours in the binary mask
            redcont, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            pid_img = cv2.drawContours(frame, redcont, -1, (0, 255, 0), 1)
            
            min_area = 4000
            
            if len(redcont) != 0 and cv2.contourArea(max(redcont, key=cv2.contourArea)) > min_area:
                
                return True
            else:
                return False
            
    def scanfortruck(self, frameorig):
        
            frame = frameorig.copy()
        
            h=200
            ## Define the coordinates of the region of interest (ROI)
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+320  # Adjust these coordinates as needed

            ## Crop the image to the ROI
            roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
            lower_truck1 = hsvConv (0, 0, 17.5)
            upper_truck1 = hsvConv (1, 1, 29)
            
            # lower_truck2 = hsvConv (0, 0, 80)
            # upper_truck2 = hsvConv (1, 1, 90)
            
            truck_masklow = cv2.inRange(hsv_image, lower_truck1, upper_truck1)
            # truck_maskhigh = cv2.inRange(hsv_image, lower_truck2, upper_truck2)

            #truck_mask = cv2.bitwise_or(truck_masklow, truck_maskhigh)
                        ## Find contours in the binary mask
            truckcont, _ = cv2.findContours(truck_masklow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 500
            max_area = 100000
            
            truckcont = [contour for contour in truckcont if min_area < cv2.contourArea(contour) < max_area]
            
            pid_img = cv2.drawContours(roi_image, truckcont, -1, (0, 255, 0), 1)
            
            cv2.imshow("truck mask", cv2.cvtColor(truck_masklow, cv2.COLOR_RGB2BGR))
            cv2.imshow("truck cont", cv2.cvtColor(pid_img, cv2.COLOR_RGB2BGR))

            cv2.waitKey(1)
            
            if len(truckcont) != 0:
                return True
            else:
                return False

            
    def scanforman(self, frameorig):
        
            frame = frameorig.copy()
        
            h=350
            ## Define the coordinates of the region of interest (ROI)
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+120  # Adjust these coordinates as needed

            ## Crop the image to the ROI
            roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
            lower_blue = hsvConv (200, 0, 0)
            upper_blue = hsvConv (210, 60, 60)
            blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
            
            # cv2.imshow("pant mask", cv2.cvtColor(blue_mask, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            
            ## Find contours in the binary mask
            bluecont, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 100
            max_area = 100000
            
            bluecont = [contour for contour in bluecont if min_area < cv2.contourArea(contour) < max_area]
            
            pid_img = cv2.drawContours(frame, bluecont, -1, (0, 255, 0), 1)
            # cv2.imshow("pant cont", cv2.cvtColor(pid_img, cv2.COLOR_RGB2BGR))
                       ## Iterate through the contours and find the position of color change within the ROI
                       
            cx = 0
            cxnet = 0
            moments = 0
            cxavg = 0
            
            for contour in bluecont:

                ## Calculate the centroid of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])

                    ## Add the ROI offset to get the position within the original image
                    cx += roi_x1
                    
                    cxnet += cx
                    moments += 1

                    #print(f"Position of color change within ROI: ({cx}, {cy})")  
            
            if moments != 0:
                cxavg = cxnet / moments
            
            if cxavg > 620 and cxavg < 660:
                return True
            else:
                return False

    def isSpace(self):
        THRESHOLD = 30
        index = 0
        # Calculate the distance between consecutive contours
        sorted_contours = sorted(self.cause_sign, key=lambda contour: np.min(contour[:, 0]))
        for i in range(len(sorted_contours) - 1):
            current_contour = sorted_contours[i].reshape(-1, 2)
            next_contour = sorted_contours[i + 1].reshape(-1,2)
            current_max_x = np.max(current_contour[:,0])
            next_min_x = np.min(next_contour[:, 0])
            
            if (next_min_x - current_max_x) > THRESHOLD:
                
                index = i  # Large space detected

        return index
    
    def makePrediction(self):

        clue_prediction = []
        cause_prediction = []

        for letter in self.clue_sign:

            clue_image = np.zeros((100, 100, 3), dtype=np.uint8)
            isoletter = plotcontour(letter,clue_image)

            img_aug = np.expand_dims(isoletter, axis=0)

            prediction_onehot = self.model.predict(img_aug)[0]
            index = np.where(prediction_onehot == 1)[0][0]
            clue_prediction.append(self.chr_vec[index])

        #space = self.isSpace()

        for letter in self.cause_sign:

            cause_image = np.zeros((100, 100, 3), dtype=np.uint8)
            isoletter = plotcontour(letter,cause_image)

            img_aug = np.expand_dims(isoletter, axis=0)

            prediction_onehot = self.model.predict(img_aug)[0]
            index = np.where(prediction_onehot == 1)[0][0]
            cause_prediction.append(self.chr_vec[index])

        #if(space > 0):
        #    cause_prediction.insert(space+1, ' ')

        #cause_prediction.insert(1, ' ')

        clue = ''.join(clue_prediction)
        cause = ''.join(cause_prediction)

        clueid = getClue(clue)+1
        message = str('TEAM16,joebot,' + str(clueid) + ',' + cause)
        print(message)
        self.score_pub.publish(message)

    
def findFullIndex(full):
        chr_vec = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N"
,"O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]
        X_dataset = []
        Y_dataset = []
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        for letter in full:
            
            isoletter = plotcontour(letter[0], empty_image)
            index = chr_vec.index(letter[1])
            Y_dataset.append(index)
            X_dataset.append(isoletter)
            
        return X_dataset, Y_dataset

def transform_X_Y(X,Y):
    Y_dataset_orig = np.array(Y)
    X_dataset_orig = np.array(X)
    Y_data = convert_to_one_hot(Y_dataset_orig)
    X_data = X_dataset_orig/255

    return X_data, Y_data

def convert_to_one_hot(Y):
    NUMBER_OF_LABELS = 36
    Y = np.eye(NUMBER_OF_LABELS)[Y.reshape(-1)]
    return Y
        # Returns two words, (clue, cause)
        # Each word is comprised of a guess and a truth
        # Ex. Clue: Clue[0] -> guess in contour form
        #           Clue[1] -> guess in chr form
        #           Clue[1][0] -> first chr
        #           Clue[0][0] -> first chr in contour form
        #           len(Clue[1]) = len(Clue[0])
def createClueCause(clue_sign,clue_truth,cause_sign,cause_truth):
    if len(clue_sign) != len(clue_truth) or len(cause_sign) != len(cause_truth):
        #print("LENGTH NO MATCHING")
        clue = []
        cause = []
        full = []
    else: 
        clue = [clue_sign,clue_truth]
        cause = [cause_sign,cause_truth]
        full = []
        ### COMMENT OUT CLUE IF WE DONT WANT ANY MORE CLUE NAMES
        #for letter, truth_letter in zip(clue_sign,clue_truth):
        #    pair = [letter, truth_letter]
        #    full.append(pair)
        for letter, truth_letter in zip(cause_sign,cause_truth):
            pair = [letter, truth_letter]
            full.append(pair)

    return clue,cause,full

def plotcontour(contour, image):
    
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Calculate the shifts needed to move the centroid to the center of the image
    shiftX = image.shape[1] // 2 - cX
    shiftY = image.shape[0] // 2 - cY

    # Shift the contour by adding the shift values to the contour points
    shifted_contour = contour + [shiftX, shiftY]

    # Draw the shifted contour on the image
    contourimage = cv2.drawContours(image.copy(), [shifted_contour], -1, (255, 255, 255), 1)
    contourgrey = cv2.cvtColor(contourimage, cv2.COLOR_RGB2GRAY)
    
    return contourgrey
    
    
def cleanLetterContours(letters,letters_hierarchy):
    min_area = 100
    max_area = 2000
    letters = [contour for i, contour in enumerate(letters) if is_outer_contour(letters_hierarchy, i)]
    letters = [contour for contour in letters if min_area < cv2.contourArea(contour) < max_area]
    upletter = []
    downletter = []
    threshold_y = 130  # Adjust the threshold as needed
    
    for letter in letters:
        x, y, w, h = cv2.boundingRect(letter)
        if y < threshold_y:
            upletter.append(letter)
        else:
            downletter.append(letter)
    
    upletter.sort(key=lambda letter: cv2.boundingRect(letter)[0])
    downletter.sort(key=lambda letter: cv2.boundingRect(letter)[0])
    return upletter,downletter

def loadCsv(path):
    clue_t = []
    cause_t = []
    # Open the CSV file and read its contents
    with open(path, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        # Read the data from the CSV file
        for row in csv_reader:
            # Remove spaces within each element in the row
            cleaned_clue_t = list(''.join(row[0].split()))
            cleaned_cause_t = list(''.join(row[1].split()))
            
            clue_t.append(cleaned_clue_t)
            cause_t.append(cleaned_cause_t)
    return clue_t, cause_t

def getClue(prediction):
    truth = ["SIZE","VICTIM","CRIME","TIME","PLACE","MOTIVE","WEAPON","BANDIT"]
    distances = [Levenshtein.distance(prediction, word) for word in truth]

    # Find the index of the minimum distance (closest match)
    closest_index = np.argmin(distances)

    return closest_index

def assign(sign,truth):
    rsign = []
    rtruth = []
    for s,t in sign,truth: 
        rsign.append(s)
        rtruth.append(t)
    return [rsign,rtruth]

def hsvConv (gimpH, gimpS, gimpV):
    
    opencvH = gimpH / 2
    opencvS = (gimpS / 100) * 255
    opencvV = (gimpV / 100) * 255
    return np.array([opencvH, opencvS, opencvV])

def is_outer_contour(hierarchy, index):
    return hierarchy[0][index][3] == -1

def rectangle_positions(approx):
    x0, y0 = approx[0][0][0], approx[0][0][1]
    x2, y2 = approx[2][0][0], approx[2][0][1]
    if(x0 < x2):
        return approx[0],approx[3],approx[1],approx[2]
    else:
        return approx[1],approx[0],approx[2],approx[3]

def is_outer_contour(hierarchy, index):
    return hierarchy[0][index][3] == -1


def main(args):
    rospy.init_node('listener', anonymous=True)
    rospy.sleep(1)
    controller = navigation()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally: cv2.destroyAllWindows()
    


if __name__ == '__main__':
    main(sys.argv)