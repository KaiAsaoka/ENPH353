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


##
# @brief Callback method
# @retval 
class navigation():


    def __init__(self):
        
        
        self.sift = cv2.SIFT_create()

		## Feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.template_path = "/home/fizzer/ros_ws/src/2023_competition/media_src/clue_banner.png"
        self.img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
		## Features
        self.kp_image, self.desc_image = self.sift.detectAndCompute(self.img, None)
        self.bridge = CvBridge()
        self.move_pub = rospy.Publisher("/R1/cmd_vel",Twist,queue_size=1)
        print("Loaded template image file: " + self.template_path)

        self.words = []
        

        rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
  

    def callback(self, data):


    
        # self.tapefollow(data)  
        WIDTH = 600
        HEIGHT = 400
        
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        
        # Apply blue color mask

        hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([115, 128, 95])
        upper_blue = np.array([120, 255, 204])
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        ### Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = frame.copy()
        dst = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        if contours:

            largest_contour = max(contours, key=cv2.contourArea)
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
                BORDER_WIDTH = 80
                BORDER_HEIGHT = 50
                dst = dst[BORDER_HEIGHT:HEIGHT-BORDER_HEIGHT,
                           BORDER_WIDTH:WIDTH-BORDER_WIDTH]
            
        ### Create Contours to find Letters

        hsv_image = cv2.cvtColor(bifilter, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([115, 128, 95])
        upper_blue = np.array([120, 255, 204])
        dstmask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        
        letters, letters_hierarchy = cv2.findContours(dstmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        clue_sign, cause_sign = cleanLetterContours(letters,letters_hierarchy)
        
        ### Load CSV data in
        csv_file_path = '/home/fizzer/ros_ws/src/2023_competition/enph353/enph353_gazebo/scripts/plates.csv'
        clue_truth,cause_truth = loadCsv(csv_file_path)

        #self.label(clue_sign,clue_truth,cause_sign,cause_truth)


        ### Showing screens

        lettermask = dstmask.copy()
        letterimage = cv2.drawContours(lettermask, letters, -1, (0, 255, 0), 1)    
        cv2.imshow("Letter Image", cv2.cvtColor(letterimage, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        #cv2.imshow("thresh", cv2.cvtColor(bifilter, cv2.COLOR_RGB2BGR))
        #cv2.waitKey(1)

        dstup = dst.copy()
        uletterimage = cv2.drawContours(dstup, clue_sign, -1, (0, 255, 0), 1)
        cv2.imshow("dst up", cv2.cvtColor(uletterimage, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
        dstdown = dst.copy()
        dletterimage = cv2.drawContours(dstdown, cause_sign, -1, (0, 255, 0), 1)
        cv2.imshow("dst down", cv2.cvtColor(dletterimage, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
        # cv2.imshow("Contour Crop", cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)
        
        # cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)

        cv2.imshow("Binary", blue_mask)
        cv2.waitKey(1)

        cv2.imshow("Contour Image", cv2.cvtColor(contour, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


    def tapefollow(self, data):
        
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        
        h=420
        
        ## Define the coordinates of the region of interest (ROI)
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+5  # Adjust these coordinates as needed
        ## Default Resolution x = 320, y = 240

        ## Crop the image to the ROI
        roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        cv2.waitKey(1)
        
        hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 250])
        upper_white = np.array([255, 30, 255])
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
        ## Define the lower and upper bounds for the color you want to detect (here, it's blue)
        sensitivity = 15
        cv2.waitKey(1)
        ## Define a threshold value for detecting grayscale change
        threshold_value = 100  # Adjust this threshold as needed

        ## Find contours in the binary mask
        pidcontours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx = 0
        cx = 0

        pid_img = cv2.drawContours(frame, pidcontours, -1, (0, 255, 0), 1)

        cxnet = 0
        cynet = 0
        moments = 0
        cxavg = 640
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
        rate = rospy.Rate(2)
        move = Twist()
        move.linear.x = .1
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        
        if moments != 0:
            cxavg = cxnet / moments
            
        if len(pidcontours) == 1:
            move.linear.x = .1
            if cxavg < 640:
                move.angular.z = 2
                cv2.putText(frame, str(cxavg) + " LEFT!!!", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            if cxavg > 640:
                move.angular.z = -2
                cv2.putText(frame, str(cxavg) + " RIGHT!!!", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            
        else:
            
            if cxavg >= 0 and cxavg < 128:
                move.angular.z = 2
                cv2.putText(frame, str(cxavg) + " LEFT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            elif cxavg >= 128 and cxavg < 256:
                move.angular.z = 1
                cv2.putText(frame, str(cxavg) + " LEft", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            elif cxavg >= 256 and cxavg < 384:
                move.angular.z = .75
                cv2.putText(frame, str(cxavg) + " Left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            elif cxavg >= 384 and cxavg < 512:
                move.angular.z = .5
                cv2.putText(frame, str(cxavg) + " left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            elif cxavg >= 512 and cxavg < 630:
                move.angular.z = .25
                cv2.putText(frame, str(cxavg) + " l", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                
            elif cxavg >= 630 and cxavg < 650:
                move.angular.z = 0
                cv2.putText(frame, str(cxavg) + " none", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                
            elif cxavg >= 650 and cxavg < 768:
                move.angular.z = -.25
                cv2.putText(frame, str(cxavg) + " r", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            elif cxavg >= 768 and cxavg < 896:
                move.angular.z = -.5
                cv2.putText(frame, str(cxavg) + " right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            elif cxavg >= 896 and cxavg < 1024:
                move.angular.z = -.75
                cv2.putText(frame, str(cxavg) + " Right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            elif cxavg >= 1024 and cxavg < 1152:
                move.angular.z = -1
                cv2.putText(frame, str(cxavg) + " RIght", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            else:
                move.angular.z = -2
                cv2.putText(frame, str(cxavg) + " RIGHT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            
        center_coordinates = (int(cxavg), int(h))  # Change the coordinates as needed
        #print (center_coordinates)
        radius = 30
        color = (0, 0, 255)  # Red color in BGR format
        thickness = -1 # Thickness of the circle's border (use -1 for a filled circle)
        # Process the frame here (you can add your tracking code or other operations)
        frame_with_circle = cv2.circle(frame.copy(), center_coordinates, radius, color, thickness)



        cv2.imshow("PID", cv2.cvtColor(frame_with_circle, cv2.COLOR_RGB2BGR))
        #cv2.imshow("pidimg", cv2.cvtColor(white_mask, cv2.COLOR_RGB2BGR))
        #cv2.imshow("hsv", cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR))

        self.move_pub.publish(move)
    
    def label(clue_sign,clue_truth,cause_sign,cause_truth,self):
        
        if len(clue_sign) != len(clue_truth) or len(cause_sign) != len(cause_truth):
            return
        else: 
            
            clue = [clue_sign,clue_truth]
            cause = [cause_sign,cause_truth]

            self.words.append([clue,cause])

def cleanLetterContours(letters,letters_hierarchy):
    min_area = 100
    max_area = 1000
    letters = [contour for i, contour in enumerate(letters) if is_outer_contour(letters_hierarchy, i)]
    letters = [contour for contour in letters if min_area < cv2.contourArea(contour) < max_area]
    upletter = []
    downletter = []
    threshold_y = 200  # Adjust the threshold as needed
    
    for letter in letters:
        x, y, w, h = cv2.boundingRect(letter)
        if y < threshold_y:
            upletter.append(letter)
        else:
            downletter.append(letter)
    
    return (upletter.sort(key=lambda letter: cv2.boundingRect(letter)[0]),
downletter.sort(key=lambda letter: cv2.boundingRect(letter)[0]))

def loadCsv(path):
    clue_t = []
    cause_t = []
    # Open the CSV file and read its contents
    with open(path, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        # Read the data from the CSV file
        for row in csv_reader:
            clue_chrs = []
            clue_chrs.append(char for char in row[0])
            clue_t.append(clue_chrs)
            cause_chrs = []
            cause_chrs.append(char for char in row[1])
            cause_t.append(cause_chrs)

    return clue_t,cause_t

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
    controller = navigation()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally: cv2.destroyAllWindows()
    


if __name__ == '__main__':
    main(sys.argv)