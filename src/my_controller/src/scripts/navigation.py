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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

##
# @brief Callback method
# @retval 
class navigation():


    def __init__(self):
        
        
        self.sift = cv2.SIFT_create()
        self.grassy = False
		## Feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        self.capture = False
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.template_path = "/home/fizzer/ros_ws/src/2023_competition/media_src/clue_banner.png"
        self.img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
		## Features
        self.kp_image, self.desc_image = self.sift.detectAndCompute(self.img, None)
        self.bridge = CvBridge()
        self.move_pub = rospy.Publisher("/R1/cmd_vel",Twist,queue_size=1)
        print("Loaded template image file: " + self.template_path)
        rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        rospy.Subscriber("/read_sign", Int32, self.callback)
        rospy.Subscriber("/predict_sign", Int32, self.predict_callback)

        #rostopic pub /read_sign std_msgs/Int32 "data: 0"

        model_path = '/home/fizzer/ros_ws/src/my_controller/src/pickle/sign_detection_weights.h5'
        self.model = tf.keras.models.load_model(model_path)
        self.chr_vec = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N"
                        ,"O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]

        print("navigation init success")

    def predict_callback(self,data):

        prediction = []

        for letter in self.clue_sign:

            clue_image = np.zeros((100, 100, 3), dtype=np.uint8)
            isoletter = plotcontour(letter,clue_image)

            img_aug = np.expand_dims(isoletter, axis=0)

            prediction_onehot = self.model.predict(img_aug)[0]
            index = np.where(prediction_onehot == 1)[0][0]
            prediction.append(self.chr_vec[index])

        prediction.append(", ")

        for letter in self.cause_sign:

            cause_image = np.zeros((100, 100, 3), dtype=np.uint8)
            isoletter = plotcontour(letter,cause_image)

            img_aug = np.expand_dims(isoletter, axis=0)

            prediction_onehot = self.model.predict(img_aug)[0]
            index = np.where(prediction_onehot == 1)[0][0]
            prediction.append(self.chr_vec[index])

        print(prediction)
        
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
        

        self.image_raw = data
        self.tapefollow(data) 
         
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
            
            signtresh = 0  # minimum size of sign vector
            
            if (cv2.contourArea(largest_contour)) > signtresh:
                
                if self.capture == False:
                    
                    #self.capture = True
                    
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
                        
                    #cv2.imshow("isoletter", isoletter)
                    #cv2.imshow("test", test)
                    #cv2.imshow("Binary", blue_mask)
                    
                    # cv2.imshow("Contour Crop", cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
                    # cv2.waitKey(1)
                    
                    # cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    # cv2.waitKey(1)
                    
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



        #cv2.imshow("Binary", blue_mask)
        #cv2.waitKey(1)

        cv2.imshow("Contour Image", cv2.cvtColor(contour, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


    def tapefollow(self, data):
        

        if self.grassy == False: #Road detection
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            
            h=430
            
            ## Define the coordinates of the region of interest (ROI)
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+10  # Adjust these coordinates as needed
            ## Default Resolution x = 320, y = 240

            ## Crop the image to the ROI
            roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            cv2.waitKey(1)
            
            hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
            lower_white = hsvConv (0, 0, 32)
            upper_white = hsvConv (0, 0, 34)
            white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
            ## Define the lower and upper bounds for the color you want to detect (here, it's blue)
            sensitivity = 15
            cv2.waitKey(1)
            ## Define a threshold value for detecting grayscale change
            threshold_value = 100  # Adjust this threshold as needed
            cv2.imshow("white mask", cv2.cvtColor(white_mask, cv2.COLOR_RGB2BGR))
            ## Find contours in the binary mask
            pidcontours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cx = 0
            cx = 0

            pid_img = cv2.drawContours(frame, pidcontours, -1, (0, 255, 0), 1)

            cxnet = 0

            moments = 0
            cxavg = 640
            ## Iterate through the contours and find the position of color change within the ROI
            
            if len(pidcontours) == 0:
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
            rate = rospy.Rate(2)
            move = Twist()
            move.linear.x = .5
            cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            
            if moments != 0:
                cxavg = cxnet / moments
            
                turn0 = 0
                turn1 = 1
                turn2 = 1.5
                turn3 = 2
                turn4 = 3
                turn5 = 4
    
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

                
        else: # grassy area
            
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            
            h=430
            
            ## Define the coordinates of the region of interest (ROI)
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, h, 1280, h+100  # Adjust these coordinates as needed
            ## Default Resolution x = 320, y = 240

            ## Crop the image to the ROI
            roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            cv2.waitKey(1)
            
            hsv_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
            lower_white = hsvConv (0, 20, 70)
            upper_white = hsvConv (360, 25, 83)
            white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
            ## Define the lower and upper bounds for the color you want to detect (here, it's blue)
            sensitivity = 15
            cv2.waitKey(1)
            ## Define a threshold value for detecting grayscale change
            threshold_value = 100  # Adjust this threshold as needed
            cv2.imshow("white mask", cv2.cvtColor(white_mask, cv2.COLOR_RGB2BGR))
            ## Find contours in the binary mask
            pidcontours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
                        
            min_area = 50
            max_area = 100000
            
            pidcontours = [contour for contour in pidcontours if min_area < cv2.contourArea(contour) < max_area]
            
            cx = 0
            cx = 0

            pid_img = cv2.drawContours(frame, pidcontours, -1, (0, 255, 0), 1)

            cxnet = 0

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
            
                
                if cxavg >= 0 and cxavg < 128:
                    move.angular.z = 3
                    cv2.putText(frame, str(cxavg) + " LEFT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 128 and cxavg < 256:
                    move.angular.z = 2
                    cv2.putText(frame, str(cxavg) + " LEft", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 256 and cxavg < 384:
                    move.angular.z = 1.5
                    cv2.putText(frame, str(cxavg) + " Left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 384 and cxavg < 512:
                    move.angular.z = 1
                    cv2.putText(frame, str(cxavg) + " left", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 512 and cxavg < 630:
                    move.angular.z = .75
                    cv2.putText(frame, str(cxavg) + " l", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
                elif cxavg >= 630 and cxavg < 650:
                    move.angular.z = 0
                    cv2.putText(frame, str(cxavg) + " none", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
                elif cxavg >= 650 and cxavg < 768:
                    move.angular.z = -.75
                    cv2.putText(frame, str(cxavg) + " r", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 768 and cxavg < 896:
                    move.angular.z = -1
                    cv2.putText(frame, str(cxavg) + " right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 896 and cxavg < 1024:
                    move.angular.z = -1.5
                    cv2.putText(frame, str(cxavg) + " Right", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                elif cxavg >= 1024 and cxavg < 1152:
                    move.angular.z = -2
                    cv2.putText(frame, str(cxavg) + " RIght", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                else:
                    move.angular.z = -3
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

    # Read sign takes a perspective transformed image and a sign to read
    #
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