import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
import pykitti
import PIL
import random
import time 
import prettytable as pt

import cv2


# Change this to the directory where you store KITTI data
Odo_Dir = 'KITTI_SAMPLE/ODOMETRY'
RAW_Dir = 'KITTI_SAMPLE/RAW'

date = '2011_09_26'
drive = '0009'

# Specify the dataset to load
sequence = '04'
data= pykitti.raw(RAW_Dir, date, drive, frames=range(0, 51, 1))


# intensity ranging 
ranging_b = np.arange(start=-30, stop=31, step=20)
ranging_c = np.arange(start=0.7, stop=1.4, step=0.2)

# scale ranging 
ranging_scale = np.arange(start = 1.1, stop = 2.4, step = 0.2)

#Rotation ranging
ranging_rotation = np.arange(start=10, stop=91, step=10)



def get_cam_intensified_additive(cam, frame, itensity):
    
    
    if cam == 3:
        Image = np.array(data.get_cam3(frame), dtype = np.float32)
    if cam == 2:
        Image = np.array(data.get_cam2(frame), dtype = np.float32)
    if cam == 1:
        Image = np.array(data.get_cam1(frame), dtype = np.float32)
    if cam == 0:
        Image = np.array(data.get_cam0(frame), dtype = np.float32)
    
    Image_intense = np.zeros(Image.shape)
       
    # adding intensite to every channel
    Image_intense[:,:,0] = Image[:,:,0] + itensity
    Image_intense[:,:,1] = Image[:,:,1] + itensity
    Image_intense[:,:,2] = Image[:,:,2] + itensity
      
    # setting limits
    Image [ Image > 255] = 255
    Image [ Image < 0] = 0
      
    
    return np.asarray(Image, dtype='uint8'), np.asarray(Image_intense, dtype='uint8')

def get_cam_intensified_multiple(cam, frame, itensity):
    
    
    if cam == 3:
        Image = np.array(data.get_cam3(frame), dtype = np.float32)
    if cam == 2:
        Image = np.array(data.get_cam2(frame), dtype = np.float32)
    if cam == 1:
        Image = np.array(data.get_cam1(frame), dtype = np.float32)
    if cam == 0:
        Image = np.array(data.get_cam0(frame), dtype = np.float32)
    
    Image_intense = np.zeros(Image.shape)
    
    # multiplying intensite to every channel
    Image_intense[:,:,0] = Image[:,:,0] * itensity
    Image_intense[:,:,1] = Image[:,:,1] * itensity
    Image_intense[:,:,2] = Image[:,:,2] * itensity
     
    # setting limits
    Image [ Image > 255] = 255
    Image [ Image < 0] = 0
      
    
    return np.asarray(Image, dtype='uint8'), np.asarray(Image_intense, dtype='uint8')

def get_cam_scale(cam, frame, scale):
    
    if cam == 3:
        Image = np.array(data.get_cam3(frame), dtype = np.float32)
    if cam == 2:
        Image = np.array(data.get_cam2(frame), dtype = np.float32)
    if cam == 1:
        Image = np.array(data.get_cam1(frame), dtype = np.float32)
    if cam == 0:
        Image = np.array(data.get_cam0(frame), dtype = np.float32)
    
    # get width and height of the image 
    rows, cols = Image.shape[:2]
    
    # calculating scale matrix by setting the centre of scalling (scale and translation)
    scale_matrix	=	cv2.getRotationMatrix2D((cols//2,	rows//2),	0,	scale) 
    # crop the image depending on the matrix
    scaled_image =	cv2.warpAffine(Image,	scale_matrix,	(cols,	rows)) 

    
    return np.asarray(Image,dtype='uint8'), np.asarray(scaled_image,dtype='uint8')

def get_stereo_couple(frame):
    
    # pykitti function that return a stereo couple (cam2 and cam3)
    left_image, right_image = data.get_rgb(frame)
    
    return np.asarray(left_image, dtype='uint8'), np.asarray(right_image, dtype='uint8')

def get_cam_rotation(camera, frame, rotation):
    
    if camera == 3:
        Image = np.array(data.get_cam3(frame), dtype = np.float32)
    if camera == 2:
        Image = np.array(data.get_cam2(frame), dtype = np.float32)
    if camera == 1:
        Image = np.array(data.get_cam1(frame), dtype = np.float32)
    if camera == 0:
        Image = np.array(data.get_cam0(frame), dtype = np.float32)
    
    # getting image shape (width and height )
    rows, cols = Image.shape[:2]
    #calculate rotation matrix by setting the angle and the centre of rotation (rotation and translation)
    rotation_matrix	=	cv2.getRotationMatrix2D((cols//2,	rows//2),	rotation,	1) 
    # crop the image depending on the matrix 
    rotated_image =	cv2.warpAffine(Image,	rotation_matrix,	(cols,	rows)) 
    
    return np.asarray(Image,dtype='uint8'), np.asarray(rotated_image,dtype='uint8')

def get_cam_scale(camera, frame, scale):
    
    if camera == 3:
        Image = np.array(data.get_cam3(frame), dtype = np.float32)
    if camera == 2:
        Image = np.array(data.get_cam2(frame), dtype = np.float32)
    if camera == 1:
        Image = np.array(data.get_cam1(frame), dtype = np.float32)
    if camera == 0:
        Image = np.array(data.get_cam0(frame), dtype = np.float32)
    
    
    
    rows, cols = Image.shape[:2]
    scale_matrix	=	cv2.getRotationMatrix2D((cols//2,	rows//2),	0,	scale) 
    scaled_image =	cv2.warpAffine(Image,	scale_matrix,	(cols,	rows)) 

    
    return np.asarray(Image,dtype='uint8'), np.asarray(scaled_image,dtype='uint8')
    
    
def get_consecutive_images(camera, frame):
    
    # get two consecutive images
    
    if camera == 3:
        left_image = np.array(data.get_cam3(frame), dtype = np.float32)
        right_image = np.array(data.get_cam3(frame + 1), dtype = np.float32)
    if camera == 2:
        left_image = np.array(data.get_cam3(frame), dtype = np.float32)
        right_image = np.array(data.get_cam3(frame + 1), dtype = np.float32)
    if camera == 1:
        left_image = np.array(data.get_cam3(frame), dtype = np.float32)
        right_image = np.array(data.get_cam3(frame + 1), dtype = np.float32)
    if camera == 0:
       left_image = np.array(data.get_cam3(frame), dtype = np.float32)
       right_image = np.array(data.get_cam3(frame + 1), dtype = np.float32)
    
    
    
    return np.asarray(left_image, dtype='uint8'), np.asarray(right_image, dtype='uint8')


def evaluate_scenario_1(imgLpoints, imgRpoints):

    # intensity scenarios:
    # in this scenarios we do not need to calculate the ground truth points positions
    # because the position stayed the same
    # so we only need to check their postion if it's the same or not 
    
    inliers = 0
    
    # check the diffrence in x and y
    for i in range(imgRpoints.shape[0]):
      
        
        diffrence_in_results_x = np.abs(imgLpoints[i, 0] - imgRpoints[i, 0])
        diffrence_in_results_y = np.abs(imgLpoints[i, 1] - imgRpoints[i, 1])
        diffrence = diffrence_in_results_x + diffrence_in_results_y
        
        if  diffrence < 1:
            inliers +=1
            
    number_of_points = imgLpoints.shape[0]
        
    
    return (inliers/number_of_points)

def evaluate_scenario_2(imgLpoints, imgRpoints, scale, image_shape):
    
    # scale scenarios:
    # in this scenarios we need to calculate the ground truth points positions
    # so we need to calculate left points positions by the scale/translation matrix 
    # and set rotation to zero 
    # then we compare the newly calculated points with the right image points 
    
    height, width = image_shape
    scale_matrix	=	cv2.getRotationMatrix2D((width//2,	height//2),	0,	scale)
    
    # create array of ones 
    ones = np.ones(imgLpoints.shape[0]).reshape(-1, 1)
    
    # make the left points homogenieus
    imgLpoints = np.concatenate((imgLpoints, ones), axis = 1)
    
    # array for the new points (rotated and translated)
    newLpoints = np.zeros((imgLpoints.shape[0], 2))
    
    # caculate new left points 
    for i in range(imgLpoints.shape[0]):
        
        # multiply every point to scale matrix to get the new points
        newLpoints[i] = scale_matrix @ imgLpoints[i]     
    
    
    inliers = 0
    
    # check the diffrence in x and y
    for i in range(imgRpoints.shape[0]):
      
        diffrence_in_results_x = np.abs(newLpoints[i, 0] - imgRpoints[i, 0])
        diffrence_in_results_y = np.abs(newLpoints[i, 1] - imgRpoints[i, 1])
        diffrence = diffrence_in_results_x + diffrence_in_results_y
        
        if  diffrence < 2:
            
            inliers +=1
    
    number_of_points = imgLpoints.shape[0]
        
    return (inliers/number_of_points)

def evaluate_scenario_3(imgLpoints, imgRpoints, rotation, image_shape):
    
    # rotation senario:
    # in this senario we need to calculate the ground truth points positions
    # so we need to calculate left points positions by the rotation/translation matrix 
    # and set scale to one 
    # then we compare the newly calculated points with the right image points 
    
    
    height, width = image_shape
    rotation_matrix	=	cv2.getRotationMatrix2D((width//2,	height//2),	rotation,	1)
    
    # make the points homogenieus
    ones = np.ones(imgLpoints.shape[0]).reshape(-1, 1)
    
    imgLpoints = np.concatenate((imgLpoints, ones), axis = 1)
    
    #rotation and translation for every point
    newLpoints = np.zeros((imgLpoints.shape[0], 2))
    
    # caculate enew left points 
    for i in range(imgLpoints.shape[0]):
        
        newLpoints[i] = rotation_matrix @ imgLpoints[i]     
    
    valide = 0
    
    for i in range(imgRpoints.shape[0]):
      
        diffrence_in_results_x = np.abs(newLpoints[i, 0] - imgRpoints[i, 0])
        diffrence_in_results_y = np.abs(newLpoints[i, 1] - imgRpoints[i, 1])
        diffrence = diffrence_in_results_x + diffrence_in_results_y
        
        if  diffrence < 2:
            valide +=1
    
        total = imgLpoints.shape[0]
    
    return (valide/total)

def evaluate_scenario_4(imgLpoints, imgRpoints):
    
    # stereo couple senario:
    # in this scenarios we do not need to calculate the ground truth points positions
    # so we just need to see if the y position stayed the same or not 
    

    inliers = 0
    
    #comparing diffrence in y values 
    for i in range(imgRpoints.shape[0]):
      
        diffrence_in_results_y = np.abs(imgLpoints[i, 1] - imgRpoints[i, 1])
        diffrence = diffrence_in_results_y
        
        if  diffrence < 1:
            inliers +=1
    
    
    number_of_points = imgLpoints.shape[0]
    inliers_percentage = inliers/number_of_points
    
    return inliers_percentage, number_of_points
        
def evaluate_scenario_5_6_7(imgLpoints, imgRpoints):
    
    # in these scenarios we use fundemental matrix to see distinguish between inliers and outliers points 
    # so we use the mask to filter out the outliers points 
    
    # to make sure the points found are more than 8 points  
    if imgLpoints.shape[0] > 8:
        
        # calculate the fundemental matrix with RASNAC & 8Points methodes
        f, mask = cv2.findFundamentalMat(imgLpoints, imgRpoints, cv2.FM_RANSAC + cv2.FM_8POINT);
        
        # get inliers number by seeing how many one in the mask 
        inliers_numbers = np.sum(mask == 1)
        
        # get the total number 
        number_of_points = imgLpoints.shape[0]
        # calculate the inliers percentage 
        inliers_percentage = inliers_numbers/number_of_points
        
    else: 
        
        inliers_percentage = 0
        number_of_points = imgLpoints.shape[0]
        
    
    return inliers_percentage, number_of_points

def costumized_detector(arguments, image_left, image_right, norm = cv2.NORM_L2):
    
    # in this function the goal was to treat the Detector, the Descriptor and the matching method seperatly
    # and by follwing this work flow we can add detectors and descriptors 
    #as much as we want without touching the code
    
    # split the arguments by comma 
    params = arguments.split(',')
    
    # variable activated FlANN is requested 
    special = False
    
    # getting first parameter (Detector)
    # in this part we get the left and right keypoints 

    if params[0] == 'SIFT':
        
        detector = cv2.SIFT_create()
        keypointsL = detector.detect(image_left)
        keypointsR = detector.detect(image_right)
    
    elif params[0] == 'AKAZE':
        
        detector = cv2.AKAZE_create()
        keypointsL = detector.detect(image_left)
        keypointsR = detector.detect(image_right)
        
    elif params[0] == 'BRISK':
        
        detector = cv2.BRISK_create()
        keypointsL = detector.detect(image_left)
        keypointsR = detector.detect(image_right)
        
    elif params[0] == 'ORB':
        
        detector = cv2.ORB_create()
        keypointsL = detector.detect(image_left)
        keypointsR = detector.detect(image_right)
        
    elif params[0] == 'KAZE':
        
        detector = cv2.KAZE_create()
        keypointsL = detector.detect(image_left)
        keypointsR = detector.detect(image_right)
        
    elif params[0] == 'FAST':
        
        detector = cv2.FastFeatureDetector_create()
        keypointsL = detector.detect(image_left)
        keypointsR = detector.detect(image_right)
        
        # detector.setNonmaxSuppression(False)
        
    elif params[0] == 'STAR':
        
        detector = cv2.xfeatures2d.StarDetector_create()
        keypointsL = detector.detect(image_left)
        keypointsR = detector.detect(image_right)
    
    
    # getting second parameter (Desriptor)
    # in this part we get the image descriptors by passing the keypoints as argument

    if params[1] == 'SIFT':
        
        descriptor = cv2.SIFT_create()
        _ , descriptorL = descriptor.compute(image_left, keypointsL)
        _ , descriptorR = descriptor.compute(image_right, keypointsR)
        
    elif params[1] == 'AKAZE':
        
        descriptor = cv2.AKAZE_create()
        _ , descriptorL = descriptor.compute(image_left, keypointsL)
        _ , descriptorR = descriptor.compute(image_right, keypointsR)
        
    elif params[1] == 'BRISK':
        
        descriptor = cv2.BRISK_create()
        _ , descriptorL = descriptor.compute(image_left, keypointsL)
        _ , descriptorR = descriptor.compute(image_right, keypointsR)
        
    elif params[1] == 'ORB':
        
        descriptor = cv2.ORB_create()
        _ , descriptorL = descriptor.compute(image_left, keypointsL)
        _ , descriptorR = descriptor.compute(image_right, keypointsR)
        
    elif params[1] == 'KAZE':
        
        descriptor = cv2.KAZE_create()
        _ , descriptorL = descriptor.compute(image_left, keypointsL)
        _ , descriptorR = descriptor.compute(image_right, keypointsR)
       
    elif params[1] == 'DAISY':
        
       descriptor = cv2.xfeatures2d.DAISY_create()
       _ , descriptorL = descriptor.compute(image_left, keypointsL)
       _ , descriptorR = descriptor.compute(image_right, keypointsR)
       
    elif params[1] == 'FREAK':
        
        descriptor = cv2.xfeatures2d.FREAK_create()
        _ , descriptorL = descriptor.compute(image_left, keypointsL)
        _ , descriptorR = descriptor.compute(image_right, keypointsR)
        
    
    # in this part we set a matcher depending on the method 
    
    if params[2] == 'L2':
        norm = cv2.NORM_L2
    
    elif params[2] == 'L1':
        norm = cv2.NORM_L1
        
    elif params[2] == 'HAMMING':
        norm = cv2.NORM_HAMMING
        
    elif params[2] == 'FLANN':
        special = True
    
    # if the matching method is L2 or L1 or HAMMING
    if not special:
        
        # create matcher with FBMATCH
        bf = cv2.BFMatcher(norm, crossCheck=True)
        
        # changing type as uint8 like CV_32F to make sure all methods work before we pass it to the match function  
        descriptorL = np.asarray(descriptorL, dtype= 'uint8')
        descriptorR = np.asarray(descriptorR, dtype= 'uint8')
        
        
        # Match descriptor
        matches = bf.match(descriptorL, descriptorR)
        
        # get left & image points 
        imgLpoints = np.float32([keypointsL[i.queryIdx].pt for i in matches]).reshape(-1, 2)
        imgRpoints = np.float32([keypointsR[i.trainIdx].pt for i in matches]).reshape(-1, 2)
        
        
    else:
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        # create flann based matcher 
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        # to make sure of the type before we pass it to KnnMatch
        descriptorL = descriptorL.astype(np.float32)
        descriptorR = descriptorR.astype(np.float32)
        
        # Count of best matches found per each query descriptor or less
        matches = flann.knnMatch(descriptorL,descriptorR, k = 2)
        
        
        # flann.Match function returns a list of lists and we want a flat list
        flat_matches = [item for sublist in matches for item in sublist]
        
        imgLpoints = np.float32([keypointsL[i.queryIdx].pt for i in flat_matches]).reshape(-1, 2)
        imgRpoints = np.float32([keypointsR[i.trainIdx].pt for i in flat_matches]).reshape(-1, 2)
         
    return imgLpoints, imgRpoints
    

def evaluate_detector_intensity(method, ranging_b, ranging_c):
    
    # variables to store evaluation
    b_evaluation = []
    c_evaluation = []
    
    for intensity in ranging_b:
        
        # getting left & right images with additive method 
        original_image, ground_truth_intensed = get_cam_intensified_additive(2, 0, intensity)
        
        # getting left and right points 
        imgLpoints, imgRpoints = costumized_detector(method, original_image, ground_truth_intensed)
        
        # evaluating with senario 1
        evaluation = evaluate_scenario_1(imgLpoints, imgRpoints)
        
        # store evaluation 
        b_evaluation.append(evaluation)
        
    
    for intensity in ranging_c:
        
        # getting left & right images with multliplying method 
        original_image, ground_truth_intensed = get_cam_intensified_multiple(2, 0, intensity)
        
        # getting left and right points 
        imgLpoints, imgRpoints = costumized_detector(method, original_image, ground_truth_intensed)
        
        # evaluating with senario 1
        evaluation = evaluate_scenario_1(imgLpoints, imgRpoints)
        
        # store evaluation 
        c_evaluation.append(evaluation)
    
    return np.array(b_evaluation), np.array(c_evaluation)

def evaluate_detector_rotation(method, ranging):
    
    # variable to store scenario evaluation
    scenario_evaluation = []

    for frame in tqdm(range(1)): # we can change the range up until 50 frames
        
        # variable to store rotation evaluation
        rotation_evaluation = []
        
        for rotation in ranging:
            
            # get original and rotated image 
            original_image, ground_truth_rotation = get_cam_rotation(2, frame, rotation)
            
            # pass it to the detector 
            imgLpoints, imgRpoints = costumized_detector(method, original_image, ground_truth_rotation)
            
            # evaluating with senario 3
            evaluation = evaluate_scenario_3(imgLpoints, imgRpoints, rotation, original_image.shape[:2])
            
            # store rotation evaluation 
            rotation_evaluation.append(evaluation)
           
        # store scenario evaluation
        scenario_evaluation.append(rotation_evaluation)
        
    # from list to array 
    scenario_evaluation = np.array(scenario_evaluation)
    
    # perpare the array to return 
    rotation_means = np.zeros(ranging.shape[0])
    
    # get every rotation mean from 0 to 50 frames 
    for i in range(rotation_means.shape[0]):
        
        rotation_means[i] = scenario_evaluation[:, i].mean()
        
    return rotation_means

def evaluate_detector_scale(method, ranging):
    
    scenario_evaluation = []

    for frame in tqdm(range(50)):
        
        rotation_evaluation = []
        
        for rotation in ranging:
            
            # get original and scaled image 
            original_image, ground_truth_scaled = get_cam_scale(2, frame, rotation)
            
            imgLpoints, imgRpoints = costumized_detector(method, original_image, ground_truth_scaled)
            
            # evaluate with scenario 2
            evaluation = evaluate_scenario_2(imgLpoints, imgRpoints, rotation, original_image.shape[:2])
         
            rotation_evaluation.append(evaluation)
           
        
        scenario_evaluation.append(rotation_evaluation)
    
    scenario_evaluation = np.array(scenario_evaluation)
    
    scale_means = np.zeros(ranging.shape[0])
    
    # get every scale mean from 0 to 50 frames 
    for i in range(scale_means.shape[0]):
        
        scale_means[i] = scenario_evaluation[:, i].mean()
        
    return scale_means


def evaluate_detector_stereo(method):
        
    scenario_evaluation = []
    
    # variable to store each detector timer 
    timers = []
    
    for frame in tqdm(range(50)):
        
        # get stereo couple 
        left_image, right_image = get_stereo_couple(frame)
        
        # start timer 
        start_time = time.time()
        imgLpoints, imgRpoints = costumized_detector(method, left_image, right_image)
        timer = time.time() - start_time 
        
        timers.append(timer)
        
        # evaluate with scenario 4
        evaluation = evaluate_scenario_4(imgLpoints, imgRpoints)
        
        scenario_evaluation.append(evaluation)
        
    return np.array(scenario_evaluation), np.array(timers).mean()

def evaluate_detector_consv_image(method):
        
    scenario_evaluation = []
    timers = []
    
    for frame in tqdm(range(50)):
        
        # get consecutive images from the same camera 
        image_t, image_t_1 = get_consecutive_images(2, frame)
        
        # start timer 
        start_time = time.time()
        imgLpoints, imgRpoints = costumized_detector(method, image_t, image_t_1)
        timer = time.time() - start_time 
        
        timers.append(timer)
        
        # evaluating with fundemental matrix 
        evaluation = evaluate_scenario_5_6_7(imgLpoints, imgRpoints)
        
        scenario_evaluation.append(evaluation)
        
    return np.array(scenario_evaluation), np.array(timers).mean()


def evaluate_detector_conv_images(method, delta_time):
        
    scenario_evaluation = []
    timers = []
    
    for frame in tqdm(range(50 - delta_time)):
        
        # get the four images 
        left_image_t, right_image_t = get_stereo_couple(frame)
        left_image_t_1, right_image_t_1 = get_stereo_couple(frame + delta_time)
        
        start_time = time.time()
        
        # pass it to the detector 
        imgLpoints, imgRpoints = costumized_detector(method, left_image_t, right_image_t_1)
        timer = time.time() - start_time 
        
        timers.append(timer)
        
        # evaluating with fundemental matrix 
        evaluation = evaluate_scenario_5_6_7(imgLpoints, imgRpoints)
        
        scenario_evaluation.append(evaluation)
        
    return np.array(scenario_evaluation), np.array(timers).mean()

    
    
def plot_senario_results(methods, senario_name, x, y):
    
    
    plt.title(senario_name)
    
    for i in range(methods.shape[0]):
        
        plt.plot(x, y[i], label = methods[i])
        plt.scatter(x,y[i])
        plt.legend(bbox_to_anchor=(1.5, 1), loc = 'upper right')
        
    plt.show()

def show_results(methods, results, clock):
    
    x = pt.PrettyTable()
    x.field_names = ["Method ", "excution Time", "image Points", "% of inliers"]
    #x.field_names = ["Method", "image Points", "% of inliers"]
    
    for i in range(methods.shape[0]):
        
        x.add_row([methods[i], clock[i], int(results[i, :, 1].mean()), results[i, :, 0].mean() *100 ])

    return x

#%%
## defining methods 

method_one = 'SIFT,SIFT,L2'
method_two = 'AKAZE,AKAZE,HAMMING'
method_three = 'ORB,ORB,L1'
method_four = 'BRISK,BRISK,L2'
method_five = 'FAST,DAISY,FLANN'
method_six = 'STAR,SIFT,L2'
method_seven = 'FAST,BRISK,L2'
method_eight = 'FAST,SIFT,L2'
method_nine = 'KAZE,BRISK,L1'
method_ten = 'ORB,FREAK,L1'
method_eleven = 'STAR,ORB,HAMMING'
method_twelve = 'KAZE,AKAZE,FLANN'

methods = np.array([method_one, method_two,
                    method_three, method_four, 
                    method_five, method_six, 
                    method_seven,method_eight, 
                    method_nine, method_ten,
                    method_eleven, method_twelve])

#%%

## senario 1 (intensity changes)

b_final_results = []
c_final_results = []

for method in methods:
    
    b_results, c_results = evaluate_detector_intensity(method, ranging_b, ranging_c)
    
    b_final_results.append(b_results)
    c_final_results.append(c_results)
    
b_final_results = np.array(b_final_results)
c_final_results = np.array(c_final_results)

plot_senario_results(methods,'Intensity Resutls with Addition', ranging_b, b_final_results)
plot_senario_results(methods,'Intensity Resutls with multiplication', ranging_c, c_final_results)

#%%

## senario 2 (scale changes)
scale_final_results = []

for method in methods:
    
    scale_evaluation = evaluate_detector_scale(method, ranging_scale)
    
    scale_final_results.append(scale_evaluation)

scale_final_results = np.array(scale_final_results)

plot_senario_results(methods,'Scale Results', ranging_scale, scale_final_results)


#%%

## senario 3 (rotation changes)
rotation_final_results = []

for method in methods:
    
    rotation_evaluation = evaluate_detector_rotation(method, ranging_rotation)
    
    rotation_final_results.append(rotation_evaluation)

rotation_final_results = np.array(rotation_final_results)

plot_senario_results(methods,'Rotation Results', ranging_rotation, rotation_final_results)

#%% 

## senario 4 (Stereo pair)

stereo_final_results = []
clock = []

for method in methods:
    
    method_evaluation, timer = evaluate_detector_stereo(method)
    
    clock.append(timer)
    stereo_final_results.append(method_evaluation)
    
stereo_final_results = np.array(stereo_final_results)
clock = np.array(clock)

table = show_results(methods, stereo_final_results, clock)

print(table)

#%% 

## senario 5 (consecutive images )

consv_final_results = []
clock = []

for method in methods:
    
    method_evaluation, timer = evaluate_detector_consv_image(method)
    
    clock.append(timer)
    consv_final_results.append(method_evaluation)
    
consv_final_results = np.array(consv_final_results)
clock = np.array(clock)

table = show_results(methods, stereo_final_results, clock)

print(table)

#%% 

## senario 6 ( t with t+1' )

consv_final_results = []
clock = []

for method in methods:
    
    
    method_evaluation, timer = evaluate_detector_conv_images(method, 1)
    
    clock.append(timer)
    consv_final_results.append(method_evaluation)
    
consv_final_results = np.array(consv_final_results)
clock = np.array(clock)

table = show_results(methods, consv_final_results, clock)

print(table)

#plot_senario_results(methods, np.arange(0, 49), stereo_final_results[:, :, 0])



#%% 

## senario 7 ( t with t+2' )
consv_final_results = []
clock = []


for method in methods:
    
    method_evaluation, timer = evaluate_detector_conv_images(method, 2)
    
    clock.append(timer)
    consv_final_results.append(method_evaluation)
    
stereo_final_results = np.array(consv_final_results)
clock = np.array(clock)

table = show_results(methods, consv_final_results, clock)

print(table)

#%%






