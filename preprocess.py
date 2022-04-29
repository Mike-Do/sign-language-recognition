# import necessary libraries
import os
import cv2
import numpy as np

# This is the method for splitting all video samples into frames.
def videosToFrames():
    directory = 'videos'
    allFrames = []
    threshold = 2000 # threshold for number of frames from each video; change if necessary
    
    # iterate over files in the directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename) # file path
        if os.path.isfile(f):
            # create a VideoCapture object to read the video
            cap = cv2.VideoCapture(f)
            currFrames = []

            # Loop until the end of the video or until the threshold has been reached
            count = 0
            while (cap.isOpened() and count < threshold):
                # capture frame by frame
                ret, frame = cap.read()
                frame = cv2.resize(frame, (540, 380), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                currFrames.append(frame)
                count += 1
            allFrames.append(currFrames)
            
            # release the video capture object
            cap.release()
            
    return allFrames