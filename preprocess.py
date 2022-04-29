# import necessary libraries
import os
import cv2
import numpy as np

# This is the method for splitting all video samples into frames.
def videosToFrames():
    directory = 'videos'
    allFrames = []
    
    # iterate over files in the directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename) # file path
        if os.path.isfile(f):
            # create a VideoCapture object to read the video
            cap = cv2.VideoCapture(f)
            currFrames = []

            # Loop until the end of the video
            while (cap.isOpened()):
                # capture frame by frame
                ret, frame = cap.read()
                frame = cv2.resize(frame, (540, 380), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                currFrames.append(frame)
            
            # release the video capture object
            cap.release()
        allFrames.append(currFrames)
    
    return allFrames