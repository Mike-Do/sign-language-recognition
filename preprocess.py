# import necessary libraries
import os
import cv2
import numpy as np
import utils
import json

# This is the method for splitting all video samples into frames.
def videosToFrames():
    directory = 'videos'
    all_frames = []
    
    # iterate over files in the directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename) # file path
        if os.path.isfile(f):
            # create a VideoCapture object to read the video
            cap = cv2.VideoCapture(f)
            curr_frames = []

            # Loop until the end of the video or until the threshold has been reached
            while (cap.isOpened()):
                # capture frame by frame, ret will be None if the frame is not valid
                ret, frame = cap.read()
                
                # check if the frame is valid
                if ret:
                    # resize the frame
                    frame = cv2.resize(frame, (540, 380), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                    # add the current frame to the list of frames
                    curr_frames.append(frame)
            
            # add all the frames to the list of all frames        
            all_frames.append(curr_frames)
            
            # release the video capture object
            cap.release()

    frame_indices = make_dataset()
    selected_frames = []
    
    # for each video in the dataset
    for index in range(len(frame_indices)):
        # get the selected frames in the video
        video_frames = all_frames[index][frame_indices[index]]
        # add the selected frames to the list of selected frames
        selected_frames.append(video_frames)

    return selected_frames

""""
This function loops through all objects in the dataset
and creates a custom data entry for each object. These data entries
are stored in the data list variable. After, it loops
over these data entries and sequentially samples the frames
and stores the indices of all frames in the all_frames list.
"""
def make_dataset():
    # store video instances into custom entries
    data = []
    # set the split file
    split = ['train', 'val']
    # set the number of samples per video
    num_samples = 50
    # store the directory of the videos
    index_file_path = "./WLASL_v0.3.json"
    # store the list of all videos
    all_frames = []

    # open the json file and read into content
    with open(index_file_path, 'r') as f:
        content = json.load(f)

     # make dataset using glosses (i.e. words)
    for gloss_entry in content:
        # store the gloss and the video instances (there are multiple videos for each gloss)
        gloss, instances = gloss_entry['gloss'], gloss_entry['instances']

        # for each video instance
        for instance in instances:
            # if the video is not in the train split
            if instance['split'] not in split:
                # skip the video instance
                continue
            
            # store the frame end and start, as well as the video id
            frame_end = instance['frame_end']
            frame_start = instance['frame_start']
            video_id = instance['video_id']

            # store the id, frame start, frame end as an entry in the data list
            instance_entry = video_id, frame_start, frame_end
            data.append(instance_entry)
    
    # go through each video and get its frame samples
    for index in range(len(data)):
        # destructure the data entry stored above
        video_id, frame_start, frame_end = data[index]
        # sequential sampling the frames
        frames = sequential_sampling(frame_start, frame_end, num_samples)
        # store all the grabbed frames
        all_frames.append(frames)
    
    # return the list of the indices all frames selected
    return all_frames
    

"""Keep sequentially ${num_samples} frames from the whole video sequence by uniformly skipping frames."""
def sequential_sampling(frame_start, frame_end, num_samples):
    # capture the number of frames in the video
    num_frames = frame_end - frame_start + 1

    # store the sampled frames
    frames_to_sample = []
    
    # if the number of frames exceeds the threshold number of frames
    if num_frames > num_samples:
        # store the number of frames to skip
        frames_skip = set()

        # the number of frames to skip is uniformly distributed between 0 and the number of frames
        num_skips = num_frames - num_samples
        interval = num_frames // num_skips

        # for each frame from start to end
        for i in range(frame_start, frame_end + 1):
            # store frames to skip uniformly at random
            if i % interval == 0 and len(frames_skip) <= num_skips:
                frames_skip.add(i)

        # loop through the frames once more and store "non-skipped" frames
        for i in range(frame_start, frame_end + 1):
            if i not in frames_skip:
                frames_to_sample.append(i)
    else:
        # if the number of samples is less than the number of frames, the frames to sample are all the frames
        frames_to_sample = list(range(frame_start, frame_end + 1))
    
    # return all the frames to sample
    return frames_to_sample