# import necessary libraries
import os
import cv2
import numpy as np
import json
import uuid

import shutil

# dictionary maps glosses to their corresponding frames
all_frames = {}
threshold = 50

# This is the method for splitting all video samples into frames.
def videosToFrames():
    directory = 'videos'

    # make frames directory to store the images
    if not os.path.exists('frames'): 
        os.mkdir('frames')
    
    # make train and test directories to store training and testing data
    train_f = os.path.join('frames', 'train')
    test_f = os.path.join('frames', 'val')
    if not os.path.exists(train_f):
        os.mkdir(train_f)
    if not os.path.exists(test_f):
        os.mkdir(test_f)

    # iterate over the sub directories in videos
    for gloss in os.listdir(directory):
        # create directory for each gloss in train and test if it doesn't exist
        train_gloss_f = os.path.join('frames/train', gloss)
        test_gloss_f = os.path.join('frames/val', gloss)

        # if gloss directory doesn't exist, create it
        if not os.path.exists(train_gloss_f):
            os.mkdir(train_gloss_f)
        if not os.path.exists(test_gloss_f):
            os.mkdir(test_gloss_f)
        
        gloss_frames = [] # frames for all videos in the same gloss

        sub_dir = os.path.join(directory, gloss)

        # check if sub_dir is a valid directory
        if os.path.isdir(sub_dir) == False:
            continue

        train = True # indicate if the video is for training or testing
    
        # iterate over files in the directory
        for filename in os.listdir(sub_dir):
            f = os.path.join(sub_dir, filename) # file path
            if os.path.isfile(f):
                # create a VideoCapture object to read the video
                cap = cv2.VideoCapture(f)
                curr_frames = []

                # get the total number of frames in the video
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print("Total frames for " + filename + " is " + str(total_frames) + " frames.")

                # Loop until the end of the video or until the threshold has been reached
                while (cap.isOpened()):
                    # capture frame by frame, ret will be None if the frame is not valid
                    ret, frame = cap.read()
                    
                    # break if the frame is not valid
                    if not ret:
                        break
                    curr_frames.append(frame)
                    
                # release the video capture object
                cap.release()
                
                # sample curr_frames to get 50 frames
                sampled_frames = sequential_sampling(curr_frames)
                
                # add the sampled frames to the gloss subdirectory
                if train:
                    os.chdir(train_gloss_f)
                else:
                    os.chdir(test_gloss_f)

                print("Saving image to directory")
                for frame in sampled_frames:
                    # check for valid frame
                    if frame is None or type(frame) is int or np.sum(frame) == 0:
                        continue

                    # generate random name for each Image
                    curr_filename = "Image" + str(uuid.uuid4()) + ".jpg"
                    cv2.imwrite(curr_filename, frame)

                train = not train
                
                # change back to root directory (3 levels back)
                os.chdir("../../../")

                # add the selected 50 frames to the list of all frames        
                gloss_frames.append(sampled_frames)

        # add all the frames to the all_frames dictionary
        all_frames[gloss] = gloss_frames

    find_missing() # find words with no train/val data and write them to a textfile

    return all_frames
    
def sequential_sampling(curr_frames):
    num_frames = len(curr_frames)
    frames_to_sample = []
    if num_frames > threshold:
        frames_skip = set()
        num_skips = num_frames - threshold
        interval = num_frames // num_skips

        for i in range(num_frames):
            if i % interval == 0 and len(frames_skip) <= num_skips:
                frames_skip.add(i)
        for i in range(num_frames):
            if i not in frames_skip:
                frames_to_sample.append(curr_frames[i])
    else:
        frames_to_sample = list(range(num_frames))
    
    return frames_to_sample


def find_missing():
    missing = []
    # loop through all the directories in train,
    # if the directory is empty, save the directory name to a list
    # delete the directory, and the corresponding directory in val
    
    train_f = os.path.join('frames', 'train')
    test_f = os.path.join('frames', 'val')
    for gloss in os.listdir(train_f):
        # check if directory is empty
        if len(os.listdir(os.path.join(train_f, gloss))) == 0:      
            # append gloss to list 
            missing.append(gloss)
            # delete the directory in train and in val
            shutil.rmtree(os.path.join(train_f, gloss))
            # recursively remove the directory in val
            shutil.rmtree(os.path.join(test_f, gloss))
        
    # loop through all the directories in val,
    # if the directory is empty, save the directory name to a list
    # delete the directory, and the corresponding directory in val
    for gloss in os.listdir(test_f):
        if len(os.listdir(os.path.join(test_f, gloss))) == 0:
            # append gloss to list
            missing.append(gloss)
            # delete the directory in val
            shutil.rmtree(os.path.join(test_f, gloss))
            # recursively remove the directory in train
            shutil.rmtree(os.path.join(train_f, gloss))
            
    # write the list to a text file called missing_words.txt
    # open the file to write to it, or create it if it doesn't exist
    if os.path.exists('missing_words.txt'):
        print('file already exists')
    
    with open('missing_words.txt', 'w') as f:
        # append the list to the file
        for gloss in missing:
            f.write(gloss + '\n')    

""""
This function loops through all objects in the dataset
and creates a custom data entry for each object. These data entries
are stored in the data list variable. After, it loops
over these data entries and sequentially samples the frames
and stores the indices of all frames in the all_frames list.
"""
# def make_dataset():
#     # store video instances into custom entries
#     data = {}
#     # set the split file
#     split = ['train', 'val']
#     # set the number of samples per video
#     num_samples = 50
#     # store the directory of the videos
#     index_file_path = "./WLASL_v0.3.json"
#     # store the list of all videos
#     all_videos = {}

#     # open the json file and read into content
#     with open(index_file_path, 'r') as f:
#         content = json.load(f)

#      # make dataset using glosses (i.e. words)
#     for gloss_entry in content:
#         # store the gloss and the video instances (there are multiple videos for each gloss)
#         gloss, instances = gloss_entry['gloss'], gloss_entry['instances']

#         # only create entires for desired glosses
#         if (gloss == "hello" or gloss == "world"):
#             gloss_data = []
#             # for each video instance
#             for instance in instances:
#                 # if the video is not in the train split
#                 if instance['split'] not in split:
#                     # skip the video instance
#                     continue
                
#                 # store the frame end and start, as well as the video id
#                 frame_end = instance['frame_end']
#                 frame_start = instance['frame_start']
#                 video_id = instance['video_id']

#                 # store the id, frame start, frame end as an entry in the data list
#                 instance_entry = video_id, frame_start, frame_end
#                 gloss_data.append(instance_entry)

#             data[gloss] = gloss_data
    
#     # go through each gloss and get its frames
#     for index in data.keys():
#         gloss_frames = []
#         # for each data instance in gloss_data
#         for datum in data[index]: 
#             # destructure the data entry stored above
#             video_id, frame_start, frame_end = datum
#             # sequential sampling the frames
#             frames = sequential_sampling(frame_start, frame_end, num_samples)
#             gloss_frames.append(frames)
        
#         all_videos[index] = gloss_frames


#     # go through each video and get its frame samples
    
#     # for index in range(len(data)):
#     #     # destructure the data entry stored above
#     #     video_id, frame_start, frame_end = data[index]
#     #     # sequential sampling the frames
#     #     frames = sequential_sampling(frame_start, frame_end, num_samples)
#     #     # store all the grabbed frames
#     #     all_videos.append(frames)
    
#     # return the list of the indices all frames selected

#     return all_videos
    

# """Keep sequentially ${num_samples} frames from the whole video sequence by uniformly skipping frames."""
# def sequential_sampling(frame_start, frame_end, num_samples):
#     # capture the number of frames in the video
#     num_frames = frame_end - frame_start + 1

#     # store the sampled frames
#     frames_to_sample = []
    
#     # if the number of frames exceeds the threshold number of frames
#     if num_frames > num_samples:
#         # store the number of frames to skip
#         frames_skip = set()

#         # the number of frames to skip is uniformly distributed between 0 and the number of frames
#         num_skips = num_frames - num_samples
#         interval = num_frames // num_skips

#         # for each frame from start to end
#         for i in range(frame_start, frame_end + 1):
#             # store frames to skip uniformly at random
#             if i % interval == 0 and len(frames_skip) <= num_skips:
#                 frames_skip.add(i)

#         # loop through the frames once more and store "non-skipped" frames
#         for i in range(frame_start, frame_end + 1):
#             if i not in frames_skip:
#                 frames_to_sample.append(i)
#     else:
#         # if the number of samples is less than the number of frames, the frames to sample are all the frames
#         frames_to_sample = list(range(frame_start, frame_end + 1))
    
#     # return all the frames to sample
#     return frames_to_sample

if __name__ == '__main__':
    videosToFrames()