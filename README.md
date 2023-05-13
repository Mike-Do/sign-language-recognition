# Robust Sign Language Detection on WLASL

## Abstract

Vision-based sign language recognition aims to identify
specific human hand gestures to convey information. For
this project, we present and analyze a system that interprets
American Sign Language (ASL) using a vocabulary set from
the Word-Level American Sign Language (WLASL) video
dataset, which contains more than 2,000 words and 21,000
videos performed by over 100 signers. Although the original
paper that introduces this dataset trains a model on all 2,000
words/glosses, we trained a binary classifier with a maxi-
mum of 12 videos per word, due to limited computational
resources

Running Instructions:
1. `python video_downloader.py` to download the raw videos. The raw videos downloaded can be found under
the raw_videos directory.
2. `python video_samples.py` to process the raw videos into video samples. The video samples can be found
under the videos directory.
3. `python preprocess.py` to split the video samples into frames, create training and testing dataset and store them in directories. The frames can be found in the frames directory, separated into training
and testing sub-directories.
4. `python model.py` to train and evaluate the model. The training and testing accuracy for each epoch will be printed out.

Problems with Data:
1. Some of the video samples contain random frames even after preprocessing, which affects the accuracy
of the model.
2. Many of the video samples include frames where the signers aren't making any signs and are simply standing still. These frames are hard for the model to classify and negatively affect the accuracy of our model.

The number of words to classify and the number of videos to download for each word can be adjusted by
changing the parameters in video_downloader.py and video_samples.py. Due to lack of computational
resources, we trained a binary classifier using 12 videos per word.  
