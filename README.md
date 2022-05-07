# sign-language-recognition

1. Binary and 

Running Instructions:
1. python video_downloader.py to download the raw videos
2. python video_samples.py to process the raw videos into video samples
3. python preprocess.py to split the video samples into frames, create training and testing dataset and store them in directories

List of Words (20):
1. again
2. also
3. ask
4. because
5. boy
6. but
7. can
8. come
9. deaf
10. different
11. drink
12. drive
13. eat
14. email
15. excuse
16. family
17. feel
18. few
19. find
20. fine

Download 10 videos for each word

Source: https://www.handspeak.com/word/most-used/

Problems with Data:
1. Some of the video samples contain random frames even after preprocessing, which affects the accuracy
of the model.

Questions for Henry:
• How should we install the dataset?
• Is it feasible to train on all 2000 words, with way more than 2000 videos? (> 100 gb storage locally)
    • Should/can we do it on GCP?
    • If not, how do we go about selecting a smaller sample of words and their corresponding videos?
• How do we only download a select number of videos?
• What are we training our model on? (how are the collection of frames individually represented) As programmatic classes? Or are we just inserting the collection of frames into the model as images?
• How do we train our model and have access to the frames?

Next Steps:
1. Select videos and words to use as data
2. Split them into frames (test if it works)
3. Create a model and train it (Understand the model architecture, run on GCP)


4. Create realtime video prediction, keystorke for starting and stopping the sign


Steps:
1. Install youtube-dl and download raw videos
    cd start_kit
    python video_downloader.py
2. Extract video samples from raw videos
    python preprocess.py
    Video samples should be under directory videos/
3. Extract frames from video samples
    Set threshold for number of frames from each video
    Threshold: Check number of frames in each video sample 

Preprocess with OpenCV
    • Make a labeling tool with OpenCV that lets us open a video file in a video of the dataset
    • Create a key-bind for sifting through frame-by-frame
    • Aim to pull an even distribution (pull at an even rate) of frames from each video
    • Adjust number of frames - from 1,000 to 10,000 frames - to deal with overfit or underfit
Starting from a subset of the dataset (5 signs) to first get good accuracy
    • Pick 5 commons signs we want good accuracy on
    • Split the videos for those signs into images using OpenCV that takes frames out of those videos















General CV Notes:
CV Notes

* Whole-body gestures rather than just hand gestures
* Start with a subset of the dataset (few signs) to first get good accuracy. Then go from there.
* If FD, map of feature vector set to sign
* Unsupervised learning approach (give model dataset of an existing sign, would need ground truth of feature vectors
    * 	AWS Mechanical Turk for labeling
* PyTorch (no Keras) and Torch Vision
    * ResNet for image classification
    * Grab ResNet and fine-tune it
        * Torch Vision has pre-trained image classification models
            * Take an already trained model and add my own few layers that are specialized for sign language classification, or take layers we want to get rid of
* Pick 5 signs we want good accuracy on
* Split the videos for those signs, split them into images using OpenCV that takes frames out of those videos
    * Now have a big 
* Partition those 5 signs and run them through models
    * Edit hyperparameters
    * divide the videos into training and testing set
* Another OpenCV pipeline for real-time
    * OpenCV processes videos frame by frame
    * IMage detection and classification model
    * Set up the app with a while loop that processes it as an array of frames, classification on each frame, calls to evaluate a function on each frame as it comes in in the while loop
    * Plot 20 and take the highest 10 for real-time classification
    * Create a state machine that determines when the sign start, then start keeping track of the results of the image classifier, as the classifier went through each image, outputs a vector of % chance that that image belongs to each of the signs. Get the first frame to first classify similar signs to get an initial sign vector, then the other 10 frames of that sign will update that first vector to get the sign. A buffer to indicate when you start a sign
    * After the sign is over, have a “control” label in our model that looks for “empty” (e.g. not making a sign), stop detecting, and return the result of the last set of frames
    * Maybe start with a keybind for starting and stopping. Ideally, process output of each model output and it would output (no sign, no sign, …), Then start assembling a vector. Then, return when the classification of the end on the next no sign.
* Make it really good with buttons. Emphasis is on DL and CV part, not on the system and state machine of the pipeline.
* Train on videos for “not signing” but not necessary (do not worry about that). Just have the buttons. If you want to go above and beyond, train on a label for “not signing” and train on video/images of not doing anything. Just stick with buttons and the data that we do have.

* Preprocess with OpenCV. Make a labeling tool with OpenCV that lets us open a video file in a video of the dataset. Make a keybind for sifting through frame by frame. See how many frames there are. Once you have a total number of frames. You don’t necessarily need tons of frames. Pull an even distribution of frames from each video. (tons of frames create overfit). Start by pulling about 1,000 of 10,000 frames, and pull them from an even rate (e.g. for every 10 frames, pick 1). If it still overfits, bring it down. If underfit, increase. Make sure the training and test set are the same amount of images. We could get accurate images if this does not work. Decide how much to pull out.
* OpenCV
* The biggest takeaway from CNN and image class really comes down to the quality of data. The data you chose and the distribution of data you have MUST be even. The same number of images for labels. Same number for training and testing. (could do button that counts seconds to make sure that frames are even)/
* Eliminate certain signs from the data set if needed. Better off with smaller sets with a high variety of videos
* Figure out how to add layers in PyTorch
* Add layers to the end of the model (do fine-tune with that model from the PyTorch tutorial on the dataset)
    * Add 2 Conv Layers and A pool and A softmax, dump it into a Sigmoid (may not need Dense or big ReLU, but don’t know a priori)
    * Start with 5 labels to see if we like our model, can decide to add more
        * The Sigmoid, spitting out 0 or 1
