# forest_health_monitoring

###################
CrownLossRanking.py
###################
Display images and boxes predicted by RTCLE and TSCLR side-by-side. Frame by frame prediction. 
Video can be either from youtube or sample video recorded from drones.

Adjustable variables
videoOption(str): video or youtube
FPVDrone3(str): any other youtube video url you want to try
weights_type(str): weights type from background subtractive methods(value, GLI or kmeans) for encoder. Note, flipping is needed (-1) for kmeans and GLI
speedup(int): set 1 to 3 to adjust how many frames to skip to speed up for demo (set to 1 if not demo)

Functions:
labelFrameAtBottomRight: Puts a text label (eg TSCLR or RTCLE) in video frame

#######################
objectDetectionModel.py
#######################
DetectionModel class. Functions name are very straightforward.

######
VAE.py
######
The most important function is build_vae. pass in the weights (set using weights_type) and this function returns vae, encoder, decoder.
encoder used in TSCLR


#################
blenderTreeGen.py
#################
Run this in Blender. Renders tree into png format. See screen recording video in blenderModels folder.

#####################
testOnSampleVideos.py
#####################
RTCLE model on sample leaf or tree video.

