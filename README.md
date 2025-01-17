# Forest Health Monitoring

## Crown Loss Ranking

### CrownLossRanking.py

Display images and boxes predicted by RTCLE and TSCLR side-by-side. Frame by frame prediction. 
Video can be either from youtube or sample video recorded from drones.

Adjustable variables
videoOption(str): video or youtube
FPVDrone3(str): any other youtube video url you want to try
weights_type(str): weights type from background subtractive methods(value, GLI or kmeans) for encoder. Note, flipping is needed (-1) for kmeans and GLI
speedup(int): set 1 to 3 to adjust how many frames to skip to speed up for demo (set to 1 if not demo)

Functions:
labelFrameAtBottomRight: Puts a text label (eg TSCLR or RTCLE) in video frame

## Object Detection Model

### objectDetectionModel.py

DetectionModel class. Functions name are very straightforward.

## Variational Autoencoder

### VAE.py

The most important function is build_vae. pass in the weights (set using weights_type) and this function returns vae, encoder, decoder.
encoder used in TSCLR

## Tree Rendering from Blender

### blenderTreeGen.py

Run this in Blender. Renders tree into png format. See screen recording video in blenderModels folder.

## Test on Sample Video

### testOnSampleVideos.py

RTCLE model on sample leaf or tree video.

#### Cite the following if you use this work:

```
@article{kocer2022vision,
  title={Vision based Crown Loss Estimation for Individual Trees With Remote Aerial Robots},
  author={Ho, Boon and Kocer, Basaran Bahadir and Kovac, Mirko},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {188},
  pages = {75-88},
  year = {2022},
  issn = {0924-2716},
  doi = {https://doi.org/10.1016/j.isprsjprs.2022.04.002},
  publisher={Elsevier}
}
```
