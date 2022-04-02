import time

from VAE import *
from objectDetectionModel import DetectionModel

import cv2


def labelFrameAtBottomRight(label, frame):
    frame_copy = frame.copy()
    cv2.putText(frame_copy, label, (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (210, 210, 210), 8)
    return frame_copy


def getRanking(croppedIms, encoder, flip=1.0):
    tmp = [encoder.predict(min_max_norm(cv2.resize(im, (128, 128))).reshape(1, 128, 128, 3))[0][0][0] for im in
           croppedIms]
    return np.argsort(np.array(tmp) * -1 * flip) #times -1 because we want in desc order


def drawBoxes(boxes, outputRanking, im, net):
    img = im.copy()
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(net.classes[outputRanking[i]])
        color = net.colors[outputRanking[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2) 
        cv2.rectangle(img, (x, y), (x + w, y + 20), color, -1)
        cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        # cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    return img


def getRankingPrediction(net, im, encoder, flip):
    boxes, confidences, class_ids, indexes = net.predict(im)
    croppedIms, _, coordBoxes = net.cropDetectedObject(boxes, confidences, class_ids, indexes, im)
    outputRanking = getRanking(croppedIms, encoder, flip)
    predictedIm = drawBoxes(coordBoxes, outputRanking, im, net)
    return predictedIm


def getCLPrediction(net, im):
    boxes, confidences, class_ids, indexes = net.predict(im)
    predictedIm = net.drawDetectionBB(boxes, confidences, class_ids, indexes, im)
    return predictedIm


# video (from sample video) and youtube (from youtube video)
videoOption = "video"  # or youtube #video
sampleVideoName = "9trial_kit"
FPVDrone3 = "https://www.youtube.com/watch?v=X8UyYegR4Lw"  # shows that viewpoint changes affect cl prediction too
url = FPVDrone3

weights_type = 'GLI' #GLI value or kmeans  #Variational autoencoder performance will be different with background substraction
WEIGHTS_PATH = "weights/"
vae, encoder, decoder = build_vae(weights_type)
weights_name = f"wsl_bg_agnostic({weights_type})_weights.hdf5"
vae.load_weights(WEIGHTS_PATH + weights_name)
if weights_type == 'kmeans' or weights_type == 'GLI':
    flip = -1
else:
    flip = 1

# RTCLE model
W, H = 512, 512
weightsPath = WEIGHTS_PATH + "bestVisuals.weights"
cfgPath = "cfg\yolov3_tiny_crownLoss.txt"
crownLossColors = [(0, 255, 0), (192, 192, 127), (0, 127, 66), (0, 0, 190), (0, 0, 255)]
with open("classes\crownLossBins.names", "r") as f:
    crownLossBins = [line.strip() for line in f.readlines()]
crownLossNet = DetectionModel(weightsPath, cfgPath, crownLossBins, crownLossColors, 0.0, 0.25)

# tree detection model
weightsPath = WEIGHTS_PATH + "treeWeights4.weights"
cfgPath = "cfg\yolov3_v2-tiny.cfg.txt"

# define ten trees
rankings = ["Rank: " + str(i) for i in range(1, 11)]
rankingColors = [(0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (192, 192, 127),
                 (0, 127, 66), (0, 0, 255)][::-1]
treeDetectionNet = DetectionModel(weightsPath, cfgPath, rankings, rankingColors, 0.0, 0.25)

# controls how many frame to skip (for demo or youtube set to 3)
speedUp = 3  # every 3 frames - make it faster with higher values

if videoOption == "youtube":
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)
else:
    cap = cv2.VideoCapture(f"sampleVideos/{sampleVideoName}.mp4")

frame_id = 0
count = 0
while True:
    _, frame = cap.read()
    frame_id += 1
    if frame is None:
        break
    if (frame_id % speedUp) != 0:
        continue
    frame = cv2.resize(frame, (W, H))
    # TSCLR
    rankingPred = getRankingPrediction(treeDetectionNet, frame, encoder, flip)
    rankingPred = labelFrameAtBottomRight("TSCLR", rankingPred)

    # RTCLE
    clPred = getCLPrediction(crownLossNet, frame)
    clPred = labelFrameAtBottomRight("RTCLE", clPred)

    # join the predictions into a frame
    finalIm = np.hstack((clPred, rankingPred))
    cv2.imshow("Image", finalIm)

    cv2.imwrite(f"C://Users//bahad//Box//Ho Boon//Software Archive//software_archive//droneProj//sampleVideos//processed17//frame_{count}.jpg", finalIm)

    count += 1
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
