import time
from objectDetectionModel import DetectionModel
import pafy
import cv2

# configure the settings
W, H = 512, 512
testFor = "tree"
if testFor == "leaves" or testFor == "leaf":
    # leaf
    weightsPath = "weights/leaf.weights"
    cfgPath = "cfg/yolov3_leaf-tiny.cfg.txt"
    videoPath = "sampleVideos/leaves.mp4"
    classPath = "classes/leaves.names"
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    thres = 0.45
    sep = 0.75
else:
    weightsPath = "weights/bestVisuals.weights"
    videoPath = "sampleVideos/droneVideo2.mp4"
    classPath = "classes/crownLossBins.names"
    colors = [(0, 255, 0), (192, 192, 127), (0, 127, 66), (0, 0, 190), (0, 0, 255)]
    thres = 0.45
    sep = 0.75
    cfgPath = "cfg/yolov3_tiny_crownLoss.txt"



with open(classPath, "r") as f:
    classes = [line.strip() for line in f.readlines()]
net = DetectionModel(weightsPath, cfgPath, classes, colors, thres, sep)

# controls the play speed of th video by skipping some frames (1 means usual speed ¬20 Fps)
speedUp = 1

cap = cv2.VideoCapture(videoPath)

starting_time = time.time()
frame_id = 0
count = 0
while True:
    _, frame = cap.read()
    if frame is None:
        break
    frame_id += 1
    if (frame_id % speedUp) != 0:
        continue
    frame = cv2.resize(frame, (W, H))
    boxes, confidences, class_ids, indexes = net.predict(frame)
    im = net.drawDetectionBB(boxes, confidences, class_ids, indexes, frame)
    im = frame

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    # cv2.putText(im, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
    cv2.imshow("Image", im)
    count += 1
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
