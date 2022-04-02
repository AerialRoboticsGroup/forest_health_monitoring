import time
from objectDetectionModel import DetectionModel
import pafy
import cv2

# configure the settings
W, H = 512, 512
COLORS = [(0, 255, 0), (192, 192, 127), (0, 127, 66), (0, 0, 190), (0, 0, 255)]
weightsPath = "weights/bestVisuals.weights"
cfgPath = "cfg/yolov3_tiny_crownLoss.txt"
savePath = "C:/Users/boony/Desktop/dta/"
with open("classes/crownLossBins.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
net = DetectionModel(weightsPath, cfgPath, classes, COLORS, 0.0, 0.25)

# configure the youtube video links
greenAshVideo = "https://www.youtube.com/watch?v=0MxWB2VJ9Yo"
snowAndDeadTrees = "https://www.youtube.com/watch?v=2tsFS-8JS7w"
beechTreeTimelapseVideo = "https://www.youtube.com/watch?v=bnwiPETX7Jw"
FPVDrone = "https://www.youtube.com/watch?v=UH7LdwJ7ReY"
FPVDrone2 = "https://www.youtube.com/watch?v=xtWaVBaccTw"
FPVDrone3 = "https://www.youtube.com/watch?v=X8UyYegR4Lw" #shows that viewpoint changes affect cl prediction too
url = FPVDrone3

# controls the play speed of th video by skipping some frames (1 means usual speed ¬20 Fps)
speedUp = 4

video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)

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
    # im = frame

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.imshow("Image", im)
    count += 1
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
