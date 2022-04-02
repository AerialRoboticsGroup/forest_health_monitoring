import time
from objectDetectionModel import DetectionModel
import pafy
import cv2

# configure the settings
W, H = 512, 512
weightsPath = "weights/leaf.weights"
cfgPath = "cfg/yolov3_leaf-tiny.cfg.txt"
savePath = "C:/Users/boony/Desktop/tmpFrames/vid2/"
with open("classes/leaves.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
net = DetectionModel(weightsPath, cfgPath, classes, 0.0, 0.25)

# configure the youtube video links
vid1 = "https://www.youtube.com/watch?v=kB1FezMffx8"
vid2 = "https://www.youtube.com/watch?v=8sI7hgFZ-4g"
vid3 = "https://www.youtube.com/watch?v=na9XeH2Cicg"
url = vid3

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
    im = net.drawDetectionBB(boxes, confidences, class_ids, indexes, frame, False)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(im, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
    cv2.imshow("Image", im)

    count += 1
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
