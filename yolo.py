import cv2
import enum
import numpy as np

class YoloResults:
    
    labels = []
    confidences = []
    centers = []
    sizes = []
    count = 0

    def add(label, confidence, center, size):
        YoloResults.labels.append(label)
        YoloResults.confidences.append(confidence)
        YoloResults.centers.append(center)
        YoloResults.sizes.append(size)
        YoloResults.count = YoloResults.count + 1

    def clear():
        YoloResults.labels.clear()
        YoloResults.confidences.clear()
        YoloResults.centers.clear()
        YoloResults.sizes.clear()
        YoloResults.count = 0

class Drawmode(enum.Enum):

    rectangle = 0
    point = 1
    off = 2

class Yolo:

    def __init__(self, cfg, names, weights, blobsize, drawmode, conf_thresh, nms_thresh):
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.blobsize = blobsize
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.drawmode = drawmode
        self.classes = open(names).read().strip().split('\n')
        np.random.seed(1)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')
        YoloResults.clear()

    def run(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, self.blobsize, 0, True, False)
        self.net.setInput(blob)
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outs = self.net.forward(ln)
        centers = []
        boxes = []
        confidences = []
        classIDs = []
        h, w = frame.shape[:2]

        for output in outs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = detection[4]
                if confidence > self.conf_thresh:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    center = [centerX, centerY]
                    centers.append(center)
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        
        YoloResults.clear()
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                YoloResults.add(self.classes[classIDs[i]], confidences[i], centers[i], boxes[i])
                if self.drawmode == Drawmode.rectangle:
                    color = [int(c) for c in self.colors[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}:{:.1f}%".format(self.classes[classIDs[i]], confidences[i] * 100)
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                elif self.drawmode == Drawmode.point:
                    color = [int(c) for c in self.colors[classIDs[i]]]
                    cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 3, color, 4)
        return YoloResults

# if __name__ == '__main__':
#     detector = Yolo("C:\\Users\\yamataku1998\\Desktop\\mulch\\mulch.cfg", "C:\\Users\\yamataku1998\\Desktop\\mulch\\mulch.names", "C:\\Users\\yamataku1998\\Desktop\\mulch\\mulch.weights", (640, 480), Drawmode.rectangle, 0.5, 0.3)
#     img = cv2.imread("C:\\Users\\yamataku1998\\Desktop\\mulch\\IMG_20200915_143816.jpg")
#     img = cv2.resize(img, (640, 480))
#     results = detector.run(img)
#     for i in range(results.count):
#         label = results.labels[i]
#         confidence = "{:.1f}%".format(results.confidences[i] * 100)
#         print(label, confidence)
#     cv2.imshow(" ", img)
#     cv2.waitKey()