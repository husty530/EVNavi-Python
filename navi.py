import os
import numpy as np
import cv2
import math
import media, kalman, yolo, tcpSocket
from copy import copy

red = (0, 0, 180)
green = (0, 160, 0)
blue = (255, 0, 0)
white = 255
fill120 = (120, 120, 120)
fill150 = (150, 150, 110)
fill180 = (180, 180, 60)
fill210 = (210, 210, 20)

class LineDetector:

    def __init__(self, filterStrength=1, isMulch=True):

        self.first = True
        self.isMulch = isMulch
        self.mNoise = filterStrength
        self.pNoise = 1.0 / filterStrength
        self.lineSegmentL = math.pi, (0, 0), (0, 0)
        self.lineSegmentR = 0, (0, 0), (0, 0)
    
    def process(self, frame, points):

        pointsL = []
        pointsR = []
        for p in points:
            if p[0] < frame.shape[1] / 2 and p[1] > frame.shape[0] / 4 and p[1] < frame.shape[0] * 9 / 10:
                pointsL.append(p)
            elif p[0] > frame.shape[1] / 2 and p[1] > frame.shape[0] / 4 and p[1] < frame.shape[0] * 9 / 10:
                pointsR.append(p)
        if (len(pointsL) == 0 or len(pointsR) == 0): return 0
        pointsL.sort(key=lambda x: x[1])
        pointsL.append([0, pointsL[-1][1]])
        pointsL.append([0, pointsL[0][1]])
        pointsR.sort(key=lambda x: x[1])
        pointsR.append([frame.shape[1], pointsR[-1][1]])
        pointsR.append([frame.shape[1], pointsR[0][1]])

        hullL = cv2.convexHull(np.array(pointsL))
        hullR = cv2.convexHull(np.array(pointsR))
        canvas = np.zeros((frame.shape[0], frame.shape[1]), dtype = 'uint8')
        cv2.fillConvexPoly(canvas, hullL, white)
        cv2.fillConvexPoly(canvas, hullR, white)

        if self.first == True:
            for i in range(hullL.shape[0] - 3):
                x, y = hullL[i + 1][0][0] - hullL[i][0][0], hullL[i + 1][0][1] - hullL[i][0][1]
                theta = math.atan2(y, x)
                if theta > math.pi / 2 and theta < self.lineSegmentL[0]:
                    self.lineSegmentL = (theta, (hullL[i][0][0], hullL[i][0][1]), (hullL[i + 1][0][0], hullL[i + 1][0][1]))
            for i in range(2, hullR.shape[0] - 1):
                x, y = hullR[i + 1][0][0] - hullR[i][0][0], hullR[i + 1][0][1] - hullR[i][0][1]
                theta = math.atan2(y, x) + math.pi
                if theta < math.pi / 2 and theta > self.lineSegmentR[0]:
                    self.lineSegmentR = (theta, (hullR[i][0][0], hullR[i][0][1]), (hullR[i + 1][0][0], hullR[i + 1][0][1]))
        else:
            error = 10000
            preTheta = self.lineSegmentL[0]
            buf = 0, (0, 0), (0, 0)
            for i in range(hullL.shape[0] - 3):
                x, y = hullL[i + 1][0][0] - hullL[i][0][0], hullL[i + 1][0][1] - hullL[i][0][1]
                theta = math.atan2(y, x)
                e = math.pow(theta - preTheta, 2)
                if (e < error):
                    error = e
                    buf = theta, (hullL[i][0][0], hullL[i][0][1]), (hullL[i + 1][0][0], hullL[i + 1][0][1])
            if error < math.pi * 50 / 180: 
                self.lineSegmentL = buf
            error = 10000
            preTheta = self.lineSegmentR[0]
            buf = 0, (0, 0), (0, 0)
            for i in range(2, hullR.shape[0] - 1):
                x, y = hullR[i + 1][0][0] - hullR[i][0][0], hullR[i + 1][0][1] - hullR[i][0][1]
                theta = math.atan2(y, x) + math.pi
                e = math.pow(theta - preTheta, 2)
                if (e < error):
                    error = e
                    buf = theta, (hullR[i][0][0], hullR[i][0][1]), (hullR[i + 1][0][0], hullR[i + 1][0][1])
            if error < math.pi * 50 / 180: 
                self.lineSegmentR = buf
        
        # cv2.circle(frame, self.lineSegmentL[1], 3, blue, 4)
        # cv2.circle(frame, self.lineSegmentL[2], 3, blue, 4)
        # cv2.circle(frame, self.lineSegmentR[1], 3, blue, 4)
        # cv2.circle(frame, self.lineSegmentR[2], 3, blue, 4)
        
        if self.isMulch == True: topY = frame.shape[0] / 3
        else: topY = frame.shape[0] * 2 / 3
        bottomY = frame.shape[0]
        interceptL = self.lineSegmentL[1][1] - math.tan(self.lineSegmentL[0]) * self.lineSegmentL[1][0]
        interceptR = self.lineSegmentR[1][1] - math.tan(self.lineSegmentR[0]) * self.lineSegmentR[1][0]
        lt = (topY - interceptL) / math.tan(self.lineSegmentL[0]), topY
        rt = (topY - interceptR) / math.tan(self.lineSegmentR[0]), topY
        lb = (bottomY - interceptL) / math.tan(self.lineSegmentL[0]), bottomY
        rb = (bottomY - interceptR) / math.tan(self.lineSegmentR[0]), bottomY

        if self.first == True:
            self.filter = kalman.Kalman(np.array([lt[0], lb[0], rt[0], rb[0]]), self.mNoise, self.pNoise)
            self.first = False
        _, pdt = self.filter.update(np.array([lt[0], lb[0], rt[0], rb[0]]))
        lt = [pdt[0], lt[1]]
        lb = [pdt[1], lb[1]]
        rt = [pdt[2], rt[1]]
        rb = [pdt[3], rb[1]]
        
        centerX = int((lt[0] + lb[0] + rt[0] + rb[0]) / 4)
        centerY = int((lt[1] + lb[1] + rt[1] + rb[1]) / 4)
        center = centerX, centerY
        cv2.circle(frame, center, 6, red, 12)
        cv2.line(frame, (int(frame.shape[1] / 2), 0), (int(frame.shape[1] / 2), frame.shape[0]), red)
        cv2.line(frame, center, (int(frame.shape[1] / 2), centerY), red)

        hull = cv2.convexHull(np.array([lt, lb, rb, rt], dtype='int32'))
        canvas = np.zeros((frame.shape[0], frame.shape[1]), dtype='uint8')
        cv2.fillConvexPoly(canvas, hull, white)
        tmp = [canvas, canvas, np.zeros((frame.shape[0], frame.shape[1]), dtype='uint8')]
        addImg = cv2.merge(tmp)
        cv2.addWeighted(frame, 0.8, addImg, 0.2, 0, frame)

        return centerX - frame.shape[1] / 2

class AreaDetector:

    def __init__(self, filterStrength=1, isMulch=True):

        self.first = True
        self.isMulch = isMulch
        self.mNoise = filterStrength
        self.pNoise = 1.0 / filterStrength

    def process(self, frame, points):

        pointsL = []
        pointsR = []
        for p in points:
            if p[0] < frame.shape[1] / 2 and p[1] > frame.shape[0] / 4 and frame.shape[0] * 9 / 10:
                pointsL.append(p)
                cv2.circle(frame, (p[0], p[1]), 3, green, 4)
            elif p[0] > frame.shape[1] / 2 and p[1] > frame.shape[0] / 4 and frame.shape[0] * 9 / 10:
                pointsR.append(p)
                cv2.circle(frame, (p[0], p[1]), 3, green, 4)
        if len(pointsL) == 0 or len(pointsR) == 0: return 0

        meanL, eigenVecL, eigenValL = cv2.PCACompute2(np.array(pointsL, dtype=np.float32), np.array([], dtype=np.float32))
        meanR, eigenVecR, eigenValR = cv2.PCACompute2(np.array(pointsR, dtype=np.float32), np.array([], dtype=np.float32))
        centerL, centerR = (meanL[0][0], meanL[0][1]), (meanR[0][0], meanR[0][1])
        valL1, valL2 = math.sqrt(eigenValL[0][0]), math.sqrt(eigenValL[1][0])
        valR1, valR2 = math.sqrt(eigenValR[0][0]), math.sqrt(eigenValR[1][0])
        angleL, angleR = math.atan2(eigenVecL[0][1], eigenVecL[0][0]) * 180 / math.pi, math.atan2(eigenVecR[0][1], eigenVecR[0][0]) * 180 / math.pi
        if angleL < 0: angleL = angleL + 180
        if angleR < 0: angleR = angleR + 180
        if math.isnan(valL2): valL2 = 0
        if math.isnan(valR2): valR2 = 0

        kalmanInputL = np.array([centerL[0], centerL[1], valL1, valL2, angleL])
        kalmanInputR = np.array([centerR[0], centerR[1], valR1, valR2, angleR])
        if self.first == True:
            self.filterL = kalman.Kalman(kalmanInputL, self.mNoise, self.pNoise)
            self.filterR = kalman.Kalman(kalmanInputR, self.mNoise, self.pNoise)
            self.first = False
        _, pdtL = self.filterL.update(kalmanInputL)
        _, pdtR = self.filterR.update(kalmanInputR)
        centerL, centerR = (int(pdtL[0]), int(pdtL[1])), (int(pdtR[0]), int(pdtR[1]))
        valL1, valL2, angleL = pdtL[2], pdtL[3], pdtL[4]
        valR1, valR2, angleR = pdtR[2], pdtR[3], pdtR[4]

        endL1 = int(centerL[0] + math.cos((angleL - 180) * math.pi / 180) * valL1), int(centerL[1] + math.sin((angleL - 180) * math.pi / 180) * valL1)
        endL2 = int(centerL[0] + math.cos((angleL - 90) * math.pi / 180) * valL2), int(centerL[1] + math.sin((angleL - 90) * math.pi / 180) * valL2)
        endR1 = int(centerR[0] + math.cos((angleR + 180) * math.pi / 180) * valR1), int(centerR[1] + math.sin((angleR + 180) * math.pi / 180) * valR1)
        endR2 = int(centerR[0] + math.cos((angleR + 90) * math.pi / 180) * valR2), int(centerR[1] + math.sin((angleR + 90) * math.pi / 180) * valR2)
        cv2.arrowedLine(frame, centerL, endL1, red, 2)
        cv2.arrowedLine(frame, centerL, endL2, red, 2)
        cv2.arrowedLine(frame, centerR, endR1, red, 2)
        cv2.arrowedLine(frame, centerR, endR2, red, 2)

        addImg = copy(frame)
        cv2.ellipse(addImg, (centerL, (valL1 * 2, valL2 * 2), angleL), fill120, cv2.FILLED)
        cv2.ellipse(addImg, (centerL, (valL1 * 1.5, valL2 * 1.5), angleL), fill150, cv2.FILLED)
        cv2.ellipse(addImg, (centerL, (valL1, valL2), angleL), fill180, cv2.FILLED)
        cv2.ellipse(addImg, (centerL, (valL1 * 0.5, valL2 * 0.5), angleL), fill210, cv2.FILLED)
        cv2.ellipse(addImg, (centerR, (valR1 * 2, valR2 * 2), angleR), fill120, cv2.FILLED)
        cv2.ellipse(addImg, (centerR, (valR1 * 1.5, valR2 * 1.5), angleR), fill150, cv2.FILLED)
        cv2.ellipse(addImg, (centerR, (valR1, valR2), angleR), fill180, cv2.FILLED)
        cv2.ellipse(addImg, (centerR, (valR1 * 0.5, valR2 * 0.5), angleR), fill210, cv2.FILLED)

        if valL2 < frame.shape[1] / 15: valL2 = frame.shape[1] / 15
        if valR2 < frame.shape[1] / 15: valR2 = frame.shape[1] / 15
        weightL, weightR = 1.0 / valL2, 1.0 / valR2
        x = int((centerL[0] * weightL + centerR[0] * weightR) / (weightL + weightR))
        y = int((centerL[1] * weightL + centerR[1] * weightR) / (weightL + weightR))
        poly = np.array([(x, y), (frame.shape[1] / 3, frame.shape[0] * 7 / 8), (frame.shape[1] * 2 / 3, frame.shape[0] * 7 / 8)], dtype='int32')
        cv2.fillConvexPoly(frame, poly, red)

        cv2.addWeighted(frame, 0.6, addImg, 0.4, 0, frame)
        cv2.line(frame, (int(frame.shape[1] / 2), 0), (int(frame.shape[1] / 2), int(frame.shape[0])), red)
        cv2.line(frame, (x, y), (int(frame.shape[1] / 2), y), red, 2)

        return x - frame.shape[1] / 2

if __name__ == '__main__':

    #cap = media.Capture(0, (1024, 576))
    cap = media.Capture("C:\\Users\\yamataku1998\\Desktop\\tsuru\\Tsuru_test.mp4", (1024, 576))
    cfg = 'tsuru_tiny\\tsuru.cfg'
    names = 'tsuru_tiny\\tsuru.names'
    weights = 'tsuru_tiny\\tsuru.weights'
    # cap = media.Capture("C:\\Users\\yamataku1998\\Desktop\\mulch\\mulch.mp4", (1024, 576))
    # cfg = 'mulch_tiny\\mulch_tiny.cfg'
    # names = 'mulch_tiny\mulch.names'
    # weights = 'mulch_tiny\\mulch_tiny.weights'
    detector = yolo.Yolo(cfg, names, weights, (640, 480), yolo.Drawmode.point, 0.25, 0.1)
    navi = LineDetector(7, False)
    #navi = AreaDetector(7, False)

    sock = tcpSocket.TcpServer()
    while cap.isOpened:
        rcv = sock.receive()
        if rcv == 'request\n':
            frame = cap.read()
            results = detector.run(frame)
            error = navi.process(frame, results.centers)
            sock.send(error)
            cv2.imshow(" ", frame)
            cv2.waitKey(1)
            print("Success")
        elif rcv == 'finish\n':
            print("Acccept Finish")
            break
        else:
            print("Message must be 'request' or 'finish'.")
            break
    sock.close()
    cap.close()


# if __name__ == '__main__':

#     #cap = media.Capture(0, (1024, 576))
#     cap = media.Capture("C:\\Users\\yamataku1998\\Desktop\\mulch\\mulch.mp4", (1024, 576))
#     cfg = 'mulch_tiny\\mulch_tiny.cfg'
#     names = 'mulch_tiny\mulch.names'
#     weights = 'mulch_tiny\\mulch_tiny.weights'
#     detector = yolo.Yolo(cfg, names, weights, (640, 480), yolo.Drawmode.off, 0.25, 0.1)
#     #navi = LineDetector(7)
#     navi = AreaDetector(7)

#     while cap.isOpened:
#         frame = cap.read()
#         results = detector.run(frame)
#         points = []
#         for i in range(results.count):
#             if results.labels[i] == "border":
#                 points.append(results.centers[i])
#         error = navi.process(frame, points)
#         cv2.imshow(" ", frame)
#         cv2.waitKey(1)
    
#     cap.close()
#     cv2.destroyAllWindows()