import cv2

class Capture:

    isOpened = False
    width = 0
    height = 0
    fps = 0
    frameCount = 0

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        Capture.isOpened = self.cap.isOpened()
        Capture.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        Capture.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        Capture.fps = self.cap.get(cv2.CAP_PROP_FPS)
        Capture.frameCount = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    def __init__(self, src, size):
        self.cap = cv2.VideoCapture(src)
        Capture.isOpened = self.cap.isOpened()
        Capture.width = size[0]
        Capture.height = size[1]
        Capture.fps = self.cap.get(cv2.CAP_PROP_FPS)
        Capture.frameCount = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    def read(self):
        _, frame = self.cap.read()
        frame = cv2.resize(frame, (Capture.width, Capture.height))
        return frame
    
    def close(self):
        self.cap.release()

class Writer:

    def __init__(self, path, fps, size):
        self.wrt = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID', fps, size))
    
    def write(self, frame):
        self.wrt.write(frame)

    def close(self):
        self.wrt.release()