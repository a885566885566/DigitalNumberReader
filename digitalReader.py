import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def tupleToInt(tu):
    return (int(tu[0]), int(tu[1]))

class DigiNumber:
    def __init__(self, location, numberSize, criteria=40):
        self.numberSize = numberSize
        self.location = location
        self.cubeSize = (numberSize[1]//6, 2* numberSize[1]//6)
        self.criteria = criteria
        self.binCode = ""
        self.number = ""
        xaxis = np.linspace(location[0], location[0]+numberSize[0], 3)
        yaxis = np.linspace(location[1], location[1]+numberSize[1], 5)
        self.roiPoints = np.array([
            #  X_axis,   Y_axis, range, sigle
            [xaxis[1], yaxis[0], 0, 0, 1], # A
            [xaxis[2], yaxis[1], 0, 0, 0], # B
            [xaxis[2], yaxis[3], 0, 0, 0], # C
            [xaxis[1], yaxis[4], 0, 0, 1], # D
            [xaxis[0], yaxis[3], 0, 0, 0], # E
            [xaxis[0], yaxis[1], 0, 0, 0], # F
            [xaxis[1], yaxis[2], 0, 0, 1] # G
        ], dtype=np.int16)
    
    def getNum(self, var):
        return {
        "1111110" : 0,
        "0110000" : 1,
        "1101101" : 2,
        "1111001" : 3,
        "0110011" : 4,
        "1011011" : 5,
        "0011111" : 6,
        "1110000" : 7,
        "1111111" : 8,
        "1111011" : 9
        }.get(var,'error')
    
    def getBound(self, roi):
        if roi[4] == 1:
            startPoint = (roi[0]-self.cubeSize[0]//2, roi[1]-self.cubeSize[1]//2)
            endPoint = (roi[0]+self.cubeSize[0]//2,   roi[1]+self.cubeSize[1]//2)
        else:
            startPoint = (roi[0]-self.cubeSize[1]//2, roi[1]-self.cubeSize[0]//2)
            endPoint = (roi[0]+self.cubeSize[1]//2,   roi[1]+self.cubeSize[0]//2)
        return startPoint, endPoint
            
    def detect(self, img):
        self.binCode = ""
        for idx, roi in enumerate(self.roiPoints):
            startPoint, endPoint = self.getBound(roi)
            roi_img = img[startPoint[1]:endPoint[1], startPoint[0]:endPoint[0]]
            nmax = np.max(roi_img)
            nmin = np.min(roi_img)
            self.roiPoints[idx, 2] = nmax - nmin
            self.roiPoints[idx, 3] = 0 if (nmax - nmin < self.criteria) else 1
            self.binCode += str(self.roiPoints[idx, 3])
        self.number = self.getNum(self.binCode)
            
    def draw(self, img):
        endPoint = (self.location[0]+self.numberSize[0], self.location[1]+self.numberSize[1])
        cv2.rectangle(img, self.location, endPoint, (0, 255, 255), 2)
        count = 0
        for roi in self.roiPoints:
            startPoint, endPoint = self.getBound(roi)
            color = (0, 0 if roi[3]>0.5 else 255, 0)
            cv2.rectangle(img, startPoint, endPoint, color, 2)
            cv2.putText(img, str(roi[2]), (self.location[0], self.location[1]+130+count), cv2.FONT_HERSHEY_SIMPLEX,
              0.5, (0, 255, 255), 1, cv2.LINE_AA)
            count += 15
            #print(roi[2])
        cv2.putText(img, self.binCode, (self.location[0], self.location[1]-30), cv2.FONT_HERSHEY_SIMPLEX,
              0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(self.number), (self.location[0], self.location[1]-60), cv2.FONT_HERSHEY_SIMPLEX,
              0.5, (0, 255, 255), 1, cv2.LINE_AA)
            

if __name__ == '__main__':
    NUM_DIGITS = 4
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    
    digiNums = []
    digitStart = (100, 100)
    digitOffset = 80
    for i in range(NUM_DIGITS):
        location = (digitStart[0] + digitOffset*i, digitStart[1])
        digiNums.append(DigiNumber(location, (45, 95)))
        
    while(True):
        ret, frame = cap.read()
        
        (h, w) = frame.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, 180, 1)
        frame = cv2.warpAffine(frame, M, (w, h))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i in range(NUM_DIGITS):
            digiNums[i].detect(gray)
            digiNums[i].draw(frame)
        cv2.imshow('My Image', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    print("Camera closed")

    cap.release()
    cv2.destroyAllWindows()