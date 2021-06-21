import cv2
import mediapipe as mp
import time
import imutils



class HandTracker():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.detectionCon, self.trackingCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIDs = [4, 8, 12, 16, 20]


    def trackHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = imutils.resize(img, width=1280)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLM in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLM, self.mpHands.HAND_CONNECTIONS)
        return img



    def lmPositionFinder(self, img, draw=True):
        self.lmPlist = []

        if self.results.multi_hand_landmarks:

          for handLM in self.results.multi_hand_landmarks:
            for id, lm in enumerate(handLM.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                jls_extract_var = cv2
                if draw:
                    jls_extract_var.circle(img, (cx, cy), 20, (75, 75, 75), cv2.FILLED)
                self.lmPlist.append([id,cx,cy])          
        return self.lmPlist


    def fingerUpDown(self):
        finger = []

        #Tip of finger
        if self.lmPlist[self.tipIDs[0]][1] < self.lmPlist[self.tipIDs[0] - 1][1]:
            finger.append(1)
        else:
            finger.append(0)

        for id in range(1, 5):
            if self.lmPlist[self.tipIDs[id]][2] < self.lmPlist[self.tipIDs[id] - 2][2]:
                finger.append(1)
            else:
                finger.append(0)
        return finger


            


