import cv2
import os
import time
import HandTrackingModule as htm

#header image
folderPath = "imgs"
imgList = os.listdir(folderPath)
overlayList = []

for img in imgList:
    image = cv2.imread(f"{folderPath}/{img}")
    overlayList.append(image)


#videoCaptture  
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)


detector = htm.HandTracker()
pTime = 0
tipIDs = [4, 8, 12, 16, 20]
while True:
    succcess, img = cap.read()
    img = cv2.flip(img, 1)

    #FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
    
    #hand detect
    img = detector.trackHands(img)
    lmList = detector.lmPositionFinder(img, draw=False)
    #print(lmList)
    if len(lmList) != 0:    
        fingers = []
        if lmList[tipIDs[0]][1] < lmList[tipIDs[0]- 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):         
          if lmList[tipIDs[id]][2] < lmList[tipIDs[id]- 1][2]:
            fingers.append(1)
          else:
            fingers.append(0)

     
         
         #counting Finger
        fingerCount = fingers.count(1)
        #print(fingerCount)


        img[10:160, 10:160] = overlayList[fingerCount]
        cv2.rectangle(img, (10, 215),(160, 415), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(fingerCount), (40, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
            
    #Output
    cv2.imshow("Image", img)
    
    #quit control
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
