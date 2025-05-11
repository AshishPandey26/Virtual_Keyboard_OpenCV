import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)

while True:
    success, img = cap.read()
    
    # Detect hands and draw landmarks
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        bboxInfo = hand["bbox"]

        print("Index finger tip:", lmList[8])

    cv2.imshow("Image", img)
    cv2.waitKey(1)