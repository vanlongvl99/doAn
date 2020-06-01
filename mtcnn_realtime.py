import cv2
import os
import time
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        result = detector.detect_faces(frame)
        for person in result:
          bounding_box = person['box']
          keypoints = person['keypoints']
          cv2.rectangle(frame,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    else:
        break

cap.release()
cv2.destroyAllWindows()