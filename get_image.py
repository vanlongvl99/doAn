import cv2
import os
import time
cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


path = "emotion_faces"
cnt = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cnt += 1
        cv2.imshow("frame", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.imwrite(path + "/" + str(cnt) + ".jpg",frame) 
        if cv2.waitKey(0) & 0xFF == ord('z'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()