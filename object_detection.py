import cv2
from mobile_net import *

cap = cv2.VideoCapture(cv2.samples.findFile("vtest.avi"))

model = ObjectRecognition()

_, first_frame = cap.read()
model.run_object_recognition(first_frame)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame_with_boxes = model.run_object_recognition(frame)

    cv2.imshow('frame',frame_with_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()