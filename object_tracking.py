import cv2
from mobile_net import *
from deep_sort.person_id_model.generate_person_features import generate_detections, init_encoder
from deep_sort.deep_sort_app import run_deep_sort, DeepSORTConfig

cap = cv2.VideoCapture(cv2.samples.findFile("vtest.avi"))

model = ObjectRecognition()
encoder = init_encoder()
config = DeepSORTConfig()

while(True):
    ret, frame = cap.read()
    boxes = model.get_boxes(frame)

    if len(boxes) > 0:
        encoding = generate_detections(encoder, np.array(boxes), frame)
        run_deep_sort(frame, encoding, config)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()