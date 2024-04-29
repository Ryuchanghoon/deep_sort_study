import torch
import cv2
import numpy as np


model = torch.load('best.pt').eval()
model.conf = 0.25  
model.iou = 0.45



with open('test_custom_dataset/data.yaml', 'r') as f:
    import yaml
    data = yaml.load(f, Loader=yaml.FullLoader)
    classes = data['names']



cap = cv2.VideoCapture('###############')


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)


    for *xyxy, conf, cls in results.xyxy[0]:
        label = classes[int(cls)] 
        x1, y1, x2, y2 = map(int, xyxy)
        color = [int(c) for c in np.random.uniform(0, 255, 3)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Image', frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break


cap.release()
cv2.destroyAllWindows()