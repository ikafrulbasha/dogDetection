import cv2
import platform
from ultralytics import YOLO
import numpy as np
import torch

if __name__ == '__main__':
    print("Starting object detection for DOGS leveraging " + str(platform.machine()) + " architecture")
    print("MPS enabled: " + str(torch.backends.mps.is_available()))
    numToClass = {}
    numToClass["16"] = "Dog"
    cap = cv2.VideoCapture("dogs.mp4")
    model = YOLO("yolov8m.pt")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, device="mps") #leverage cpu, graphic cards, etc
        result = results[0]
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        for cls, bbox in zip(classes, bboxes):
            (x,y,x2,y2) = bbox
            cv2.rectangle(frame,(x,y),(x2,y2),(0,0,255),4)
            classification = "???"
            if str(cls) in numToClass:
                classification = numToClass[str(cls)]
            cv2.putText(frame, classification, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 4)
            print("x",x,"y",y)

        cv2.imshow("Img", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


