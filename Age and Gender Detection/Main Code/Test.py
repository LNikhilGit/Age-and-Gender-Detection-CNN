import tensorflow as tf
import cv2
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

Model = tf.keras.models.load_model('./Model.h5')

def projimg(image,Model):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (48, 48))

  pred_1=Model.predict(np.array([image]))

  sex_f = ['Female', 'Male']
  age=int(np.round(pred_1[1][0]))
  sex=int(np.round(pred_1[0][0]))

  return "Gender: "+sex_f[sex]+", Age: "+str(age)[0:2]


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]
    label=projimg(frame,Model)
   
    cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(gray, 1.2, 3,minSize=(50, 50))

    if len(rects) > 0:
        # Draw a rectangle around the faces
        for (x, y, w, h) in rects:
            draw_border(frame, (x, y), (x + w, y + h), (255, 0, 105),4, 15, 10)
        
        cv2.putText(frame, str(label), (100,height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, str("No Face Detected"), (100,height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)

    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()