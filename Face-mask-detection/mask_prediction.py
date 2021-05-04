import cv2 as cv
import numpy as np
import tensorflow as tf

model=tf.keras.models.load_model('/home/umar/ML-AI/DL_Projects/Face-mask-detection/maskDetect_model')
dim=256

haar_cascade=cv.CascadeClassifier('/home/umar/ML-AI/haar_face.xml')
cap=cv.VideoCapture(0)

while True:
    _,frame = cap.read()
    frame=cv.resize(frame,(600,600))
    frame=cv.flip(frame,1)
    copy_frame=frame.copy()
    # print(frame.shape)

    face_rect=haar_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=3)
    for (x,y,w,h) in face_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,20,147),thickness=2) 
        copy_frame=frame[x:x+w,y:y+h]

    copy_frame=cv.resize(copy_frame,(dim,dim))
    copy_frame=np.asarray(copy_frame)
    copy_frame=copy_frame/255.0
    copy_frame=copy_frame.reshape(1,dim,dim,3)
    prediction=model.predict(copy_frame)
    # classes = np.argmax(prediction, axis = 1)
    print(prediction)
    if(prediction<=0.5):
        cv.putText(frame,'Mask',(30,30),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv.LINE_AA)
    else:
        cv.putText(frame,'No Mask',(30,30),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv.LINE_AA)

    # print(classes)
    # print(image.shape)  
    
    cv.imshow('Detected Mask',frame)
    
    key=cv.waitKey(1)
    if key==27:
        break

cap.release() 
cv.destroyAllWindows()