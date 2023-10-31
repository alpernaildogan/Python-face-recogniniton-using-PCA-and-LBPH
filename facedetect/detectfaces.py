#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import os

# Specify the filename of the file in the same directory as your script
script_dir = os.path.dirname(os.path.abspath(__file__))
cascPath = os.path.join(script_dir, 'haarcascade_frontalface_alt.xml')  # Replace with the actual filename of your image

faceDetect = cv2.CascadeClassifier(cascPath)
cam = cv2.VideoCapture(0)

while True:
    (ret, img) = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        data = gray[y:y + h, x:x + w] ##original gray data
        r = 375.0 / img.shape[1]
        dim = (375, int(img.shape[0] * r))
        resized = cv2.resize(data, dim, interpolation=cv2.INTER_AREA) ##resized gray data
        hist1 =cv2.equalizeHist(resized)  ##resize and hist. eq.
        hist2 =cv2.equalizeHist(data)  ##only hist. eq.
        edges = cv2.Canny(resized,0,100) ##edges of the captured image
        cv2.imshow('received data', data)
        cv2.imshow('resized data', resized)
        cv2.imshow('Hist eq', hist2)
        cv2.imshow('Resized & Histogram', hist1)
        cv2.imshow('Edges', edges)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0xFF), 2)
        data = gray[y:y + h, x:x + w]

        cv2.imshow('Face', img)
        #print (data.shape)


    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()

			
