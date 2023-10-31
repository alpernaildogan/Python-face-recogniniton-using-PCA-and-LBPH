#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import os

if __name__ == "__main__":
    
    # Specify the filename of the file in the same directory as your script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cascPath = os.path.join(script_dir, 'haarcascade_frontalface_alt.xml') #Cascade path for face detection
    faceDetect = cv2.CascadeClassifier(cascPath)
    cam = cv2.VideoCapture(0)

    # Create dataset folders if they don't already exist
    folder_names = ['dataset/Original', 'dataset/Resized', 'dataset/Resizedandhist','dataset/Hist','dataset/Edge','dataset/ResizedandSmall','dataset/ResizedSmallHist']
    for folder_name in folder_names:
        folder_path = os.path.join(script_dir, folder_name)
        if not os.path.exists(folder_path):
            try:
                os.mkdir(folder_path)
            except Exception as e:
                print(f'Error creating folder {folder_path}: {str(e)}')


    id = input('user id:')
    maxsamplenumber = input('How many samples (higher the better):')
    maxsamplenumber = int(maxsamplenumber)
    samplenum = 0

    while True:
        cv2.waitKey(200)
        (ret, img) = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            samplenum = samplenum + 1
            data = gray[y:y + h, x:x + w] ##original gray data
            r = 400.0 / img.shape[1]
            dim = (400, int(img.shape[0] * r))
            resized = cv2.resize(data, dim, interpolation=cv2.INTER_AREA) ##resized gray data
            r = 120.0 / img.shape[1]
            dim = (120, int(img.shape[0] * r))
            resized1 = cv2.resize(data, dim, interpolation=cv2.INTER_AREA)
            smallhist = cv2.equalizeHist(resized1)
            hist1 =cv2.equalizeHist(resized)  ##resize and hist. eq.
            hist2 =cv2.equalizeHist(data)  ##only hist. eq.
            edges = cv2.Canny(resized,40,40)   ######## threshold
            cv2.imshow('Received data', data)
            cv2.imshow('Resized data', resized)
            cv2.imshow('Hist equalization', hist2)
            cv2.imshow('Resized & Histogram eq', hist1)
            cv2.imshow('Edges', edges)
            cv2.imshow('Resized Small',resized1)
            cv2.imshow('Resized Small & Hist eq',smallhist)
            print (samplenum ,'pictures taken')
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(script_dir, 'dataset/Original/User.') + str(id) + '.' + str(samplenum)+ '.jpg', data)
            cv2.imwrite(os.path.join(script_dir, 'dataset/Resized/User.') + str(id) + '.' + str(samplenum)+ '.jpg', resized)
            cv2.imwrite(os.path.join(script_dir, 'dataset/Resizedandhist/User.') + str(id) + '.' + str(samplenum)+ '.jpg', hist1)
            cv2.imwrite(os.path.join(script_dir, 'dataset/Hist/User.') + str(id) + '.' + str(samplenum)+ '.jpg', hist2)
            cv2.imwrite(os.path.join(script_dir, 'dataset/Edge/User.') + str(id) + '.' + str(samplenum)+ '.jpg', edges)
            cv2.imwrite(os.path.join(script_dir, 'dataset/ResizedandSmall/User.') + str(id) + '.' + str(samplenum)+ '.jpg', resized1)
            cv2.imwrite(os.path.join(script_dir, 'dataset/ResizedSmallHist/User.') + str(id) + '.' + str(samplenum)+ '.jpg', smallhist)
        cv2.imshow('Face', img) 
        cv2.waitKey(1);
        if samplenum >= maxsamplenumber:
            break
    cam.release()
    cv2.destroyAllWindows()
                
