#!/usr/bin/python    
# -*- coding: utf-8 -*- 

import cv2    ##opencv lib version 3.3.0
import numpy as np ##numpy lib version 1.10 (??)
from pca_library import predict
import os



def most_common(lst):
    return max(set(lst), key=lst.count)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print ('ID maps are loading...')
    IDresizedandhist= np.loadtxt(os.path.join(script_dir, 'recognizer/IDsreziedandhistpca.yml'),delimiter=',')
    IDsreziedpca= np.loadtxt(os.path.join(script_dir, 'recognizer/IDsreziedpca.yml'),delimiter=',')

    print ('Mean values are loading...')
    mureziedandhistpca= np.loadtxt(os.path.join(script_dir, 'recognizer/mureziedandhistpca.yml'),delimiter=',')
    mureziedpca= np.loadtxt(os.path.join(script_dir, 'recognizer/mureziedpca.yml'),delimiter=',')

    print ('Eigen vectors are loading...')
    WWreziedandhistpca= np.loadtxt(os.path.join(script_dir, 'recognizer/WWreziedandhistpca.yml'),delimiter=',')
    WWreziedpca= np.loadtxt(os.path.join(script_dir, 'recognizer/WWreziedpca.yml'),delimiter=',')


    print ('Feature vectors are loading... ')
    yjreziedandhistpca= np.loadtxt(os.path.join(script_dir, 'recognizer/yjreziedandhistpca.yml'),delimiter=',')
    yjreziedpca= np.loadtxt(os.path.join(script_dir, 'recognizer/yjreziedpca.yml'),delimiter=',')



    ##LBPH Threshold
    threshold = 50 

    rec1 = cv2.face.LBPHFaceRecognizer_create(1, 8, 8,8)
    rec2 = cv2.face.LBPHFaceRecognizer_create(1, 8, 8,8)
    rec3 = cv2.face.LBPHFaceRecognizer_create(1, 8, 8,8)

    trainingfile1 =os.path.join(script_dir, 'recognizer/originallbp.yml')
    trainingfile2 =os.path.join(script_dir, 'recognizer/resizedlbp.yml')
    trainingfile3 =os.path.join(script_dir, 'recognizer/resizeandhistlbp.yml')

    rec1.read(trainingfile1)
    rec1.setThreshold(threshold)
    rec2.read(trainingfile2)
    rec2.setThreshold(threshold)
    rec3.read(trainingfile3)
    rec3.setThreshold(threshold)



    cascPath = os.path.join(script_dir, 'haarcascade_frontalface_alt.xml')  ##default cascade for detecting face
    faceDetect = cv2.CascadeClassifier(cascPath)  
    cam = cv2.VideoCapture(0)

    id = 0  ## default id
    id_1 = 0
    id_2 = 0
    id_3 = 0
    id_4 = 0
    id_5 = 0


    while True:

        (ret, img) = cam.read()   ##webcam opens
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  ##converts taken rgb frame to gray
        faces = faceDetect.detectMultiScale(gray, scaleFactor=1.1,  ## detection face scaling etc.
                minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:  
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0xFF), 2)
            
            data = gray[y:y + h, x:x + w] ##original gray data
            
            r = 400.0 / img.shape[1]
            dim = (400, int(img.shape[0] * r))
            resized = cv2.resize(data, dim, interpolation=cv2.INTER_AREA) ##resized gray data
            r0 = 120.0 / img.shape[1]
            dim0 = (120, int(img.shape[0] * r0))
            resized1 = cv2.resize(data, dim0, interpolation=cv2.INTER_AREA)
            smallhist = cv2.equalizeHist(resized1)
            
            hist1 =cv2.equalizeHist(resized)  ##resize and hist. eq.
            
            hist2 =cv2.equalizeHist(data)  ##only hist. eq.
            
            edges = cv2.Canny(resized,40,40) ######## threshold
            
            #cv2.imshow('receiveddata', data)  ##showing compared data
            #cv2.waitKey(150);   ###for slow motion or slow capture and hold q instead of just pushing pls
            


            (id_1,conf1) = rec1.predict(data)
            (id_2,conf2) = rec2.predict(resized)
            (id_3,conf3) = rec3.predict(hist1)
            id_4 = predict(IDresizedandhist ,WWreziedandhistpca ,hist1 ,yjreziedandhistpca , mureziedandhistpca)         
            id_5 = predict(IDsreziedpca ,WWreziedpca ,resized ,yjreziedpca , mureziedpca)


            
            lst =(id_1,id_2,id_3,id_4,id_5)
            id = most_common(lst)

    ###################### Result

            if id == -1:
                id = 'Unknown'
            if id == 0:
                id = 'Error'
            if id == 1:
                id = 'User 1'
            elif id == 2:
                id = 'User 2'
            elif id == 3:
                id = 'User 3'
            elif id == 4:
                id = 'User 4'
            elif id == 5:
                id = 'User 5'


            fontface = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = 1
            fontcolor = (0xFF, 0xFF, 0xFF)
            thickness = 4
            cv2.putText(img,str(id),(x, y + h),fontface,fontscale,fontcolor,thickness,)

        cv2.imshow('Detector', img) ##showing frame
        if cv2.waitKey(1) & 0xFF == ord('q'): ## Hold button q for quit
            break
        
    cam.release()
    cv2.destroyAllWindows()

                
