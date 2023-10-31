#!/usr/bin/python
# -*- coding: utf-8 -*-
import os  # for take pics from dataset
import cv2  # as usual
import numpy as np  # as usual
from PIL import Image  # new lib, same purpose as os lib
from pca_library import asColumnMatrix
from pca_library import pca

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    radius = 1  # default 1
    # The radius used for building the Circular Local Binary Pattern.
    # The greater the radius, the smoother the image but more spatial information you can get.

    neighbors = 8  # default 8
    # The number of sample points to build a Circular Local Binary Pattern from.
    # An appropriate value is to use 8 sample points.
    # Keep in mind: the more sample points you include, the higher the computational cost.

    grid_x = 8  # default 8
    # The number of cells in the horizontal direction, 8 is a common value used in publications.
    # The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector.

    grid_y = 8  # default 8
    # The number of cells in the vertical direction, 8 is a common value used in publications
    # The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector.

    ##threshold = DBL_MAX
    # The threshold applied in the prediction
    # If the distance to the nearest neighbor is larger than the threshold, this method returns -1.


    rec = cv2.face.LBPHFaceRecognizer_create(radius, neighbors, grid_x,grid_y) #recognizer for LBPH


    def get_image(path):
        imagepaths = [os.path.join(path, f) for f in os.listdir(path)]

        faces = []
        IDs = []
        for imagepath in imagepaths:
            faceimg = Image.open(imagepath).convert('L')
            facenp = np.array(faceimg, 'uint8')
            ID = int(os.path.split(imagepath)[-1].split('.')[1])
            faces.append(facenp)

            IDs.append(ID)
            cv2.imshow('training', facenp)
            cv2.waitKey(10)
        return (IDs, faces)


    print ('Step 1/3  (Original Data)')
    path1 = os.path.join(script_dir, 'dataset/Original')
    (Ids, faces) = get_image(path1)
    rec.train(faces, np.array(Ids))
    print ('Saving LBPH train data using raw face images')
    rec.write(os.path.join(script_dir, 'recognizer/originallbp.yml'))


    print ('Step 2/3  (Resized Data)')
    path2 = os.path.join(script_dir, 'dataset/Resized')
    (Ids, faces) = get_image(path2)
    rec.train(faces, np.array(Ids))
    print ('Saving LBPH train data using resized face images')
    rec.write(os.path.join(script_dir, 'recognizer/resizedlbp.yml'))

    aj = asColumnMatrix(np.array(faces))
    ( WW, mu, yj) = pca(aj, Ids)
    print ('Saving PCA train data using resized face images')
    np.savetxt(os.path.join(script_dir, 'recognizer/IDsreziedpca.yml'),Ids, newline='\n', delimiter = ',')
    np.savetxt(os.path.join(script_dir, 'recognizer/WWreziedpca.yml'),WW, newline='\n', delimiter = ',')
    np.savetxt(os.path.join(script_dir, 'recognizer/yjreziedpca.yml'),yj, newline='\n', delimiter = ',')
    np.savetxt(os.path.join(script_dir, 'recognizer/mureziedpca.yml'),mu, newline='\n', delimiter = ',')


    print ('Step 3/3  (Resize and Histogram Equization Process)')
    path3 = os.path.join(script_dir, 'dataset/Resizedandhist')
    (Ids, faces) = get_image(path3)
    rec.train(faces, np.array(Ids))
    print ('Saving LBPH train data using resized face images with histogram equalization')
    rec.write(os.path.join(script_dir, 'recognizer/resizeandhistlbp.yml'))

    aj = asColumnMatrix(np.array(faces))
    ( WW, mu, yj) = pca(aj, Ids)
    print ('Saving PCA train data using resized face images with histogram equalization')
    np.savetxt(os.path.join(script_dir, 'recognizer/IDsreziedandhistpca.yml'),Ids, newline='\n', delimiter = ',')
    np.savetxt(os.path.join(script_dir, 'recognizer/WWreziedandhistpca.yml'),WW, newline='\n', delimiter = ',')
    np.savetxt(os.path.join(script_dir, 'recognizer/yjreziedandhistpca.yml'),yj, newline='\n', delimiter = ',')
    np.savetxt(os.path.join(script_dir, 'recognizer/mureziedandhistpca.yml'),mu, newline='\n', delimiter = ',')

    ## Skip other steps. they were only used to examine the effect of different image processing on the result.

    # print ('Step 4/7 (Histogram Equization Process)')
    # path4 = os.path.join(script_dir, 'dataset/Hist')
    # (Ids, faces) = get_image(path4)
    # rec.train(faces, np.array(Ids))
    # print ('Saving LBPH train data')
    # rec.write(os.path.join(script_dir, 'recognizer/histlbp.yml'))


    # print ('Step 5/7  (Edge Maps)')
    # path5 = os.path.join(script_dir, 'dataset/Edge')
    # (Ids, faces) = get_image(path5)
    # rec.train(faces, np.array(Ids))
    # print ('Saving LBPH train data')
    # rec.write(os.path.join(script_dir, 'recognizer/edgelbp.yml'))


    # print ('Step 6/7  (Minimized data)')
    # path6 = os.path.join(script_dir, 'dataset/ResizedandSmall')
    # (Ids, faces) = get_image(path6)
    # rec.train(faces, np.array(Ids))
    # print ('Saving LBPH train data')
    # rec.write(os.path.join(script_dir, 'recognizer/resizedsmalllbp.yml'))

    # aj = asColumnMatrix(np.array(faces))
    # ( WW, mu, yj) = pca(aj, Ids)
    # print ('Saving PCA train data')
    # np.savetxt(os.path.join(script_dir, 'recognizer/IDsreziedsmallpca.yml'),Ids, newline='\n', delimiter = ',')
    # np.savetxt(os.path.join(script_dir, 'recognizer/WWreziedsmallpca.yml'),WW, newline='\n', delimiter = ',')
    # np.savetxt(os.path.join(script_dir, 'recognizer/yjreziedsmallpca.yml'),yj, newline='\n', delimiter = ',')
    # np.savetxt(os.path.join(script_dir, 'recognizer/mureziedsmallpca.yml'),mu, newline='\n', delimiter = ',')


    # print ('Step 7/7  (Minimized data Hist Eq.)')
    # path7 = os.path.join(script_dir, 'dataset/ResizedSmallHist')
    # (Ids, faces) = get_image(path7)
    # rec.train(faces, np.array(Ids))
    # print ('Saving LBPH train data')
    # rec.write(os.path.join(script_dir, 'recognizer/resizedsmallhistlbp.yml'))

    # aj = asColumnMatrix(np.array(faces))
    # ( WW, mu, yj) = pca(aj, Ids)
    # print ('Saving PCA train data')
    # np.savetxt(os.path.join(script_dir, 'recognizer/IDsreziedsmallhistpca.yml'),Ids, newline='\n', delimiter = ',')
    # np.savetxt(os.path.join(script_dir, 'recognizer/WWreziedsmallhistpca.yml'),WW, newline='\n', delimiter = ',')
    # np.savetxt(os.path.join(script_dir, 'recognizer/yjreziedsmallhistpca.yml'),yj, newline='\n', delimiter = ',')
    # np.savetxt(os.path.join(script_dir, 'recognizer/mureziedsmallhistpca.yml'),mu, newline='\n', delimiter = ',')


    print ('Done!!')
    cv2.destroyAllWindows()
                
