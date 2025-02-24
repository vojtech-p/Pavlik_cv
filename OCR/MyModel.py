# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:03:17 2022

@author: xredin00
"""
import numpy as np
import cv2 as cv
import joblib

def feature_extract(image, num_blocks=18, num_blocks_vert=6):
    image_canny = cv.Canny(image, 50, 200, None, 3)
    features = []
    image_h = image.shape[0]
    image_w = image.shape[1]
    num_blocks_hor = num_blocks//num_blocks_vert
    block_h = image_h//num_blocks_vert
    block_w = image_w//num_blocks_hor

    for i in range(num_blocks_vert+1):
        for j in range(num_blocks_hor):
            block = image_canny[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            if block.any():
                features.append(np.uint32(np.mean(block)))
            else:
                features.append(np.uint32(0))
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    features.append(len(keypoints))
    features.append(image_w)
    features.append(np.mean(image_canny))

    # Compute Hu Moments
    moments = cv.moments(image_canny)
    hu_moments = cv.HuMoments(moments).flatten()
    # Log scale transform to bring the values closer to each other
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-6)
    for i in hu_moments:
        features.append(i)

    idx_features = [0, 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29]

    return [features[i] for i in idx_features]


def MyModel(data):

#     Funkce slouzi k implementaci nauceneho modelu. Vas model bude ulozen v samostatne promenne a se spustenim se aplikuje
#     na vstupni data. Tedy, model se nebude pri kazdem spousteni znovu ucit. Ostatni kod, kterym doslo k nauceni modelu,
#     take odevzdejte v ramci projektu.

#Vstup:             data:           vstupni data reprezentujici 1
#                                   objekt (1 pacienta, 1 obrazek, apod.). 

#Vystup:            output:         zarazeni objektu do tridy

    features = feature_extract(data)

    knn = joblib.load('knn_model.pkl')

    output = knn.predict([features])
    
    return output