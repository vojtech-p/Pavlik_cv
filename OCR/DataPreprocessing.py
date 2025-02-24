# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 18:46:32 2022

@author: rredi
"""
import numpy as np
import cv2 as cv

def matrix_mean(matrix: list):
    suma_matrix = 0.0
    for n in matrix:
        for z in n:
            suma_matrix = suma_matrix+z
    return suma_matrix/(matrix.shape[0]*matrix.shape[1])

def CE_Clahe(image, clip=2.0, mask=(6,8)):
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    # Applying CLAHE to L-channel 
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=mask)
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color space
    return cv.cvtColor(limg, cv.COLOR_LAB2BGR)

def threshholding(matrix, num_layers=4, threshholds=(0.1,0.5)):
    step = (threshholds[1]-threshholds[0])/num_layers
    maximum = np.max(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i,j] < maximum*threshholds[0]:
                matrix[i,j] = 0
            elif matrix[i,j] > maximum*threshholds[1]:
                matrix[i,j] = maximum
            else:
                n = 0
                while n < num_layers:
                    n = n + 1
                    if matrix[i,j] < maximum*step*n:
                        matrix[i,j] = maximum*step*n
                        break
    return matrix

def hugh_lines_erase(image, image_canny, thresh=50, color=0, width=2):
    lines = cv.HoughLines(image_canny, 1, np.pi / 180, thresh, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(image, pt1, pt2, color, width, cv.LINE_AA)
    return image    

def rectangle_find(image, width=5, height=5):
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (width,height))
    dilation = cv.dilate(image,rect_kernel,iterations=1)
    max_rect_white = 0
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        rect = cv.rectangle(image, (x,y),(x+w,y+h), (0),2)
        sum_rect_white = np.sum(cnt)
        if max_rect_white < sum_rect_white:
            max_rect_white = sum_rect_white
            max_rect = [x,y,w,h]
    return image[max_rect[1]:max_rect[1]+max_rect[3], max_rect[0]:max_rect[0]+max_rect[2]]

def image_read(number):
    if number == "nine":
        img = cv.imread("./trainDataOCR/trainData/nine_208.png")
        img2 = cv.imread("./trainDataOCR/trainData/nine_008.png")
        img3 = cv.imread("./trainDataOCR/trainData/nine_009.png")
    elif number == "eight":
        img = cv.imread("./trainDataOCR/trainData/eight_063.png")
        img2 = cv.imread("./trainDataOCR/trainData/eight_007.png")
        img3 = cv.imread("./trainDataOCR/trainData/eight_030.png")
    elif number == "four":
        img = cv.imread("./trainDataOCR/trainData/four_150.png")
        img2 = cv.imread("./trainDataOCR/trainData/four_222.png")
        img3 = cv.imread("./trainDataOCR/trainData/four_143.png")
    elif number == "one":
        img = cv.imread("./trainDataOCR/trainData/one_266.png")
        img2 = cv.imread("./trainDataOCR/trainData/one_263.png")
        img3 = cv.imread("./trainDataOCR/trainData/one_279.png")
    return img, img2, img3

def DataPreprocessing(inputData):
    """
    Funkce slouzi pro predzpracovani dat, ktera slouzi k testovani modelu. Veskery kod, ktery vedl k nastaveni
    jednotlivych kroku predzpracovani (vcetne vypoctu konstant, prumeru, smerodatnych odchylek, atp.) budou odevzdany
    spolu s celym projektem.

    :parameter inputData:
        Vstupni data, ktera se budou predzpracovavat.
    :return preprocessedData:
        Predzpracovana data na vystup
    """
    preprocessedData = inputData.copy()
    
    preprocessedData = CE_Clahe(preprocessedData)

    preprocessedData = cv.cvtColor(preprocessedData, cv.COLOR_BGRA2GRAY)
    
    preprocessedData = cv.normalize(preprocessedData, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    
    preprocessedData = cv.bilateralFilter(preprocessedData, 9, 80,80)
    
    preprocessedData = threshholding(preprocessedData, 3, (0.05, 0.45))
    
    preprocessedData = cv.bitwise_not(preprocessedData)
    
    kernel = np.ones((1,2),np.uint8)
    preprocessedData = cv.morphologyEx(preprocessedData, cv.MORPH_CLOSE, kernel)
    
    (thresh, preprocessedData) = cv.threshold(preprocessedData, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    preprocessedData_canny = cv.Canny(preprocessedData, 50, 200, None, 3)
    
    preprocessedData = hugh_lines_erase(preprocessedData, preprocessedData_canny, thresh=25, color=0, width=2)
    
    preprocessedData = cv.bilateralFilter(preprocessedData, 5,75,75)

    preprocessedData = rectangle_find(preprocessedData)

    return preprocessedData
