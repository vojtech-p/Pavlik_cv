# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 12:39:26 2022

@author: xredin00
"""
import os
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import cv2 as cv
from MyModel import MyModel
from GetScoreOcr import GetScoreOcr
from DataPreprocessing import DataPreprocessing


def Main(filePath):
    
    # Funkce slouzi pro overeni klasifikacnich schopnosti navrzeneho modelu.
    # Model bude overovan na skryte mnozine dat, v odevzdanem projektu je proto
    # nutne dodrzet sktrukturu tohoto kodu. 
    
    # ======================= POZOR ==========================================
    # Odevzdany model se jiz NEBUDE ucit. Naucite jej vy, odevzdate a my ho
    # pri testovani jiz budeme POUZE volat a overovat jeho funkcnost.
    # ========================================================================
    
    # Vstup:     filePath:           Nazev slozky (textovy retezec) obsahujici data

    # Výstup:    partialFscore:      F1 skore modelu pro jednotlive tridy
    #            totalFscore:        Vysledny F1 skore modelu
    #            confusionMatrix:    Matice zamen
    
    # Funkce:
    #            DataPreprocessing:  Funkce pro predzpracovani dat
    
    #            MyModel:            Funkce pro implementaci modelu. Nauceny model se bude nacitat z externiho souboru,
    #            nebude se ucit pri kazdem spusteni kodu. Veskery kod, ktery vedl k nauceni modelu,
    #            vsak bude soucasti odevzdaneho projektu. Do funkce vstupuje vzdy jen 1 objekt (casova rada, obrazek, apod.)
    
    #            GetScoreOcr:         Funkce pro vyhodnoceni uspesnosti
    #            modelu. 

    if os.path.isdir(filePath)==False:
        print("Wrong directory")

    #%% 1 - Nacteni dat
    refFile = pd.read_csv(f"{filePath}\\references.csv")
    numRecords = refFile.shape[0]
    confMatrix = np.zeros((10,10))
    
    for idx in range(numRecords):
        fileName = refFile.name[idx]
        inputData = cv.imread(f'{filePath}/{fileName}')

        #%% 2 - Predzpracovani dat
        preprocessedData = DataPreprocessing(inputData)
        
        targetClass = refFile.target[idx]
        if targetClass == 10:
            targetClass = 0

        #%% 3 - Vybaveni natrenovaneho modelu
        outputClass = MyModel(preprocessedData)
        
        if outputClass == 0 or outputClass == 1 or outputClass == 2 or outputClass == 3 or outputClass == 4 or outputClass == 5 or outputClass == 6 or outputClass == 7 or outputClass == 8 or outputClass == 9:
            confMatrix[outputClass,targetClass] += 1
        else:
             print('Invalid class number. Operation aborted.')   
             
    partialFscore, totalFscore  = GetScoreOcr(confMatrix)
    print(partialFscore, totalFscore)
    return partialFscore, totalFscore, confMatrix
        
partialFscore, totalFscore, confMatrix = Main('./trainDataOCR/trainData/')