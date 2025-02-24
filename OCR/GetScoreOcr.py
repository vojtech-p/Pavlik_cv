# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:07:34 2022

@author: xredin00
"""

import numpy as np

def GetScoreOcr( confusionMatrix ):
# %Funkce pro vyhodnocení úspěšnosti modelu

# %Vstup:         confusionMatrix:            Matice záměn z funkce Main()

# %Výstup:        partialFscore:              částečné F1 skóre pro jednu
# %klasifikační třídu. Pořadí (číslo indexu) je dáno číselným označením třídy

# %               totalFscore:                celkové F1 skóre modelu


    partialFscore = np.zeros(10)   
    
    for idx in range(10):
        partialFscore[idx] = 2*confusionMatrix[ idx, idx ]/( np.sum( confusionMatrix[ idx, : ]) + np.sum( confusionMatrix [:, idx] ))
    
    totalFscore = ( 1/10 )*np.sum( partialFscore )
    
    return partialFscore,totalFscore