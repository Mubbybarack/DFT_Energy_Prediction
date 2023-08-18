# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 00:18:35 2022

@author: barak bello

Description: *This code generates 56 features to describe adsorbed chemical species from their respective SMILES.\
    This code contains both saturated and unsaturated linear and branched hydrocarbons.\
        The code works well for C1 - C7 with up to C2 branching
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk


#%% Huge function containing code

def extract_feat (smile):
### collecting smile for each specie
    smstr = smile

### Counting total number of carbon
    C_total = 0
    for i in range(len(smstr)):
        if smstr[i] == "C":
            C_total += 1

### Counting number of hydrogen            
    C_count = 0
    H_in_C = [0]*C_total
    smstr = smstr + str(0) + str(0) # to allow i+1 or i+2 work for last C
    
    for i in range(len(smstr)):
        if smstr[i] == "C":
            C_count += 1
    
    # Counting Hydrogen for first C        
            if C_count == 1:
                if smstr[i+1] == "H" and smstr[i+2] == "]":
                    H_in_C[C_count-1] = 1
                elif smstr[i+1] == "H" and smstr[i+2] == ("2" or "3"):
                    H_in_C[C_count-1] = int(smstr[i+2])
                elif smstr[i+1] == "]":
                    H_in_C[C_count-1] = 0
                else:
                    H_in_C[C_count-1] = 3
                    
    # Counting Hydrogen for C linearly bonded and branched
            elif C_count < C_total:
                if smstr[i+1] == "(":
                    H_in_C[C_count-1] = 1
                elif smstr[i-1] == "[" and smstr[i+1] == "]":
                    H_in_C[C_count-1] = 0
                elif smstr[i-1] == "[" and smstr[i+1] == "H":
                    H_in_C[C_count-1] = 1                
                elif smstr[i+1] == ")":
                    H_in_C[C_count-1] = 3
                else:
                    H_in_C[C_count-1] = 2
            elif smstr[i-1] == "[" and smstr[i-2] == "(" and smstr[i+1] == "H" and smstr[i+2] == "]":
                    H_in_C[C_count-1] = 1
            elif smstr[i-1] == "[" and smstr[i-2] == "(" and smstr[i+1] == "H" and smstr[i+2] == "2":
                    H_in_C[C_count-1] = 2
            elif smstr[i-2] == "(" and smstr[i-1] == "[" and smstr[i+1] == "]":
                H_in_C[C_count-1] = 0
            elif smstr[i-1] == "[" and smstr[i+1] == "H" and smstr[i+2] == "]" and smstr[i+3] == ")":
                    H_in_C[C_count-1] = 1
            elif smstr[i-1] == "[" and smstr[i+1] == "H" and smstr[i+2] == "2" and smstr[i+4] == ")":
                    H_in_C[C_count-1] = 2
            elif smstr[i-1] == "[" and smstr[i+1] == "]" and smstr[i+2] == "(":
                H_in_C[C_count-1] = 0                 
    
    # Counting Hydrogen for last C
            elif C_count == C_total:
                if smstr[i+1] == "]":
                    H_in_C[C_count-1] = 0
                elif smstr[i+1] == "H" and smstr[i+2] == "]":
                    H_in_C[C_count-1] = 1
                elif smstr[i+1] == "H" and smstr[i+2] == "2":
                    H_in_C[C_count-1] = 2
                else:
                    H_in_C[C_count-1] = 3    
                    
            if C_total == 1:
                if smstr[i+1] == "0":
                    H_in_C[C_count-1] = 4                    
                    
    #Factoring in double and triple bonds
    C_count = 0
    for i in range(len(smstr)):
        if smstr[i] == "C":
            C_count += 1
        if smstr[i] == "=" and smstr[i+1] == ("C" or "["):
            if H_in_C[C_count] > 0:
                H_in_C[C_count] -= 1
        if smstr[i] == "=" and (smstr[i-1] == "C" or smstr[i-1] == "]" or smstr[i-1] == "("):
            if H_in_C[C_count-1] > 0:    
                H_in_C[C_count-1] -= 1           
        if smstr[i] == "#" and (smstr[i-1] == "C" or smstr[i-1] == "]" or smstr[i-1] == "("):
            if H_in_C[C_count-1] > 1:        
                H_in_C[C_count-1] -= 2
        if smstr[i] == "#" and smstr[i+1] == ("C" or "]"):  
            if H_in_C[C_count] > 1:        
                H_in_C[C_count] -= 2 
                
        if smstr[i] =="H" and smstr[i-1] == "[" and smstr[i+1] == "]":
            H_in_C = [1]                
                    
    # Exceptions due to branching  
        C_count_temp = 0   
                
        if smstr[i] == ")" and smstr[i+1] == "=":
            for j in range (i,0,-1):
                C_count_temp += 1
                if smstr[j] == "(":
                    break
            H_in_C[C_count_temp -2] -= 1     
            
            
        if smstr[i] == ")" and smstr[i+1] == "#":
            for j in range (i,0,-1):
                C_count_temp += 1
                if smstr[j] == "(":
                    break
            H_in_C[C_count_temp -2] -= 2  


### Checking linearity of each C
    Linearity = [True]*C_total
    smstr = smstr + str(0) + str(0)
    C_count = -1
    for i in range(len(smstr)):
        if smstr[i] == "C":
            C_count += 1
            if smstr[i-1] == "(" or (smstr[i-1] == "[" and smstr[i-2] == "(") or\
                (smstr[i-1] == "=" and smstr[i-2] == "(") or\
                    (smstr[i-1] == "[" and smstr[i-2] == "=" and smstr[i-3] == "("):
                Linearity[C_count] = False
            elif smstr[i+1] == ")" or (smstr[i+1] == "]" and smstr[i+2] == ")") or\
                (smstr[i+1] == "H" and smstr[i+2] == "]" and smstr[i+3] == ")") or\
                    (smstr[i+1] == "H" and smstr[i+2] == "2" and smstr[i+3] == "]" and smstr[i+4] == ")"):
                Linearity[C_count] = False  
                
### Checking double bond for each C
    DoubleBond = [False]*C_total
    C_count = -1
    for i in range(len(smstr)):
        if smstr[i] == "C":
            C_count += 1
        if smstr[i] == "=":
            if Linearity[C_count] == True and Linearity[C_count+1] == True:
                DoubleBond[C_count] = True
                DoubleBond[C_count+1] = True
            elif Linearity[C_count] == True and Linearity[C_count+1] == False:
                DoubleBond[C_count] = True
                DoubleBond[C_count+1] = True    
            elif Linearity[C_count] == False and Linearity[C_count+1] == True and Linearity[C_count-1] == True:
                DoubleBond[C_count-1] = True
                DoubleBond[C_count+1] = True    
            elif Linearity[C_count] == False and Linearity[C_count+1] == True and Linearity[C_count-1] == False:
                DoubleBond[C_count-2] = True
                DoubleBond[C_count+1] = True    
                
### Counting triple bond            
    TripleBond = [False]*C_total
    C_count = -1
    for i in range(len(smstr)):
        if smstr[i] == "C":
            C_count += 1
        if smstr[i] == "#":
            if Linearity[C_count] == True and Linearity[C_count+1] == True:
                TripleBond[C_count] = True
                TripleBond[C_count+1] = True
            elif Linearity[C_count] == True and Linearity[C_count+1] == False:
                TripleBond[C_count] = True
                TripleBond[C_count+1] = True    
            elif Linearity[C_count] == False and Linearity[C_count+1] == True and Linearity[C_count-1] == True:
                TripleBond[C_count-1] = True
                TripleBond[C_count+1] = True    
            elif Linearity[C_count] == False and Linearity[C_count+1] == True and Linearity[C_count-1] == False:
                TripleBond[C_count-2] = True
                TripleBond[C_count+1] = True                    
    
### Counting valence
    
    C_valence = [0]*C_total 
    C_count = -1
    for i in range(len(smstr)):
        if smstr[i] == "C":
            C_count += 1
    # Identifying valence for first and last C        
            if C_count == 0 or C_count == C_total-1:
                if smstr[i+1] == "]" and smstr[i-1] == "[" :
                    C_valence[C_count] = 3
                elif smstr[i+1] == "H" and smstr[i+2] == "]":
                    C_valence[C_count] = 2
                elif smstr[i+1] == "H" and smstr[i+2] == "2":
                    C_valence[C_count] = 1
                else:
                    C_valence[C_count] = 0
    # Identifying valence for lone C                
            if C_total == 1:
                if smstr[i+1] == "]" and smstr[i-1] == "[" :
                    C_valence[C_count] = 4
                elif smstr[i+1] == "H" and smstr[i+2] == "]":
                    C_valence[C_count] = 3
                elif smstr[i+1] == "H" and smstr[i+2] == "2":
                    C_valence[C_count] = 2
                elif smstr[i+1] == "H" and smstr[i+2] == "3":
                    C_valence[C_count] = 1
                else:
                    C_valence[C_count] = 0
     # Identifying valence for middle C with consideration of branched C               
            if C_count < C_total-1 and C_count > 0:
                if smstr[i+1] == "]" and smstr[i-1] == "[" and smstr[i+2] != ")" and smstr[i+2] != "(":
                    C_valence[C_count] = 2
                elif smstr[i+1] == "]" and smstr[i-1] == "[" and smstr[i+2] == ")":
                    C_valence[C_count] = 3                
                elif smstr[i+1] == "H" and smstr[i+2] == "2" and smstr[i+3] == "]" and smstr[i+4] == ")":
                    C_valence[C_count] = 1               
                elif smstr[i+1] == "H" and smstr[i+2] == "]" and smstr[i+3] != ")" and smstr[i+3] != "(":
                    C_valence[C_count] = 1
                elif smstr[i+1] == "H" and smstr[i+2] == "]" and smstr[i+3] == ")":
                    C_valence[C_count] = 2 
                elif smstr[i+1] == "]" and smstr[i-1] == "[" and smstr[i+2] == "(":
                    C_valence[C_count] = 1   
                elif smstr[i+1] == "H" and smstr[i+2] == "]" and smstr[i+3] == "(":
                    C_valence[C_count] = 0                
                else:
                    C_valence[C_count] = 0
    
        # Correcting valence in the presence of double or triple bond           
            if DoubleBond[C_count] == True:
                if C_valence[C_count] > 0:
                    C_valence[C_count] -= 1
            if TripleBond[C_count] == True:
                if C_valence[C_count] > 1:
                    C_valence[C_count] -= 2

                            
### Counting C, H, C0, C1, C2, C3, C-H0, C-H, C-H2, C-H3, C-H4
    
    C = C_total
    H = sum(H_in_C)
    C0 = C1 = C2 = C3 = C4 = CbH0 = CbH = CbH2 = CbH3 = CbH4 = 0
    for i in range(len(C_valence)):
        if C_valence[i] == 0:
            C0 += 1 
        if C_valence[i] == 1:
            C1 += 1           
        if C_valence[i] == 2:
            C2 += 1                     
        if C_valence[i] == 3:
            C3 += 1  
        if C_valence[i] == 4:
            C4 += 1              
    for i in range(len(H_in_C)):
        if H_in_C[i] == 0:
            CbH0 += 1
        if H_in_C[i] == 1 and C_total > 0:
            CbH += 1        
        if H_in_C[i] == 2:
            CbH2 += 1        
        if H_in_C[i] == 3:
            CbH3 += 1        
        if H_in_C[i] == 4:
            CbH4 += 1                  
            
### Counting linear single C-C bond at different valence
    
    CtoC = np.zeros((4,4))
    for i in [0,1,2,3]:
        for j in [0,1,2,3]:
            for count in range(len(C_valence)):
                if count == len(C_valence) -1:
                    break
                else:
                    if count == 0:
                        if C_valence[count] == i and C_valence[count+1] == j:
                            CtoC[i,j] += 1                    
            
                    else:
                        if Linearity[count] == False and Linearity[count+1] == True and\
                            Linearity[count-1] == True:
                                if C_valence[count-1] == i and C_valence[count+1] == j:
                                    CtoC[i,j] += 1                    
                        elif Linearity[count] == False and Linearity[count+1] == True and\
                            Linearity[count-1] == False:
                                if C_valence[count-2] == i and C_valence[count+1] == j:
                                    CtoC[i,j] += 1    
                        else:
                            if C_valence[count] == i and C_valence[count+1] == j:
                                CtoC[i,j] += 1
    
    
### Counting linear carbon-carbon double bonds across carbon valencies
    CdoubC = np.zeros((4,4))                       
    for i in [0,1,2,3]:
        for j in [0,1,2,3]:
            for count in range(len(C_valence)):
                if count == len(C_valence) -1:
                    break
                else: 
                    if DoubleBond[count] == True and DoubleBond[count+1] == True:
                        if Linearity[count+1] == True:
                            if C_valence[count] == i and C_valence[count+1] == j:
                                CdoubC[i,j] += 1 
                    if count < len(C_valence)-2:        
                        if DoubleBond[count] == True and DoubleBond[count+1] == False and DoubleBond[count+2] == True:
                            if C_valence[count] == i and C_valence[count+2] == j:
                                CdoubC[i,j] += 1
                    if count < len(C_valence)-3:                                       
                        if DoubleBond[count] == True and DoubleBond[count+1] == False and DoubleBond[count+2] == False\
                            and DoubleBond[count+3] == True:
                            if C_valence[count] == i and C_valence[count+3] == j:
                                CdoubC[i,j] += 1    
    
### Counting linear carbon-carbon triple bonds across carbon valencies                        
    CtriC = np.zeros((4,4))                       
    for i in [0,1,2,3]:
        for j in [0,1,2,3]:
            for count in range(len(C_valence)):
                if count == len(C_valence) -1:
                    break
                else: 
                    if TripleBond[count] == True and TripleBond[count+1] == True:
                        if Linearity[count+1] == True:                    
                            if C_valence[count] == i and C_valence[count+1] == j:
                                CtriC[i,j] += 1  
                    if count < len(C_valence)-2:                                
                        if TripleBond[count] == True and TripleBond[count+1] == False and TripleBond[count+2] == True:
                            if C_valence[count] == i and C_valence[count+2] == j:
                                CtriC[i,j] += 1
                    if count < len(C_valence)-3:                                          
                        if TripleBond[count] == True and TripleBond[count+1] == False and TripleBond[count+2] == False\
                            and TripleBond[count+3] == True:
                            if C_valence[count] == i and C_valence[count+3] == j:
                                CtriC[i,j] += 1                          
    
    
### Counting branched carbon-carbon single bonds across carbon valencies
    CbC = np.zeros((4,4))
    for i in [0,1,2,3]:
        for j in [0,1,2,3]:
            for count in range(len(C_valence)):
                    if count == len(C_valence) -1:
                        break
                    else:
                        if Linearity[count] == True and Linearity[count+1] == False:
                            if C_valence[count] == i and C_valence[count+1] == j:
                                CbC[i,j] += 1
    
## Counting branched carbon-carbon double bonds across carbon valencies                           
    C_branchedDoub_C = np.zeros((4,4))
    for i in [0,1,2,3]:
        for j in [0,1,2,3]:
            for count in range(len(C_valence)):
                    if count == len(C_valence) -1:
                        break
                    else:
                        if Linearity[count] == True and Linearity[count+1] == False:
                            if DoubleBond[count] == True and DoubleBond[count+1] == True:
                                if C_valence[count] == i and C_valence[count+1] == j:
                                    C_branchedDoub_C[i,j] += 1 
                                    
### Counting branched carbon-carbon triple bonds across carbon valencies                           
    C_branchedTrip_C = np.zeros((4,4))
    for i in [0,1,2,3]:
        for j in [0,1,2,3]:
            for count in range(len(C_valence)):
                    if count == len(C_valence) -1:
                        break
                    else:
                        if Linearity[count] == True and Linearity[count+1] == False:
                            if TripleBond[count] == True and TripleBond[count+1] == True:
                                if C_valence[count] == i and C_valence[count+1] == j:
                                    C_branchedTrip_C[i,j] += 1                                 
                                
### Describing position of double bond, triple bond and branching
    
    doub = [0]*C_total
    triple = [0]*C_total
    C_count_temp = 0
    count = -1
    for i in range(len(smstr)):
        if smstr[i] == "C":
            count += 1
        if smstr[i] == "=":
            if smstr[i-1] == "C" or smstr[i-1] == "]" or smstr[i-1] == "(":
                doub[count] += 1
                
        if smstr[i] == ")" and smstr[i+1] == "=":
            for j in range (i,0,-1):
                if smstr[j] == "C":
                    C_count_temp += 1
                if smstr[j] == "(":
                    break   
            doub[count - C_count_temp] += 1
    
        if smstr[i] == "#":
            if smstr[i-1] == "C" or smstr[i-1] == "]" or smstr[i-1] == "(":
                triple[count] += 1
    
    methyl = [0]*C_total
    ethyl = [0]*C_total
    
    for i in range(len(Linearity)-1):
        if Linearity[i] == True and Linearity[i+1] == False and Linearity[i+2] == True:
            methyl[i] += 1
        elif Linearity[i] == True and Linearity[i+1] == False and Linearity[i+2] == False:
            ethyl[i] += 1
            
### Correcting length of double, triple, methyl and ethyl
    for i in [doub, triple, methyl, ethyl]: 
        if len(i) < 7:
            corr = [0]*(7-len(i))
            i.extend(corr)
        
     
### Extracting features
    
    # C-C linear bond
    C0b0 = CtoC[0,0]
    C0bC1 = CtoC[0,1] + CtoC[1,0]; C0bC2 = CtoC[0,2] + CtoC[2,0]; C0bC3 = CtoC[0,3] + CtoC[3,0]
    C0dbC0 = CdoubC[0,0]; C0dbC1 = CdoubC[0,1] + CdoubC[1,0]; C0dbC2 = CtoC[0,2] + CtoC[2,0]
    C0tbC0 = CtriC[0,0]; C0tbC1 = CtriC[0,1] + CtriC[1,0]; C1tbC1 = CtoC[1,1]
    
    # C-C branched bond
    C0b0_br = CbC[0,0]
    C0bC1_br = CbC[0,1] + CbC[1,0]; C0bC2_br = CbC[0,2] + CbC[2,0]; C0bC3_br = CbC[0,3] + CbC[3,0]
    C0dbC0_br = C_branchedDoub_C[0,0]; C0dbC1_br = C_branchedDoub_C[0,1] + C_branchedDoub_C[1,0];\
        C0dbC2_br = C_branchedDoub_C[0,2] + C_branchedDoub_C[2,0]
    C0tbC0_br = C_branchedTrip_C[0,0]; C0tbC1_br = C_branchedTrip_C[0,1] + C_branchedTrip_C[1,0];\
        C1tbC1_br = C_branchedTrip_C[1,1]
    
       # Position of double and triple bond
    C1db, C2db, C3db, C4db, C5db, C6db = doub[:6]
    C1tb, C2tb, C3tb, C4tb, C5tb, C6tb = triple[:6]
    C1_met, C2_met, C3_met, C4_met, C5_met, C6_met = methyl[:6]
    C1_eth, C2_eth, C3_eth, C4_eth, C5_eth, C6_eth = ethyl[:6]
    
    
    features = [0]*56
    features[:12] = C, H, C0, C1, C2, C3, C4, CbH0, CbH, CbH2, CbH3, CbH4
    features[12:22] = C0b0, C0bC1, C0bC2, C0bC3, C0dbC0, C0dbC1, C0dbC2, C0tbC0, C0tbC1, C1tbC1
    features[22:32] = C0b0_br, C0bC1_br, C0bC2_br, C0bC3_br, C0dbC0_br, C0dbC1_br,\
        C0dbC2_br, C0tbC0_br, C0tbC1_br, C1tbC1_br
    features[32:] = C1db, C2db, C3db, C4db, C5db, C6db, C1tb, C2tb, C3tb,\
        C4tb, C5tb, C6tb ,C1_met, C2_met, C3_met, C4_met, C5_met, C6_met,\
            C1_eth, C2_eth, C3_eth, C4_eth, C5_eth, C6_eth 
            
    return features

#%% load data, call function and save features as csv

Hydrocarbon = pd.read_csv("C:/Users/mabello/Desktop/Research/ML_paper/hydrocarbon_data1.csv", header=None)
data = Hydrocarbon.values.tolist()

extrt = np.zeros((155,56))

for i in range(len(data)):
    smile = data[i][0]
    extrt[i,:] = extract_feat(smile)

ex_1 = pd.DataFrame(data=extrt)
ex_1.columns = ['C', 'H', 'C0', 'C1', 'C2', 'C3', 'C4', 'C', 'C-H', 'C-H2', 'C-H3', 'C-H4', 'C0-C0', 'C0-C1', 'C0-C2', 'C0-C3',\
                'C0=C0', 'C0=C1', 'C0=C2', 'C0#C0', 'C0#C1', 'C1#C1', 'C0:C0', 'C0:C1', 'C0:C2', 'C0:C3', 'C0=:C0',\
                    'C0=:C1', 'C0=:C2', 'C0#:C0', 'C0#:C1', 'C1#:C1', 'C1=', 'C2=', 'C3=', 'C4=', 'C5=', 'C6=', 'C1#',\
                        'C2#', 'C3#', 'C4#', 'C5#', 'C6#','C1-C', 'C2-C', 'C3-C', 'C4-C', 'C5-C', 'C6-C', 'C1-CC',\
                            'C2-CC','C3-CC', 'C4-CC', 'C5-CC', 'C6-CC']
ex_1.insert(0, "SMILES", data, True)
ex_2 = pd.concat([Hydrocarbon, ex_1],axis=1)
ex_3 = ex_2.drop(columns=['SMILES'])
ex_4 = ex_3.rename(columns={0:"SMILES"})
ex_4.to_csv('features.csv',index=False)
