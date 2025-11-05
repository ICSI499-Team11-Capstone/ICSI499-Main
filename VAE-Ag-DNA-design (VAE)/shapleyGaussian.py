#Elham Sadeghi March2023

import os
import time
import torch
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sequenceModel import SequenceModel
from sequenceTrainer import SequenceTrainer
from sequenceDataset import SequenceDataset
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
import pandas as pd
from matplotlib.pyplot import axis
import sys
import json
from itertools import combinations
from math import comb
import math

def gaussian(x, mu, sigma):

    return (1 / (sigma * math.sqrt(2*math.pi))) * math.exp(-(1/2)*((x-mu)/sigma)**2)


def get_wavelength_arr(data):
    """Basic wrapper for accessing wavelength array from data array"""
    wavelength = data['Wavelen']
    return wavelength


def get_lii_arr(data):
    """Basic wrapper for accessing local integrated intensity array from data array"""
    lii = data["LII"]
    return lii


def get_ohe_data(data):
    """Wrapper function that accesses one hot encoded array from data array and returns it as a pytorch tensor"""
    # ohe_data = torch.from_numpy(data['ohe'])
    ohe_data = data['ohe']
    return ohe_data

def unpack_and_load_data(path_to_file: str, path_to_model: str):
    """This function is used as a wrapper function to load the .npz file that stores the wavelength, LII and one
    hot encoded arrays and load the trained model used for sampling. Both the data file and model are returned
    as objects."""
    data_file = np.load(path_to_file)
    model_state = torch.load(path_to_model)
    model = SequenceModel()
    model.load_state_dict(model_state)
    model.eval()
    return data_file, model

def encode_data(ohe_sequences: object, model: object):
    """This is a wrapper function for the encode() function that can be found in sequenceModel.py. This simply calls
    that function and returns the latent distribution that is produced in the latent space."""
    ohe_sequences = torch.from_numpy(ohe_sequences)
    print(len(ohe_sequences))
    model_state = torch.load("./models/weighted/a0.003lds17b0.007g2d1h15.pt")
    model = SequenceModel()  # You may need to pass arguments if your SequenceModel requires them
    model.load_state_dict(model_state)
    model.eval()
    return latent_dist

model = torch.load("./models/weighted/a0.003lds17b0.007g2d1h15.pt")

dataset = SequenceDataset(datafile = './data-and-cleaning/cleandata.csv')

#filtering nir and green
sequences = dataset.dataset.loc[dataset.dataset["Wavelen"]<590,'Sequence']

G_mean = -3.547799 #dataset.dataset.loc[dataset.dataset["Wavelen"]<590,'Wavelen'].mean()
G_std  = 5.002617 #dataset.dataset.loc[dataset.dataset["Wavelen"]<590,'Wavelen'].std()

#filtering other wave
#dataset1 = dataset.dataset.loc[dataset.dataset["Wavelen"]>590,['Sequence','Wavelen']]
#sequences = dataset1.loc[dataset1["Wavelen"]<=660,'Sequence']


oheSeqs = dataset.transform_sequences(sequences.apply(lambda x: pd.Series([c for c in x])).to_numpy())



        
latent_dist = encode_data(oheSeqs, model)

mean_matrix = latent_dist.mean.detach().cpu().numpy()

z_wav = mean_matrix[:,0]
z_lii = mean_matrix[:,1]


l = len(oheSeqs)
nir_ohe = []
cols = ['p1','p2','p3','p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
events = pd.DataFrame(columns=(cols))
nir_z = []

# Wavelen[i] > 800:
for i in range(l):
    if True:
        nir_ohe.append(oheSeqs[i])
        nir_z.append(z_wav[i])
        dict = {}
        for j in range(10):
            col = 'p'+ str(j+1)
            dict[col] = oheSeqs[i][j]
            # print(col, ohe_data[i][j])
        events = pd.concat([events, pd.DataFrame([dict])], ignore_index=True)


def powerset(string):
    ret = []
    for i in range(0,len(string)+1):
        for element in combinations(string,i):
            # print(''.join(element))
            ret.append(list(element))
    return ret

def calcV(arr, nir):
    # ohe = [ [0]*4 for _ in range(10) ]
    ohe = np.zeros([1,10,4])
    if len(arr) > 0:
        for i in range(len(arr)):
            ohe[0][ arr[i]-1 ] = nir[ arr[i]-1 ]
    
    ohe_sequences = torch.from_numpy(ohe)
    latent_dist = model.encode(ohe_sequences)

    mean_matrix = latent_dist.mean.detach().cpu().numpy()
    
    z_wav = mean_matrix[:,0]
    # print(z_wav[0])
    return z_wav[0]
    # print(ohe)
    # return ohe

N = list(range(1,10+1))
S = powerset(N)

nir_shap = [[] for _ in range(len(oheSeqs))]


nir_sub = []

# nir_ohe = ["ABC"]
# S = [[], [1],[1,2],[1,3]]
# N = list(range(1,3+1))


for nd in range(len(nir_ohe)):
    shap = []


    for p in S:
        if len(p) == 0:
            shap.append(0)
            

            continue
        res = 0


        for j in range(len(S)):
            check =  any(item in S[j] for item in p)
            if check == False:
                s = S[j].copy()
                if len(s) == 0:
                    s = p
                else:
                    s.extend(p)
                if len(S[j]) == len(N):
                    norm = 1
                else:
                    norm = comb(len(N), len(S[j])) * (len(N) - len(S[j]))
                vp = calcV(s, nir_ohe[nd])
                v = calcV(S[j], nir_ohe[nd])
                V = (gaussian(vp, G_mean, G_std) - gaussian(v, G_mean, G_std)) / norm
                res = res + V                
        shap.append(res / len(N))
    nir_sub.append(shap)




df = pd.DataFrame(nir_sub, dtype=(float))

df.to_csv("./data-and-cleaning/Plate14_Group1_less_than590nm_gaussian_shapley_values.csv", index = False)

