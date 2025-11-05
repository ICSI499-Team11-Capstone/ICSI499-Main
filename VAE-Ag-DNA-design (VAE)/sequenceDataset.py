#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:07:00 2020

@author: alexandergorovits
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from probabilityBin import AttributeProbabilityBin

import sys
os.chdir(sys.path[0]) #compatability hack

class SequenceDataset:
    
    def __init__(self,datafile = './data-and-cleaning/cleandata.csv', seqlen=10, split=(0.85, 0.15), noofbuckets = 7):
        self.dataset = pd.read_csv(datafile)
        if "Unnamed: 0" in self.dataset.columns:
            self.dataset.drop(columns=["Unnamed: 0"],inplace=True)
        self.ALPHABET = ['A','C','G','T']
        self.seqlen = seqlen
        self.split = split
        self.noofbuckets = noofbuckets
        
    def transform_sequences(self,seqs):
        enc = OneHotEncoder()
        enc.fit(np.array(self.ALPHABET).reshape(-1,1))
        return enc.transform(seqs.reshape(-1,1)).toarray().reshape(
            -1, self.seqlen, len(self.ALPHABET))
        
    # @staticmethod
    # def cust_collate(batch):
    #     #print(list(batch))
    #     #print(len(batch))
    #     return [x for x in batch]
        
    def data_loaders(self, batch_size):
        seqs = self.transform_sequences(
            self.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).to_numpy())
        Wavelen = self.dataset['Wavelen'].to_numpy(dtype="float").reshape(-1,1) 
        localII = self.dataset['LII'].to_numpy(dtype="float").reshape(-1,1)
        attribs = np.append(Wavelen, localII, axis=1)
        wavelenBin = AttributeProbabilityBin(Wavelen, self.noofbuckets, (450, 800))
        liiBin = AttributeProbabilityBin(localII, self.noofbuckets, (1, 8))
        
        nval = seqs.shape[0]
        split1 = int(self.split[0]*nval)
        split2 = int(self.split[1]*nval)
        perm = torch.randperm(nval)
        
        print(nval, split1,split2)
        
        train_ds = TensorDataset(
            torch.from_numpy(seqs[perm[:split1],:,:]),
            torch.from_numpy(attribs[perm[:split1],:]),
            #torch.from_numpy(localII[perm[:split1],:])
        )
        val_ds = TensorDataset(
            torch.from_numpy(seqs[perm[split1:],:,:]),
            torch.from_numpy(attribs[perm[split1:],:]),
            # torch.from_numpy(seqs[perm[:int(.05*split1)],:,:]),
            # torch.from_numpy(attribs[perm[:int(.05*split1)],:]),
            #torch.from_numpy(localII[perm[split1:split1+split2],:])
        )

        valdict = perm[split1:]
        np.save('utils/valdict.npy',valdict)
        # val_ds = TensorDataset(
        #     torch.from_numpy(seqs[perm[split1+split2:],:,:]),
        #     torch.from_numpy(attribs[perm[split1+split2:],:]),
        #     #torch.from_numpy(localII[perm[split1+split2:],:])
        # )
        
        print(len(train_ds),len(val_ds))
        
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True
            )
        # test_dl = DataLoader(
        #     test_ds,
        #     batch_size=batch_size,
        #     shuffle=False
        #     )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False
            )
        
        print(len(train_dl),len(val_dl))
        
        return train_dl, val_dl, None, [wavelenBin, liiBin]
