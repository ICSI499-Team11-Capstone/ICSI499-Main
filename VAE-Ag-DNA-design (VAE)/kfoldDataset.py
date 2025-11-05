"""
Created on Wed Dec 3 2021

@author: vyeruva@albany.edu
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from probabilityBin import AttributeProbabilityBin
from sklearn.model_selection import StratifiedKFold


import sys
os.chdir(sys.path[0]) #compatability hack
# def y_wav:
#     tmp = []
    
class KFoldDataset:
    
    def __init__(self,datafile = './data-and-cleaning/cleandata.csv', seqlen=10, noofbuckets = 7, kfolds = 10, curfold=0):
        self.dataset = pd.read_csv(datafile)
        if "Unnamed: 0" in self.dataset.columns:
            self.dataset.drop(columns=["Unnamed: 0"],inplace=True)
        self.ALPHABET = ['A','C','G','T']
        self.seqlen = seqlen
        self.kfolds = kfolds
        self.curfold = curfold
        self.seqs = self.transform_sequences(
            self.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).to_numpy())
        self.perm = torch.randperm(self.seqs.shape[0])
        self.noofbuckets = noofbuckets
        self.y_true = []
        for i in self.dataset['Wavelen']:
            if  i <= 590:
                self.y_true.append('G')
            if i > 590 and i <= 660:
                self.y_true.append('R')
            if i > 660 and i <= 800:
                self.y_true.append('VR')
            if i > 800 :
                self.y_true.append('N')

        self.list_st = []
        seqGr = StratifiedKFold(n_splits=self.kfolds)
        for i, (train_index, test_index) in enumerate(seqGr.split(self.seqs,self.y_true)):
            self.list_st.append((train_index,test_index))

    def updatefold(self,curfold):
        self.curfold = curfold

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
        
    def data_loaders(self, batch_size, split=(0.85, 0.1)):
        
        Wavelen = self.dataset['Wavelen'].to_numpy(dtype="float").reshape(-1,1) 
        localII = self.dataset['LII'].to_numpy(dtype="float").reshape(-1,1)
        attribs = np.append(Wavelen, localII, axis=1)

        wavelenBin = AttributeProbabilityBin(Wavelen, self.noofbuckets, (450, 800))
        liiBin = AttributeProbabilityBin(localII, self.noofbuckets, (1, 8))

        nval = self.seqs.shape[0]

        foldsize = nval//self.kfolds
        splitstart = foldsize * self.curfold
        splitend = foldsize + splitstart
          
        train_ds = TensorDataset(
            torch.from_numpy(self.seqs[self.list_st[self.curfold][0],:,:]),
            torch.from_numpy(attribs[self.list_st[self.curfold][0],:]),
            #torch.from_numpy(localII[perm[:split1],:])
        )

        test_ds = TensorDataset(
            torch.from_numpy(self.seqs[self.perm[self.list_st[self.curfold][1]],:,:]),
            torch.from_numpy(attribs[self.perm[self.list_st[self.curfold][1]],:]),
            #torch.from_numpy(localII[perm[split1:split1+split2],:])
        )
        
        val_ds = TensorDataset(
            torch.from_numpy(self.seqs[self.perm[self.list_st[self.curfold][1]],:,:]),
            torch.from_numpy(self.seqs[self.perm[self.list_st[self.curfold][1]],:]),
            #torch.from_numpy(localII[perm[split1+split2:],:])
        )

        valdict = self.perm[splitstart:splitend:]
        np.save('utils/valdict.npy',valdict)

        print(len(train_ds),len(test_ds),len(val_ds))
        
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True
            )
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False
            )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False
            )
        
        print(len(train_dl),len(test_dl),len(val_dl))
        
        return train_dl, test_dl, val_dl, [wavelenBin, liiBin]
