import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import sys


class RegressionDataset(Dataset):
    def __init__(self, X_data, emb_data, labels, lengths, class_labels):
        self.X_data = X_data
        self.emb_data = emb_data
        self.labels = labels
        self.lengths = lengths
        self.class_labels = class_labels

    def __getitem__(self, index):
        return self.X_data[index], self.emb_data[index], self.labels[index], self.lengths[index], self.class_labels[index], index

    def __len__(self):
        return len(self.X_data)

class SequenceDataset:
    def __init__(self, embedder=None, datafile='./data-and-cleaning/peptide_with_activity_labels-May6.xlsx', seqlen=10, split=(0.85, 0.15), noofbuckets=7):
        self.dist_file = pd.read_excel('./data-and-cleaning/L1distance_peptide_with_activity_labels-May6.xlsx').values
        self.embedder = embedder

        if datafile[-4:] == 'xlsx':
            self.dataset = pd.read_excel(datafile)
        else:
            self.dataset = pd.read_csv(datafile)
        if "Unnamed: 0" in self.dataset.columns:
            self.dataset.drop(columns=["Unnamed: 0"], inplace=True)
        self.ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.seqlen = seqlen
        self.split = split
        self.noofbuckets = noofbuckets
        ####
        pad_size = np.max([len(i) for i in self.dataset['Sequence'].to_numpy().tolist()])
        print("sssss    ", pad_size)
        self.pad_size = int(pad_size)
        ######
        self.embedder.set_max_length(self.pad_size)
        self.embedder.set_models()
        
    def transform_sequences(self, seqs):
        # print("seqs   ", seqs[:10])
        # print("split    ", list(seqs[0]))
        # print("split    ", seqs[0].split())
        lengths = [len(i) for i in seqs]
        seqs = [list(i) for i in seqs]
        unravel = [i for k in seqs for i in k]
        unq = np.unique(unravel)
        self.ALPHABET = unq
        # self.ALPHABET.append('nl')
        # print("unq  ", unq)
        # kkk += 1
        enc = OneHotEncoder()
        enc.fit(np.array(self.ALPHABET).reshape(-1, 1))
        # ret = enc.transform(seqs.reshape(-1, 1)).toarray()
        # ret = ret.reshape(-1, self.seqlen, len(self.ALPHABET))
        pad_size = np.max(lengths)
        # print("pad size    ", pad_size)
        ret = []
        ret_emb = []
        for i in range(len(seqs)):
            inp = np.array(seqs[i]).reshape(-1,1)
            # print("inp    ", inp)
            ################################
            encoded = enc.transform(inp).toarray()
            # print("encoded   ", encoded.shape)
            for j in range(pad_size-encoded.shape[0]):
                app = np.array([-1]*len(self.ALPHABET)).reshape(1,-1)
                # print("app   ", app.shape)
                encoded = np.concatenate((encoded, app), axis=0)
            # print("encoded 2  ", encoded.shape)
            # print("--------")
            ret.append(encoded)
            #################################
            x = inp.reshape(-1).tolist()
            x = ''.join(x)
            # print("x     ", x, "  ---   ", len(x))
            for j in range(pad_size-len(x)):
                if j == 0:
                    x += '<eos>'
                else:
                    x += '<pad>'
            # print("x22    ", x   ," --   ", len(x))
            encoded = self.embedder.encode(x, max_length=self.pad_size)
            # print("enc    ", encoded.shape)
            # print("encccc    ", encoded.argmax(dim=-1))
            # print("--------------------------------------------------------------")
            ret_emb.append(encoded)
            #################################
            # ret.append(inp.reshape(-1).tolist())
            ##
        # print("ret    ", ret)
        # kkkk += 6
        ret = np.array(ret)
        ret_emb = np.array(ret_emb)
        # print("ret   ", ret.shape)
        # print("rert0    ", ret[0])
        # kkk += 1
        return ret, ret_emb, np.array(lengths)
        
    def data_loaders(self, batch_size, drop_indices):
        # Transform sequences and labels
        seqs, seqs_emb, seq_lengths = self.transform_sequences(
            self.dataset['Sequence'].to_numpy().tolist())
        Dlables = self.dataset['Label'].to_numpy(dtype="float").reshape(-1, 1)
        class_lables = self.dataset['class'].values
        #Dlables = np.zeros((len(seqs),1))

        # print("seqs    ",seqs[:10])
        
        if self.split[1] != 0:
            print("---------------   1")   
            indices = np.arange(len(seqs))
            self.train_seq, self.val_seq, self.train_label, self.val_label,self.train_clabel, self.val_clabel, train_data, val_data, self.train_ind, self.val_ind, self.train_lengths, self.val_lengths = train_test_split(
                seqs, Dlables, class_lables, self.dataset, indices, seq_lengths, test_size=self.split[1], random_state=42)
            self.train_emb = seqs_emb[self.train_ind,:]
            self.val_emb = seqs_emb[self.val_ind,:]

        else:
            self.train_seq = seqs
            self.val_seq = np.zeros_like(seqs)
            self.train_label = Dlables
            self.val_label = np.zeros_like(Dlables)
            self.train_clabel = class_lables
            self.val_clabel = class_lables
            train_data = self.dataset.copy()
            val_data = pd.DataFrame()  # Empty dataframe for validation since no split
            self.train_lengths = seq_lengths
            self.val_lengths = np.array([])


        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)


        d1 = self.dist_file[self.train_ind, :]
        d1 = d1[:, self.train_ind]
        # print("d1    ", d1)
        # print("ddddd    ", self.dist_file[0,:])
        d1 = d1.astype(np.float64)
        if len(self.val_ind) == 0:
            d2 = np.array([])
        else:
            d2 = self.dist_file[self.val_ind, :]
            d2 = d2[:, self.val_ind]
        
        d2 = d2.astype(np.float64)

        train_ds2 = RegressionDataset(
            torch.from_numpy(self.train_seq), 
            torch.from_numpy(self.train_emb),
            torch.from_numpy(self.train_label),
            torch.from_numpy(self.train_lengths),
            self.train_clabel
        )
        val_ds2 = RegressionDataset(
            torch.from_numpy(self.val_seq), 
            torch.from_numpy(self.val_emb),
            torch.from_numpy(self.val_label),
            torch.from_numpy(self.val_lengths),
            self.val_clabel
        )

        # # Create DataLoaders

        train_dl2 = DataLoader(
            train_ds2,
            batch_size=batch_size,
            shuffle=False
        )

        val_dl2 = DataLoader(
            val_ds2,
            batch_size=batch_size,
            shuffle=False
        )

        print("d1    ", d1.shape)
        return train_dl2, val_dl2, torch.from_numpy(d1), torch.from_numpy(d2), None
        

