import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU setups
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class RegressionDataset(Dataset):
    def __init__(self, X_data, labels, c4_labels, distances):
        self.X_data = X_data
        self.labels = labels
        self.c4_labels = c4_labels
        self.distances = distances

    def __getitem__(self, index):
        return self.X_data[index], self.labels[index], self.c4_labels[index], self.distances, index

    def __len__(self):
        return len(self.X_data)



class FixedBatchSampler:
    def __init__(self, y, batch_size, fixed_batches=None):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, "Label array must be 1D"
        self.batch_size = batch_size
        self.y = y

        if fixed_batches is None:
            self.batch_indices = self._generate_fixed_batches()
        else:
            self.batch_indices = fixed_batches

    def _generate_fixed_batches(self):
        """
        Generate fixed batches ensuring each sample is used exactly once.
        - Batches prioritize the same `Class 4` label.
        - If a batch is too small, it fills with the closest labels.
        """
        np.random.seed(42) 

        # Group indices by `Class 4` label
        label_partitions = defaultdict(list) 
        for idx, label in enumerate(self.y): 
            label_partitions[label].append(idx) 

        # Sort labels for consistency
        sorted_labels = sorted(label_partitions.keys()) 

        # Track used samples
        used_indices = set() 
        batches = [] 

        for label in sorted_labels:
            indices = label_partitions[label] 
            remaining = [idx for idx in indices if idx not in used_indices]  

            while remaining:
                batch = remaining[:self.batch_size] 
                remaining = remaining[self.batch_size:] 
                used_indices.update(batch) 

                # If batch is too small, try adding more from similar labels
                if len(batch) < self.batch_size:
                    for similar_label in sorted_labels: # another label that comes after the current one in the sorted list
                        if similar_label == label:
                            continue  

                        extra_indices = [idx for idx in label_partitions[similar_label] if idx not in used_indices] 

                        needed = self.batch_size - len(batch) 
                        batch.extend(extra_indices[:needed]) 
                        used_indices.update(extra_indices[:needed]) # Update the used indices with the new samples added to the batch

                        if len(batch) == self.batch_size:
                            break  

                batches.append(batch)

        # Ensure no samples are left out
        all_indices = set(range(len(self.y)))
        missing_samples = list(all_indices - used_indices)
        if missing_samples:
            print(f"Warning: Some samples were not included in any batch")

            for i in range(0, len(missing_samples), self.batch_size):
                batch = missing_samples[i : i + self.batch_size]
                batches.append(batch)

        return batches

    def __iter__(self):
        for batch in self.batch_indices:
            yield batch

    def __len__(self):
        return len(self.batch_indices)


class SequenceDataset:
    def __init__(self, datafile='./data-and-cleaning/SORTED-250116-shuffled-final-labeled-data-log10.xlsx',
                 seqlen=10, split=(0.85, 0.15), noofbuckets=10):

        self.dist_file = pd.read_excel('./data-and-cleaning/CSdistance-250116-shuffled-final-labeled-data-log10.xlsx').values

        if datafile.endswith('xlsx'):
            self.dataset = pd.read_excel(datafile)
        else:
            self.dataset = pd.read_csv(datafile)

        if "Unnamed: 0" in self.dataset.columns:
            self.dataset.drop(columns=["Unnamed: 0"], inplace=True)

        self.ALPHABET = ['A', 'C', 'G', 'T']
        self.seqlen = seqlen
        self.split = split
        self.noofbuckets = noofbuckets

        self.fixed_train_batches = None
        self.fixed_val_batches = None

    def transform_sequences(self, seqs):
        enc = OneHotEncoder()
        enc.fit(np.array(self.ALPHABET).reshape(-1, 1))
        return enc.transform(seqs.reshape(-1, 1)).toarray().reshape(-1, self.seqlen, len(self.ALPHABET))

    def data_loaders(self, batch_size, drop_indices):
        nBuckets = self.noofbuckets

        seqs = self.transform_sequences(
            self.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).to_numpy())
        Dlables = self.dataset['Label'].to_numpy(dtype="float").reshape(-1, 1)
        c4_lables = self.dataset['class 4'].values  # Keep as string labels

        indices = np.arange(len(seqs))

        if self.split[1] != 0:
            (self.train_seq, self.val_seq, self.train_label, self.val_label,
             self.train_c4_label, self.val_c4_label, train_data, val_data,
             self.train_ind, self.val_ind) = train_test_split(seqs, Dlables, c4_lables, self.dataset, indices,
                                                              test_size=self.split[1], random_state=42)
        else:
            self.train_seq = seqs
            self.val_seq = np.zeros_like(seqs)
            self.train_label = Dlables
            self.train_c4_label = c4_lables
            self.val_label = np.zeros_like(Dlables)
            self.val_c4_label = np.zeros_like(Dlables)
            train_data = self.dataset.copy()
            val_data = pd.DataFrame()
            self.train_ind = np.arange(len(seqs))
            self.val_ind = np.array([])

    
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)

        # train_data.to_excel("train_data-sorted.xlsx", index=False, engine='openpyxl')
        # val_data.to_excel("val_data-sorted.xlsx", index=False, engine='openpyxl')


        d1 = self.dist_file[self.train_ind, :][:, self.train_ind]
        d2 = self.dist_file[self.val_ind, :][:, self.val_ind] if len(self.val_ind) > 0 else np.array([])

        # 🔹 Generate fixed batches strictly based on `Class 4` labels
        train_ds2 = RegressionDataset(torch.from_numpy(self.train_seq),
                                      torch.from_numpy(self.train_label),
                                      self.train_c4_label, 0)
        train_sampler = FixedBatchSampler(self.train_c4_label, batch_size=batch_size, fixed_batches=self.fixed_train_batches)
        train_dl2 = DataLoader(train_ds2, batch_sampler=train_sampler)

        val_ds2 = RegressionDataset(torch.from_numpy(self.val_seq),
                                    torch.from_numpy(self.val_label),
                                    self.val_c4_label, 0)
        val_sampler = FixedBatchSampler(self.val_c4_label, batch_size=batch_size, fixed_batches=self.fixed_val_batches)
        val_dl2 = DataLoader(val_ds2, batch_sampler=val_sampler)

        # ** Collect batch data for saving**
        batch_info_train = []
        batch_info_val = []

        for batch_idx, (_, _, batch_c4_labels, _, batch_indices) in enumerate(train_dl2):
            for i, index in enumerate(batch_indices.numpy()):
                batch_info_train.append([index, batch_idx, batch_c4_labels[i]])

        for batch_idx, (_, _, batch_c4_labels, _, batch_indices) in enumerate(val_dl2):
            for i, index in enumerate(batch_indices.numpy()):
                batch_info_val.append([index, batch_idx, batch_c4_labels[i]])

        # ** Save batch distributions to Excel**
        df_train_batches = pd.DataFrame(batch_info_train, columns=["Sample Index", "Batch Number", "Class 4 Label"])
        df_val_batches = pd.DataFrame(batch_info_val, columns=["Sample Index", "Batch Number", "Class 4 Label"])

        # df_train_batches.to_excel("train_batches.xlsx", index=False, engine="openpyxl")
        # df_val_batches.to_excel("val_batches.xlsx", index=False, engine="openpyxl")


        return train_dl2, val_dl2, torch.from_numpy(d1), torch.from_numpy(d2), None



