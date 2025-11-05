from matplotlib.pyplot import axis
import torch
import numpy as np
import pandas as pd
from sequenceModel import SequenceModel
import sequenceDataset as sd
import time
import os
import sys
import json
import filter_sampled_sequences as filt

def process_data_file(path_to_dataset: str, sequence_length=10, prepended_name="processed", path_to_put=None, return_path=False):
    """Takes in a filepath to desired dataset and the sequence length of the sequences for that dataset,
    saves .npz file with arrays for one hot encoded sequences, array of wavelengths and array of local
    integrated intensities"""
    data = sd.SequenceDataset(path_to_dataset, sequence_length)
    ohe_sequences = data.transform_sequences(data.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).
                                             to_numpy())  #One hot encodings in the form ['A', 'C', 'G', 'T']
    Wavelen = np.array(data.dataset['Wavelen'])
    LII = np.array(data.dataset['LII'])

    if path_to_put is not None:
        file_path = f"{path_to_put}/{prepended_name}-{time.time()}.npz"
    else:
        file_path = f"{prepended_name}-{time.time()}.npz"

    np.savez(file_path, Wavelen=Wavelen, LII=LII,ohe=ohe_sequences)
    if return_path:
        return file_path

process_data_file(path_to_dataset='data-and-cleaning/cleandata.csv')