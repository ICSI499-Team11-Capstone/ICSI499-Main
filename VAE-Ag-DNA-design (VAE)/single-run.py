# James Oswald 1/12/20

#This file generates logs of runs with various parameters
import sequenceDataset as sd
import os
import sys
import numpy as np
import torch
import pandas as pd
import json
from sklearn.model_selection import ParameterGrid
from sequenceModel import SequenceModel
from sequenceTrainer import SequenceTrainer
from sequenceDataset import SequenceDataset
from plotRun import genPlotForRun

import pcaVisualization

import sys

#CPU or GPU?
os.environ["CUDA_VISIBLE_DEVICES"]=""

def process_parameters(filepath: str = 'hyperparameters.json') -> dict:
    try:
        with open(filepath) as f:
            data = json.load(f)
        return data['Parameters']
    except Exception:
        print(f"Cannot process parameter file, please make sure {filename} is correctly configured.")
        sys.exit(1)


def process_data_file(path_to_dataset: str, sequence_length=10, prepended_name="processed", path_to_put=None, return_path=False):
    """Takes in a filepath to desired dataset and the sequence length of the sequences for that dataset,
    saves .npz file with arrays for one hot encoded sequences, array of wavelengths and array of local
    integrated intensities"""
    data = sd.SequenceDataset(path_to_dataset, sequence_length)
    ohe_sequences = data.transform_sequences(data.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).
                                             to_numpy())  #One hot encodings in the form ['A', 'C', 'G', 'T']
    Wavelen = np.array(data.dataset['Wavelen'])
    LII = np.array(data.dataset['LII'])

    return {'Wavelen':Wavelen, "LII":LII,"ohe":ohe_sequences}
    
# batchSize = 32

# numEpochs = 2000
# alphas = [0.005, 0.02]
# betas = [0.006, 0.008]
# gammas = [1]
# deltas = [1]
# dropout =  [0]
# latentDims= [16,17,19]
# lstmLayers = [1]
# hiddenSize = [13,14,15] #features inside lstm

# train = 1.0
# trainTestSplit = (train, 1-train)
# weighted = True

params = process_parameters()

#Post processing of log data, avarages metrics for each epoch
def averageCols(logMat, numEpochs=params.get("number of epochs")):
    # print("logMat")
    print(logMat.shape)
    rv = np.zeros((numEpochs, 6))
    for epoch in range(numEpochs):
        for col in range(1, 6):
            num = 0
            for row in range(logMat.shape[0]):
                if(logMat[row, 0] == epoch):
                    rv[epoch, col] += logMat[row, col]
                    num += 1
            rv[epoch, col] /= num
        rv[epoch,0] = epoch
    # print("rv", rv)
    return rv

batchSize = params.get('batch size')
numEpochs = params.get('number of epochs')
trainTestSplit = (params.get('training percentage'), 1.0 - params.get('training percentage'))
weighted = params.get('weighting')

#set up the model and trainer
model = SequenceModel(hidden_layers=params["number of LSTM layers"], emb_dim=params["number of latent dimensions"],
lin_dim=params["number of linear dimensions"], dropout=params["dropout"], hidden_size=params["hidden size"]) 
data = SequenceDataset(datafile=params['path to training data'], split=trainTestSplit)
if torch.cuda.is_available(): 
    print('cuda available')
    model.cuda()
trainer = SequenceTrainer(data, model, alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"], delta=params["delta"], logTerms=True, IICorVsEpoch=True)
if torch.cuda.is_available(): 
    trainer.cuda()

#train model, use internal logging
print("Training Model")
#trainer.train_model(batchSize, numEpochs, log=False)

#train model, using weighted loss function
trainer.train_model(batchSize, numEpochs, log=False, weightedLoss=weighted)

filename = "a" + str(params["alpha"]) + "lds"+str(params["number of latent dimensions"])+"b"+str(params["beta"])+"g" +str(params["gamma"])+"d"+str(params["delta"])+"h"+str(params["hidden size"])+"lin_dim"+str(params["number of linear dimensions"])
#save the model
if weighted:
    torch.save(model, "./models/weighted/" + filename + ".pt")
else:
    torch.save(model, "./models/" + filename + ".pt")
print("Avging stats for batches")
#training accuricies at each epoch
tl = averageCols(trainer.trainList)
#validation accuricies at each epoch
vl = averageCols(trainer.validList)

print("Computing final correlations")
#Corellation on II at every 50 epochs, if it exists
wldims = trainer.WLCorList if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))
liidims = trainer.LIICorList if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))

npdata = process_data_file('data-and-cleaning/cleandata.csv')
res = pcaVisualization.conduct_visualizations(npdata, model, (False, False, True))

print("Saving file")

par = np.array([params["alpha"], params["beta"], params["gamma"], params["delta"], 
    params["number of latent dimensions"], params["number of LSTM layers"], params["dropout"], params["hidden size"], params["number of linear dimensions"]])
if weighted:
    np.savez("./runs/weighted/" + filename + ".npz", par=par, tl=tl, vl=vl, wldims=wldims, liidims=liidims)
else:
    np.savez("./runs/" + filename + ".npz", par=par, tl=tl, vl=vl, wldims=wldims, liidims=liidims)

if weighted:
    genPlotForRun(','.join(res[:5])+'\n'+','.join(res[5:]),runsPath="./runs/weighted/", run=filename + ".npz", graphsPath="./graphs/weighted", graph=filename + ".png")
else:
    genPlotForRun(','.join(res[:5])+'\n'+','.join(res[5:]),runsPath="./runs/", run=filename + ".npz", graphsPath="./graphs", graph=filename + ".png")

print(res)
