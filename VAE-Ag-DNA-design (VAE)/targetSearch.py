
# James Oswald 1/12/20

#This file generates logs of runs with various parameters

import os
import time
import numpy as np
import torch
from sequenceModel import SequenceModel
from sequenceTrainer import SequenceTrainer
from sequenceDataset import SequenceDataset

#CPU or GPU?
os.environ['CUDA_VISIBLE_DEVICES']=""

batchSize = 32

numEpochs = 2000
beta =          [ 0.1, 0.01,0.001,0.001]
gamma =         [   3,    3,    3,    3]
delta =         [   1,    1,    1,    1]
dropout =       [   0,    0,    0,    0]
latentDims =    [   8,    8,   10,   10]
lstmLayers =    [   1,    1,    1,    1]
hiddenSize =    [   5,   10,    5,    8]


#Post processing of log data, avarages metrics for each epoch
def averageCols(logMat):
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
    return rv

for i in range(len(beta)):   #gridsearch
    #set up the model and trainer
    model = SequenceModel(hidden_layers=lstmLayers[i], emb_dim=latentDims[i], dropout=dropout[i], hidden_size=hiddenSize[i]) 
    data = SequenceDataset()
    if torch.cuda.is_available(): 
        print('cuda available')
        model.cuda()
    trainer = SequenceTrainer(data, model, beta=beta[i], gamma=gamma[i], delta=delta[i], logTerms=True, IICorVsEpoch=True)
    if torch.cuda.is_available(): 
        trainer.cuda()
    
    #train model, use internal logging
    print("Training Model")
    trainer.train_model(batchSize, numEpochs, log=False)

    unixTimestamp = str(int(time.time()))
    #save the model
    torch.save(model, "./models/" + unixTimestamp + ".pt")

    print("Avging stats for batches")
    #training accuricies at each epoch
    tl = averageCols(trainer.trainList)
    #validation accuricies at each epoch
    vl = averageCols(trainer.validList)

    print("Computing final correlations")
    #Corellation on II at every 50 epochs, if it exists
    wldims = trainer.WLCorList if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))
    liidims = trainer.LIICorList if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))

    print("Saving file")
    par = np.array([beta[i], gamma[i], delta[i], 
        latentDims[i], lstmLayers[i], dropout[i], hiddenSize[i]])
    np.savez("./runs/" + unixTimestamp + ".npz", par=par, tl=tl, vl=vl, wldims=wldims, liidims=liidims)