
import os
import time
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sequenceModel import SequenceModel
from kfoldsequenceTrainer import SequenceTrainer
from kfoldDataset import KFoldDataset
from kfoldPlotter import Plotter

#CPU or GPU?
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

batchSize = 32
numEpochs = 20
alphas = [0.005]#, 0.01, 0.02]
betas = [0.005]#, 0.007, 0.008]
gammas = [0.6]
deltas = [1.0]
dropout =  [0]
latentDims= [66]#,19]
lstmLayers = [1]
hiddenSize = [6]#,15]
kfolds = 2
weighted = False
purity_interval = 2


def averageCols(logMat):
    # print("logMat")
    print(logMat.shape)
    dim = logMat.shape[1]
    rv = np.zeros((numEpochs, dim))
    for epoch in range(numEpochs):
        for col in range(1, dim):
            num = 0
            for row in range(logMat.shape[0]):
                if(logMat[row, 0] == epoch):
                    rv[epoch, col] += logMat[row, col]
                    num += 1
            rv[epoch, col] /= num
        rv[epoch,0] = epoch
    # print("rv", rv)
    return rv



paramDict = {"beta": betas, "gamma": gammas, "delta": deltas,
     "latentDims": latentDims, "lstmLayers": lstmLayers, "dropout":dropout, "hiddenSize":hiddenSize}
for params in list(ParameterGrid(paramDict)):   #gridsearch
    #set up the model and trainer
    data = KFoldDataset(kfolds=kfolds)
    runfiles = []

    plotter = Plotter()
    if weighted:
        plotter = Plotter(metricsfolder="./runs/weighted/kfold",graphsfolder="./graphs/weighted/kfold")
    else:
        plotter = Plotter(metricsfolder="./runs/kfold",graphsfolder="./graphs/kfold")

    for i in range(kfolds):
        data.updatefold(i)
        model = SequenceModel(hidden_layers=params["lstmLayers"], emb_dim=params["latentDims"], dropout=params["dropout"], hidden_size=params["hiddenSize"]) 
        if torch.cuda.is_available(): 
            print('cuda available')
            model.cuda()
        trainer = SequenceTrainer(data, model, beta=params["beta"], gamma=params["gamma"], delta=params["delta"], logTerms=True, IICorVsEpoch=True)
        if torch.cuda.is_available(): 
            trainer.cuda()
        filename = "a" + str(params["latentDims"]) + "lds"+str(params["latentDims"])+"b"+str(params["beta"])+"g" +str(params["gamma"])+"d"+str(params["delta"])+"h"+str(params["hiddenSize"]) + "fold" + str(i)

        #train model, use internal logging
        print("Training Model")
        distence ,distence_m,correlation,correlation_valid = trainer.train_model(batchSize, numEpochs,filename,params, log=False, weightedLoss=weighted, purity_interval=purity_interval)

        unixTimestamp = str(int(time.time()))

        unixTimestamp = str(int(time.time()))
        #save the model
        if weighted:
            torch.save(model, "./models/weighted/kfold/" + filename + ".pt")
        else:
            torch.save(model, "./models/kfold/" + filename + ".pt")
    
        print("Avging stats for batches")
        #training accuricies at each epoch
        tl = averageCols(trainer.trainList)
        #validation accuricies at each epoch
        vl = averageCols(trainer.validList)

        
        print("Saving file")
        par = np.array([params["beta"], params["gamma"], params["delta"], 
            params["latentDims"], params["lstmLayers"], params["dropout"], params["hiddenSize"]])
        
        if weighted:
            np.savez("./runs/weighted/kfold/" + filename + ".npz", par=par, tl=tl, vl=vl,distence=distence,distence_m=distence_m,correlation=correlation,correlation_valid=correlation_valid,trainLabel = data.train_label,validLabel = data.val_label)
        else:
            np.savez("./runs/kfold/" + filename + ".npz", par=par, tl=tl, vl=vl,  distence=distence,distence_m=distence_m,correlation=correlation,correlation_valid=correlation_valid,trainLabel = data.train_label,validLabel = data.val_label)
        runfiles.append(filename + ".npz")
        plotter.genFigure(filename + ".npz", filename + ".png")
    plotter.genAvgFigure(runfiles, "lds"+str(params["latentDims"])+"b"+str(params["beta"])+"g" +str(params["gamma"])+"d"+str(params["delta"])+"h"+str(params["hiddenSize"]) +"_avg" + ".png")
