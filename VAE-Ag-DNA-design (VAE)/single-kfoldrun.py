
# James Oswald 1/12/20

#This file generates logs of runs with various parameters

import os
import time
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import ParameterGrid

from sequenceModel import SequenceModel
from sequenceTrainer import SequenceTrainer
from kfoldDataset import KFoldDataset
from kfoldPlotter import Plotter

#CPU or GPU?
os.environ["CUDA_VISIBLE_DEVICES"]=""

batchSize = 32

numEpochs = 2000
# alphas = [0.005, 0.01, 0.02]
# betas = [0.006, 0.007, 0.008]
# gammas = [1.0]
# deltas = [1.0]
# dropout =  [0]
# latentDims= [17,19]
# lstmLayers = [1]
# hiddenSize = [13,15]
kfolds = 10
weighted = True

param_dict_1 = {"alpha":0.02,"beta": 0.08, "gamma": 1.0, "delta": 1.0,
        "latentDims": 19, "lstmLayers": 1, "dropout":0, "hiddenSize":13}

param_dict_2 = {"alpha":0.01,"beta": 0.07, "gamma": 1.0, "delta": 1.0,
        "latentDims": 19, "lstmLayers": 1, "dropout":0, "hiddenSize":13}

param_dict_3 = {"alpha":0.02,"beta": 0.06, "gamma": 1.0, "delta": 1.0,
        "latentDims": 17, "lstmLayers": 1, "dropout":0, "hiddenSize":15}

param_dict_4 = {"alpha":0.005,"beta": 0.08, "gamma": 1.0, "delta": 1.0,
        "latentDims": 19, "lstmLayers": 1, "dropout":0, "hiddenSize":13}

param_dict_list = [param_dict_1, param_dict_2, param_dict_3, param_dict_4]


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


for params in param_dict_list:

    paramDict = params

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
        trainer = SequenceTrainer(data, model, alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"], delta=params["delta"], logTerms=True, IICorVsEpoch=True)
        if torch.cuda.is_available(): 
            trainer.cuda()

        #train model, use internal logging
        print("Training Model")
        trainer.train_model(batchSize, numEpochs, log=False, weightedLoss=weighted)

        filename = "a" + str(params["latentDims"]) + "lds"+str(params["latentDims"])+"b"+str(params["beta"])+"g" +str(params["gamma"])+"d"+str(params["delta"])+"h"+str(params["hiddenSize"]) + "fold" + str(i)
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

        print("Computing final correlations")
        #Corellation on II at every 50 epochs, if it exists
        wldims = trainer.WLCorList if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))
        liidims = trainer.LIICorList if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))

        print("Saving file")
        par = np.array([params["alpha"], params["beta"], params["gamma"], params["delta"], 
            params["latentDims"], params["lstmLayers"], params["dropout"], params["hiddenSize"]])
        
        if weighted:
            np.savez("./runs/weighted/kfold/" + filename + ".npz", par=par, tl=tl, vl=vl, wldims=wldims, liidims=liidims)
        else:
            np.savez("./runs/kfold/" + filename + ".npz", par=par, tl=tl, vl=vl, wldims=wldims, liidims=liidims)
        runfiles.append(filename + ".npz")
        plotter.genFigure(filename + ".npz", filename + ".png")
    plotter.genAvgFigure(runfiles, "lds"+str(params["latentDims"])+"b"+str(params["beta"])+"g" +str(params["gamma"])+"d"+str(params["delta"])+"h"+str(params["hiddenSize"]) +"_avg" + ".png")
