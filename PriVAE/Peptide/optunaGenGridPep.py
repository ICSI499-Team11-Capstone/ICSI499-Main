
import os
import time
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sequenceModelPep import SequenceModel
from sequenceTrainer import SequenceTrainer
from sequenceDatasetPep import SequenceDataset
from plotRun import genPlotForRun
from embedder import Embedder
import optuna
import json
import time
import joblib


#CPU or GPU?
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

batchSize = 32
numEpochs = 1000
betas = [0.007, 0.007]
gammas = [0,0]#[0.5890989775345152, 0.5890989775345152]
deltas = [1, 1]
dropout = [0,0]
latentDims= [32,32]
lstmLayers = [1,1]
hiddenSize = [16,16]
knnSizes = [25, 25]
trainTestSplit = (0.85, 0.15)
weighted = False
purity_interval = 100





#Post processing of log data, avarages metrics for each epoch
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
     "latentDims": latentDims, "lstmLayers": lstmLayers, "dropout":dropout, "hiddenSize":hiddenSize, "knnSizes": knnSizes}


def objective(trial):
    # Hyperparameters to tune
    start_time = time.time()
    hidden_size = trial.suggest_int('hiddenSize', paramDict['hiddenSize'][0],paramDict['hiddenSize'][1])
    latent_dim = trial.suggest_int('latentDims', paramDict['latentDims'][0],paramDict['latentDims'][1])
    lstm_layers = trial.suggest_int('lstmLayers', paramDict['lstmLayers'][0],paramDict['lstmLayers'][1])
    knn_size = trial.suggest_int('knnSizes', paramDict['knnSizes'][0],paramDict['knnSizes'][1])
    beta = trial.suggest_float('beta', paramDict['beta'][0],paramDict['beta'][1])
    gamma = trial.suggest_float('gamma', paramDict['gamma'][0], paramDict['gamma'][1])
    delta = trial.suggest_float('delta', paramDict['delta'][0],paramDict['delta'][1])
    dropout = trial.suggest_float('dropout', paramDict['dropout'][0],paramDict['dropout'][1])
    
    params = {"beta": float(beta), "gamma": float(gamma), "delta": float(delta), 
     "latentDims": int(latent_dim), "lstmLayers": int(lstm_layers), "dropout":float(dropout), "hiddenSize":int(hidden_size), "knnSize": int(knn_size)}
    print("params     ", params)
    ########################
    embedder = Embedder()
    data = SequenceDataset(split=trainTestSplit, embedder=embedder)
    print(" -----    padsize    ", data.pad_size)
    model = SequenceModel(hidden_layers=params["lstmLayers"], emb_dim=params["latentDims"], dropout=params["dropout"], hidden_size=params["hiddenSize"], seq_len=data.pad_size)#+2)
    filename = "a" + str(params["latentDims"]) + "lds"+str(params["latentDims"])+"b"+str(params["beta"])+"g" +str(params["gamma"])+"d"+str(params["delta"])+"h"+str(params["hiddenSize"])
    if torch.cuda.is_available(): 
        print('cuda available')
        model.cuda()
    trainer = SequenceTrainer(data, model, beta=params["beta"], gamma=params["gamma"], delta=params["delta"], knn_size=params["knnSize"], logTerms=True, IICorVsEpoch=True,)
    if torch.cuda.is_available(): 
        trainer.cuda()
    
    #train model, use internal logging
    print("Training Model")
    # trainer.train_model(batchSize, numEpochs, log=False)


    #train model, using weighted loss function
    distence ,distence_m,correlation, smoothness_train, smoothness_val = trainer.train_model(batchSize, numEpochs,filename,params, log=False, weightedLoss=weighted, purity_interval=purity_interval)

    #save the model
    b = list(params.values())
    app_name = '_'.join(str(x) for x in b) 
    torch.save(model, "./models/optuna/" + filename+ app_name + ".pt")
    print("Avging stats for batches")
    #training accuricies at each epoch
    tl = averageCols(trainer.trainList) 
    #validation accuricies at each epoch
    vl = averageCols(trainer.validList)

    

    print("Saving file")
    par = np.array([ params["beta"], params["gamma"], params["delta"], 
        params["latentDims"], params["lstmLayers"], params["dropout"], params["hiddenSize"], purity_interval])
    
    np.savez("./runs/optuna/" + filename + app_name + ".npz", par=par, tl=tl, vl=vl, distence=distence,distence_m=distence_m,correlation=correlation,correlation_valid=None,trainLabel = data.train_label,validLabel = data.val_label)

    genPlotForRun(runsPath="./runs/optuna/", run=filename + app_name + ".npz", graphsPath="./graphs/optuna/", graph=filename + app_name + ".png")
    

    train_accuracy = tl[-1, 5]
    train_purity = tl[:, 8]
    train_temp = np.where(train_purity > 0)
    train_pur = train_purity[train_temp][-1]
    train_smoothness = np.sum(smoothness_train)
    #######
    accuracy = vl[-1, 5]
    purity = vl[:, 8]
    temp = np.where(purity > 0)
    pur = purity[temp][-1]
    smoothness = np.sum(smoothness_val)
    run_time = time.time() - start_time
    ##### add logs
    with open("./optuna_live_results.csv","a") as f:
        f.write(f"{app_name},{train_accuracy},{train_pur},{train_smoothness},{accuracy},{pur},{smoothness},{run_time}\n")
    #####
    print("hyperopt      ", sampler.hyperopt_parameters())
    joblib.dump(sampler.hyperopt_parameters(), 'optuna_sampler_hyperopt.pkl')
    # with open("./optuna_live_hyperopt.csv","a") as f:
    #     f.write(f"{app_name},{accuracy},{pur},{smoothness},{run_time}\n")
    return accuracy, pur, smoothness



n_trials = 10

##
a = list(paramDict.keys())
par_names = '_'.join(str(x) for x in a) 
with open("./optuna_live_results.csv","w") as f:
    # f.write("name,accuracy,purity,smoothness\n")
    f.write(f"name({par_names}),train accuracy,train purity,train smoothness,val accuracy,val purity,val smoothness,time(seconds)\n")
##
if os.path.isfile('./optuna_sampler_hyperopt.pkl'):
    print("  --  use saved hyperopts for sampler --  ")
    hyperopts = joblib.load('optuna_sampler_hyperopt.pkl')
    sampler = optuna.samplers.TPESampler(**hyperopts)
else:
    sampler = optuna.samplers.TPESampler()
study = optuna.create_study(directions=["maximize","minimize"], sampler=sampler)
study.optimize(objective, n_trials=n_trials)

print("Best Trials:", study.best_trials)
for i in study.best_trials:
    print("--------------------------------------------------------------")
    print("Best Trial pars:", i.params)
    print("Best Trial values:", i.values)
    print("--------------------------------------------------------------")

#### 
with open("./optuna_pareto_outputs.txt", "w") as f:
    for i in study.best_trials:
        # print("--------------------------------------------------------------")
        # print("Best Trial pars:", i.params)
        # print("Best Trial values:", i.values)
        # print("--------------------------------------------------------------")
        f.write(json.dumps(i.params))
        f.write("\n")
        vals = '_'.join(str(x) for x in i.values)
        f.write(vals)
        f.write("\n")
        f.write("-------------------------------------------")
        f.write("\n")
# print("Best Hyperparameters:", study.best_params)