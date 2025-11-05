import numpy as np
import matplotlib.pyplot as plt
import argparse
import os.path as path

#takes a file path and returns a matplot figure object

class Plotter:

    def __init__(self, metricsfile="", graphname=""):
        self.metrics = metricsfile
        self.outputFigure = graphname
    
    def plot(self):
        f = self.genAvgFigure(path.join(".","runs",self.metrics))
        f.savefig(path.join(".","graphs",self.outputFigure))
    
    def genAvgFigure(self, filePaths, outputFile):
        data =[];
        for filePath in filePaths:
            data.append(np.load(filePath))

        beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo = data[0]["par"]
        
        rLossM = []
        kdLossM = []
        regLossM = []
        lossM = []
        accM = []
        purM = []
        for datasub in data:
            trainMat = datasub["tl"]
            rLossM = np.append(rLossM, trainMat[:,1])
            kdLossM = np.append(kdLossM, trainMat[:,2])
            regLossM = np.append(regLossM, trainMat[:,3])
            lossM = np.append(lossM, trainMat[:,4])
            accM = np.append(accM, trainMat[:,5])
            purM = np.append(purM, trainMat[:,8])

        epochsT = data[0]["tl"][:,0]
        rLossT = np.mean(np.reshape(rLossM, (len(data), len(epochsT))), axis=0)
        kdLossT = np.mean(np.reshape(kdLossM, (len(data), len(epochsT))), axis=0)
        regLossT =  np.mean(np.reshape(regLossM, (len(data), len(epochsT))), axis=0)
        lossT =  np.mean(np.reshape(lossM, (len(data), len(epochsT))), axis=0)
        accT =  np.mean(np.reshape(accM, (len(data),len(epochsT))), axis=0)
        purT =  np.mean(np.reshape(purM, (len(data),len(epochsT))), axis=0)

        tempT = np.where(purT > 0)
        purityT = purT[tempT]

        rLossM = []
        kdLossM = []
        regLossM = []
        lossM = []
        accM = []
        purM = []
        for datasub in data:
            validMat = data[0]["vl"]
            rLossM = np.append(rLossM, validMat[:,1])
            kdLossM = np.append(kdLossM, validMat[:,2])
            regLossM = np.append(regLossM, validMat[:,3])
            lossM = np.append(lossM, validMat[:,4])
            accM = np.append(accM, validMat[:,5])
            purM = np.append(purM, validMat[:,8])
        
        #validation losses vs epoch
        epochsV = data[0]["vl"][:,0]
        rLossV = np.mean(np.reshape(rLossM, (len(data), len(epochsV))), axis=0)
        kdLossV = np.mean(np.reshape(kdLossM, (len(data), len(epochsV))), axis=0)
        regLossV =  np.mean(np.reshape(regLossM, (len(data), len(epochsV))), axis=0)
        lossV =  np.mean(np.reshape(lossM, (len(data), len(epochsV))), axis=0)
        accV =  np.mean(np.reshape(accM, (len(data), len(epochsV))), axis=0)
        purV =  np.mean(np.reshape(purM, (len(data), len(epochsV))), axis=0)

        tempV = np.where(purV > 0)
        purityV = purV[tempV]

        fvaccuray = accV[len(epochsV)-1]
        #corelations with each dimention vs epoch
        wldims = data[0]["wldims"]
        lidims = data[0]["liidims"]
        wldimsM = np.reshape(wldims, np.shape(wldims)+(1,))
        lidimsM = np.reshape(wldims, np.shape(lidims)+(1,))

        for datasub in data:
            wldims = datasub["wldims"]
            lidims = datasub["liidims"]
            wldimsM = np.concatenate((wldimsM, np.reshape(wldims, np.shape(wldims)+(1,))), axis=2)
            lidimsM = np.concatenate((lidimsM, np.reshape(lidims, np.shape(lidims)+(1,))), axis=2)

        wldimCors = np.mean(wldimsM, axis=2) 
        lidimCors = np.mean(lidimsM, axis=2) 
        dimRange = [j * 50 for j in range(wldimCors.shape[0])]
        finalWIIcor = wldimCors[:,0][len(wldimCors[:,0])-1]
        finalLIIcor = wldimCors[:,1][len(wldimCors[:,1])-1]  
        #fig, ax = plt.subplots(3)
        fig, ax = plt.subplots(6)
        fig.set_size_inches(8, 7)
        fig.suptitle(r"$\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo), fontsize=12)

        ax[0].plot(epochsT, rLossT, label="r Loss")
        ax[0].plot(epochsT, kdLossT, label="kld Loss")
        ax[0].plot(epochsT, regLossT, label="reg Loss")
        ax[0].plot(epochsT, lossT, label="Training Loss")
        ax[0].plot(epochsV, lossV, label="Valid Loss", ls="--")
        ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax[0].set(xlabel="epoch", ylabel="Training Loss")
        ax[0].label_outer()

        ax[1].plot(epochsT, rLossV, label="r Loss")
        ax[1].plot(epochsT, kdLossV, label="kld Loss")
        ax[1].plot(epochsT, regLossV, label="reg Loss")
        ax[1].plot(epochsV, lossV, label="Valid Loss")
        ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax[1].set(xlabel="epoch", ylabel="Validation Loss")
        ax[1].label_outer()

        ax[2].plot(epochsT, accT, label="Train Accuracy")
        ax[2].plot(epochsV, accV, label="Valid Accuracy")
        ax[2].text(400, 0.5, "V accuracy="+ str(fvaccuray))
        ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax[2].set(xlabel="epoch",ylabel="Accuracy")
        ax[2].label_outer()

        for i in range(int(latentDims)):
            ax[3].plot(dimRange, wldimCors[:,i], label=str(i), ls="-" if i == 0 else "--")
            ax[4].plot(dimRange, lidimCors[:,i], label=str(i), ls="-" if i == 1 else "--")
        ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
        ax[3].set(xlabel="epoch",ylabel="WL Cor")
        ax[3].text(400, 0.2, "WL z0 corr="+ str(finalWIIcor))
        ax[3].label_outer()
        ax[4].legend(loc="center left", bbox_to_anchor=(1, 0.3), ncol=2)
        ax[4].set(xlabel="epoch",ylabel="LII Cor")
        ax[4].text(400, 0.1, "LII z1 corr="+ str(finalLIIcor))
        ax[4].label_outer()

        ax[5].plot(tempT[0], purityT, label="Train purity")
        ax[5].plot(tempV[0], purityV, label="Validation purity", ls="--")
        ax[5].set_ylabel("Purity")
        ax[5].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        fig.subplots_adjust(right=0.75)
        fig.savefig(path.join(".","graphs",outputFile))
        return fig


filepaths = ["./runs/lds19b0.007g1.0d1.0h12fold0.npz",
             "./runs/lds19b0.007g1.0d1.0h12fold1.npz",
             "./runs/lds19b0.007g1.0d1.0h12fold2.npz",
             "./runs/lds19b0.007g1.0d1.0h12fold3.npz",
             "./runs/lds19b0.007g1.0d1.0h12fold4.npz",
             "./runs/lds19b0.007g1.0d1.0h12fold5.npz",
             "./runs/lds19b0.007g1.0d1.0h12fold6.npz",
             "./runs/lds19b0.007g1.0d1.0h12fold7.npz",
             "./runs/lds19b0.007g1.0d1.0h12fold8.npz",
             "./runs/lds19b0.007g1.0d1.0h12fold9.npz",
            ]

plotter = Plotter()
plotter.genAvgFigure(filepaths,"lds19b0.007g1.0d1.0h12_avg.png")