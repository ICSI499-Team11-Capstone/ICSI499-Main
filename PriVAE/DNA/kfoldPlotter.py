import numpy as np
import matplotlib.pyplot as plt
import os.path as path

class Plotter:

    def __init__(self, metricsfolder=path.join(".", "runs", "kfold"), graphsfolder=path.join(".", "graphs", "kfold")):
        self.inputfolder = metricsfolder
        self.outputfolder = graphsfolder
        self.avg_corrT = []
        self.avg_corrV = []

    # Generate figure averaging across multiple k-fold runs
    def genAvgFigure(self, inputFiles, outputFigure):
        data = []
        for file in inputFiles:
            data.append(np.load(path.join(self.inputfolder, file)))

        # Extract parameters (assuming they're the same across files)
        beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo = data[0]["par"]

        rLossM, kdLossM, regLossM, lossM, accM, purityM = [], [], [], [], [], []
        for datasub in data:
            trainMat = datasub["tl"]
            rLossM = np.append(rLossM, trainMat[:, 1])
            kdLossM = np.append(kdLossM, trainMat[:, 2])
            regLossM = np.append(regLossM, trainMat[:, 3])
            lossM = np.append(lossM, trainMat[:, 4])
            accM = np.append(accM, trainMat[:, 5])
            purityM = np.append(purityM, trainMat[:, 8])  # Purity column

        epochsT = data[0]["tl"][:, 0]
        rLossT = np.mean(np.reshape(rLossM, (len(data), len(epochsT))), axis=0)
        kdLossT = np.mean(np.reshape(kdLossM, (len(data), len(epochsT))), axis=0)
        regLossT = np.mean(np.reshape(regLossM, (len(data), len(epochsT))), axis=0)
        lossT = np.mean(np.reshape(lossM, (len(data), len(epochsT))), axis=0)
        accT = np.mean(np.reshape(accM, (len(data), len(epochsT))), axis=0)
        purityT = np.mean(np.reshape(purityM, (len(data), len(epochsT))), axis=0)

        tempT = np.where(purityT > 0)
        purityT = purityT[tempT]

        rLossM, kdLossM, regLossM, lossM, accM, purityM = [], [], [], [], [], []
        for datasub in data:
            validMat = datasub["vl"]
            rLossM = np.append(rLossM, validMat[:, 1])
            kdLossM = np.append(kdLossM, validMat[:, 2])
            regLossM = np.append(regLossM, validMat[:, 3])
            lossM = np.append(lossM, validMat[:, 4])
            accM = np.append(accM, validMat[:, 5])
            purityM = np.append(purityM, validMat[:, 8])  # Purity column

        epochsV = data[0]["vl"][:, 0]
        rLossV = np.mean(np.reshape(rLossM, (len(data), len(epochsV))), axis=0)
        kdLossV = np.mean(np.reshape(kdLossM, (len(data), len(epochsV))), axis=0)
        regLossV = np.mean(np.reshape(regLossM, (len(data), len(epochsV))), axis=0)
        lossV = np.mean(np.reshape(lossM, (len(data), len(epochsV))), axis=0)
        accV = np.mean(np.reshape(accM, (len(data), len(epochsV))), axis=0)
        purityV = np.mean(np.reshape(purityM, (len(data), len(epochsV))), axis=0)

        tempV = np.where(purityV > 0)
        purityV = purityV[tempV]

        # Adjust the figure to have 4 subplots
        fig, ax = plt.subplots(4, figsize=(10, 12))
        fig.suptitle(r" $\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) +
                     " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) +
                     " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo), fontsize=12)

        # First subplot: Training losses (log scale)
        ax[0].plot(epochsT, rLossT, label="r Loss (Train)")
        ax[0].plot(epochsT, kdLossT, label="kld Loss (Train)")
        ax[0].plot(epochsT, regLossT, label="reg Loss (Train)")
        ax[0].plot(epochsT, lossT, label="Total Loss (Train)")
        ax[0].set_yscale('log')  # Log scale
        ax[0].set_ylabel("Train Loss (log scale)")
        ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Second subplot: Validation losses (log scale)
        ax[1].plot(epochsV, rLossV, label="r Loss (Valid)")
        ax[1].plot(epochsV, kdLossV, label="kld Loss (Valid)")
        ax[1].plot(epochsV, regLossV, label="reg Loss (Valid)")
        ax[1].plot(epochsV, lossV, label="Total Loss (Valid)")
        ax[1].set_yscale('log')  # Log scale
        ax[1].set_ylabel("Validation Loss (log scale)")
        ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Third subplot: Accuracy
        ax[2].plot(epochsT, accT, label="Train Accuracy")
        ax[2].plot(epochsV, accV, label="Validation Accuracy", ls="--")
        ax[2].set_ylabel("Accuracy")
        ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Fourth subplot: Purity
        # ax[3].plot(epochsT, purityT, label="Train Purity")
        # ax[3].plot(epochsV, purityV, label="Validation Purity", ls="--")
        ax[3].plot(tempT[0], purityT, label="Train purity")
        ax[3].plot(tempV[0], purityV, label="Validation purity", ls="--")
        ax[3].set_ylabel("Purity")
        ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        fig.subplots_adjust(right=0.75)
        fig.savefig(path.join(self.outputfolder, outputFigure))
        print("save path avg   ", path.join(self.outputfolder, outputFigure))
        return fig

    # Generates a figure for a single run
    def genFigure(self, inputFile, outputFigure):
        data = np.load(path.join(self.inputfolder, inputFile))
        beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo = data["par"]

        trainMat = data["tl"]
        epochsT = trainMat[:, 0]
        rLossT = trainMat[:, 1]
        kdLossT = trainMat[:, 2]
        regLossT = trainMat[:, 3]
        lossT = trainMat[:, 4]
        accT = trainMat[:, 5]
        purityT = trainMat[:, 8]

        tempT = np.where(purityT > 0)
        purityT = purityT[tempT]

        validMat = data["vl"]
        epochsV = validMat[:, 0]
        rLossV = validMat[:, 1]
        kdLossV = validMat[:, 2]
        regLossV = validMat[:, 3]
        lossV = validMat[:, 4]
        accV = validMat[:, 5]
        purityV = validMat[:, 8]  # Purity column

        tempV = np.where(purityV > 0)
        purityV = purityV[tempV]

        fig, ax = plt.subplots(4, figsize=(10, 12))
        fig.suptitle(r" $\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) +
                     " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) +
                     " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo), fontsize=12)

        # First subplot: Training losses
        ax[0].plot(epochsT, rLossT, label="r Loss (Train)")
        ax[0].plot(epochsT, kdLossT, label="kld Loss (Train)")
        ax[0].plot(epochsT, regLossT, label="reg Loss (Train)")
        ax[0].plot(epochsT, lossT, label="Total Loss (Train)")
        ax[0].set_yscale('log')
        ax[0].set_ylabel("Train Loss")
        ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Second subplot: Validation losses
        ax[1].plot(epochsV, rLossV, label="r Loss (Valid)")
        ax[1].plot(epochsV, kdLossV, label="kld Loss (Valid)")
        ax[1].plot(epochsV, regLossV, label="reg Loss (Valid)")
        ax[1].plot(epochsV, lossV, label="Total Loss (Valid)")
        ax[1].set_yscale('log')
        ax[1].set_ylabel("Validation Loss")
        ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Third subplot: Accuracy
        ax[2].plot(epochsT, accT, label="Train Accuracy")
        ax[2].plot(epochsV, accV, label="Validation Accuracy", ls="--")
        ax[2].set_ylabel("Accuracy")
        ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Fourth subplot: Purity
        # ax[3].plot(epochsT, purityT, label="Train Purity")
        # ax[3].plot(epochsV, purityV, label="Validation Purity", ls="--")
        ax[3].plot(tempT[0], purityT, label="Train purity")
        ax[3].plot(tempV[0], purityV, label="Validation purity", ls="--")
        ax[3].set_ylabel("Purity")
        ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        fig.subplots_adjust(right=0.75)
        fig.savefig(path.join(self.outputfolder, outputFigure))
        print("save path    ", path.join(self.outputfolder, outputFigure))
        return fig
