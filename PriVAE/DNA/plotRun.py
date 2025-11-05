import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import matplotlib.ticker as mtick

# Takes a file path and returns a matplot figure object
def genFigure(filePath):
    data = np.load(filePath)
    countOfClustersTrain = dict(pd.DataFrame({'Label': data['trainLabel'].reshape(-1)})['Label'].value_counts().sort_index())
    countOfClustersVal = dict(pd.DataFrame({'Label': data['validLabel'].reshape(-1)})['Label'].value_counts().sort_index())
    _ = [countOfClustersVal.update({i: 0}) for i in countOfClustersTrain.keys() if i not in countOfClustersVal.keys()]
    countOfClustersTrainFilter = list(countOfClustersTrain.values())
    countOfClustersValFilter = list(zip(*sorted(countOfClustersVal.items())))[1]

    # print("filepath    ", filePath)
    # print("params    ",data["par"])
    beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo, purity_interval = data["par"]

    trainMat = data["tl"]
    print("trainMat   ", trainMat.shape)
    epochsT = trainMat[:, 0]
    rLossT = trainMat[:, 1]
    kdLossT = trainMat[:, 2]
    regLossT = trainMat[:, 3]
    lossT = trainMat[:, 4]
    accT = trainMat[:, 5]
    traDistT = trainMat[:, 6]
    terDistT = trainMat[:, 7]
    purityT = trainMat[:, 8]

    tempT = np.where(purityT > 0)
    purT = purityT[tempT]
    print("tempx   ", tempT)
    print("purt   ", purT)


    # Validation losses vs epoch
    validMat = data["vl"]
    epochsV = validMat[:, 0]
    rLossV = validMat[:, 1]
    kdLossV = validMat[:, 2]
    regLossV = validMat[:, 3]
    lossV = validMat[:, 4]
    accV = validMat[:, 5]
    traDistV = validMat[:, 6]
    terDistV = validMat[:, 7]
    purityV = validMat[:, 8]

    tempV = np.where(purityV > 0)
    purV = purityV[tempV]
    print("purv    ", purV)

    fig, ax = plt.subplots(4)
    fig.set_size_inches(8, 7)

    fig.suptitle(
    r"$\beta$=" + f"{beta:.3f}" +
    r" $\gamma$=" + f"{gamma:.3f}" +
    " latentDims=" + str(latentDims) +
    " lstmLayers=" + str(lstmLayers) +
    " Dropout=" + f"{drop:.1f}" +
    " lstmInfo=" + str(lstmInfo) +
    " linearWidth", 
    fontsize=12)

    # Plot training and validation losses
    ax[0].plot(epochsT, rLossT, label="r Loss")
    ax[0].plot(epochsT, kdLossT, label="kld Loss")
    ax[0].plot(epochsT, regLossT, label="reg Loss")
    ax[0].plot(epochsT, lossT, label="Training Loss")
    ax[0].plot(epochsV, lossV, label="Valid Loss", ls="--")
    ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[0].set(xlabel="epoch", ylabel="T Loss")
    ax[0].label_outer()
    ax[0].set_yscale('log')

    # Validation losses
    ax[1].plot(epochsT, rLossV, label="r Loss")
    ax[1].plot(epochsT, kdLossV, label="kld Loss")
    ax[1].plot(epochsT, regLossV, label="reg Loss")
    ax[1].plot(epochsV, lossV, label="Valid Loss")
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[1].set(xlabel="epoch", ylabel="Val Loss")
    ax[1].label_outer()
    ax[1].set_yscale('log')

    # Accuracy plot
    ax[2].plot(epochsT, accT, label="Train Accuracy")
    ax[2].plot(epochsV, accV, label="Valid Accuracy")
    ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[2].set(xlabel="epoch", ylabel="Accuracy")
    ax[2].set_ylim([0, 1])  

    ax[3].plot(tempT[0], purT, label="Train purity")
    ax[3].plot(tempV[0], purV, label="Validation purity")
    ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[3].set(xlabel="epoch", ylabel="Purity")

        
    x_mid_purity = (tempT[0][0] + tempT[0][-1]) / 2  
    if len(purT) > 0 and len(purV) > 0:
        ax[3].text(x=x_mid_purity, y=(purT[-1] + purV[-1]) / 2 + 0.005, 
                s=f"Train Purity: {'{:.3f}'.format(purT[-1])}", ha='center', va='center')
        ax[3].text(x=x_mid_purity, y=(purT[-1] + purV[-1]) / 2 - 0.005, 
                s=f"Validation Purity: {'{:.3f}'.format(purV[-1])}", ha='center', va='center')

    
    x_mid_acc = (epochsT[0] + epochsT[-1]) / 2  
    y_mid_acc = 0.5  

    if len(accT) > 0 and len(accV) > 0:
        ax[2].text(x=x_mid_acc, y=y_mid_acc + 0.05, s=f"Training Accuracy: {'{:.3f}'.format(accT[-1])}", ha='center', va='center')
        ax[2].text(x=x_mid_acc, y=y_mid_acc - 0.05, s=f"Validation Accuracy: {'{:.3f}'.format(accV[-1])}", ha='center', va='center')

    ax[2].label_outer()

    
    for axis in ax:
        axis.set_xticks(list(range(0, len(epochsT), 500)))  
        axis.set_xticklabels(list(range(0, len(epochsT), 500)), rotation=45)

    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    fig.subplots_adjust(right=0.75)

    return fig


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="i", default="None")
parser.add_argument("-o", "--output", dest="o", default="None")
args = parser.parse_args()

# Uncomment the following block to enable command line arguments for input/output files
"""
if args.i != "None":
    f = genFigure(args.i)
    name = ""
    if args.o == "None":
        raise Exception("No output name provided")
    else:
        name = args.o
    f.savefig(name)
else:
    fileName = "./runs/1623311492.npz"
    f = genFigure(fileName)
    f.savefig("./graphs/latest.png")
"""

def genPlotForRun(runsPath, run, graphsPath, graph):
    f = genFigure(runsPath + "/" + run)
    f.savefig(graphsPath + "/" + graph)


# Uncomment this for testing a specific run:
# genFigure("all-results/1-18-22-res/runs/a19lds19b0.007g1.0d1.0h13.npz")
# f = genFigure('./runs/a20lds20b0.007g0.5d1h13.npz')
# f.savefig('./graphs/test.png')
