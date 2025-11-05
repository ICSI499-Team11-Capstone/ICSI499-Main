
#James Oswald 1/12/21

import numpy as np
import matplotlib.pyplot as plt
import argparse

#takes a file path and returns a matplot figure object
def genFigure(filePath):
    data = np.load(filePath)
    alpha, beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo = data["par"]

    trainMat = data["tl"]
    epochsT = trainMat[:,0]
    rLossT = trainMat[:,1]
    kdLossT = trainMat[:,2]
    regLossT = trainMat[:,3]
    lossT = trainMat[:,4]
    accT = trainMat[:,5]

    #validation losses vs epoch
    validMat = data["vl"]
    epochsV = validMat[:,0]
    rLossV = validMat[:,1]
    kdLossV = validMat[:,2]
    regLossV = validMat[:,3]
    lossV = validMat[:,4]
    accV = validMat[:,5]

    #corelations with each dimention vs epoch
    wldimCors = data["wldims"]
    lidimCors = data["liidims"]
    wldimCorsv = data["wldimsv"]
    lidimCorsv = data["liidimsv"]

    wldimt = data["wtldims"]
    lidimt = data["liitdims"]
    wldimtv = data["wtldimsv"]
    lidimtv = data["liitdimsv"]

    wavg = data["wavg"]
    wavgv =data["wavgv"]
    liiavg = data["liiavg"]
    liiavgv = data["liiavgv"]
    dimRange = [j * 50 for j in range(wldimCors.shape[0])]
    dimRangev = [j * 50 for j in range(wldimCorsv.shape[0])]
    #fig, ax = plt.subplots(3)
    fig, ax = plt.subplots(8,2,sharex=False,sharey=False,)
    fig.set_size_inches(15, 15)
    fig.suptitle(r"$\alpha$=" + str(alpha) + r" $\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo) + " linearWidth" , fontsize=12)
    ax[0,0].plot(epochsT, rLossT, label="r Loss")
    ax[0,0].plot(epochsT, kdLossT, label="kld Loss")
    ax[0,0].plot(epochsT, regLossT, label="reg Loss")
    ax[0,0].plot(epochsT, lossT, label="Training Loss")
    ax[0,0].plot(epochsV, lossV, label="Valid Loss", ls="--")
    #ax[0,0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[0,0].set(xlabel="epoch", ylabel="Training Loss")
    #ax[0,0].label_outer()

    ax[0,1].plot(epochsT, rLossV, label="r Loss")
    ax[0,1].plot(epochsT, kdLossV, label="kld Loss")
    ax[0,1].plot(epochsT, regLossV, label="reg Loss")
    ax[0,1].plot(epochsV, lossV, label="Valid Loss")
    ax[0,1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[0,1].set(xlabel="epoch", ylabel="Validation Loss")
    #ax[0,1].label_outer()
    
    ax[1,1].remove()
    #gs=ax[1,0].get_gridspec()
    #ax[1,0] =fig.add_subplot(gs[1,:])
    ax[1,0].plot(epochsT, accT, label="Train Accuracy")
    ax[1,0].plot(epochsV, accV, label="Valid Accuracy")
    ax[1,0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[1,0].set(xlabel="epoch",ylabel="Accuracy")
    ax[1,0].text(x=1300, y=0.62, s=f"Training Accuracy: {'{:.3f}'.format(accT[len(accT) - 1])}")
    ax[1,0].text(x=1300, y=0.50, s=f"Testing Accuracy: {'{:.3f}'.format(accV[len(accV) - 1])}")
    # ax[1,0].label_outer()


    for i in range(int(latentDims)):
        ax[2,0].plot(dimRange, wldimCors[:,i], label=str(i), ls="-" if i == 0 else "--")
        ax[3,0].plot(dimRange, lidimCors[:,i], label=str(i), ls="-" if i == 1 else "--")
    #ax[2,0].legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
    ax[2,0].set(xlabel="epoch",ylabel="WL Cor")
    #ax[2,0].label_outer()
    #ax[3,0].legend(loc="center left", bbox_to_anchor=(1, 0.3), ncol=2)
    ax[3,0].set(xlabel="epoch",ylabel="LII Cor")
    #ax[3,0].label_outer()

    for i in range(int(latentDims)):
        ax[2,1].plot(dimRangev, wldimCorsv[:,i], label=str(i), ls="-" if i == 0 else "--")
        ax[3,1].plot(dimRangev, lidimCorsv[:,i], label=str(i), ls="-" if i == 1 else "--")
    ax[2,1].legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
    ax[2,1].set(xlabel="epoch",ylabel="WL Cor validation")
    #ax[2,1].label_outer()
    ax[3,1].legend(loc="center left", bbox_to_anchor=(1, 0.3), ncol=2)
    ax[3,1].set(xlabel="epoch",ylabel="LII Cor validation")
    #ax[3,1].label_outer()


    for i in range(int(latentDims)):
        ax[4,0].plot(dimRange, wldimt[:,i], label=str(i), ls="-" if i == 0 else "--")
        ax[5,0].plot(dimRange, lidimt[:,i], label=str(i), ls="-" if i == 1 else "--")
    #ax[2,0].legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
    ax[4,0].set(xlabel="epoch",ylabel="WL kendalltau")
    #ax[2,0].label_outer()
    #ax[3,0].legend(loc="center left", bbox_to_anchor=(1, 0.3), ncol=2)
    ax[5,0].set(xlabel="epoch",ylabel="LII kendalltau")
    #ax[3,0].label_outer()

    for i in range(int(latentDims)):
        ax[4,1].plot(dimRangev, wldimtv[:,i], label=str(i), ls="-" if i == 0 else "--")
        ax[5,1].plot(dimRangev, lidimtv[:,i], label=str(i), ls="-" if i == 1 else "--")
    ax[4,1].legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
    ax[4,1].set(xlabel="epoch",ylabel="WL kendalltau validation")
    #ax[2,1].label_outer()
    ax[5,1].legend(loc="center left", bbox_to_anchor=(1, 0.3), ncol=2)
    ax[5,1].set(xlabel="epoch",ylabel="LII kendalltau validation")
    #ax[3,1].label_outer()


    help1= { 0:'New Mean', 1:'New Std', 2:'Green Mean', 3:'Green Std', 4:'Red Mean', 5:'Red Std', 6:'Very Red Mean', 7:'Very Red Std', 8:'Near-IR Mean', 9:'Near-IR Std'}
    wavg = np.array(wavg).astype('float')
    y1 = np.array(wavg[:,0])
    y2 = np.array(wavg[:,1])
    y3 = np.array(wavg[:,2])
    y4 = np.array(wavg[:,3])
    y5 = np.array(wavg[:,4])
    y6 = np.array(wavg[:,5])
    y7 = np.array(wavg[:,6])
    y8 = np.array(wavg[:,7])
    y9 = np.array(wavg[:,8])
    y10 = np.array(wavg[:,9])
   
    
    ax[6,0].plot(y3 ,'g', label=help1[2])
    #ax[4,0].plot(y4 ,'darkgreen', label=help1[3])
    ax[6,0].plot(y5 ,'coral', label=help1[4])
    #ax[4,0].plot(y6 ,'tomato', label=help1[5])
    ax[6,0].plot(y7 ,'tab:red', label=help1[6])
    #ax[4,0].plot(y8 ,'brown', label=help1[7])
    ax[6,0].plot(y9 ,'k', label=help1[8])
    #ax[4,0].plot(y10,'gray', label=help1[9])
    ax[6,0].set(xlabel="epoch",ylabel='wavelengh avg all: '+str(round(np.mean(wavg),4)))
    #ax[4,0].legend(loc="center left", bbox_to_anchor=(1, 0.7), ncol=2)

    #ax[4,0].label_outer()
    try:
        help1= { 0:'New Mean', 1:'New Std', 2:'Green Mean', 3:'Green Std', 4:'Red Mean', 5:'Red Std', 6:'Very Red Mean', 7:'Very Red Std', 8:'Near-IR Mean', 9:'Near-IR Std'}
        wavgv = np.array(wavgv).astype('float')
        yv1 = np.array(wavgv[:,0])
        yv2 = np.array(wavgv[:,1])
        yv3 = np.array(wavgv[:,2])
        yv4 = np.array(wavgv[:,3])
        yv5 = np.array(wavgv[:,4])
        yv6 = np.array(wavgv[:,5])
        yv7 = np.array(wavgv[:,6])
        yv8 = np.array(wavgv[:,7])
        yv9 = np.array(wavgv[:,8])
        yv10 = np.array(wavgv[:,9])
        
   
    
        ax[6,1].plot(yv3 ,'g', label=help1[2])
        #ax[8].plot(yv4 ,'darkgreen', label=help1[3])
        ax[6,1].plot(yv5 ,'coral', label=help1[4])
        #ax[8].plot(yv6 ,'tomato', label=help1[5])
        ax[6,1].plot(yv7 ,'tab:red', label=help1[6])
        #ax[8].plot(yv8 ,'brown', label=help1[7])
        ax[6,1].plot(yv9 ,'k', label=help1[8])
        #ax[8].plot(yv10,'gray', label=help1[9])
        ax[6,1].set(xlabel="epoch",ylabel='wavelengh avg valid: '+str(round(np.mean(wavgv),4)))
        ax[6,1].legend(loc="center left", bbox_to_anchor=(1, 0.7), ncol=2)

    #ax[4,1].label_outer()
    except:
        pass
    help1= { 0:'New Mean', 1:'New Std', 2:'Green Mean', 3:'Green Std', 4:'Red Mean', 5:'Red Std', 6:'Very Red Mean', 7:'Very Red Std', 8:'Near-IR Mean', 9:'Near-IR Std'}
    liiavg = np.array(liiavg).astype('float')
    y1 = np.array(liiavg[:,0])
    y2 = np.array(liiavg[:,1])
    y3 = np.array(liiavg[:,2])
    y4 = np.array(liiavg[:,3])
    y5 = np.array(liiavg[:,4])
    y6 = np.array(liiavg[:,5])
    y7 = np.array(liiavg[:,6])
    y8 = np.array(liiavg[:,7])
    y9 = np.array(liiavg[:,8])
    y10 =np.array(liiavg[:,9])
   
    
    ax[7,0].plot(y3 ,'g', label=help1[2])
    #ax[4,0].plot(y4 ,'darkgreen', label=help1[3])
    ax[7,0].plot(y5 ,'coral', label=help1[4])
    #ax[4,0].plot(y6 ,'tomato', label=help1[5])
    ax[7,0].plot(y7 ,'tab:red', label=help1[6])
    #ax[4,0].plot(y8 ,'brown', label=help1[7])
    ax[7,0].plot(y9 ,'k', label=help1[8])
    #ax[4,0].plot(y10,'gray', label=help1[9])
    ax[7,0].set(xlabel="epoch",ylabel='lii avg all: '+str(round(np.mean(liiavg),4)))
    #ax[4,0].legend(loc="center left", bbox_to_anchor=(1, 0.7), ncol=2)

    #ax[4,0].label_outer()
    try:
        help1= { 0:'New Mean', 1:'New Std', 2:'Green Mean', 3:'Green Std', 4:'Red Mean', 5:'Red Std', 6:'Very Red Mean', 7:'Very Red Std', 8:'Near-IR Mean', 9:'Near-IR Std'}
        liiavgv = np.array(liiavgv).astype('float')
        yv1 = np.array(liiavgv[:,0])
        yv2 = np.array(liiavgv[:,1])
        yv3 = np.array(liiavgv[:,2])
        yv4 = np.array(liiavgv[:,3])
        yv5 = np.array(liiavgv[:,4])
        yv6 = np.array(liiavgv[:,5])
        yv7 = np.array(liiavgv[:,6])
        yv8 = np.array(liiavgv[:,7])
        yv9 = np.array(liiavgv[:,8])
        yv10 =np.array(liiavgv[:,9])
    
        
        ax[7,1].plot(yv3 ,'g', label=help1[2])
        #ax[8].plot(yv4 ,'darkgreen', label=help1[3])
        ax[7,1].plot(yv5 ,'coral', label=help1[4])
        #ax[8].plot(yv6 ,'tomato', label=help1[5])
        ax[7,1].plot(yv7 ,'tab:red', label=help1[6])
        #ax[8].plot(yv8 ,'brown', label=help1[7])
        ax[7,1].plot(yv9 ,'k', label=help1[8])
        #ax[8].plot(yv10,'gray', label=help1[9])
        ax[7,1].set(xlabel="epoch",ylabel='lii avg valid: '+str(round(np.mean(liiavgv),4)))
        ax[7,1].legend(loc="center left", bbox_to_anchor=(1, 0.7), ncol=2)

        #ax[4,1].label_outer()
    except:
        pass
    fig.subplots_adjust(right=0.75)
    
    
    return fig

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="i", default="None")
parser.add_argument("-o", "--output", dest="o", default="None")
args = parser.parse_args()

""" if args.i != "None":
    f = genFigure(args.i)
    name = ""
    if args.o == "None":
        raise Exception("No output name provided")
        #name = "./graphs/b"+args.i+".png"
    else:
        name = args.o
    f.savefig(name)
else:
    fileName = "./runs/1623311492.npz"
    f = genFigure(fileName)
    f.savefig("./graphs/latest.png") """

def genPlotForRun(runsPath, run, graphsPath, graph):
    f = genFigure(runsPath+"/"+run)
    f.savefig(graphsPath+"/"+graph)


#genFigure("all-results/1-18-22-res/runs/a19lds19b0.007g1.0d1.0h13.npz")