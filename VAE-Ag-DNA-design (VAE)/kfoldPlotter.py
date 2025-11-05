
#James Oswald 1/12/21
import numpy as np
import matplotlib.pyplot as plt
import os.path as path

#takes a file path and returns a matplot figure object
class Plotter:

    def __init__(self, metricsfolder=path.join(".","runs","kfold"), graphsfolder=path.join(".","graphs","kfold")):
        self.inputfolder = metricsfolder
        self.outputfolder = graphsfolder
     
    def genAvgFigure(self, inputFiles, outputFigure):
        data =[]
        for file in inputFiles:
            data.append(np.load(path.join(self.inputfolder, file)))

        alpha, beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo = data[0]["par"]
        
        rLossM = []
        kdLossM = []
        regLossM = []
        lossM = []
        accM = []
        for datasub in data:
            trainMat = datasub["tl"]
            rLossM = np.append(rLossM, trainMat[:,1])
            kdLossM = np.append(kdLossM, trainMat[:,2])
            regLossM = np.append(regLossM, trainMat[:,3])
            lossM = np.append(lossM, trainMat[:,4])
            accM = np.append(accM, trainMat[:,5])

        epochsT = data[0]["tl"][:,0]
        rLossT = np.mean(np.reshape(rLossM, (len(data), len(epochsT))), axis=0)
        kdLossT = np.mean(np.reshape(kdLossM, (len(data), len(epochsT))), axis=0)
        regLossT =  np.mean(np.reshape(regLossM, (len(data), len(epochsT))), axis=0)
        lossT =  np.mean(np.reshape(lossM, (len(data), len(epochsT))), axis=0)
        accT =  np.mean(np.reshape(accM, (len(data),len(epochsT))), axis=0)

        rLossM = []
        kdLossM = []
        regLossM = []
        lossM = []
        accM = []

        for datasub in data:
            validMat = data[0]["vl"]
            rLossM = np.append(rLossM, validMat[:,1])
            kdLossM = np.append(kdLossM, validMat[:,2])
            regLossM = np.append(regLossM, validMat[:,3])
            lossM = np.append(lossM, validMat[:,4])
            accM = np.append(accM, validMat[:,5])
        
        #validation losses vs epoch
        epochsV = data[0]["vl"][:,0]
        rLossV = np.mean(np.reshape(rLossM, (len(data), len(epochsV))), axis=0)
        kdLossV = np.mean(np.reshape(kdLossM, (len(data), len(epochsV))), axis=0)
        regLossV =  np.mean(np.reshape(regLossM, (len(data), len(epochsV))), axis=0)
        lossV =  np.mean(np.reshape(lossM, (len(data), len(epochsV))), axis=0)
        accV =  np.mean(np.reshape(accM, (len(data), len(epochsV))), axis=0)
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
        finalLIIcor = wldimCors[:,1][len(lidimCors[:,1])-1]  
        
        
       #corelations with each dimention vs epoch
        wavg = data[0]["wavg"]
        wavgv = data[0]["wavgv"]
        wavgM = np.reshape(wavg, np.shape(wavg)+(1,))
        wavgvM = np.reshape(wavgv, np.shape(wavgv)+(1,))

        for datasub in data:
            wavg = datasub["wavg"]
            wavgv = datasub["wavgv"]
            wavgM = np.concatenate((wavgM, np.reshape(wavg, np.shape(wavg)+(1,))), axis=2)
            wavgvM = np.concatenate((wavgvM, np.reshape(wavgv, np.shape(wavgv)+(1,))), axis=2)
        
        wavgM = np.array(wavgM,dtype=np.float)
        wavgvM = np.array(wavgvM,dtype=np.float)

        wavgCors = np.mean(wavgM, axis=2) 
        wavgvCors = np.mean(wavgvM, axis=2)

        #corelations with each dimention vs epoch
        liiavg = data[0]["liiavg"]
        liiavgv = data[0]["liiavgv"]
        liiavgM = np.reshape(wavg, np.shape(liiavg)+(1,))
        liiavgvM = np.reshape(liiavgv, np.shape(liiavgv)+(1,))

        for datasub in data:
            liiavg = datasub["liiavg"]
            liiavgv = datasub["liiavgv"]
            liiavgM = np.concatenate((liiavgM, np.reshape(liiavg, np.shape(liiavg)+(1,))), axis=2)
            liiavgvM = np.concatenate((liiavgvM, np.reshape(liiavgv, np.shape(liiavgv)+(1,))), axis=2)
        
        liiavgM = np.array(liiavgM,dtype=np.float)
        liiavgvM = np.array(liiavgvM,dtype=np.float)

        liiavgCors = np.mean(liiavgM, axis=2) 
        liiavgvCors = np.mean(liiavgvM, axis=2) 
   

        #corelations with each dimention vs epoch
        wldimsv = data[0]["wldimsv"]
        lidimsv = data[0]["liidimsv"]
        wldimsMv = np.reshape(wldimsv, np.shape(wldims)+(1,))
        lidimsMv = np.reshape(lidimsv, np.shape(lidims)+(1,))

        for datasub in data:
            wldimsv = datasub["wldimsv"]
            lidimsv = datasub["liidimsv"]
            wldimsMv = np.concatenate((wldimsMv, np.reshape(wldimsv, np.shape(wldimsv)+(1,))), axis=2)
            lidimsMv = np.concatenate((lidimsMv, np.reshape(lidimsv, np.shape(lidimsv)+(1,))), axis=2)
        wldimCorsv = np.mean(wldimsMv, axis=2) 
        lidimCorsv = np.mean(lidimsMv, axis=2) 
        dimRangev = [j * 50 for j in range(wldimCors.shape[0])]
        finalWIIcorv = wldimCorsv[:,0][len(wldimCorsv[:,0])-1]
        finalLIIcorv = wldimCorsv[:,1][len(lidimCorsv[:,1])-1]  
        #corelations with each dimention vs epoch
        wldimst = data[0]["wtldims"]
        lidimst = data[0]["liitdims"]
        wldimsMt = np.reshape(wldimst, np.shape(wldims)+(1,))
        lidimsMt = np.reshape(lidimst, np.shape(lidims)+(1,))

        for datasub in data:
            wldimst = datasub["wtldims"]
            lidimst = datasub["liitdims"]
            wldimsMt = np.concatenate((wldimsMt, np.reshape(wldimst, np.shape(wldimst)+(1,))), axis=2)
            lidimsMt = np.concatenate((lidimsMt, np.reshape(lidimst, np.shape(lidimst)+(1,))), axis=2)
        wldimCorst = np.mean(wldimsMt, axis=2) 
        lidimCorst = np.mean(lidimsMt, axis=2) 
        dimRangev = [j * 50 for j in range(wldimCorst.shape[0])]
        finalWIIcortv = wldimCorst[:,0][len(wldimCorst[:,0])-1]
        finalLIIcortv = wldimCorst[:,1][len(lidimCorst[:,1])-1]  
        
        #corelations with each dimention vs epoch
        wldimstv = data[0]["wtldimsv"]
        lidimstv = data[0]["liitdimsv"]
        wldimsMtv = np.reshape(wldimstv, np.shape(wldims)+(1,))
        lidimsMtv = np.reshape(lidimstv, np.shape(lidims)+(1,))

        for datasub in data:
            wldimsv = datasub["wtldimsv"]
            lidimsv = datasub["liitdimsv"]
            wldimsMtv = np.concatenate((wldimsMtv, np.reshape(wldimstv, np.shape(wldimstv)+(1,))), axis=2)
            lidimsMtv = np.concatenate((lidimsMtv, np.reshape(lidimstv, np.shape(lidimstv)+(1,))), axis=2)
        wldimCorstv = np.mean(wldimsMtv, axis=2) 
        lidimCorstv = np.mean(lidimsMtv, axis=2) 
        dimRangev = [j * 50 for j in range(wldimCorstv.shape[0])]
        finalWIIcortv = wldimCorstv[:,0][len(wldimCorstv[:,0])-1]
        finalLIIcortv = wldimCorstv[:,1][len(lidimCorstv[:,1])-1]  
        

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
            ax[4,0].plot(dimRange, wldimCorst[:,i], label=str(i), ls="-" if i == 0 else "--")
            ax[5,0].plot(dimRange, lidimCorst[:,i], label=str(i), ls="-" if i == 1 else "--")
        #ax[2,0].legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
        ax[4,0].set(xlabel="epoch",ylabel="WL kendalltau")
        #ax[2,0].label_outer()
        #ax[3,0].legend(loc="center left", bbox_to_anchor=(1, 0.3), ncol=2)
        ax[5,0].set(xlabel="epoch",ylabel="LII kendalltau")
        #ax[3,0].label_outer()

        for i in range(int(latentDims)):
            ax[4,1].plot(dimRangev, wldimCorstv[:,i], label=str(i), ls="-" if i == 0 else "--")
            ax[5,1].plot(dimRangev, lidimCorstv[:,i], label=str(i), ls="-" if i == 1 else "--")
        ax[4,1].legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
        ax[4,1].set(xlabel="epoch",ylabel="WL kendalltau validation")
        #ax[2,1].label_outer()
        ax[5,1].legend(loc="center left", bbox_to_anchor=(1, 0.3), ncol=2)
        ax[5,1].set(xlabel="epoch",ylabel="LII kendalltau validation")
        #ax[3,1].label_outer()


        help1= { 0:'New Mean', 1:'New Std', 2:'Green Mean', 3:'Green Std', 4:'Red Mean', 5:'Red Std', 6:'Very Red Mean', 7:'Very Red Std', 8:'Near-IR Mean', 9:'Near-IR Std'}
        wavgCors = np.array(wavgCors).astype('float')
        y1 = np.array(wavgCors[:,0])
        y2 = np.array(wavgCors[:,1])
        y3 = np.array(wavgCors[:,2])
        y4 = np.array(wavgCors[:,3])
        y5 = np.array(wavgCors[:,4])
        y6 = np.array(wavgCors[:,5])
        y7 = np.array(wavgCors[:,6])
        y8 = np.array(wavgCors[:,7])
        y9 = np.array(wavgCors[:,8])
        y10 = np.array(wavgCors[:,9])
    
        
        ax[6,0].plot(y3 ,'g', label=help1[2])
        #ax[4,0].plot(y4 ,'darkgreen', label=help1[3])
        ax[6,0].plot(y5 ,'coral', label=help1[4])
        #ax[4,0].plot(y6 ,'tomato', label=help1[5])
        ax[6,0].plot(y7 ,'tab:red', label=help1[6])
        #ax[4,0].plot(y8 ,'brown', label=help1[7])
        ax[6,0].plot(y9 ,'k', label=help1[8])
        #ax[4,0].plot(y10,'gray', label=help1[9])
        ax[6,0].set(xlabel="epoch",ylabel='avg all: '+str(round(np.mean(wavgCors),4)))
        #ax[4,0].legend(loc="center left", bbox_to_anchor=(1, 0.7), ncol=2)

        #ax[4,0].label_outer()

        help1= { 0:'New Mean', 1:'New Std', 2:'Green Mean', 3:'Green Std', 4:'Red Mean', 5:'Red Std', 6:'Very Red Mean', 7:'Very Red Std', 8:'Near-IR Mean', 9:'Near-IR Std'}
        wavgvCors = np.array(wavgvCors).astype('float')
        yv1 = np.array(wavgvCors[:,0])
        yv2 = np.array(wavgvCors[:,1])
        yv3 = np.array(wavgvCors[:,2])
        yv4 = np.array(wavgvCors[:,3])
        yv5 = np.array(wavgvCors[:,4])
        yv6 = np.array(wavgvCors[:,5])
        yv7 = np.array(wavgvCors[:,6])
        yv8 = np.array(wavgvCors[:,7])
        yv9 = np.array(wavgvCors[:,8])
        yv10 = np.array(wavgvCors[:,9])
    
        
        ax[6,1].plot(yv3 ,'g', label=help1[2])
        #ax[8].plot(yv4 ,'darkgreen', label=help1[3])
        ax[6,1].plot(yv5 ,'coral', label=help1[4])
        #ax[8].plot(yv6 ,'tomato', label=help1[5])
        ax[6,1].plot(yv7 ,'tab:red', label=help1[6])
        #ax[8].plot(yv8 ,'brown', label=help1[7])
        ax[6,1].plot(yv9 ,'k', label=help1[8])
        #ax[8].plot(yv10,'gray', label=help1[9])
        ax[6,1].set(xlabel="epoch",ylabel='avg valid: '+str(round(np.mean(wavgvCors),4)))
        ax[6,1].legend(loc="center left", bbox_to_anchor=(1, 0.7), ncol=2)

        #ax[4,1].label_outer()


        help1= { 0:'New Mean', 1:'New Std', 2:'Green Mean', 3:'Green Std', 4:'Red Mean', 5:'Red Std', 6:'Very Red Mean', 7:'Very Red Std', 8:'Near-IR Mean', 9:'Near-IR Std'}
        liiavgCors = np.array(liiavgCors).astype('float')
        y1 = np.array(liiavgCors[:,0])
        y2 = np.array(liiavgCors[:,1])
        y3 = np.array(liiavgCors[:,2])
        y4 = np.array(liiavgCors[:,3])
        y5 = np.array(liiavgCors[:,4])
        y6 = np.array(liiavgCors[:,5])
        y7 = np.array(liiavgCors[:,6])
        y8 = np.array(liiavgCors[:,7])
        y9 = np.array(liiavgCors[:,8])
        y10 =np.array(liiavgCors[:,9])
    
        
        ax[7,0].plot(y3 ,'g', label=help1[2])
        #ax[4,0].plot(y4 ,'darkgreen', label=help1[3])
        ax[7,0].plot(y5 ,'coral', label=help1[4])
        #ax[4,0].plot(y6 ,'tomato', label=help1[5])
        ax[7,0].plot(y7 ,'tab:red', label=help1[6])
        #ax[4,0].plot(y8 ,'brown', label=help1[7])
        ax[7,0].plot(y9 ,'k', label=help1[8])
        #ax[4,0].plot(y10,'gray', label=help1[9])
        ax[7,0].set(xlabel="epoch",ylabel='avg all: '+str(round(np.mean(wavgCors),4)))
        #ax[4,0].legend(loc="center left", bbox_to_anchor=(1, 0.7), ncol=2)

        #ax[4,0].label_outer()

        help1= { 0:'New Mean', 1:'New Std', 2:'Green Mean', 3:'Green Std', 4:'Red Mean', 5:'Red Std', 6:'Very Red Mean', 7:'Very Red Std', 8:'Near-IR Mean', 9:'Near-IR Std'}
        liiavgvCors = np.array(liiavgvCors).astype('float')
        yv1 = np.array(liiavgvCors[:,0])
        yv2 = np.array(liiavgvCors[:,1])
        yv3 = np.array(liiavgvCors[:,2])
        yv4 = np.array(liiavgvCors[:,3])
        yv5 = np.array(liiavgvCors[:,4])
        yv6 = np.array(liiavgvCors[:,5])
        yv7 = np.array(liiavgvCors[:,6])
        yv8 = np.array(liiavgvCors[:,7])
        yv9 = np.array(liiavgvCors[:,8])
        yv10 =np.array(liiavgvCors[:,9])
    
        
        ax[7,1].plot(yv3 ,'g', label=help1[2])
        #ax[8].plot(yv4 ,'darkgreen', label=help1[3])
        ax[7,1].plot(yv5 ,'coral', label=help1[4])
        #ax[8].plot(yv6 ,'tomato', label=help1[5])
        ax[7,1].plot(yv7 ,'tab:red', label=help1[6])
        #ax[8].plot(yv8 ,'brown', label=help1[7])
        ax[7,1].plot(yv9 ,'k', label=help1[8])
        #ax[8].plot(yv10,'gray', label=help1[9])
        ax[7,1].set(xlabel="epoch",ylabel='avg valid: '+str(round(np.mean(wavgvCors),4)))
        ax[7,1].legend(loc="center left", bbox_to_anchor=(1, 0.7), ncol=2)

        #ax[4,1].label_outer()

        fig.subplots_adjust(right=0.75)
        fig.savefig(path.join(self.outputfolder, outputFigure))
        return fig

    def genFigure(self, inputFile , outputFigure):
        data = np.load(path.join(self.inputfolder, inputFile))
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
        ax[6,0].set(xlabel="epoch",ylabel='avg all: '+str(round(np.mean(wavg),4)))
        #ax[4,0].legend(loc="center left", bbox_to_anchor=(1, 0.7), ncol=2)

        #ax[4,0].label_outer()

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
        ax[6,1].set(xlabel="epoch",ylabel='avg valid: '+str(round(np.mean(wavgv),4)))
        ax[6,1].legend(loc="center left", bbox_to_anchor=(1, 0.7), ncol=2)

        #ax[4,1].label_outer()
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



        fig.subplots_adjust(right=0.75)
        fig.savefig(path.join(self.outputfolder, outputFigure))
        return fig
