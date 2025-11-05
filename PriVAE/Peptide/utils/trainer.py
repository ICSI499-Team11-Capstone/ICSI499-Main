from contextlib import nullcontext
import os
import time
import datetime
from tqdm import tqdm
import pandas as pd
from scipy.stats import norm
from abc import ABC, abstractmethod
import torch
from torch import nn  # Q: is this necessary to do?
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from utils.helpers import to_numpy  # Disregard syntax errors on this line
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from utils.helpers import to_cuda_variable_long, to_cuda_variable, to_numpy
import sys
from pathlib import Path
import matplotlib.pyplot as plt
pre='.'

GmmData = pd.read_excel(pre+'/data-and-cleaning/peptide_with_activity_labels-May6.xlsx')
GmmDData = pd.read_excel(pre+'/data-and-cleaning/L1distance_peptide_with_activity_labels-May6.xlsx')

print("Distances Data loaded")
# ALPHABET = ['A','C','G','T']
ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
GmmDataS = GmmData['Sequence'].to_list()
# clusternums = len(set(GmmData['Label'].to_list()))
enc = OneHotEncoder()
enc.fit(np.array(ALPHABET).reshape(-1,1))
GMMSums = sum(sum(GmmDData.to_numpy()))
scale = 10

def Distencecs(sequence):
    basesequence = [''.join(enc.inverse_transform(e).reshape(-1).tolist()) for e in sequence] 
    filterdDdata = np.zeros((len(basesequence),len(basesequence)))
    for idx,clmns in enumerate(basesequence):
        for jdx,rows in enumerate(basesequence):
            if np.isnan(GmmDData[GmmDataS.index(clmns)][GmmDataS.index(rows)]):
                raise ValueError('GmmDData is nan')
            filterdDdata[idx][jdx] =  GmmDData[GmmDataS.index(clmns)][GmmDataS.index(rows)]

    return filterdDdata



def compute_pairwise_distances(sequences, GmmDData, GmmDataS):
    '''Function to compute pairwise distances and save them
    sequences: one-hot encoded sequences (e.g., from train or validation dataset)
    GmmDData: precomputed pairwise distances (matrix of distances)
    GmmDataS: list of sequences corresponding to indices in GmmDData'''

    # Convert one-hot encoded sequences back to strings
    basesequence = [''.join(enc.inverse_transform(e).reshape(-1).tolist()) for e in sequences]
    
    # Initialize distance matrix
    pairwise_distances = np.zeros((len(basesequence), len(basesequence)))
    for idx, clmns in enumerate(basesequence):
        for jdx, rows in enumerate(basesequence):
            # Look up distance from GmmDData
            pairwise_distances[idx][jdx] = GmmDData[GmmDataS.index(clmns)][GmmDataS.index(rows)]
    
    return pairwise_distances


def rbf(inputs,delta):
    return torch.exp(- inputs ** 2 / (2. * delta ** 2))



def find_knn(data, k):

    num_samples = data.shape[0]
    # print("data   ", data.shape)
    # distances = data.values
    distances = data.cpu().numpy()
    # print("dist   ", distances.shape)
    ret = np.argpartition(distances, k + 1, axis=1)[:, :k + 1]

    weights = np.exp(-distances / (2 * 1**2))
    weights[np.arange(num_samples), np.arange(num_samples)] = 0
    # print("weights    ", weights)

    # weights /= np.max(weights, axis=1)[:, None]
    weights /= np.max(weights)

    adj = np.zeros_like(distances)
    adj[np.arange(num_samples)[:, None], ret] = 1

    return ret, weights, adj


connection_cache = {}

# def cached_extract_connections(indices, raw_dataset, X_batch, X_emb, label_batch, adj_inds, adj_weights,seq_lengths, knn_distances, knn_size):
def cached_extract_connections(indices, raw_dataset, X_batch, X_emb, label_batch, clabel_batch,seq_lengths, knn_distances, knn_size):
    key = tuple(indices.cpu().numpy())
    # print("--")
    if key not in connection_cache:
        # print("------------------")
        adj_inds, adj_weights, adj_matrix = find_knn(data=knn_distances,k=knn_size)
        result = extract_connections(indices, raw_dataset, X_batch, X_emb, label_batch, clabel_batch, adj_inds, adj_weights,seq_lengths)
        connection_cache[key] = result
    return connection_cache[key]

def extract_connections(indices, raw_dataset, X_batch, X_emb, label_batch, clabel_batch, adj_inds, adj_weights,seq_lengths):
    # print(" \n *****-------*****   \n ")
    translation = {}
    for i in range(len(indices)):
        translation[int(indices[i].detach().numpy())] = i
    # translation = {int(i.detach().numpy()): ind for ind, i in enumerate(indices)}

    trans_keys = translation.keys()


    X_prime_inds = []
    for i in range(len(indices)):
        row = int(indices[i].detach().numpy())
        # print("row    ", row)
        # print("adjs   ", adj_inds[row])
        for j in range(len(adj_inds[row])):
            # print("$$$$       ", adj_inds[row][j]," -- ", row)
            if adj_inds[row][j] != row:
                if adj_inds[row][j] not in trans_keys:
                    X_prime_inds.append(adj_inds[row][j])
    X_prime_inds = list(set(X_prime_inds))
    # print(" ^^^^      ", len(X_prime_inds))
    # X_prime_inds = [adj_ind for row in adj_inds for adj_ind in row]
    # X_prime_inds = list(set(X_prime_inds))

    if len(X_prime_inds) == 0:
        new_adj_inds = []
        for i in range(len(adj_inds)):
            # print("sssssss     ", adj_weights[i])
            st = adj_inds[i][0]
            en = adj_inds[i][1]
            wg = adj_weights[st, en]
            new_adj_inds.append([st,en,wg])
        return X_batch, X_emb, label_batch, clabel_batch, new_adj_inds,translation



    X_prime = []
    emb_prime = []
    label_prime = []
    clabel_prime = []
    seq_len_prime = []
    ln_inds = int(len(translation))
    cnt = 0


    for i in range(len(X_prime_inds)):

        tx, emb_tx, lbl, clbl, seq_len = raw_dataset[X_prime_inds[i]]
        if X_prime_inds[i] not in trans_keys:
            translation[X_prime_inds[i]] = cnt+ln_inds
            cnt += 1
        X_prime.append(tx)
        emb_prime.append(emb_tx)
        label_prime.append(lbl)
        clabel_prime.append(clbl)
        seq_len_prime.append(seq_len)
    X_prime = torch.from_numpy(np.array(X_prime))
    emb_prime = torch.from_numpy(np.array(emb_prime))
    label_prime = torch.stack(label_prime, dim=0)
    # clabel_prime = torch.stack(clabel_prime, dim=0)
    # c4_label_prime = torch.stack(c4_label_prime, dim=0)
    # print("c4 label prime    ", c4_label_prime)
    ln = len(trans_keys)
    # print(" - ln     ", ln)
    new_adj = np.zeros((ln, ln))
    new_adj_inds = []
    for i in range(len(indices)):
        row = int(indices[i].detach().numpy())
        for j in range(len(adj_inds[row])):
            start = int(indices[i].detach().numpy())
            end = adj_inds[row][j]
            wg = adj_weights[row][j]
            tr_start = translation[start]
            tr_end = translation[end]
            new_adj[tr_start, tr_end] = 1
            new_adj_inds.append([tr_end, tr_start, wg])
    new_X = torch.cat((X_batch, X_prime), axis=0)
    new_emb = torch.cat((X_emb, emb_prime), axis=0)
    new_label = torch.cat((label_batch, label_prime), axis=0)
    # new_clabel = torch.cat((clabel_batch, clabel_prime), axis=0)
    clabel_batch = list(clabel_batch)
    new_clabel = []
    new_clabel.extend(clabel_batch)
    new_clabel.extend(clabel_prime)
    # print("new label    ", new_label)
    new_seq_len = []
    new_seq_len.extend(seq_lengths)
    new_seq_len.extend(seq_len_prime)
    new_seq_len = [i.item() for i in new_seq_len]
    # print("c4 labels   ", c4_labels)
    # print("new c4    ", new_c4_label)
    # return new_X, new_label, new_adj_inds
    return new_X, new_emb, new_label, new_clabel, new_seq_len, new_adj_inds,translation



# all_label_dis_sum_train = [0]*clusternums
# all_label_dis_sum_val = [0]*clusternums
clusters_distance_latent_train = []
clusters_distance_latent_valid = []
all_label_code = []
  
class Trainer(ABC):
    """
    Abstract base class which will serve as a NN trainer
    """
    def __init__(self, dataset,
                 model,
                 knn_size,
                 lr=1e-4):
        """
        Initializes the trainer class
        :param dataset: torch Dataset object
        :param model: torch.nn object
        :param lr: float, learning rate
        """
        self.dataset = dataset
        self.model = model
        self.knn_size = knn_size
        self.optimizer = torch.optim.Adam(  # Adam is an alternate method for minimizing loss function
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        self.global_iter = 0
        self.trainer_config = ''
        self.writer = None
    

    def train_model(self, batch_size, num_epochs,filename,params, log=False, weightedLoss=False, purity_interval=100):
        """
        Trains the model
        :param batch_size: int,
        :param num_epochs: int,
        :param log: bool, logs epoch stats for viewing in tensorboard if TRUE
        :return: None
        """

        # set-up log parameters
        if log:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime(
                '%Y-%m-%d_%H:%M:%S'
            )
            # configure tensorboardX summary writer
            self.writer = SummaryWriter(
                logdir=os.path.join('runs/' + self.model.__repr__() + st)
            )


        # (generator_train, generator_val, _,) = self.dataset.data_loaders(batch_size=batch_size, drop_indices=False)
        (generator_train, generator_val, dist_train, dist_val, _,) = self.dataset.data_loaders(batch_size=batch_size, drop_indices=False)

        print(f'Num Train Batches: {len(generator_train)}')
        print(f'Num Validation Batches: {len(generator_val)}')

        
        print('Num Train Batches: ', len(generator_train))
        print('Num Valid Batches: ', len(generator_val))

        print('Uniqe Label Train: ', set(self.dataset.train_label.reshape(-1)))
        print('Uniqe Label Valid: ', set(self.dataset.val_label.reshape(-1)))

    

        distence = []
        distence_mun = []

        correlation = []
        correlation_valid = []
        def averageCols(logMat):
            dim = logMat.shape[1]
            rv = np.zeros((num_epochs, dim))
            for epoch in range(num_epochs):
                for col in range(1, dim):
                    num = 0
                    for row in range(logMat.shape[0]):
                        if(logMat[row, 0] == epoch):
                            rv[epoch, col] += logMat[row, col]
                            num += 1
                    rv[epoch, col] /= num
                rv[epoch,0] = epoch
            return rv
        # train epochs
        batch_pairs_train=[]
        batch_pairs_val=[]
        batch_pairs_full=[]
        p=0
        for i, (batch_inputs_train, batch_emb_train, batch_labels_train, batch_lengths_train, clabel_train, batch_indices_train) in enumerate(generator_train.dataset):
            input_tensor = batch_inputs_train
            index_tensor = batch_indices_train
            batch_pairs_train.append((input_tensor, batch_emb_train, batch_labels_train, clabel_train, batch_lengths_train))

        for i, (batch_inputs_val, batch_emb_val, batch_labels_val, batch_lengths_val, clabel_val, batch_indices_val) in enumerate(generator_val.dataset):
            input_tensor_val = batch_inputs_val
            index_tensor_val = batch_indices_val
            batch_pairs_val.append((input_tensor_val, batch_emb_val, batch_labels_val, clabel_val, batch_lengths_val))
        
        sum_w2_loss_smoothness_train=[]
        sum_w2_loss_smoothness_val=[]
        epoch_list=[]
        epoch_num_path = "epoch_num.txt"
        sum_w2_loss_smoothness_train_path = "sum_w2_loss_smoothness_train.txt"
        sum_w2_loss_smoothness_val_path = "sum_w2_loss_smoothness_val.txt"
        for epoch_index in range(num_epochs):
            purity_flag = False
            if purity_interval != -1 and epoch_index % purity_interval == 0:
                purity_flag = True
            # update training scheduler
            self.update_scheduler(epoch_index)
			
            # run training loop on training data
            self.model.train()
            w2_loss_smoothness=[]
##          epoch on train
            mean_loss_train, mean_accuracy_train,w2_loss_smoothness = self.loss_and_acc_on_epoch(w2_loss_smoothness,
                data_loader=generator_train,
                epoch_num=epoch_index,
                train=True,
                weightedLoss=weightedLoss,
                raw_dataset=batch_pairs_train, 
                adj_inds=None, 
                adj_weights=None,
                purity_flag=purity_flag,
                csdist=dist_train
            )
            #sum_w2_loss_smoothness_train.append(sum(w2_loss_smoothness))
            sum_w2_loss_smoothness_train.append(np.mean(w2_loss_smoothness))
            self.model.eval()

##          epoch on validation     
            mean_loss_val, mean_accuracy_val,w2_loss_smoothness = self.loss_and_acc_on_epoch(w2_loss_smoothness,
                data_loader=generator_val,
                epoch_num=epoch_index,
                train=False,
                weightedLoss=weightedLoss,
                raw_dataset=batch_pairs_val,
                adj_inds=None,
                adj_weights=None,
                purity_flag=purity_flag,
                csdist=dist_val
            )
            epoch_list.append(epoch_index)
            #sum_w2_loss_smoothness_val.append(sum(w2_loss_smoothness))
            sum_w2_loss_smoothness_val.append(np.mean(w2_loss_smoothness))
            #print("sum sumoothnes for validation on epoch",epoch_index+1,"is : " ,sum_w2_loss_smoothness_val)
            self.eval_model(
                data_loader=generator_val,
                epoch_num=epoch_index,
            )

            # log parameters
            if log:
                # log value in tensorboardX for visualization
                self.writer.add_scalar('loss/train', mean_loss_train, epoch_index)
                self.writer.add_scalar('loss/valid', mean_loss_val, epoch_index)
                self.writer.add_scalar('acc/train', mean_accuracy_train, epoch_index)
                self.writer.add_scalar('acc/valid', mean_accuracy_val, epoch_index)

            # print epoch stats
            data_element = {
                'epoch_index': epoch_index,
                'num_epochs': num_epochs,
                'mean_loss_train': mean_loss_train,
                'mean_accuracy_train': mean_accuracy_train,
                'mean_loss_val': mean_loss_val,
                'mean_accuracy_val': mean_accuracy_val
                }
            self.print_epoch_stats(**data_element)
            

            if epoch_index % 100 == (99):
                # pre = './GCN-modifiedVAE'
                pre = '.'
                torch.save(self.model, pre+"/models/weighted/" + filename +'e '+ str(epoch_index) + ".pt")
                par = np.array([ params["beta"], params["gamma"], params["delta"], 
                params["latentDims"], params["lstmLayers"], params["dropout"], params["hiddenSize"], purity_interval])
                tl = averageCols(self.trainList) 
                vl = averageCols(self.validList)
                print(" -- tl    ", tl.shape)
                print(" -- vl    ", vl.shape)
                np.savez(pre+"/runs/weighted/" + filename  +'e '+ str(epoch_index) + ".npz", par=par, tl=tl, vl=vl, distence=distence,distence_m=distence_mun,correlation=correlation,correlation_valid=correlation_valid,trainLabel = self.dataset.train_label,validLabel = self.dataset.val_label)
    
        with open(epoch_num_path, 'w') as epoch_file:
            for value in epoch_list:
                epoch_file.write(f"{value+1}\n")


        # print("smoothness   ", sum_w2_loss_smoothness_train)

        with open(sum_w2_loss_smoothness_train_path, 'w') as train_file:
            for value in sum_w2_loss_smoothness_train:
                train_file.write(f"{value}\n")

        with open(sum_w2_loss_smoothness_val_path, 'w') as val_file:
            for value in sum_w2_loss_smoothness_val:
                val_file.write(f"{value}\n")
            #for batch_num, batch in tqdm(enumerate(data_loader)):
             
        return distence,distence_mun,correlation, sum_w2_loss_smoothness_train, sum_w2_loss_smoothness_val
    def loss_and_acc_on_epoch(self,w2_loss_smoothness, data_loader, epoch_num=None, train=True, weightedLoss=False, probBins=[], raw_dataset=None, adj_inds=None, adj_weights=None, purity_flag=False, csdist=None):
         
        """
        Computes the loss and accuracy for an epoch
        :param data_loader: torch dataloader object
        :param epoch_num: int, used to change training schedule
        :param train: bool, performs the backward pass and gradient descent if TRUE
        :return: loss values and accuracy percentages
        """
        
        mean_loss = 0
        mean_accuracy = 0
        
        list_w2_loss_smoothness=[]
        
        # for batch_num, batch  in tqdm(enumerate(data_loader)):
        for batch_num, (batch_inputs_train, batch_emb_train, batch_labels_train, batch_lengths_train, batch_clabels_train, batch_indices_train) in tqdm(enumerate(data_loader)):
            ##################
            csdistance = csdist
            # new_X_train, new_emb_train, new_labels, new_seq_len, new_adj_inds_train,translation = cached_extract_connections(indices=batch_indices_train, raw_dataset=raw_dataset, X_batch=batch_inputs_train, X_emb=batch_emb_train, label_batch=batch_labels_train, adj_inds=adj_inds, adj_weights=adj_weights, seq_lengths=batch_lengths_train, knn_distances=csdistance, knn_size=self.knn_size)
            new_X_train, new_emb_train, new_labels, new_clabels, new_seq_len, new_adj_inds_train,translation = cached_extract_connections(indices=batch_indices_train, raw_dataset=raw_dataset, X_batch=batch_inputs_train, X_emb=batch_emb_train, label_batch=batch_labels_train, clabel_batch=batch_clabels_train, seq_lengths=batch_lengths_train, knn_distances=csdistance, knn_size=self.knn_size)
            ############################
            # process batch data
            batch_data = self.process_batch_data((new_X_train, new_labels))
            new_emb_train = to_cuda_variable_long(new_emb_train)
            # batch_data = to_cuda_variable_long(new_X_train)
            # print("epoch loop 1    ", len(new_seq_len))
            # print("epoch loop 2    ", len(batch_lengths_train))
            # zero the gradients
            self.zero_grad()

            # compute loss for batch
            loss, accuracy,myz = self.loss_and_acc_for_batch(
                batch_data,
                new_clabels,
                new_emb_train, 
                new_adj_inds_train,
                new_seq_len,
                epoch_num, batch_num, train=train,
                weightedLoss=weightedLoss,
                probBins = probBins,
                purity_flag = purity_flag
            )
            st_en=np.array(new_adj_inds_train)[:,:2]
            st_en_weight=np.array(new_adj_inds_train)[:,2:3]
            reverse_tr = {v: k for k, v in translation.items()}
            ## Find true index in our raw data
            orginal_edge_index = [[reverse_tr[int(src)], reverse_tr[int(dst)]] for src, dst in st_en]
            
            num=0
            w2_loss_smoothness=0
            squared_euclidean_distance=0


            
            for num in range(len(st_en)):    

                z_i = myz[int(st_en[num][0])]  
                z_j = myz[int(st_en[num][1])]  

                # Compute the squared Euclidean distance
                squared_euclidean_distance = torch.norm(z_i - z_j, p=2)**2
                w2_loss_smoothness=st_en_weight[num]*(squared_euclidean_distance.item())
                #list_w2_loss_smoothness.append(w2_loss_smoothness.item())
                list_w2_loss_smoothness.append(w2_loss_smoothness.item()/len(myz))
            if train:
                #loss.register_hook(lambda grad: print(grad))
                loss.backward()
                # self.plot_grad_flow()
                self.step()

            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            if accuracy is not None:
                mean_accuracy += to_numpy(accuracy)

        if len(data_loader) == 0:
            mean_loss = 0
            mean_accuracy = 0
        else:
            mean_loss /= len(data_loader) 
            mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy,list_w2_loss_smoothness
        )



    def cuda(self):
        """
        Convert the model to cuda
        """
        self.model.cuda()


    def zero_grad(self):
        """
        Zero the grad of the relevant optimizers
        :return:
        """
        self.optimizer.zero_grad()

    def step(self):

        """
        Perform the backward pass and step update for all optimizers
        :return:
        """
         # Clip gradients to prevent exploding gradients
         # Use clip_grad_norm_ or clip_grad_value_ depending on your needs
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        self.optimizer.step()

    def eval_model(self, data_loader, epoch_num):
        """
        This can contain any method to evaluate the performance of the mode
        Possibly add more things to the summary writer
        """
        pass

    def load_model(self):
        is_cpu = False if torch.cuda.is_available() else True
        self.model.load(cpu=is_cpu)
        if not is_cpu:
            self.model.cuda()
    



    @abstractmethod
    



    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True, weightedLoss=False, probBins = []):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param batch_num: int,
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        pass


    @abstractmethod
    # def process_batch_data(self, batch):
    #     batch_inputs, batch_labels, batch_indices = batch  # Unpack the batch
    #     return batch_inputs, batch_labels, batch_indices  # Return inputs, labels, and indices

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: torch Variable or tuple of torch Variable objects
        """
        pass

    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        pass

    @staticmethod
    def print_epoch_stats(
            epoch_index,
            num_epochs,
            mean_loss_train,
            mean_accuracy_train,
            mean_loss_val,
            mean_accuracy_val
    ):
        """
        Prints the epoch statistics
        :param epoch_index: int,
        :param num_epochs: int,
        :param mean_loss_train: float,
        :param mean_accuracy_train:float,
        :param mean_loss_val: float,
        :param mean_accuracy_val: float
        :return: None
        """
        print(
            f'Train Epoch: {epoch_index + 1}/{num_epochs}')
        print(f'\tTrain Loss: {mean_loss_train}'
              f'\tTrain Accuracy: {mean_accuracy_train * 100} %'
              )
        print(
            f'\tValid Loss: {mean_loss_val}'
            f'\tValid Accuracy: {mean_accuracy_val* 100} %'
        )

    @staticmethod
    def mean_crossentropy_loss(weights, targets):
        """
        Evaluates the cross entropy loss
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return: float, loss
        """
        criteria = nn.CrossEntropyLoss(reduction='mean')
        batch_size, seq_len, num_notes = weights.size()
        # print("weights    ", weights.shape)
        # print("targets    ", targets.shape)
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        # print("weights    ", weights)
        # print("targets    ", targets)
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_accuracy(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return float, accuracy
        """
        _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)

        #https://pytorch.org/docs/stable/generated/torch.max.html#torch.max
        _, max_indices = weights.max(1)
        correct = max_indices == targets
        return torch.sum(correct.float()) / targets.size(0)

    @staticmethod
    def mean_l1_loss_rnn(weights, targets):
        """
        Evaluates the mean l1 loss
        :param weights: torch Variable,
                (batch_size, seq_len, hidden_size)
        :param targets: torch Variable
                (batch_size, seq_len, hidden_size)
        :return: float, l1 loss
        """
        criteria = nn.L1Loss()
        batch_size, seq_len, hidden_size = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        assert (hidden_size == targets.size(2))
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_mse_loss_rnn(weights, targets):
        """
        Evaluates the mean mse loss
        :param weights: torch Variable,
                (batch_size, seq_len, hidden_size)
        :param targets: torch Variable
                (batch_size, seq_len, hidden_size)
        :return: float, l1 loss
        """
        criteria = nn.MSELoss()
        batch_size, seq_len, hidden_size = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        assert (hidden_size == targets.size(2))
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_crossentropy_loss_alt(weights, targets):
        """
        Evaluates the cross entropy loss
        :param weights: torch Variable,
                (batch_size, num_measures, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, num_measures, seq_len)
        :return: float, loss
        """
        criteria = nn.CrossEntropyLoss(reduction='mean')
        _, _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_accuracy_alt(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        :param weights: torch Variable,
                (batch_size, num_measures, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, num_measures, seq_len)
        :return float, accuracy
        """
        _, _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        _, max_indices = weights.max(1)
        correct = max_indices == targets
        return torch.sum(correct.float()) / targets.size(0)

    @staticmethod
    def compute_kld_loss(z_dist, prior_dist, beta, c=0.0):
        """

        :param z_dist: torch.distributions object
        :param prior_dist: torch.distributions
        :param beta: weight for kld loss
        :param c: capacity of bottleneck channel
        :return: kl divergence loss
        """
        kld = torch.distributions.kl.kl_divergence(z_dist, prior_dist)
        kld = kld.sum(1).mean()
        kld = beta * (kld - c).abs()
        return kld

    @staticmethod
    def compute_reg_loss(z, inputs,label,latent_code_mean, latent_code_std, edge_index, gamma, train, factor=1.0):
        """
        Computes the regularization loss
        """
        
        reg_loss = Trainer.reg_loss_sign(z, inputs,label, latent_code_mean, latent_code_std, edge_index,train= train, factor=factor)
        # print("sggggggsssssss      ", reg_loss.item())
        # print("gamma    ", gamma)
        return gamma * reg_loss
    

    @staticmethod
    def reg_loss_sign(latent_code, sequence,label, latent_code_mean, latent_code_std, edge_index,train, factor=1.0):
        """
        Computes the regularization loss given the latent code and attribute
        Args:
            latent_code: torch Variable, (N,)
            attribute: torch Variable, (N,)
            factor: parameter for scaling the loss
        Returns
            scalar, loss
        """

        # print("latent    ", latent_code.shape)
        # print("label    ", label.shape)
        # jjjjj += 9

        st_en=np.array(edge_index)[:,:2]
        st_en_weight=np.array(edge_index)[:,2:3]

        
        num=0
        losses=[]
        squared_euclidean_distance=0



        st_en_tensor = torch.tensor(st_en, dtype=torch.long)
        st_en_weight_tensor = torch.tensor(st_en_weight, dtype=torch.float32)

        if torch.cuda.is_available():
            st_en_tensor = st_en_tensor.cuda()
            st_en_weight_tensor = st_en_weight_tensor.cuda()

        z_i = latent_code[st_en_tensor[:, 0]]
        z_j = latent_code[st_en_tensor[:, 1]]

        squared_euclidean_distance = torch.norm(z_i - z_j, p=2, dim=1)**2

        losses = st_en_weight_tensor.squeeze() * squared_euclidean_distance / len(st_en)

        reg_loss = losses.sum()

        return reg_loss


    @staticmethod
    def compute_reg_loss_weighted(self, z,inputs, labels, reg_dim, gamma, alpha, factor=1.0, probBins=[]):
        """
        Computes the regularization loss
        """
        x = z[:, reg_dim]
        reg_loss = Trainer.reg_loss_sign_weighted(x, inputs, factor=factor)
        return gamma * reg_loss

    @staticmethod
    def reg_loss_sign_weighted(latent_code,sequence, factor=1.0):
        """
        Computes the regularization loss given the latent code and attribute
        Args:
            latent_code: torch Variable, (N,)
            attribute: torch Variable, (N,)
            factor: parameter for scaling the loss
        Returns
            scalar, loss
        """

        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        # attribute = Distencecs(attribute)#.view(-1, 1).repeat(1, attribute.shape[0])
        attribute_dist_mat = Distencecs(sequence)
        attribute_dist_mat = torch.tensor(attribute_dist_mat).reshape(-1, 1)

        # compute regularization loss
        loss_fn = torch.nn.L1Loss(reduction = "none")
        lc_tanh = torch.tanh(lc_dist_mat * factor)
        attribute_sign = torch.sign(attribute_dist_mat)
        #{ln}= { ∣xn−yn∣ }
        elementwise_L1loss = loss_fn(lc_tanh, attribute_sign.float())

        return torch.mean(elementwise_L1loss)#norm_weighted_loss

    
    

    @staticmethod
    def get_save_dir(model, sub_dir_name='results'):
        path = os.path.join(
            os.path.dirname(model.filepath),
            sub_dir_name
        )
        if not os.path.exists(path):
            os.makedirs(path)
        return path
