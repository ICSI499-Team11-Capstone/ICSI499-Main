#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:32:05 2020

@author: alexandergorovits
"""
#import os
#import json
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats

from utils.trainer import Trainer
#from sequenceModel import SequenceModel
from utils.helpers import to_cuda_variable_long, to_cuda_variable, to_numpy
#from utils.evaluation import *

#The Inherited Trainer Class
class SequenceTrainer(Trainer):
    REG_TYPE = {"integrated_intensity": 0}
    def __init__(
            self,
            dataset,
            model,
            lr=1e-4,
            reg_type = 'all',
            reg_dim = tuple([0, 1]),
            beta=0.001,
            gamma=1.0,
            capacity=0.0,
            rand=0,
            delta=10.0,
            logTerms=False, #James's NPZ logging system
            IICorVsEpoch=False, #Flag to disable faulty cuda IICorVsEpoch Logging when running on GPU
            alpha = 5.0
    ):
        super(SequenceTrainer, self).__init__(dataset, model, lr)
        
        self.attr_dict = self.REG_TYPE  
        self.reverse_attr_dict = {  #map of regularized dimension indices to their names
            v: k for k, v in self.attr_dict.items()
        }
        self.metrics = {}
        self.beta = beta    #The loss hyperparameters, 
        self.gamma = 0.0    # g and d are reset later in this constructor, update them there not here
        self.delta = 0.0
        self.capacity = to_cuda_variable(torch.FloatTensor([capacity])) 
        self.cur_epoch_num = 0      #The current Epoch while training
        self.warm_up_epochs = 10    #This dosn't do anything anywhere? CTRL+F
        self.reg_type = reg_type    #configures regularization settings later on
        self.reg_dim = ()           #The dimentions we're looking to regularize
        self.use_reg_loss = False   #use regularization loss, without this its just Beta VAE
        self.rand_seed = rand
        self.logTerms = logTerms 
        if logTerms: #use James's logging model
            self.trainList = np.zeros((0,6)) #Training accuracy after each epoch
            self.validList = np.zeros((0,6)) #validation accuracy after each epoch
            self.IICorVsEpoch = IICorVsEpoch                   #Do we log II vs epoch or no?
            if IICorVsEpoch:
                self.WLCorList = np.zeros((0, self.model.emb_dim)) #II corelation of all dims after each epoch
                self.LIICorList = np.zeros((0, self.model.emb_dim)) #II corelation of all dims after each epoch
                self.WLCorListv = np.zeros((0, self.model.emb_dim)) #II corelation of all dims after each epoch
                self.LIICorListv = np.zeros((0, self.model.emb_dim)) #II corelation of all dims after each epoch

                self.LIItauList = np.zeros((0, self.model.emb_dim)) #II corelation of all dims after each epoch
                self.WLtauList  = np.zeros((0, self.model.emb_dim)) #II corelation of all dims after each epoch
                self.LIItauListv = np.zeros((0, self.model.emb_dim)) #II corelation of all dims after each epoch
                self.WLtauListv  = np.zeros((0, self.model.emb_dim)) #II corelation of all dims after each epoch


        
        torch.manual_seed(self.rand_seed)
        np.random.seed(self.rand_seed)
        self.trainer_config = f'_r_{self.rand_seed}_b_{self.beta}_'
        if capacity != 0.0:
            self.trainer_config += f'c_{capacity}_'
        self.model.update_trainer_config(self.trainer_config)
        if len(self.reg_type) != 0: # meaning we're using an ARVAE, not just beta VAE
            self.use_reg_loss = True
            self.reg_dim = reg_dim
            self.gamma = gamma
            self.delta = delta
            self.alpha = alpha
            self.trainer_config += f'g_{self.gamma}_d_{self.delta}_'
            reg_type_str = '_'.join(self.reg_type)
            self.trainer_config += f'{reg_type_str}_'
            self.model.update_trainer_config(self.trainer_config)

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        #score tensor is actually one hot encodings
        score_tensor, attribTesnsor = batch
        # convert input to torch Variables
        batch_data = (
            to_cuda_variable_long(score_tensor),
            to_cuda_variable_long(attribTesnsor)
        )
        return batch_data


    #This is our primary loss function 
    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True, weightedLoss=False, probBins=[]):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param batch_num: int
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        if self.cur_epoch_num != epoch_num:
            flag = True
            self.cur_epoch_num = epoch_num
        else:
            flag = False

        inputs, labels = batch
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs, latent_dist, prior, latent_sample, _ = self.model(inputs)
        
        # compute accuracy
        accuracy = self.mean_accuracy(weights=outputs,target=inputs)
        r_loss = self.reconstruction_loss(inputs, outputs)
        kld_loss = self.compute_kld_loss(latent_dist, prior, beta=self.beta, c=self.capacity)
        loss = r_loss + kld_loss

        # compute and add regularization loss if needed, otherwise its just beta-VAE
        if self.use_reg_loss:
            reg_loss = 0.0
            if type(self.reg_dim) == tuple:
                for dim in self.reg_dim:
                    #our supervized training
                    if weightedLoss == False:
                        reg_loss += self.compute_reg_loss(
                            latent_sample, labels[:, dim], dim, gamma=self.gamma, factor=self.delta)
                    else:
                        reg_loss += self.compute_reg_loss_weighted(
                        self, latent_sample, labels[:, dim], 
                        dim, gamma=self.gamma, alpha = self.alpha,
                        factor=self.delta,
                        probBins=probBins
                    )
            else:
                raise TypeError("Regularization dimension must be a tuple of integers")
            loss += reg_loss

            #James's "worry about it later by throwing it all into a matrix" logging system for loss terms
            if self.logTerms and train:
                self.trainList = np.vstack((self.trainList, 
                    [self.cur_epoch_num, r_loss.item(), kld_loss.item(),
                    reg_loss.item(), loss.item(), accuracy.item()]))
            if self.logTerms and not train:
                self.validList = np.vstack((self.validList, 
                    [self.cur_epoch_num, r_loss.item(), kld_loss.item(),
                    reg_loss.item(), loss.item(), accuracy.item()]))

        return loss, accuracy

    def compute_representations(self, data_loader, num_batches=None):
        latent_codes = []
        attributes = []
        if num_batches is None:
            num_batches = 200
        for batch_id, batch in tqdm(enumerate(data_loader)):
            inputs, metadata = self.process_batch_data(batch)
            _, _, _, z_tilde, _ = self.model(inputs)
            latent_codes.append(to_numpy(z_tilde.cpu()))
            labels = metadata
            attributes.append(to_numpy(labels))
            if batch_id == num_batches:
                break
        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0)
        attr_list = [
            attr for attr in self.attr_dict.keys()
        ]
        return latent_codes, attributes, attr_list

    #This function is called after each epoch, 
    #we can use it for computing metrics of the model after each epoch.
    def eval_model(self, data_loader,data_loader_tr, epoch_num=0):
        if self.IICorVsEpoch:
            if self.logTerms and epoch_num % 50 == 0:  #compute this metric every 50 epochs
                print("Computing Corelation")
                #Compute the corelation with II after Each epoch
                dataset = data_loader.dataset
                datasett = data_loader_tr.dataset
                print(type(dataset))
                s = np.concatenate((dataset.tensors[0].numpy(),datasett.tensors[0].numpy()),axis=0)
                #one hot encodings of each sequence
                #Attribs for each sequence
                Wavelength = np.concatenate((dataset.tensors[1][:,0].numpy(),datasett.tensors[1][:,0].numpy()),axis=0)
                print('--------')
                print(Wavelength.shape)
                print('--------')

                LocalII = np.concatenate((dataset.tensors[1][:,1].numpy(),datasett.tensors[1][:,1].numpy()),axis=0)
                #Get latent representations from model
                #array of q(z|x) for each sequence
                q = self.model.encode(torch.from_numpy(s).float())
                #array of the mean of q(z|x) for each sequence, aka: 
                # the most likely sampled point for each q(z|x)
                #Note that since q(z|x) is a 10 dimensional probability distribution,
                # the mean will also be 10 dimensional. We are interested in seeing if one of
                #  these dimensions corresponds to integrated intensity
                z = q.loc.detach().numpy()
                #array of the corelation coefcients of each dimention with II for each sequence
                LIIcorrCoefs = np.zeros((self.model.emb_dim)) 
                WLcorrCoefs = np.zeros((self.model.emb_dim)) 
                for latentDimensionIndex in range(self.model.emb_dim):
                    WLcorrCoefs[latentDimensionIndex] = abs(np.corrcoef(Wavelength,z[:,latentDimensionIndex])[0, 1])
                    LIIcorrCoefs[latentDimensionIndex] = abs(np.corrcoef(LocalII,z[:,latentDimensionIndex])[0, 1])
                self.LIICorList = np.vstack((self.LIICorList, LIIcorrCoefs))
                self.WLCorList = np.vstack((self.WLCorList, WLcorrCoefs))
                print("Correlations")
                print("Wavelength: " + str(WLcorrCoefs[0]))
                print("Local II: " + str(LIIcorrCoefs[1]))

                sv = dataset.tensors[0].numpy()
                #one hot encodings of each sequence
                #Attribs for each sequence
                Wavelengthv = dataset.tensors[1][:,0].numpy()
                

                LocalIIv = dataset.tensors[1][:,1].numpy()
                #Get latent representations from model
                #array of q(z|x) for each sequence
                qv = self.model.encode(torch.from_numpy(sv).float())
                #array of the mean of q(z|x) for each sequence, aka: 
                # the most likely sampled point for each q(z|x)
                #Note that since q(z|x) is a 10 dimensional probability distribution,
                # the mean will also be 10 dimensional. We are interested in seeing if one of
                #  these dimensions corresponds to integrated intensity
                zv = qv.loc.detach().numpy()
                #array of the corelation coefcients of each dimention with II for each sequence
                LIIcorrCoefsv = np.zeros((self.model.emb_dim)) 
                WLcorrCoefsv = np.zeros((self.model.emb_dim)) 
                for latentDimensionIndexv in range(self.model.emb_dim):
                    WLcorrCoefsv[latentDimensionIndexv] = abs(np.corrcoef(Wavelengthv,zv[:,latentDimensionIndexv])[0, 1])
                    LIIcorrCoefsv[latentDimensionIndexv] = abs(np.corrcoef(LocalIIv,zv[:,latentDimensionIndexv])[0, 1])
                self.LIICorListv = np.vstack((self.LIICorListv, LIIcorrCoefsv))
                self.WLCorListv = np.vstack((self.WLCorListv, WLcorrCoefsv))
                

                
                
                LIItau = np.zeros((self.model.emb_dim)) 
                WLtau = np.zeros((self.model.emb_dim)) 
                for latentDimensionIndex in range(self.model.emb_dim):
                    WLtau[latentDimensionIndex] = abs(stats.kendalltau(Wavelength,z[:,latentDimensionIndex])[0])
                    LIItau[latentDimensionIndex] = abs(stats.kendalltau(LocalII,z[:,latentDimensionIndex])[0])
                self.LIItauList = np.vstack((self.LIItauList, LIItau))
                self.WLtauList = np.vstack((self.WLtauList, WLtau))
                print("tau")
                print("Wavelength: " + str(WLtau[0]))
                print("Local II: " + str(LIItau[1]))

               
                #array of the corelation coefcients of each dimention with II for each sequence
                LIItauv = np.zeros((self.model.emb_dim)) 
                WLtauv = np.zeros((self.model.emb_dim))
                for latentDimensionIndexv in range(self.model.emb_dim):
                    WLtauv[latentDimensionIndexv] = abs(stats.kendalltau(Wavelengthv,zv[:,latentDimensionIndexv])[0])
                    LIItauv[latentDimensionIndexv] = abs(stats.kendalltau(LocalIIv,zv[:,latentDimensionIndexv])[0])
                self.LIItauListv = np.vstack((self.LIItauListv, LIItauv))
                self.WLtauListv = np.vstack((self.WLtauListv, WLtauv))
                

    def loss_and_acc_test(self, data_loader):
        mean_loss = 0
        mean_accuracy = 0

        for _, batch in tqdm(enumerate(data_loader)):
            inputs, _ = self.process_batch_data(batch)
            inputs = to_cuda_variable(inputs)
            # compute forward pass
            outputs, _, _, _, _ = self.model(inputs)
            # compute loss
            recons_loss = self.reconstruction_loss(
                x=inputs, x_recons=outputs
            )
            loss = recons_loss
            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            accuracy = self.mean_accuracy(
                weights=outputs,
                target=inputs
            )
            mean_accuracy += to_numpy(accuracy)
        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    @staticmethod
    def reconstruction_loss(x, x_recons):
        return Trainer.mean_crossentropy_loss(weights=x_recons, targets=x.argmax(dim=2))

    @staticmethod
    def mean_accuracy(weights, target):
        _,_,nn = weights.size()
        weights = weights.view(-1,nn)
        target = target.argmax(dim=2).view(-1)

        _, best = weights.max(1)
        correct = (best==target)
        return torch.sum(correct.float())/target.size(0) 
