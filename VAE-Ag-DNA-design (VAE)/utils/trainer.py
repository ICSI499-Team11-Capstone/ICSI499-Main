#from contextlib import nullcontext
import os
import time
import datetime
from tqdm import tqdm

from abc import ABC, abstractmethod
import torch
from torch import nn  # Q: is this necessary to do?
import sequenceDataset as sd

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pcaVisualization
from utils.helpers import to_numpy  # Disregard syntax errors on this line
import numpy as np
import pandas as pd
import math

def process_data_file(path_to_dataset: str, sequence_length=10, prepended_name="processed", path_to_put=None, return_path=False):
    """Takes in a filepath to desired dataset and the sequence length of the sequences for that dataset,
    saves .npz file with arrays for one hot encoded sequences, array of wavelengths and array of local
    integrated intensities"""
    data = sd.SequenceDataset(path_to_dataset, sequence_length)
    ohe_sequences = data.transform_sequences(data.dataset['Sequence'][:].apply(lambda x: pd.Series([c for c in x])).
                                             to_numpy())  #One hot encodings in the form ['A', 'C', 'G', 'T']
    Wavelen = np.array(data.dataset['Wavelen'][:])
    LII = np.array(data.dataset['LII'][:])

    return {'Wavelen':Wavelen, "LII":LII,"ohe":ohe_sequences}
def process_data_file2(path_to_dataset: str, sequence_length=10, prepended_name="processed", path_to_put=None, return_path=False):
    """Takes in a filepath to desired dataset and the sequence length of the sequences for that dataset,
    saves .npz file with arrays for one hot encoded sequences, array of wavelengths and array of local
    integrated intensities"""
    valdict = np.load('utils/valdict.npy',allow_pickle=True)
    data = sd.SequenceDataset(path_to_dataset, sequence_length)
    ohe_sequences = data.transform_sequences(data.dataset['Sequence'][valdict].apply(lambda x: pd.Series([c for c in x])).
                                             to_numpy())  #One hot encodings in the form ['A', 'C', 'G', 'T']
    Wavelen = np.array(data.dataset['Wavelen'][valdict])
    LII = np.array(data.dataset['LII'][valdict])

    return {'Wavelen':Wavelen, "LII":LII,"ohe":ohe_sequences}

class Trainer(ABC):
    """
    Abstract base class which will serve as a NN trainer
    """
    def __init__(self, dataset,
                 model,
                 lr=1e-4):
        """
        Initializes the trainer class
        :param dataset: torch Dataset object
        :param model: torch.nn object
        :param lr: float, learning rate
        """
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(  # Adam is an alternate method for minimizing loss function
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        self.global_iter = 0
        self.trainer_config = ''
        self.writer = None

    def train_model(self, batch_size, num_epochs,filename, log=False, weightedLoss=False):
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

        # get dataloaders
        (generator_train,
         generator_val,
         _, probBins) = self.dataset.data_loaders(
            batch_size=batch_size
        )
        
        print('Num Train Batches: ', len(generator_train))
        print('Num Valid Batches: ', len(generator_val))
        train_dev_sets = torch.utils.data.ConcatDataset([generator_train, generator_val])
        print('Num all Batches: ', len(train_dev_sets))

        allres = []
        aliires = []
        allresv = []
        aliiresv = []
        # train epochs
        for epoch_index in range(num_epochs):
            # update training scheduler
            self.update_scheduler(epoch_index)

            # run training loop on training data
            self.model.train()
            mean_loss_train, mean_accuracy_train = self.loss_and_acc_on_epoch(
                data_loader=generator_train,
                epoch_num=epoch_index,
                train=True,
                weightedLoss=weightedLoss, 
                probBins = probBins
            )

            # run evaluation loop on validation data
            self.model.eval()
            mean_loss_val, mean_accuracy_val = self.loss_and_acc_on_epoch(
                data_loader=generator_val,
                epoch_num=epoch_index,
                train=False,
                weightedLoss=weightedLoss,
                probBins = probBins
            )

            self.eval_model(
                data_loader=generator_val,
                data_loader_tr= generator_train,
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


            npdata = process_data_file('data-and-cleaning/cleandata.csv')
            res,liires = pcaVisualization.conduct_visualizations(npdata, self.model, (False, False, True),epoch_index,'y',filename)
            allres.append(res)
            aliires.append(liires)
            try:
                npdata2 = process_data_file2('data-and-cleaning/cleandata.csv')
                resv,liiresv = pcaVisualization.conduct_visualizations(npdata2, self.model, (False, False, True),epoch_index,'b',filename)
                allresv.append(resv)
                aliiresv.append(liiresv)
            except:
                allresv.append(0)
                aliiresv.append(0)
            # save model
            #self.model.save()
        
            
        return allres,allresv,aliires,aliiresv
        
    def loss_and_acc_on_epoch(self, data_loader, epoch_num=None, train=True, weightedLoss=False, probBins=[]):
        """
        Computes the loss and accuracy for an epoch
        :param data_loader: torch dataloader object
        :param epoch_num: int, used to change training schedule
        :param train: bool, performs the backward pass and gradient descent if TRUE
        :return: loss values and accuracy percentages
        """
        mean_loss = 0
        mean_accuracy = 0
        for batch_num, batch in tqdm(enumerate(data_loader)):
            # process batch data
            batch_data = self.process_batch_data(batch)

            # zero the gradients
            self.zero_grad()

            # compute loss for batch
            loss, accuracy = self.loss_and_acc_for_batch(
                batch_data, epoch_num, batch_num, train=train,
                weightedLoss=weightedLoss,
                probBins = probBins
            )

            # compute backward and step if train
            if train:
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
            mean_accuracy
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
        self.optimizer.step()

    def eval_model(self, data_loader,data_loader_tr, epoch_num):
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
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
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
    def compute_reg_loss(z, labels, reg_dim, gamma, factor=1.0):
        """
        Computes the regularization loss
        """
        x = z[:, reg_dim]
        reg_loss = Trainer.reg_loss_sign(x, labels, factor=factor)
        return gamma * reg_loss

    @staticmethod
    def reg_loss_sign(latent_code, attribute, factor=1.0):
        """
        Computes the regularization loss given the latent code and attribute
        Args:
            latent_code: torch Variable, (N,)
            attribute: torch Variable, (N,)
            factor: parameter for scaling the loss
        Returns
            scalar, loss
        """
        # compute latent distance matrix
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
        attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

        # compute regularization loss
        loss_fn = torch.nn.L1Loss()
        lc_tanh = torch.tanh(lc_dist_mat * factor)
        attribute_sign = torch.sign(attribute_dist_mat)
        sign_loss = loss_fn(lc_tanh, attribute_sign.float())

        return sign_loss

    @staticmethod
    def compute_reg_loss_weighted(self, z, labels, reg_dim, gamma, alpha, factor=1.0, probBins=[]):
        """
        Computes the regularization loss
        """
        x = z[:, reg_dim]
        reg_loss = Trainer.reg_loss_sign_weighted(x, labels, probBins[reg_dim], alpha, factor=factor)
        return gamma * reg_loss

    @staticmethod
    def reg_loss_sign_weighted(latent_code, attribute, probability_bin, alpha, factor=1.0):
        """
        Computes the regularization loss given the latent code and attribute
        Args:
            latent_code: torch Variable, (N,)
            attribute: torch Variable, (N,)
            factor: parameter for scaling the loss
        Returns
            scalar, loss
        """

        attr_in_batch = attribute.numpy()

        attr_probability = []
        for i in range(len(attr_in_batch)):
            x = attr_in_batch[i]
            attr_probability.append(probability_bin.getProbability(x))

        # computer pairwise probability multiplication (p(i) * p(j))
        attr_probability = torch.tensor(np.array(attr_probability))
        mult_attr_prob = torch.outer(attr_probability, attr_probability)

        # multiply by alpha
        mult_attr_prob = torch.mul(mult_attr_prob, (-1 * alpha))

        # rise to the power of e
        weight_tensor = torch.exp(mult_attr_prob)

        # set the diagonal elements to zero to remove pi * pi in sum
        weights = weight_tensor.fill_diagonal_(0).view(-1, 1)
        # normalizing factor
        S = torch.sum(weights)

         # compute latent distance matrix
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
        attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

        # compute regularization loss
        loss_fn = torch.nn.L1Loss(reduction = "none")
        lc_tanh = torch.tanh(lc_dist_mat * factor)
        attribute_sign = torch.sign(attribute_dist_mat)
        #{ln}= { ∣xn−yn∣ }
        elementwise_L1loss = loss_fn(lc_tanh, attribute_sign.float())

        # multiply by weights
        elementwise_Weighted_loss = torch.mul(weights, elementwise_L1loss)

        # sum all weighted values
        total_weighted_loss = torch.sum(elementwise_Weighted_loss)

        # normalize by S
        norm_weighted_loss = total_weighted_loss / S

        return norm_weighted_loss

    @staticmethod
    def get_save_dir(model, sub_dir_name='results'):
        path = os.path.join(
            os.path.dirname(model.filepath),
            sub_dir_name
        )
        if not os.path.exists(path):
            os.makedirs(path)
        return path
