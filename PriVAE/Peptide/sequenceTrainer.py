
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import norm
from utils.trainer import Trainer
from utils.helpers import to_cuda_variable_long, to_cuda_variable, to_numpy
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances


#The Inherited Trainer Class
class SequenceTrainer(Trainer):
    REG_TYPE = {"integrated_intensity": 0}
    def __init__(
            self,
            dataset,
            model,
            lr=1e-4,
            reg_type = 'all',
            reg_dim = tuple([0]),
            beta=0.0015,
            gamma=1.0,
            capacity=0.0,
            rand=0,
            delta=10.0,
            knn_size=30,
            logTerms=False, #James's NPZ logging system
            IICorVsEpoch=False, #Flag to disable faulty cuda IICorVsEpoch Logging when running on GPU
            alpha = 5.0
    ):
        super(SequenceTrainer, self).__init__(dataset, model, knn_size, lr)
        
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
        if logTerms: 
            self.trainList = np.zeros((0,9)) #Training accuracy after each epoch
            self.validList = np.zeros((0,9)) #validation accuracy after each epoch
            self.IICorVsEpoch = IICorVsEpoch                   #Do we log II vs epoch or no?
            if IICorVsEpoch:
                self.WLCorList = np.zeros((0, self.model.emb_dim)) #II corelation of all dims after each epoch
                self.LIICorList = np.zeros((0, self.model.emb_dim)) #II corelation of all dims after each epoch
        
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


    @staticmethod
    def calculate_distances(latent_code, labels):
        unique_labels = np.unique(labels)
        # print("unique labels    ", unique_labels)
        intra_cluster_distances = []
        inter_cluster_distances = []

        for label in unique_labels:
            cluster_points = latent_code[labels == label]
            if len(cluster_points) > 1:
                intra_dist = cdist(cluster_points, cluster_points, metric='euclidean')
                intra_cluster_distances.append(np.mean(intra_dist))
        
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                cluster1 = latent_code[labels == label1]
                cluster2 = latent_code[labels == label2]
                inter_dist = cdist(cluster1, cluster2, metric='euclidean')
                inter_cluster_distances.append(np.mean(inter_dist))
        
        # return intra_cluster_distances, inter_cluster_distances
        return np.sum(intra_cluster_distances), np.sum(inter_cluster_distances)

    #This is our primary loss function 
    def loss_and_acc_for_batch(self, batch, new_clabels, emb_batch,edge_index, batch_lengths, epoch_num=None, batch_num=None, train=True, weightedLoss=False, purity_flag=False, probBins=[]):
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

        # print("batch     ", batch)
        inputs, labels = batch
        emb_inputs = emb_batch
        # print("inputs   ", inputs.shape)
        # print("emb inputs   ", emb_inputs.shape)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # outputs, latent_sample, z_dist, prior, z_mean, z_log_std = self.model(inputs, edge_index)
        outputs, latent_sample, z_dist, prior, z_mean, z_log_std = self.model(emb_inputs, edge_index)

        accs = []
        for i in range(len(inputs)):
            inp = inputs[i,:batch_lengths[i],:]
            inp = inp.unsqueeze(0)
            out = outputs[i,:batch_lengths[i],:]
            out = out.unsqueeze(0)
            # print("1    ", inp.shape)
            # print("2    ", out.shape)
            acc = self.mean_accuracy(out, inp)
            accs.append(acc.item())
            # accs.append(acc)
        # print("accs    ", accs)
        accuracy = np.array(np.mean(accs))
        # print("acc1   ", accuracy)
        accuracy = torch.from_numpy(accuracy)
        # print("acc2   ", accuracy)
        if torch.cuda.is_available():
            accuracy = accuracy.cuda()

        r_loss = 0
        for i in range(len(inputs)):
            inp = inputs[i,:batch_lengths[i],:]
            inp = inp.unsqueeze(0)
            out = outputs[i,:batch_lengths[i],:]
            out = out.unsqueeze(0)
            # print("1    ", inp.shape)
            # print("2    ", out.shape)
            r_l = self.reconstruction_loss(inp, out)
            r_loss += r_l
        #################################################################
        kld_loss = self.compute_kld_loss(z_dist, prior, beta=self.beta, c=self.capacity)
        loss = r_loss + kld_loss
        myz=z_mean
        #smmplDistence = self.Distencecs(labels)
        # compute and add regularization loss if needed, otherwise its just beta-VAE
        # if self.use_reg_loss:
        reg_loss = 0.0
        if type(self.reg_dim) == tuple:
            #for dim in self.reg_dim:
                #our supervized training
            if weightedLoss == False:
                # print("non weighted")
                temp = self.compute_reg_loss(
                    latent_sample,inputs,labels, z_mean, z_log_std, edge_index, gamma=self.gamma ,train = train, factor=self.delta )
                # print(" - rggggg loss    ", temp.item())
                reg_loss += temp
            else:
                # print("weighted")
                reg_loss += self.compute_reg_loss_weighted(
                self, latent_sample,inputs, labels 
                , gamma=self.gamma, alpha = self.alpha,
                factor=self.delta,
                probBins=probBins
                )
        else:
            raise TypeError("Regularization dimension must be a tuple of integers")
        
        # print("reg loss    ", reg_loss.item())
        loss += reg_loss

        # kkkk += 3
        #####
        intra_distances, inter_distances = SequenceTrainer.calculate_distances(latent_sample.detach().cpu().numpy(), labels.cpu().reshape(-1))

        purity = 0
        if purity_flag:
            # purity = SequenceTrainer.calculate_purity(latent_sample.detach().cpu().numpy(), labels.cpu().reshape(-1))
            purity = SequenceTrainer.calculate_purity(latent_sample.detach().cpu().numpy(), new_clabels)

        if self.logTerms and train:
            self.trainList = np.vstack((self.trainList, 
                [self.cur_epoch_num, r_loss.item(), kld_loss.item(),
                reg_loss.item(), loss.item(), accuracy.item(),intra_distances,inter_distances, purity]))
        if self.logTerms and not train:
            self.validList = np.vstack((self.validList, 
                [self.cur_epoch_num, r_loss.item(), kld_loss.item(),
                reg_loss.item(), loss.item(), accuracy.item(),intra_distances,inter_distances, purity]))

        return loss, accuracy,myz
    

    @staticmethod
    def calculate_purity(latent_code, class4_labels, k=30):
        """
        Calculate the Jaccard-based purity of each sample based on class4 labels.
        Args:
            latent_code: Latent representations of data points.
            class4_labels: Corresponding class4 labels (list of strings).
            k: Number of nearest neighbors to consider.
        Returns:
            Average Jaccard-based purity across all samples.
        """
        distances = pairwise_distances(latent_code)

        neighbor_purity_scores = []

        for i in range(latent_code.shape[0]):
            # print("args    ", np.argsort(distances[i]))
            neighbor_indices = np.argsort(distances[i])[1:k+1]
            # neighbor_labels = [set(label) for label in class4_labels[neighbor_indices]]
            neighbor_labels = []
            for lbl in neighbor_indices:
                neighbor_labels.append(class4_labels[lbl])
            # print(" neigjbour labels    ", neighbor_labels)
            neighbor_labels = list(set(neighbor_labels))
            current_label = set(class4_labels[i])
            # print("current label before  ", class4_labels[i])
            # print("current label   ", current_label)
            # print("neighbor label   ", neighbor_labels)
            jaccard_similarities = [
                len(current_label.intersection(neighbor)) / len(current_label.union(neighbor))
                for neighbor in neighbor_labels
            ]

            # print("sim 1   ", jaccard_similarities)
            # print("sim 2   ", np.mean(jaccard_similarities))
            # print("-----------------")

            neighbor_purity_scores.append(np.mean(jaccard_similarities))

        return np.mean(neighbor_purity_scores)



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
        # print("input 1   ", weights.shape)
        # print("input 2   ", target.shape)
        _,_,nn = weights.size()
        weights = weights.view(-1,nn)
        target = target.argmax(dim=2).view(-1)

        _, best = weights.max(1)
        # print("best    ", best)
        # print("target    ", target)
        correct = (best==target)
        # print("correct    ", torch.sum(correct.float()))
        # print("target    ", target.shape)
        # print("target    ", target.size(0))
        return torch.sum(correct.float())/target.size(0) 
