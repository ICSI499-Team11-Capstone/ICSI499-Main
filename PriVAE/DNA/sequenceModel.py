
import torch
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
from utils.model import Model

class SequenceModel(Model):
    ALPHABET = ['A', 'C', 'G', 'T']

    def __init__(
        self,
        n_chars=4,
        seq_len=10,
        bidirectional=True,
        batch_size=32,
        hidden_layers=1,
        hidden_size=32,
        lin_dim=16,
        emb_dim=10,
        dropout=0,
    ):
        super(SequenceModel, self).__init__()
        self.n_chars = n_chars          # Number of DNA Bases, 4
        self.seq_len = seq_len          # Number of bases in a sequence, 10
        self.hidden_size = hidden_size  # Number of features hidden in the LSTM
        self.emb_dim = emb_dim          # Width of the embedded dimension
        self.lin_dim = lin_dim          # Width of the linear layer to transform the input and output of the latent space
        self.batch_size = batch_size    # Number of sequences in a batch

        # The encoder
        self.emb_lstm = torch.nn.LSTM(
            input_size=n_chars,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Two GCN layers
        # self.gcn1 = GCNConv(hidden_size * seq_len * 2, hidden_size * seq_len * 2) #GCNConv(hidden_size * seq_len * 2, 64)
        # self.gcn2 = GCNConv(hidden_size * seq_len * 2, hidden_size * seq_len * 2)  # Second GCN layer, GCNConv(64, 32), then torch.nn.Linear(32, lin_dim)

        self.gcn1 = GATConv(hidden_size * seq_len * 2, hidden_size * seq_len * 2)
        self.gcn2 = GATConv(hidden_size * seq_len * 2, hidden_size * seq_len * 2)

        self.latent_linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * seq_len * 2, lin_dim),
            torch.nn.ReLU()
        )

        self.latent_mean = torch.nn.Linear(lin_dim, emb_dim)
        self.latent_log_std = torch.nn.Linear(lin_dim, emb_dim)

        self.dec_lin = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU()
        )

        # The decoder
        self.dec_lstm = torch.nn.LSTM(
            input_size=1,
            num_layers=hidden_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
            hidden_size=hidden_size
        )

        self.dec_final = torch.nn.Linear(hidden_size * emb_dim * 2, n_chars * seq_len)

        self.xavier_initialization()

    # Encoder forward pass, takes one-hot encoded sequences x and returns q(x|z)
    def encode(self, x, edges):
        hidden, _ = self.emb_lstm(x.float())  # The _ contains unnecessary hidden and cell state info
        a, b, c = hidden.shape
        hidden = hidden.reshape(a, b * c)

        weights = np.array(edges)[:, 2].reshape(-1, 1)
        weights = torch.from_numpy(weights).float()
        ed = np.array(edges)[:, :2].astype(np.uint8).T
        ed = torch.from_numpy(ed).long()

        if torch.cuda.is_available():
            weights = weights.cuda()
            ed = ed.cuda()

        ##################################################
        # hidden = hidden.unsqueeze(0)
        # # Apply the first GCN layer
        # hidden = self.gcn1(hidden, edge_index=ed, edge_weight=weights)
    
        # # Apply the second GCN layer
        # hidden = self.gcn2(hidden, edge_index=ed, edge_weight=weights)

        ##################################################
        # Apply the first GAT layer
        # print(" -- hidd  ", hidden.dim())
        # print(" -- hidd  ", hidden.shape)
        hidden = self.gcn1(hidden, edge_index=ed, edge_attr=weights)
    
        # Apply the second GAT layer
        hidden = self.gcn2(hidden, edge_index=ed, edge_attr=weights)
        ##################################################

        hidden = hidden.squeeze(0)
        hidden = self.latent_linear(torch.flatten(hidden, 1))
        z_mean = self.latent_mean(hidden)
        z_log_std = self.latent_log_std(hidden)
        #z_log_std = torch.clamp(z_log_std, min=-10, max=10)

        return torch.distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std)), z_mean, z_log_std

    # Decoder forward pass, takes a latent sample z and returns x^hat encoded sequences
    def decode(self, z):
        hidden = self.dec_lin(z)
        hidden, _ = self.dec_lstm(hidden.view(-1, self.emb_dim, 1))
        out = self.dec_final(torch.flatten(hidden, 1))
        return out.view(-1, self.seq_len, self.n_chars)

    # Reparameterization trick for backwards pass
    def reparametrize(self, dist):
        sample = dist.rsample()
        prior = torch.distributions.Normal(torch.zeros_like(dist.loc), torch.ones_like(dist.scale))
        prior_sample = prior.sample()
        return sample, prior_sample, prior

    # Full forward pass for entire model
    def forward(self, x, edges):
        dist, z_mean, z_log_std = self.encode(x, edges)
        z, prior_sample, prior = self.reparametrize(dist)
        dec = self.decode(z)
        return dec, z, dist, prior, z_mean, z_log_std

    def __repr__(self):
        return 'SequenceVAE' + self.trainer_config
