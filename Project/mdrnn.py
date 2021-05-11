"""
Define MDRNN model, refactored copy from 
https://github.com/ctallec/world-models/blob/master/models/mdrnn.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

class _MDRNNBase(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_gaussian):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_gaussian = n_gaussian

        self.gmm_linear = nn.Linear(
            n_hidden, (2 * n_input + 1) * n_gaussian + 2)

    def forward(self, *inputs):
        pass

class MDRNN(_MDRNNBase):
    """ MDRNN model for multi steps forward """
    def __init__(self, n_input, n_output, n_hidden, n_gaussian):
        super().__init__(n_input, n_output, n_hidden, n_gaussian)
        self.rnn = nn.LSTM(n_input + n_output, n_hidden)

    def forward(self, n_output, n_input):
        """ MULTI STEPS forward.
        :args n_output: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args n_input: (SEQ_LEN, BSIZE, LSIZE) torch tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = n_output.size(0), n_output.size(1)

        ins = torch.cat([n_output, n_input], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.n_gaussian * self.n_input

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.n_gaussian, self.n_input)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.n_gaussian, self.n_input)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.n_gaussian]
        pi = pi.view(seq_len, bs, self.n_gaussian)
        logpi = f.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds