import torch
import torch.nn as nn
import torch.nn.functional as F
import math

################### DeepSets #######################

class DeepSets(nn.Module):
    def __init__(self, nfeat=32, nheads=256):
        super(DeepSets, self).__init__()

        self.nfeat = nfeat

        self.encoder = nn.Sequential(
            nn.Linear(2, nheads),
            nn.ELU(),
            nn.Linear(nheads, nheads),
            nn.ELU(),
            nn.Linear(nheads, nheads),
            nn.ELU(),
            nn.Linear(nheads, nheads),
            nn.ELU(),
            nn.Linear(nheads, nheads),
            nn.ELU(),
            nn.Linear(nheads, nfeat)
        )

        self.decoder = nn.Sequential(
            nn.Linear(nfeat + 1, 2*nheads),
            nn.ELU(),
            nn.Linear(2*nheads, 2*nheads),
            nn.ELU(),
            nn.Linear(2*nheads, 2*nheads),
            nn.ELU(),
            nn.Linear(2*nheads, 2*nheads),
            nn.ELU(),
            nn.Linear(2*nheads, 2*nheads),
            nn.ELU(),
            nn.Linear(2*nheads, 2)
        )

    def forward(self, x1, x2):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # shape x1 : batch, 3, 2
        # Doubles and quads separation
        count = torch.count_nonzero(x1[:, :, 0] + 1, dim=1)
        # Doubles
        if torch.any(count == 1):
            ind2 = torch.where(count == 1)
            doubles = x1[ind2][:, :-2]
            doubles = doubles.reshape(doubles.size(0), 2)
            doubles = self.encoder(doubles)
            doubles = doubles.view(-1, 1, self.nfeat)
            doubles = torch.mean(doubles, dim=1)
            x = doubles
        # Quads
        if torch.any(count == 3):
            ind4 = torch.where(count == 3)
            quads = x1[ind4]
            quads = quads.reshape(3 * quads.size(0), 2)
            quads = self.encoder(quads)
            quads = quads.view(-1, 3, self.nfeat)
            quads = torch.mean(quads, dim=1)
            x = quads

        # Doubles and quads recombination
        if torch.any(count == 1) and torch.any(count == 3):
            x = torch.zeros((x1.size(0), self.nfeat), device=device, dtype=doubles.dtype)
            x[ind2] = doubles
            x[ind4] = quads

        # Decoder
        x = torch.cat((x, x2), dim=1)

        # Decoding
        x = self.decoder(x)

        return x


################### Multilayer Perceptron #######################

class MLP(nn.Module):
    def __init__(self, nheads=256):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(3, nheads),
            nn.ELU(),
            nn.Linear(nheads, nheads),
            nn.ELU(),
            nn.Linear(nheads, nheads),
            nn.ELU(),
            nn.Linear(nheads, nheads),
            nn.ELU(),
            nn.Linear(nheads, nheads),
            nn.ELU(),
            nn.Linear(nheads, 2)
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.layers(x)

        return x


################### Set Transformer #########################

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(self, dim_input=2, num_outputs=1, dim_output=2,
                 num_inds=32, dim_heads=64, num_heads=6, ln=False):
        super(SetTransformer, self).__init__()
        dim_hidden = dim_heads * num_heads
        self.enc = nn.Sequential(
            SAB(dim_input, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        )
        self.pool = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden + 1, 2 * dim_hidden),
            nn.ELU(),
            nn.Linear(2 * dim_hidden, 2 * dim_hidden),
            nn.ELU(),
            nn.Linear(2 * dim_hidden, dim_output)
        )
        self.dim_hidden = dim_hidden

    def forward(self, x1, x2):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        #x1 shape = (batch size, 3, 2)

        # Doubles and quads separation
        count = torch.count_nonzero(x1[:, :, 0] + 1, dim=1)
        if torch.any(count == 1):
            ind2 = torch.where(count == 1)
            doubles = x1[ind2][:, :-2, :]
            doubles = self.enc(doubles)
            doubles = self.pool(doubles)
            x = doubles.squeeze(1)
        if torch.any(count == 3):
            ind4 = torch.where(count == 3)
            quads = x1[ind4]
            quads = self.enc(quads)
            quads = self.pool(quads)
            x = quads.squeeze(1)

        # Doubles and quads recombination
        if torch.any(count == 1) and torch.any(count == 3):
            x = torch.zeros((x1.size(0), self.dim_hidden), device=device, dtype=doubles.dtype)
            x[ind2] = doubles.squeeze(1)
            x[ind4] = quads.squeeze(1)

        # Concatening H0
        x = torch.cat((x, x2), dim=1)

        # Decoder
        x = self.dec(x)

        return x