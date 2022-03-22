import torch
import torch.nn as nn


class DeepSets(nn.Module):
    def __init__(self, nfeat=32, p_drop=0):
        super(DeepSets, self).__init__()

        self.nfeat = nfeat

        self.encoder = nn.Sequential(
            nn.Linear(1, int(nfeat / 8)),
            nn.ReLU(),
            nn.Linear(int(nfeat / 8), int(nfeat / 4)),
            nn.ReLU(),
            nn.Linear(int(nfeat / 4), int(nfeat / 2)),
            nn.ReLU(),
            nn.Linear(int(nfeat / 2), nfeat)
        )
        self.decoder = nn.Sequential(
            nn.Linear(nfeat, int(2 * nfeat)),
            nn.ReLU(),
            nn.Linear(int(2 * nfeat), int(4 * nfeat)),
            nn.ReLU(),
            nn.Linear(int(4 * nfeat), int(2 * nfeat)),
            nn.ReLU(),
            nn.Linear(int(2 * nfeat), 2)
        )

    def forward(self, x):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Doubles and quads separation
        count = torch.count_nonzero(x + 1, dim=1)
        ind2 = torch.where(count == 2)
        ind4 = torch.where(count == 4)
        doubles = x[ind2][:, :2]
        quads = x[ind4]

        # Reshape to apply the encoder
        doubles = doubles.reshape(2 * doubles.size(0), 1)
        quads = quads.reshape(4 * quads.size(0), 1)

        # Encoding
        doubles = self.encoder(doubles)
        quads = self.encoder(quads)

        # Reshape to retrieve the time delay sets
        doubles = doubles.view(-1, 2, self.nfeat)
        quads = quads.view(-1, 4, self.nfeat)

        # Pooling over the time delays
        doubles = torch.sum(doubles, dim=1)
        quads = torch.sum(quads, dim=1)

        # Doubles and quads recombination
        x = torch.zeros((x.size(0), self.nfeat), device=device)
        x[ind2] = doubles
        x[ind4] = quads

        # Decoding
        x = self.decoder(x)

        return x


class DeepSets1(nn.Module):
    def __init__(self, nfeat=16, p_drop=0):
        super(DeepSets1, self).__init__()

        self.nfeat = nfeat

        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, nfeat)
        )
        
        self.latent = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(nfeat+1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x1, x2):

        # Shuffle time delays
        indices = torch.argsort(torch.rand(*x1.shape), dim=-1)
        x1 = x1[torch.arange(x1.shape[0]).unsqueeze(-1), indices]

        # Encoder
        batch_size = x1.size(0)
        x1 = x1.reshape(batch_size * x1.size(1), 1)
        x1 = self.encoder(x1)

        # Pooling
        x1 = x1.view(batch_size, -1, self.nfeat)
        x1 = torch.mean(x1, dim=1)
        
        # Conditionning on latent variable
        x2 = self.latent(x2)
        
        # Decoder
        x = torch.cat((x1, x2), dim=1)
        x = self.decoder(x)

        return x


class MLP(nn.Module):
    def __init__(self, p_drop=0):
        super(MLP, self).__init__()

        self.dt_layers = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.H0_layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.class_layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x1, x2):
        x1 = self.dt_layers(x1)
        x2 = self.H0_layers(x2)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.class_layers(x)

        return x
