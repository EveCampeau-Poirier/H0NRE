import torch
import torch.nn as nn


class DeepSets(nn.Module):
    def __init__(self, nfeat=16, nhead=128):
        super(DeepSets, self).__init__()

        self.nfeat = nfeat

        self.encoder = nn.Sequential(
            nn.Linear(1, nhead),
            nn.ReLU(),
            nn.Linear(nhead, nhead),
            nn.ReLU(),
            nn.Linear(nhead, nhead),
            nn.ReLU(),
            nn.Linear(nhead, nfeat)
        )

        self.latent = nn.Sequential(
            nn.Linear(1, nhead),
            nn.ReLU(),
            nn.Linear(nhead, nhead),
            nn.ReLU(),
            nn.Linear(nhead, nhead),
            nn.ReLU(),
            nn.Linear(nhead, 1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(nfeat + 1, 2*nhead),
            nn.ReLU(),
            nn.Linear(2*nhead, 2*nhead),
            nn.ReLU(),
            nn.Linear(2*nhead, 2*nhead),
            nn.ReLU(),
            nn.Linear(2*nhead, 2*nhead),
            nn.ReLU(),
            nn.Linear(2*nhead, 2*nhead),
            nn.ReLU(),
            nn.Linear(2*nhead, 2)
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