import torch
import torch.nn as nn


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
            doubles = x1[ind2][:, :2]
            doubles = doubles.reshape(doubles.size(0), 2)
            doubles = self.encoder(doubles)
            doubles = doubles.view(-1, 1, self.nfeat)
            doubles = torch.mean(doubles, dim=1)
        # Quads
        if torch.any(count == 3):
            ind4 = torch.where(count == 3)
            quads = x1[ind4]
            quads = quads.reshape(3 * quads.size(0), 2)
            quads = self.encoder(quads)
            quads = quads.view(-1, 3, self.nfeat)
            quads = torch.mean(quads, dim=1)

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