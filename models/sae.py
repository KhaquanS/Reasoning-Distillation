import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=65536):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        return F.relu(self.encoder(x))

    def forward(self, x):
        acts = self.encode(x)
        recon = self.decoder(acts)
        return recon, acts