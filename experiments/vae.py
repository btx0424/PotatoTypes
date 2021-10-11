import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
import pytorch_lightning as pl
from src.api import *
from torchvision.utils import make_grid



class VAE(pl.LightningModule):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.encoder = e
        self.decoder = d
        self.pz = pz

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_, q = self.forward(x)

        complexity = dists.kl_divergence(q, self.pz).mean(1).mean()
        likelihood = -F.binary_cross_entropy_with_logits(x_, x)

        loss = complexity - likelihood

        with torch.no_grad():
            self.log("train_complexity", complexity)
            self.log("train_likelihood", likelihood)
            if batch_idx%100==0:
                image = make_grid(
                    torch.cat((x[:4], torch.sigmoid(x_[:4])), dim=0).cpu(),
                    nrow=4,
                )
                self.logger.experiment.add_image("train_results", image, self.global_step)
        return loss

    def forward(self, x):
        q: dists.Distribution = self.encoder(x)
        z   = q.rsample().to(x.device)
        x_  = self.decoder(z)
        return x_, q
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def to(self, *args, **kwargs):
        self.pz = self.pz.__class__(
            self.pz.loc.to(*args, **kwargs),
            self.pz.scale.to(*args, **kwargs)
        )
        
        return super().to(*args, **kwargs)

