import torch
from dataclasses import dataclass
from transformers.utils import ModelOutput

@dataclass
class VAEOutput(ModelOutput):
    loss: torch.FloatTensor = None
    recon_loss: torch.FloatTensor = None
    kl_loss: torch.FloatTensor = None