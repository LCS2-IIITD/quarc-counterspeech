import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook, num_embeddings, embedding_dim, config):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._codebook_cost = config.codebook_cost
        
        self.codebook = nn.Parameter(codebook)
        self._commitment_cost = config.commitment_cost
        
        self.VERBOSE = config.verbose
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def forward(self, inputs, categories):
        if self.VERBOSE:
            print(f'In quantizer forward, inputs shape={inputs.shape}, categories shape={categories.shape}')
            
        input_shape = inputs.shape
        assert len(input_shape) == 2

        bs = input_shape[0]

        if len(categories.size()) == 2:
            categories.squeeze()

        categories_ohe = F.one_hot(categories, num_classes=self._num_embeddings).to(self.device).float()
        quantized = torch.matmul(categories_ohe, self.codebook)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = self._codebook_cost * q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss
