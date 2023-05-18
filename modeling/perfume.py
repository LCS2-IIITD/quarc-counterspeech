import torch
from torch import nn


class PerFuMe(nn.Module):
    def __init__(self, m1_size, m2_size, k=3, dropout=0.15, config=None):
        """
        the fusion module

        :param m1_size: the semantic embedding size
        :param m2_size: the syntactic embedding size
        :param k: number of persistent steps, defaults to 3
        :param dropout: the dropout value, defaults to 0.15
        """
        super().__init__()
        
        self.k = k
        self.w_s = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()
        
        for i in range(k):
            self.w_s.append(nn.Linear(m1_size+m2_size, m1_size))
            self.norms.append(nn.LayerNorm(m1_size))
    
        self.sem_weight1 = nn.Linear(m1_size, m1_size)
        self.sem_weight2 = nn.Linear(m1_size, m1_size)
        
        self.syn_weight1 = nn.Linear(m2_size, m1_size)
        self.syn_weight2 = nn.Linear(m2_size, m1_size)
        
        self.sigmoid = nn.Sigmoid()
        
        self.VERBOSE = config.verbose if config is not None else False
        
        
    def forward(self, sem_input, quantized):
        if self.VERBOSE: 
            print(f'Sem Input Shape = {sem_input.shape}, quantized_shape={quantized.shape}')
        
        z = sem_input
        quantized = quantized.squeeze()
        if len(quantized.size()) == 2:
            quantized = quantized.unsqueeze(1).repeat(1, sem_input.size(1), 1)
            
        for i in range(self.k):
            z = torch.cat([z, quantized], -1)
            z = self.w_s[i](z)
            z = self.act(z)
            z = self.dropout(z)
            z = z + sem_input
            z = self.norms[i](z)

        mu_sem = self.sigmoid(self.sem_weight1(sem_input)+self.syn_weight1(quantized))
        mu_form = self.sigmoid(self.sem_weight2(sem_input)+self.syn_weight2(quantized))

        z1 = mu_sem*sem_input + (1-mu_sem)*quantized
        z2 = (1-mu_form)*sem_input + mu_form*quantized
            
        z = z + z1*z2
        return z