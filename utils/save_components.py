import torch
import os

def save_clime_components(model, save_path):
    sem_encoder = model.model.encoder.sem_encoder.to('cpu')
    fusion = model.model.encoder.fusion.to('cpu')
    decoder = model.model.decoder.to('cpu')
    codebook = model.model.encoder.quantizer.codebook.to('cpu')
    
    comp_path = os.path.join(save_path, 'components')
    if not os.path.exists(comp_path):
        os.makedirs(comp_path)
        
    torch.save(sem_encoder, os.path.join(comp_path, 'encoder.pt'))
    torch.save(fusion, os.path.join(comp_path, 'fusion.pt'))
    torch.save(decoder, os.path.join(comp_path, 'decoder.pt'))
    torch.save(codebook, os.path.join(comp_path, 'codebook.pt'))
    
    
def save_cogent_components(model, save_path):
    encoder = model.model.encoder.to('cpu')
    sem_encoder = model.model.encoder.sem_encoder.to('cpu')
    fusion = model.model.encoder.fusion.to('cpu')
    decoder = model.model.decoder.to('cpu')
    codebook = model.model.encoder.codebook.to('cpu')
    
    
    comp_path = os.path.join(save_path, 'components')
    if not os.path.exists(comp_path):
        os.makedirs(comp_path)
    
    torch.save(encoder, os.path.join(comp_path, 'encoder.pt'))
    torch.save(sem_encoder, os.path.join(comp_path, 'sem_encoder.pt'))
    torch.save(fusion, os.path.join(comp_path, 'fusion.pt'))
    torch.save(decoder, os.path.join(comp_path, 'decoder.pt'))
    torch.save(codebook, os.path.join(comp_path, 'codebook.pt'))