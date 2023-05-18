def get_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("[INFO] Total parameters: ", pytorch_total_params)
    pytorch_total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[INFO] Total trainable parameters: ", pytorch_total_train_params)