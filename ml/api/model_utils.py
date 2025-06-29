import torch
import yaml

def load_model(model, model_name, device='cpu'):
    checkpoint = torch.load(model_name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() 
    return model

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
