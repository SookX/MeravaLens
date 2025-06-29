import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataset import LoveDa
from unet.model import UNet
from unet.resunet import ResUNet

from utils import load_config, train_step, save_model
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load the configuration file
config = load_config("./config.yaml")

# Configure computation device
try:
    device = torch.device(config["training"]["device"])
except RuntimeError:
    device = torch.device("cpu")

print(f"[INFO] Training will run on: {device.type.upper()}")


# Set up training and validation datasets and data loaders
train_dataset = LoveDa("./dataset/dist", "train")
val_dataset = LoveDa("./dataset/dist", "val")

train_dataloder = DataLoader(train_dataset, config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_dataset, config["training"]["batch_size"], shuffle=False)

print("[INFO] All data loaders have been successfully initialized.")


# Initialize the model
model = ResUNet(
    in_channels=config['model']['in_channels'],
    out_channels=config['model']['out_channels'],
    initial_feature=config['model']['initial_feature'],
    steps=config['model']['steps']
).to(device)

print(f"[INFO] UNet model initialized with input channels={config['model']['in_channels']}, "
      f"output channels={config['model']['out_channels']}, and moved to {device.type.upper()}.")




optimizer = torch.optim.Adam(model.parameters(), float(config["training"]["learning_rate"]))
classification_loss = nn.BCEWithLogitsLoss()
segmentation_loss = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


train_step(model, config["training"]["epochs"], optimizer, classification_loss, segmentation_loss, scheduler , train_dataloder, val_dataloader, device)
