import torch
from unet.model import UNet
from unet.resunet import ResUNet
from utils import load_model, load_config
from dataset.dataset import LoveDa
import matplotlib.pyplot as plt


config = load_config("./config.yaml")

# Load model
model = ResUNet(    in_channels=config['model']['in_channels'],
    out_channels=config['model']['out_channels'],
    initial_feature=config['model']['initial_feature'],
    steps=config['model']['steps'])
model = load_model(model, "resunet-loveda")

# Load test dataset
test_dataset = LoveDa("./dataset/dist", "test")

image, _, label = test_dataset[65]


model.eval()
with torch.no_grad():
    input_tensor = image.unsqueeze(0).to(next(model.parameters()).device) 
    pred_cls, pred_seg = model(input_tensor)
    
    if pred_cls.dim() > 1:
        predicted_label = torch.argmax(pred_cls, dim=1).item()
    else:
        predicted_label = (pred_cls.squeeze() > 0.5).long().item()
    
    predicted_mask = torch.argmax(pred_seg, dim=1).squeeze().cpu().numpy()
    
    img_np = image.permute(1, 2, 0).cpu().numpy()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(img_np)
axs[0].set_title(f"Input Image\nTrue Class: {label.item()} | Pred Class: {predicted_label}")
axs[0].axis('off')

axs[1].imshow(predicted_mask, cmap='jet', interpolation='nearest')
axs[1].set_title("Predicted Segmentation Mask")
axs[1].axis('off')

plt.tight_layout()
plt.show()
