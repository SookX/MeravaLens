import yaml
from tqdm import tqdm
import torch
import time
import logging
import gc



def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def valid_step():
    pass

def train_step(model, epochs, optimizer, classification_loss, segmentation_loss, train_dataloder, val_dataloader, device):
    model.train()
    logging.info("Initializing model training...\n")
    for epoch in tqdm(range(epochs)):
        start_time = time.time()

        total_loss_cls = 0.0
        total_loss_seg = 0.0
        total_correct_cls = 0
        total_samples = 0
        total_correct_seg = 0
        total_pixels_seg = 0

        for input_image, output_mask, label in train_dataloder:
            input_image = input_image.to(device)
            output_mask = output_mask.to(device)
            label = label.to(device)

            with torch.amp.autocast("cuda"):
                pred, output = model(input_image)
                
                loss_cls = classification_loss(pred.squeeze(1), label)
                loss_seg = segmentation_loss(output, output_mask)

                loss = loss_cls + loss_seg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_cls += loss_cls.item() * input_image.size(0)
            total_loss_seg += loss_seg.item()

            predicted_cls = (pred.squeeze(1) > 0.5).long() 
            total_correct_cls += (predicted_cls == label).sum().item()
            total_samples += input_image.size(0)

            predicted_seg = torch.argmax(output, dim=1)
            total_correct_seg += (predicted_seg == output_mask).sum().item()
            total_pixels_seg += output_mask.numel()

        elapsed = time.time() - start_time
        avg_loss_cls = total_loss_cls / total_samples
        avg_loss_seg = total_loss_seg / total_samples
        accuracy_cls = total_correct_cls / total_samples
        accuracy_seg = total_correct_seg / total_pixels_seg

        print(f"\nEpoch {epoch + 1}/{epochs} completed in {elapsed:.2f} seconds")
        print(f"Classification Loss: {avg_loss_cls:.4f}, Classification Accuracy: {accuracy_cls:.4f}")
        print(f"Segmentation Loss: {avg_loss_seg:.4f}, Segmentation Pixel Accuracy: {accuracy_seg:.4f}")
