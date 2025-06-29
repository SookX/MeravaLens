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

def valid_step(model, classification_loss, segmentation_loss, val_dataloader, device):
    model.eval()
    total_loss_cls = 0.0
    total_loss_seg = 0.0
    total_correct_cls = 0
    total_samples = 0
    total_correct_seg = 0
    total_pixels_seg = 0

    with torch.no_grad():
        for input_image, output_mask, label in val_dataloader:
            input_image = input_image.to(device)
            output_mask = output_mask.to(device)
            label = label.to(device)

            with torch.amp.autocast("cuda"):
                pred, output = model(input_image)
                
                loss_cls = classification_loss(pred.squeeze(1), label)
                loss_seg = segmentation_loss(output, output_mask)

            total_loss_cls += loss_cls.item() * input_image.size(0)
            total_loss_seg += loss_seg.item() * input_image.size(0)

            predicted_cls = (torch.sigmoid(pred.squeeze(1)) > 0.5).long() 
            total_correct_cls += (predicted_cls == label).sum().item()
            total_samples += input_image.size(0)

            predicted_seg = torch.argmax(output, dim=1)
            total_correct_seg += (predicted_seg == output_mask).sum().item()
            total_pixels_seg += output_mask.numel()

    avg_loss_cls = total_loss_cls / total_samples
    avg_loss_seg = total_loss_seg / total_samples
    accuracy_cls = total_correct_cls / total_samples
    accuracy_seg = total_correct_seg / total_pixels_seg

    print(f"Classification Validation Loss: {avg_loss_cls:.4f} | Classification Validation Accuracy: {accuracy_cls:.4f}")
    print(f"Segmentation Validation Loss: {avg_loss_seg:.4f} | Segmentation Pixel Validation Accuracy: {accuracy_seg:.4f}")


def train_step(model, epochs, optimizer, classification_loss, segmentation_loss, scheduler , train_dataloder, val_dataloader, device):
    model.train()
    torch.autograd.set_detect_anomaly(True)
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

                loss = 25*loss_seg + 5*loss_cls

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_cls += loss_cls.item() * input_image.size(0)
            total_loss_seg += loss_seg.item()

            predicted_cls = (pred.squeeze(1) > 0.5).long() 
            total_correct_cls += (predicted_cls == label).sum().item()
            total_samples += input_image.size(0)

            predicted_seg = torch.argmax(output, dim=1)
            total_correct_seg += (predicted_seg == output_mask).sum().item()
            total_pixels_seg += output_mask.numel()
        scheduler.step()
        elapsed = time.time() - start_time
        avg_loss_cls = total_loss_cls / total_samples
        avg_loss_seg = total_loss_seg / total_samples
        accuracy_cls = total_correct_cls / total_samples
        accuracy_seg = total_correct_seg / total_pixels_seg

        print(f"\nEpoch {epoch + 1}/{epochs} completed in {elapsed:.2f} seconds")
        print(f"Classification Train Loss: {avg_loss_cls:.4f} | Classification Train Accuracy: {accuracy_cls:.4f}")
        print(f"Segmentation Train Loss: {avg_loss_seg:.4f} | Segmentation Pixel Train Accuracy: {accuracy_seg:.4f}")
        valid_step(model, classification_loss, segmentation_loss, val_dataloader, device)
        save_model(model, "resunet-loveda")


def save_model(model, model_name):
    checkpoint = {
        'model_state_dict': model.state_dict()
    }
    torch.save(checkpoint, model_name)

def load_model(model, model_name, device='cpu'):
    checkpoint = torch.load(model_name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() 
    return model