from fastapi import FastAPI
from pydantic import BaseModel
import requests
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import torch
from model_structure import ResUNet
from model_utils import load_model, load_config
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from dotenv import load_dotenv

load_dotenv()

class AnalysisModel(BaseModel):
    url: str

app = FastAPI()
config = load_config("./config.yaml")
model = ResUNet(       
    in_channels=config['model']['in_channels'],
    out_channels=config['model']['out_channels'],
    initial_feature=config['model']['initial_feature'],
    steps=config['model']['steps']
    )
model = load_model(model, "resunet-loveda") 

def llm(stats):
    API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv("HUGGING_FACE")}",
    }

    stats_lines = []
    for cls, data in stats.items():
        area = data["area_m2"]
        percent = data["percent"]
        stats_lines.append(f"{cls}: {area:.2f} square meters ({percent:.2f}%)")
    stats_text = "\n".join(stats_lines)

    prompt = (
        "Given the following land use statistics, write a detailed summary of around 200 words "
        "Use clear numbered sections as follows:\n"
        "1. Introduction to the area\n"
        "2. Detailed explanation for each land cover type\n"
        "3. Comparative analysis of dominant land covers\n"
        "4. Conclusion\n\n"
        f"Land use statistics:\n{stats_text}"
    )          


    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    response = query({
        "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional geospatial analyst. Always respond with a detailed, well-structured, "
                        "and comprehensive summary of land use statistics. Use numbered sections, elaborate on each class, "
                        "and maintain a consistent format."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "meta-llama/llama-3.3-70b-instruct"})

    return response


@app.post("/api/service/analysis")
def analysis(item: AnalysisModel):
    response = requests.get(item.url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image_resized = image.resize((512, 512), Image.BILINEAR)
    image_np = np.array(image_resized, dtype=np.float32) / 255.0
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)

    loveda_classes = [
        "Background", "Building", "Road", "Water", "Barren", "Forest", "Agriculture"
    ]

    with torch.inference_mode():
        label, output = model(image_tensor)

    label = torch.round(torch.sigmoid(label))
    probs = torch.sigmoid(label)      
    pred = torch.round(probs)
    pred = pred.item()

    output = output.squeeze(0)
    class_map = torch.argmax(output, dim=0)
    mask_np = class_map.cpu().numpy()

    pixel_area_m2 = 0.6 * 0.6
    total_pixels = mask_np.size
    total_area_m2 = total_pixels * pixel_area_m2

    unique, counts = np.unique(mask_np, return_counts=True)
    area_stats = {}
    for cls, count in zip(unique, counts):
        cls = int(cls)
        if cls >= len(loveda_classes):
            continue
        area = count * pixel_area_m2
        percent = (area / total_area_m2) * 100
        area_stats[loveda_classes[cls]] = {
            "area_m2": round(area, 2),
            "percent": round(percent, 2)
        }

    colormap = cm.get_cmap("tab20", len(loveda_classes)) 
    rgba_img = colormap(mask_np)[:, :, :3]
    rgb_img = (rgba_img * 255).astype(np.uint8)
    output_image = Image.fromarray(rgb_img)

    buffer = BytesIO()
    output_image.save(buffer, format="PNG")
    buffer.seek(0)
    base64_img = base64.b64encode(buffer.read()).decode("utf-8")

    analysis = llm(area_stats)

    model_output = analysis['choices'][0]['message']['content']

    return {
        "status": "Success",
      #  "label": pred,
        "analysis": model_output,
        "area_stats": area_stats,
        "segmentation_mask_base64": base64_img
    }