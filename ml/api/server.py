from fastapi import FastAPI
from pydantic import BaseModel
import requests
from PIL import Image
import numpy as np
import io
import torch
from model_structure import ResUNet
from model_utils import load_model, load_config


app = FastAPI()
config = load_config("./config.yaml")
model = ResUNet(       
    in_channels=config['model']['in_channels'],
    out_channels=config['model']['out_channels'],
    initial_feature=config['model']['initial_feature'],
    steps=config['model']['steps']
    )
model = load_model(model, "resunet-loveda") 


class AnalysisModel(BaseModel):
    url: str

@app.post("/api/service/analysis")
def analysis(item: AnalysisModel):
    response = requests.get(item.url)
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    image = image.resize((512, 512), Image.BILINEAR)
    image = np.array(image, dtype=np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1)
    image = image.unsqueeze(0)
    label, output = model(image)
    

