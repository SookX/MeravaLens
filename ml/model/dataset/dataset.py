import torch
from torch.utils.data import Dataset
import os
import torchvision

class LoveDa(Dataset):
    def __init__(self, folder_name, data_split, transform = None):
        super().__init__()
        self.folder_name = folder_name
        self.data_split = data_split
        self.transform = transform
        self.path = os.path.join(folder_name, data_split)

    def __len__(self):
        urban_path = os.path.join(self.path, "Urban", "images_png")
        urban_length = len([f for f in os.listdir(urban_path) if os.path.isfile(os.path.join(urban_path, f))])

        rural_path = os.path.join(self.path, "Rural", "images_png")
        rural_length  = len([f for f in os.listdir(rural_path) if os.path.isfile(os.path.join(rural_path, f))])

        return urban_length + rural_length

    def __getitem__(self, index):
        pass        

if __name__ == "__main__":
    dataset = LoveDa("./dist", "train")
    print(len(dataset))