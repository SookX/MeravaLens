import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class LoveDa(Dataset):
    """
    LoveDa Dataset for semantic segmentation of rural and urban satellite images.

    This dataset loads image-mask pairs along with their labels (rural=0, urban=1)
    from a specified directory structure. Images and masks are expected in PNG format.

    Directory structure example:
        folder_name/
            train/
                Rural/
                    images_png/
                    masks_png/
                Urban/
                    images_png/
                    masks_png/

    Args:
        folder_name (str): Root directory containing dataset splits.
        data_split (str): Dataset split to load, e.g., "train", "val", or "test".

    Attributes:
        samples (list): List of tuples (image_path, mask_path, label).
    """

    def __init__(self, folder_name, data_split):
        """
        Initialize the LoveDa dataset by scanning image and mask file paths.

        Args:
            folder_name (str): Root folder of the dataset.
            data_split (str): Dataset split, such as 'train', 'val', or 'test'.
        """
        super().__init__()
        self.folder_name = folder_name
        self.data_split = data_split
        self.path = os.path.join(folder_name, data_split)

        self.samples = []
        for label, region in enumerate(["Rural", "Urban"]):
            img_dir = os.path.join(self.path, region, "images_png")
            mask_dir = os.path.join(self.path, region, "masks_png")

            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith(".png"):
                    img_path = os.path.join(img_dir, fname)
                    mask_path = os.path.join(mask_dir, fname)
                    self.samples.append((img_path, mask_path, label))

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of image-mask-label samples.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Retrieve a sample from the dataset at the specified index.

        Loads the image and mask, converts to torch tensors, and returns them along
        with the associated label.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (image_tensor, mask_tensor, label)
                - image_tensor (torch.FloatTensor): RGB image tensor of shape (3, H, W) normalized to [0, 1].
                - mask_tensor (torch.LongTensor): Mask tensor of shape (H, W) with class indices.
                - label (int): Integer label representing the region class (0 for Rural, 1 for Urban).
        """
        img_path, mask_path, label = self.samples[index]

        image = Image.open(img_path).convert("RGB")
        image = image.resize((512, 512), Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((512, 512), Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.long)


        return image, mask, torch.tensor(label).to(torch.float32)


if __name__ == "__main__":
    dataset = LoveDa("./dist", "train")
    image, mask, label = dataset[0]

