import os
import requests
from tqdm import tqdm

DATASET_FOLDER = "./dist/"
os.makedirs(DATASET_FOLDER, exist_ok=True)

def download_with_progress(url: str, filename: str) -> None:
    file_path = os.path.join(DATASET_FOLDER, filename)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 

    print(f"Downloading {filename}...")

    with open(file_path, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            f.write(data)
            bar.update(len(data))

    print(f"\n✅ Download complete: {file_path}")

data_types = ["Train", "Test", "Val"]
for data in data_types:
    download_with_progress(f"https://zenodo.org/records/5706578/files/{data}.zip?download=1", f"{data.lower()}.zip")
