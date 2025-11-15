import os
import json
from PIL import Image
import requests
from io import BytesIO

def load_products(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_products(products, path):
    with open(path, 'w') as f:
        json.dump(products, f, indent=2)

def download_image(url, out_path):
    # small helper to download images
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert('RGB')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return out_path
