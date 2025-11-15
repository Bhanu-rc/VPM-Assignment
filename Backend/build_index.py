import os
import json
import random
import requests
from PIL import Image
import torch
from torchvision import transforms
import faiss

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
PRODUCTS_FILE = os.path.join(DATA_DIR, "products.json")

MODEL_DIR = os.path.join(BASE_DIR, "model_data")
INDEX_FILE = os.path.join(MODEL_DIR, "index.faiss")
EMB_FILE = os.path.join(MODEL_DIR, "embeddings.npy")
IDS_FILE = os.path.join(MODEL_DIR, "ids.json")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- SAMPLE PRODUCT GENERATOR ----------
def create_sample_products(n=50):
    products = []
    for i in range(n):
        products.append({
            "id": i,
            "name": f"Product {i}",
            "category": "demo",
            "image_url": f"https://picsum.photos/seed/{i}/300/300"
        })
    with open(PRODUCTS_FILE, "w") as f:
        json.dump(products, f, indent=2)
    return products

# ---------- IMAGE DOWNLOAD ----------
def download_image(url, save_path):
    try:
        data = requests.get(url).content
        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except:
        return False

# ---------- MODEL LOADING ----------
def load_model():
    from torchvision.models import resnet50
    from torchvision.models import ResNet50_Weights
    
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    return torch.nn.Sequential(*list(model.children())[:-1])  # remove last layer

# ---------- EMBEDDING CREATION ----------
def extract_embedding(model, image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(x).squeeze().numpy()
    return emb

# ---------- MAIN BUILD FUNCTION ----------
def build_index():
    print("Generating sample products...")
    products = create_sample_products(50)

    print("Downloading images...")
    for p in products:
        save_path = os.path.join(IMAGES_DIR, f"{p['id']}.jpg")
        download_image(p["image_url"], save_path)

    print("Loading model...")
    model = load_model()

    print("Extracting embeddings...")
    embeddings = []
    ids = []

    for p in products:
        img_path = os.path.join(IMAGES_DIR, f"{p['id']}.jpg")
        emb = extract_embedding(model, img_path)
        embeddings.append(emb)
        ids.append(p["id"])

    embeddings = torch.tensor(embeddings).numpy()

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Saving index & metadata...")
    faiss.write_index(index, INDEX_FILE)
    with open(EMB_FILE, "wb") as f:
        import numpy as np
        np.save(f, embeddings)
    with open(IDS_FILE, "w") as f:
        json.dump(ids, f)

    print("Done! Index built successfully.")

# ---------- ENTRY ----------
if __name__ == "__main__":
    build_index()
