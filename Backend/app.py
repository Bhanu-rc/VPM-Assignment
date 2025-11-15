import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models
import faiss
from utils import load_products, download_image

app = Flask(__name__)
CORS(app)

DATA_DIR = "data"
PRODUCTS_FILE = os.path.join(DATA_DIR, "products.json")
INDEX_FILE = os.path.join("model_data", "index.faiss")
IDS_FILE = os.path.join("model_data", "ids.json")

# load products & index
products = load_products(PRODUCTS_FILE)
with open(IDS_FILE, 'r') as f:
    ids = json.load(f)
index = faiss.read_index(INDEX_FILE)

# model init
device = 'cpu'
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def image_to_embedding_pil(img_pil):
    x = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        v = model(x).squeeze().cpu().numpy()
    v = v / np.linalg.norm(v)
    return v.astype('float32')

@app.route('/products')
def get_products():
    # return metadata (censored path to be served by frontend or static hosting)
    # convert image path to server accessible URL if needed
    simplified = []
    for p in products:
        simplified.append({
            "id": p["id"],
            "name": p["name"],
            "category": p.get("category",""),
            "image": "/static_images/" + os.path.basename(p["image"]),
            "price": p.get("price", "")
        })
    return jsonify(simplified)

@app.route('/static_images/<filename>')
def static_image(filename):
    return send_from_directory(os.path.join(DATA_DIR,"images"), filename)

@app.route('/search', methods=['POST'])
def search():
    top_k = int(request.form.get('k', 10))
    # get image: file or URL
    if 'image' in request.files:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    else:
        url = request.form.get('image_url')
        if not url:
            return jsonify({"error":"No image provided"}), 400
        # download to temporary
        path = download_image(url, os.path.join("data","temp_query.jpg"))
        img = Image.open(path).convert('RGB')
    try:
        q = image_to_embedding_pil(img)
    except Exception as e:
        return jsonify({"error":str(e)}), 500
    q = np.expand_dims(q, axis=0).astype('float32')
    D, I = index.search(q, top_k)
    # D are inner products (cosine since normalized)
    results = []
    for score, idx in zip(D[0], I[0]):
        pid = ids[idx]
        # find product metadata
        p = next((x for x in products if x['id'] == pid), None)
        if p:
            results.append({
                "id": p['id'],
                "name": p['name'],
                "category": p.get('category'),
                "image": "/static_images/" + os.path.basename(p['image']),
                "score": float(score)
            })
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
