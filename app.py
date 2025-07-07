from flask import Flask, render_template, request
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from model.uvnet.models import Contrast
from model.retrieval.vector_db import VectorDatabase
from dgl.data.utils import load_graphs
from torch import FloatTensor
from model.datasets import util

app = Flask(__name__)

data_path = "data/FABWave"
vector_db_name = "FaBWave"
vector_db_folder = "model/retrieval"
model_weights = "model/best.ckpt"

model = Contrast.load_from_checkpoint(model_weights)
db = VectorDatabase(vector_db_folder, vector_db_name)

def get_categories():
    return [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

def get_jpeg_models(category):
    jpeg_path = os.path.join(data_path, category, "JPEG")
    if not os.path.exists(jpeg_path):
        return []
    return [f for f in os.listdir(jpeg_path) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

def get_models_in_category(category):
    bin_path = os.path.join(data_path, category, "bin")
    if not os.path.exists(bin_path):
        return []
    return [f.replace('.bin', '') for f in os.listdir(bin_path) if f.endswith('.bin')]


@app.route("/", methods=['GET', 'POST'])

def index():
    categories = get_categories()
    selected_category = None
    models = []
    selected_model = None
    top_k = 5
    similar_models = []
    error = None

    return render_template(
        "index.html",
        categories=categories,
        selected_category=selected_category,
        models=models,
        selected_model=selected_model,
        top_k=top_k,
        similar_models=similar_models,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7336)
