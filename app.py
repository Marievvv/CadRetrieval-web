from flask import Flask, render_template, request, send_from_directory, url_for
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

@app.route('/data/FABWave/<path:category>/<path:filename>')
def serve_model_image(category, filename):
    return send_from_directory(os.path.join(data_path, category), filename)

def get_categories():
    return [d for d in os.listdir(data_path) 
            if os.path.isdir(os.path.join(data_path, d))]

def get_jpeg_models(category):
    jpeg_path = os.path.join(data_path, category, "JPEG")
    if not os.path.exists(jpeg_path):
        return []
    return [
        {
            'filename': f,
            'name': f.replace('.jpg', ''),
            'url': url_for('serve_model_image', category=f'{category}/JPEG', filename=f)
        }
        for f in os.listdir(jpeg_path)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ]

def get_models_in_category(category):
    bin_path = os.path.join(data_path, category, "bin")
    if not os.path.exists(bin_path):
        return []
    return [f.replace('.bin', '') for f in os.listdir(bin_path) if f.endswith('.bin')]

def find_similar_models(category, model_name, top_k):
    similar_models = []
    
    model_bin_path = os.path.join(data_path, category, "bin", f"{model_name}.bin")

    graph = load_graphs(model_bin_path)[0][0]
    # center_and_scale
    graph.ndata["x"], center, scale = util.center_and_scale_uvgrid(
        graph.ndata["x"], return_center_scale=True
    )
    graph.edata["x"][..., :3] -= center
    graph.edata["x"][..., :3] *= scale
    graph.ndata["x"] = graph.ndata["x"].type(FloatTensor)
    graph.edata["x"] = graph.edata["x"].type(FloatTensor)

    query_vector = model.predict_one(graph).cpu().numpy()
    retrieval_topk = db.search(query_vector, k=top_k)
    return retrieval_topk

def format_similar_models(results):
    formatted_list = []

    for item in results[0]: 
        model_name = item['name'].replace('.bin', '')
        category = item['label']
        jpeg_path = os.path.join(category, "JPEG", f"{model_name}.jpg")
        
        if os.path.exists(os.path.join(data_path, jpeg_path)):
            formatted_list.append({
                'name': model_name,
                'category': category,
                'jpeg_path': jpeg_path,
                'distance': item['distance']
            })
    return formatted_list

@app.route("/", methods=['GET', 'POST'])

def index():
    categories = get_categories()
    selected_category = None
    models = [] # для списка моделей, которые надо загружить после выбора категории
    selected_model = None
    top_k = 5
    similar_models = []
    error = None

    if request.method == 'POST':
        selected_category = request.form.get('category') or request.form.get('selected_category')
        top_k = int(request.form.get('top_k'))

        if selected_category:
            models = get_jpeg_models(selected_category)
            if not models:
                error = f"No jpeg models found in category {selected_category}"

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
