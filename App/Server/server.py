from flask import Flask, jsonify, request
import torch
from utils import merge_ids_and_amounts

class Server:
    def __init__(self, model):
        self.model = model
        self.app = Flask(__name__)
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route("/")
        def home():
            return jsonify({"message": "Welcome to the NutriPlan API!"})
        
        @self.app.route("/wakeup", methods=["GET"])
        def wakeup():
            return jsonify({"message": "Server is awake!"})

        @self.app.route("/predict", methods=["POST"])
        def predict():
            data = request.json

            vec = []

            for key in data:
                vec.append(data[key])

            vec = torch.tensor([vec], dtype=torch.float32)

            pred_id, pred_amount = self.model(vec)

            pred_id, pred_amount = pred_id[0], pred_amount[0]

            pred_id = torch.argmax(pred_id, dim=-1)

            pred_amount = pred_amount.squeeze(-1)

            merged_pred = merge_ids_and_amounts(pred_id, pred_amount)

            return jsonify({"output": merged_pred.tolist()})

    def run(self, host="0.0.0.0", port=5000):
        self.app.run(host=host, port=port, debug=False)
