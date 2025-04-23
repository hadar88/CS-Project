import torch
import threading
import time
import requests
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import merge_ids_and_amounts
from flask import Flask, jsonify, request, send_file
from datetime import datetime, timedelta
import io


class Server:
    def __init__(self, model, food_names, char_vec, char_nn, word_vec, word_nn):
        self.model = model
        self.app = Flask(__name__)
        self.setup_routes()
        self.start_wakeup_thread()
        self.food_names = food_names
        self.char_vec = char_vec
        self.char_nn = char_nn
        self.word_vec = word_vec
        self.word_nn = word_nn

    def setup_routes(self):
        @self.app.route("/")
        def home():
            return jsonify({"message": "Welcome to the NutriPlan API!"})

        @self.app.route("/wakeup", methods=["GET"])
        def wakeup():
            return jsonify({"message": "Server is awake!"})

        @self.app.route("/search", methods=["GET"])
        def find_closest_foods():
            data = request.json
            query = data.get("query", None)

            if not query:
                return jsonify({"error": "Missing query parameter"}), 400

            query_char_vec = self.char_vec.transform([query])
            char_distances, char_indices = self.char_nn.kneighbors(query_char_vec)
            query_word_vec = self.word_vec.transform([query])
            word_distances, word_indices = self.word_nn.kneighbors(query_word_vec)

            combined = defaultdict(float)

            for i in range(len(char_distances[0])):
                char_food = self.food_names[char_indices[0][i]]
                word_food = self.food_names[word_indices[0][i]]

                char_distance = 1 - char_distances[0][i]
                word_distance = 1 - word_distances[0][i]

                combined[char_food] += 0.5 * char_distance
                combined[word_food] += 0.5 * word_distance

            sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)

            return jsonify({"results": [food for food, _ in sorted_results][:10]})

        @self.app.route("/wgraph", methods=["GET"])
        def make_graph():
            data: dict = request.json

            weights = data.get("weights", None)
            bmis = data.get("bmis", None)
            times = data.get("times", None)

            if not weights or not bmis or not times:
                return jsonify({"error": "Missing weights/bmis/times"}), 400

            times = [datetime.strptime(date, "%Y-%m-%d") for date in times]

            plt.figure()

            if len(bmis) == 1:
                # Plot one dot
                plt.scatter(
                    times[0], weights[0], color=self.__bmi_decs_and_color(bmis[0])[1]
                )

            for i in range(len(bmis) - 1):
                # color = self.__bmi_decs_and_color(bmis[i + 1])[1]

                b_m = (bmis[i + 1] - bmis[i]) / (
                    times[i + 1] - times[i]
                ).total_seconds()
                w_m = (weights[i + 1] - weights[i]) / (
                    times[i + 1] - times[i]
                ).total_seconds()

                t_0, w_0, b_0 = times[i], weights[i], bmis[i]
                t_1, w_1, b_1 = times[i + 1], weights[i + 1], bmis[i + 1]

                if b_0 < b_1:
                    levels = [16, 18.5, 25, 30, 40]
                else:
                    levels = [40, 30, 25, 18.5, 16]

                for level in levels:
                    # We should split the line into two lines if the BMI level is crossed
                    if b_0 < level < b_1 or b_1 < level < b_0:
                        # Calculate the intersection point
                        b_ = level
                        t_delta = (
                            b_ - b_0
                        ) / b_m  # This gives the time difference in seconds
                        t_ = t_0 + timedelta(
                            seconds=t_delta
                        )  # Add the time difference to t_0
                        w_ = w_0 + w_m * (t_ - t_0).total_seconds()

                        plt.plot(
                            [t_0, t_],
                            [w_0, w_],
                            color=self.__bmi_decs_and_color((b_0 + b_) / 2)[1],
                        )

                        t_0, w_0, b_0 = t_, w_, b_

                plt.plot(
                    [t_0, t_1],
                    [w_0, w_1],
                    color=self.__bmi_decs_and_color((b_0 + b_1) / 2)[1],
                )

            plt.xticks(
                [times[0], times[-1]],
                [times[0].strftime("%d-%m-%Y"), times[-1].strftime("%d-%m-%Y")],
                fontsize=15,
            )
            plt.yticks(fontsize=15)

            img_buffer = io.BytesIO()

            plt.savefig(img_buffer, format="png")
            plt.close()

            img_buffer.seek(0)

            return send_file(img_buffer, mimetype="image/png")

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

    def __bmi_decs_and_color(self, bmi_val):
        if bmi_val < 16:
            return ("Severely underweight", (0, 0, 1, 1))  # blue
        elif bmi_val < 18.5:
            return ("Underweight", (0, 1, 0.9, 1))  # cyan
        elif bmi_val < 25:
            return ("Healthy", (0, 0.5, 0, 1))  # green
        elif bmi_val < 30:
            return ("Overweight", (1, 0.9, 0, 1))  # yellow
        elif bmi_val < 40:
            return ("Obese", (1, 0.5, 0, 1))  # orange
        else:
            return ("Extremely obese", (1, 0, 0, 1))  # red

    def start_wakeup_thread(self):
        def send_wakeup_request():
            while True:
                try:
                    requests.get("http://cs-project-m5hy.onrender.com/wakeup")
                except Exception as e:
                    print(f"Failed to send wakeup request: {e}")
                time.sleep(720)

        # Start the thread as a daemon so it doesn't block server shutdown
        threading.Thread(target=send_wakeup_request, daemon=True).start()

    def run(self, host="0.0.0.0", port=5000):
        self.app.run(host=host, port=port, debug=False)
