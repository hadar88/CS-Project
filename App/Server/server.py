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
import random
from scipy.optimize import minimize
import numpy as np
import json

FOODS_DATA_PATH = "FoodsByID.json"
FOODS_ALTERNATIVES_PATH = "FoodAlternatives.json"
MEALS_PATH = "MealsByType.json"

BREAKFAST = 0.25
LUNCH = 0.4
DINNER = 0.35

with open(FOODS_DATA_PATH, "r") as file:
    foods_data = json.load(file)

with open(FOODS_ALTERNATIVES_PATH, "r") as file:
    foods_alternatives = json.load(file)

with open(MEALS_PATH, "r") as file:
    meals_by_type = json.load(file)

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
        self.breakfast_meals = meals_by_type["Breakfast"]
        self.lunch_meals = meals_by_type["Lunch"]
        self.dinner_meals = meals_by_type["Dinner"]

        self.foods_values = {}
        for food_id in foods_data:
            values = []
            food = foods_data[food_id]
            values.append(food["Calories"])
            values.append(food["Carbohydrate"])
            values.append(food["Sugars"])
            values.append(food["Fat"])
            values.append(food["Protein"])
            self.foods_values[food_id] = values
        self.combinations = {
            '141': ['172'], 
            '172': ['141'], 
            '173': ['174', '90'], 
            '174': ['173', '90'], 
            '183': ['141'], 
            '33': ['88'], 
            '88': ['33'], 
            '10': ['88'], 
            '4': ['3'], 
            '36': ['3'], 
            '17': ['2'], 
            '50': ['2'], 
            '79': ['2'], 
            '71': ['90'], 
            '24': ['123'], 
            '84': ['130'], 
            '62': ['3'], 
            '116': ['172'], 
            '60': ['170'], 
            '170': ['60']
            }
        only_one_of = {
            "Milk": ["1", "52", "72", "130", "200"],
            "Yogurt": ["3", "129", "192"],
            "Egg": ["37", "56", "68", "149", "186"],
            "Meat": ["6", "12", "26", "47", "73", "82", "105", "163"],
            "Bread": ["2", "48", "64", "90", "106"],
            "Pancake": ["172", "183"],
            "Tortilla": ["28", "65", "196"],
            "Fish": ["18", "54", "108", "159", "160"],
            "Cheese": ["22", "27", "38", "63", "80", "87", "95", "148", "152", "153"]
        }
        self.food_to_category = {}
        for category, ids in only_one_of.items():
            for fid in ids:
                self.food_to_category[fid] = category

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
            
            result = self.generate_menu(data)
            return jsonify(result)
        
        @self.app.route("/predict2", methods=["POST"])
        def predict2():
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

    def generate_menu(self, nutrition_goals):
        nutrition_goals = [int(nutrition_goals[key]) for key in nutrition_goals]

        breakfast_values = [round(BREAKFAST * goal) for goal in nutrition_goals[:5]] 
        lunch_values = [round(LUNCH * goal) for goal in nutrition_goals[:5]]
        dinner_values = [round(DINNER * goal) for goal in nutrition_goals[:5]]

        menu = []

        for i in range(7):
            day_plan = []

            for meal_type in ["breakfast", "lunch", "dinner"]:
                goal_values = breakfast_values if meal_type == "breakfast" else lunch_values if meal_type == "lunch" else dinner_values
                meal_foods = self.breakfast_meals if meal_type == "breakfast" else self.lunch_meals if meal_type == "lunch" else self.dinner_meals

                foods = set()
                used_categories = set()

                while len(foods) != 4:
                    food_id = random.choice(meal_foods)
                    category = self.food_to_category.get(food_id)
                    if category:
                        if category in used_categories:
                            continue
                        used_categories.add(category)
                    foods.add(food_id)

                foods_temp = set(foods)
                for food_id in foods_temp:
                    if food_id in self.combinations and len(foods) < 10:
                        food_combination = self.combinations[food_id]
                        foods.update(food_combination)

                foods_temp = set(foods)
                if nutrition_goals[5]:
                    for food_id in foods_temp:
                        if foods_data[food_id]["Vegetarian"] == 0:
                            food_alternatives = self.foods_alternatives[food_id]["Vegetarian"]
                            food = random.choice(food_alternatives)
                            foods.remove(food_id)
                            foods.add(str(food))

                foods_temp = set(foods)
                if nutrition_goals[6]:
                    for food_id in foods_temp:
                        if foods_data[food_id]["Vegan"] == 0:
                            food_alternatives = self.foods_alternatives[food_id]["Vegan"]
                            food = random.choice(food_alternatives)
                            foods.remove(food_id)
                            foods.add(str(food))
                        
                foods = list(foods)
                        
                parts = []
                for food_id in foods:
                    upper = foods_data[food_id]["Upper bound part"]
                    lower = foods_data[food_id]["Lower bound part"]
                    part = random.uniform(lower, upper)
                    parts.append(round(part, 1))
                sum_parts = sum(parts)
                parts = [round(part / sum_parts, 2) for part in parts]

                temp_food_values = {food_id: self.foods_values[food_id] for food_id in foods}

                goal_values = np.array(goal_values)
                parts = np.array(parts)
                amounts = np.round(self.getAmounts(temp_food_values, parts, goal_values))
                amounts = [int(amount) for amount in amounts]
                
                foods = [int(food_id) for food_id in foods]
                meal = [[food_id, amount] for food_id, amount in zip(foods, amounts)]
                day_plan.append(meal)
                
            menu.append(day_plan)
            
        return {"output": menu}
        
    def getAmounts(self, temp_food_values, parts, goal_values):
        # Precompute the nutrition matrix (transpose only once)
        A = np.array([v for v in temp_food_values.values()]).T / 100

        # Precompute repeated parts
        parts = np.array(parts)
        goal_values = np.array(goal_values)
        parts_outer = np.outer(parts, parts)

        # Precompute constants
        five = 5.0

        def cost(w):
            # Nutrition error
            Aw = A @ w
            nutrition_error = np.sum((Aw - goal_values) ** 2)

            # Ratio error
            alpha = np.sum(w)
            w_diff = w - parts * alpha
            ratio_error = np.dot(w_diff, w_diff)  # faster than sum of squares

            return nutrition_error + five * ratio_error

        # Initial guess
        initial_alpha = 700
        w0 = parts * initial_alpha

        # Use L-BFGS-B (faster convergence, still derivative-free)
        res = minimize(cost, w0, method='powell', bounds=[(0, None)] * len(w0))

        return res.x

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
