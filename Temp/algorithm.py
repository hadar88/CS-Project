import random
import json
import numpy as np
from scipy.optimize import minimize

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

breakfast_meals = meals_by_type["Breakfast"]
lunch_meals = meals_by_type["Lunch"]
dinner_meals = meals_by_type["Dinner"]

foods_values = {}
for food_id in foods_data:
    values = []
    food = foods_data[food_id]
    values.append(food["Calories"])
    values.append(food["Carbohydrate"])
    values.append(food["Sugars"])
    values.append(food["Fat"])
    values.append(food["Protein"])
    foods_values[food_id] = values

def generate_menu(nutrition_goals):
    combinations = {
        "146": "181",
        "181": "146",
        "166": "132",
        "182": "183",
        "183": "182"
    }

    breakfast_values = [round(BREAKFAST * goal) for goal in nutrition_goals[:5]] 
    lunch_values = [round(LUNCH * goal) for goal in nutrition_goals[:5]]
    dinner_values = [round(DINNER * goal) for goal in nutrition_goals[:5]]

    menu = []

    for day in range(7):
        day_plan = []

        for meal_type in ["breakfast", "lunch", "dinner"]:
            values_totals = [0, 0, 0, 0, 0]

            goal_values = breakfast_values if meal_type == "breakfast" else lunch_values if meal_type == "lunch" else dinner_values
            meal_foods = breakfast_meals if meal_type == "breakfast" else lunch_meals if meal_type == "lunch" else dinner_meals

            foods = set()

            while len(foods) != 5:
                food_id = random.choice(meal_foods)
                foods.add(food_id)

            foods_temp = set(foods)
            for food_id in foods_temp:
                if food_id in combinations:
                    foods.add(combinations[food_id])

            foods_temp = set(foods)
            if nutrition_goals[5]:
                for food_id in foods_temp:
                    if foods_data[food_id]["Vegetarian"] == 0:
                        food_alternatives = foods_alternatives[food_id]["Vegetarian"]
                        food = random.choice(food_alternatives)
                        foods.remove(food_id)
                        foods.add(str(food))

            foods_temp = set(foods)
            if nutrition_goals[6]:
                for food_id in foods_temp:
                    if foods_data[food_id]["Vegan"] == 0:
                        food_alternatives = foods_alternatives[food_id]["Vegan"]
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

            temp_food_values = {food_id: foods_values[food_id] for food_id in foods}

            goal_values = np.array(goal_values)
            parts = np.array(parts)
            amounts = np.round(getAmounts(temp_food_values, parts, goal_values))
            amounts = [int(amount) for amount in amounts]

            for i, food_id in enumerate(foods):
                for j in range(5):
                    values_totals[j] += amounts[i] * foods_values[food_id][j] / 100

            values_totals = np.round(values_totals)
            values_totals = [int(value) for value in values_totals]
            
            foods = [int(food_id) for food_id in foods]
            meal = [[food_id, amount] for food_id, amount in zip(foods, amounts)]
            day_plan.append(meal)
            
        menu.append(day_plan)
        
    return {"output": menu}

def getAmounts(temp_food_values, parts, goal_values):
    A = np.array([np.array(v) / 100 for v in temp_food_values.values()]).T 

    def cost(w, A, goal_values, parts, lambd):
        nutrition_error = np.sum((A @ w - goal_values)**2)
        alpha = np.sum(w)
        ratio_error = np.sum((w - parts * alpha)**2)
        return nutrition_error + lambd * ratio_error

    initial_alpha = 700
    w0 = parts * initial_alpha

    lambd = 5
    res = minimize(cost, w0, args=(A, goal_values, parts, lambd), bounds=[(0, None)]*len(w0), method='L-BFGS-B')

    final_weights = res.x
    return final_weights

# Example usage

nutrition_goals = [2826, 326, 27, 72, 190, 1, 0]
menu = generate_menu(nutrition_goals)
print(f"Menu: {menu}")