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

combinations = {
        "146": "181",
        "181": "146",
        "166": "132",
        "182": "183",
        "183": "182",
        "192": "146", 
        "217": "146",
        "33": "90",
        "90": "33",
    }

def generate_menu(nutrition_goals):
    nutrition_goals = [int(nutrition_goals[key]) for key in nutrition_goals]

    breakfast_values = [round(BREAKFAST * goal) for goal in nutrition_goals[:5]] 
    lunch_values = [round(LUNCH * goal) for goal in nutrition_goals[:5]]
    dinner_values = [round(DINNER * goal) for goal in nutrition_goals[:5]]

    menu = []

    for _ in range(7):
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
                if food_id in combinations and len(foods) < 10:
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


######


def transform(menu, food_data):
    output = [0.0] * 7
    daily_calories = [0.0] * 7

    output[5], output[6] = 1, 1  

    for didx, day in enumerate(menu):
        for midx, meal in enumerate(menu[day]):
            for food in menu[day][meal]:
                food_id = food
                food_amount = menu[day][meal][food_id] / 100.0 

                if food_id == 0:
                    continue

                food_nut = food_data[str(food_id)]
                food_calories = food_nut["Calories"] * food_amount

                # Update accumulators
                daily_calories[didx] += food_calories
                output[0] += food_calories
                output[1] += food_nut["Carbohydrate"] * food_amount
                output[2] += food_nut["Sugars"] * food_amount
                output[3] += food_nut["Fat"] * food_amount
                output[4] += food_nut["Protein"] * food_amount
                output[5] *= food_nut["Vegetarian"]
                # print(output[5])
                output[6] *= food_nut["Vegan"]

    output_final = output
    output_final[:5] = [val / 7.0 for val in output[:5]]
    output_final[5] = output[5]
    output_final[6] = output[6]

    return [int(round(val)) for val in output_final]

def convert_to_dictId(menu):
    days = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    meals = ["breakfast", "lunch", "dinner"]

    nested_data = menu.get("output", [])
    structured_dict = {}

    for i, group in enumerate(nested_data):
        group_key = days[i]
        structured_dict[group_key] = {}

        for j, sub_group in enumerate(group):
            sub_group_key = meals[j]
            structured_dict[group_key][sub_group_key] = {}

            for k, pair in enumerate(sub_group):
                structured_dict[group_key][sub_group_key][pair[0]] = round(pair[1])

    return structured_dict

def convert_to_dictName(menu):
    days = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    meals = ["breakfast", "lunch", "dinner"]

    nested_data = menu.get("output", [])
    structured_dict = {}

    for i, group in enumerate(nested_data):
        group_key = days[i]
        structured_dict[group_key] = {}

        for j, sub_group in enumerate(group):
            sub_group_key = meals[j]
            structured_dict[group_key][sub_group_key] = {}

            for k, pair in enumerate(sub_group):
                food_id = str(pair[0])
                name = foods_data[food_id]["Name"]
                structured_dict[group_key][sub_group_key][name] = round(pair[1])

    return structured_dict

def checkMenu(menu):
    with open("check_menu.json", "w") as file:
        json.dump(menu, file, indent=4)

# Example usage

nutrition_goals = {'calories': 2826.6875, 'carbohydrates': 326.16, 'sugar': 27.18, 'fat': 72.48, 'protein': 190.26, 'vegetarian': 0, 'vegan': 0}
menu = generate_menu(nutrition_goals)

menuId = convert_to_dictId(menu)
menuName = convert_to_dictName(menu)

m = transform(menuId, foods_data)
n = [round(ng) for ng in nutrition_goals.values()]
values = ["Calories", "Carbohydrate", "Sugars", "Fat", "Protein", "Vegetarian", "Vegan"]
for i, (m1, m2) in enumerate(zip(m, n)):
    print(f"{values[i]}: {m1:.0f} vs {m2:.0f}")

checkMenu(menuName)