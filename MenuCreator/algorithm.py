import random
import json
import numpy as np
from scipy.optimize import minimize
import pandas as pd

FOODS_DATA_PATH = "FoodsByID.json"
FOODS_ALTERNATIVES_PATH = "FoodAlternatives.json"
MEALS_PATH = "Meals.json"

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
    "141": ["172"], 
    "172": ["141"], 
    "173": ["174", "90"], 
    "174": ["173", "90"], 
    "183": ["141"], 
    "33": ["88"], 
    "88": ["33"], 
    "10": ["33", "88"], 
    "4": ["3"], 
    "36": ["3"], 
    "17": ["2"], 
    "50": ["2"], 
    "79": ["2"], 
    "71": ["90"], 
    "24": ["123"], 
    "84": ["1"], 
    "62": ["3", "8"], 
    "116": ["172"], 
    "60": ["170"], 
    "170": ["60"],
    "70": ["6"],
    "108": ["77", "107"],
    "6": ["21", "49"],
    "12": ["124"],
    "105": ["124"],
    "163": ["124"],
    "31": ["34", "76"], 
    "76": ["19", "34"],
    "110": ["19", "34"],
    "207": ["34", "76"],
    "7": ["33", "88"],
    "74": ["33", "88"],
    "89": ["33", "88"]
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
food_to_category = {}
for category, ids in only_one_of.items():
    for fid in ids:
        food_to_category[fid] = category

def generate_menu(nutrition_goals):
    nutrition_goals = [int(nutrition_goals[key]) for key in nutrition_goals]

    breakfast_values = [round(BREAKFAST * goal) for goal in nutrition_goals[:5]] 
    lunch_values = [round(LUNCH * goal) for goal in nutrition_goals[:5]]
    dinner_values = [round(DINNER * goal) for goal in nutrition_goals[:5]]

    menu = []

    for _ in range(7):
        day_plan = []

        for meal_type in ["breakfast", "lunch", "dinner"]:
            goal_values = breakfast_values if meal_type == "breakfast" else lunch_values if meal_type == "lunch" else dinner_values
            meal_foods = breakfast_meals if meal_type == "breakfast" else lunch_meals if meal_type == "lunch" else dinner_meals

            foods = set()
            used_categories = set()

            while len(foods) != 5:
                food_id = random.choice(meal_foods)
                category = food_to_category.get(food_id)
                if category:
                    if category in used_categories:
                        continue
                    used_categories.add(category)
                foods.add(food_id)

            foods_temp = set(foods)
            for food_id in foods_temp:
                if food_id in combinations:
                    food_combination = combinations[food_id]
                    foods.update(food_combination)

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
            
            foods = [int(food_id) for food_id in foods]
            meal = [[food_id, amount] for food_id, amount in zip(foods, amounts)]
            day_plan.append(meal)
            
        menu.append(day_plan)
        
    return {"output": menu}

def getAmounts(temp_food_values, parts, goal_values):
    A = np.array([np.array(v) / 100 for v in temp_food_values.values()]).T 
    
    def cost(w):
        nutrition_error = np.sum((A @ w - goal_values)**2)
        alpha = np.sum(w)
        ratio_error = np.sum((w - parts * alpha)**2)
        return nutrition_error + ratio_error 

    initial_alpha = 700
    w0 = parts * initial_alpha

    res = minimize(
        cost,
        w0,
        bounds=[(0, 300)]*len(w0),
        method='L-BFGS-B',
        options={'ftol': 1e-3, 'gtol': 1e-3}
    )
    
    return res.x

###### checking functions ######

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

def evaluate_menu(menu, nutrition_goals):
    score = 0

    ####### Load the menu into file #######

    menuName = convert_to_dictName(menu)
    with open("check_menu.json", "w") as file:
        json.dump(menuName, file, indent=4)

    ####### Print nutrition values comparison #######

    menuId = convert_to_dictId(menu)
    m = transform(menuId, foods_data)
    n = [round(ng) for ng in nutrition_goals.values()]
    values = ["Calories", "Carbohydrate", "Sugars", "Fat", "Protein", "Vegetarian", "Vegan"]

    calories_goal = nutrition_goals['calories']
    veggie = nutrition_goals['vegetarian']
    vegan = nutrition_goals['vegan']

    max_values = [
        int(1 * calories_goal), 
        int(0.65 * calories_goal / 4), 
        int(0.1 * calories_goal / 4), 
        int(0.35 * calories_goal / 9), 
        int(0.35 * calories_goal / 4),
        int(veggie),
        int(vegan)
    ]

    min_values = [
        int(1 * calories_goal), 
        int(0.45 * calories_goal / 4), 
        int(0 * calories_goal / 4), 
        int(0.25 * calories_goal / 9), 
        int(0.1 * calories_goal / 4),
        int(veggie),
        int(vegan)
    ]

    table = []

    for i, (m1, m2) in enumerate(zip(m, n)):
        diff = abs(m1 - m2)
        diff_max = abs(max_values[i] - m1)
        diff_min = abs(min_values[i] - m1)

        min_diff = min(diff_max, diff_min, diff)

        in_range = min_values[i] <= m1 <= max_values[i]

        if i == 0:
            in_range = None

        table.append((values[i], m1, m2, max_values[i], min_values[i], min_diff, in_range))

        if not in_range:
            score += min_diff ** 2

    df = pd.DataFrame(table, columns=["Nutrition", "Generated", "Goal", "Max", "Min", "Min Difference", "In Range"])

    print(df.to_string(index=False))

    ###### Evaluate the menu ######


    ########################


    score = np.sqrt(score)

    print(f"\nTotal Score: {score:.0f}")

# Example usage

nutrition_goals = {'calories': 2826.6875, 'carbohydrates': 326.16, 'sugar': 27.18, 'fat': 72.48, 'protein': 190.26, 'vegetarian': 0, 'vegan': 0}
menu = generate_menu(nutrition_goals)
evaluate_menu(menu, nutrition_goals)