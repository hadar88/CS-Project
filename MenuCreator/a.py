import json

with open("FoodsByID.json", "r") as file:
    foods = json.load(file)

breakfast = []
lunch = []
dinner = []

for food_id in foods:
    meals = foods[food_id]["Meals"]
    if "Breakfast" in meals:
        breakfast.append(food_id)
    if "Lunch" in meals:
        lunch.append(food_id)
    if "Dinner" in meals:
        dinner.append(food_id)

    
with open("MealsByType.json", "w") as file:
    json.dump({
        "Breakfast": breakfast,
        "Lunch": lunch,
        "Dinner": dinner
    }, file, indent=4)