import json

with open("FoodsById.json", "r") as file:
    foods = json.load(file)

breakfast = []
lunch = []
dinner = []

for food in foods:
    meals = foods[food]["Meals"]
    if "Breakfast" in meals:
        breakfast.append(food)
    if "Lunch" in meals:
        lunch.append(food)
    if "Dinner" in meals:
        dinner.append(food)

# load into json file
with open("MealsByType.json", "w") as file:
    json.dump({
        "Breakfast": breakfast,
        "Lunch": lunch,
        "Dinner": dinner
    }, file, indent=4)


