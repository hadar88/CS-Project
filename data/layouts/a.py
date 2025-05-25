import json

with open("MenusInput copy.json", "r") as file:
    data = json.load(file)

# new_data = {}

# for food_id in data:
#     values = data[food_id]["Initial"]

#     new_data[food_id] = {
#         "Calories": values["Calories"],
#         "Carbohydrate": values["Carbohydrate"],
#         "Sugars": values["Sugars"],
#         "Fat": values["Fat"],
#         "Protein": values["Protein"],
#         "Vegetarian": values["Vegetarian"],
#         "Vegan": values["Vegan"]
#     }


# with open("MenusInput copy.json", "w") as file:
#     json.dump(new_data, file, indent=4)